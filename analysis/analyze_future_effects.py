import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import os
import json
import re
import shutil
import random
import argparse
from typing import Optional

import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import LegalDataset
from lib.datasets.legal import (
    CLASSIFICATION_OPTIONS,
    ARABIC_TO_ENGLISH_CLASSIFICATION,
    normalize_classification,
)
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, get_lm_head, set_eval


# ── Module-level config (set from CLI in main) ────────────────────────
# Used by helper _make_dataset() so every place we instantiate
# LegalDataset gets the same language/path consistently.
_language: str = "english"
_data_path: Optional[str] = None


def _make_dataset() -> LegalDataset:
    """Build a LegalDataset using the language/data_path chosen via CLI."""
    kwargs = {"language": _language}
    if _data_path is not None:
        kwargs["json_path"] = _data_path
    return LegalDataset(**kwargs)


# ── Plotting helpers ──────────────────────────────────────────────────

def plot_layer_diffs(dall):
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(dall.float().cpu().numpy(), vmin=0, vmax=1, interpolation="nearest")
    plt.ylabel("Layer skipped")
    plt.xlabel("Effect @ layer")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.2, pad=0.1)
    fig.colorbar(im, cax=cax, label='Relative change')
    return fig


def plot_logit_diffs(dall):
    fig = plt.figure(figsize=(6, 3))
    dall = dall.squeeze()
    plt.bar(list(range(dall.shape[0])), dall)
    plt.xlim(-1, dall.shape[0])
    plt.xlabel("Layer")
    plt.ylabel("Output change norm")
    return fig


def plot_accuracy_bars(accuracies, baseline_acc, n_samples, model_name):
    """Plot classification accuracy per layer skipped."""
    fig = plt.figure(figsize=(8, 3))
    plt.bar(list(range(len(accuracies))), accuracies)
    plt.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Baseline (no skip): {baseline_acc:.2f}')
    plt.xlim(-0.5, len(accuracies) - 0.5)
    plt.ylim(0, 1.05)
    plt.xlabel("Layer skipped")
    plt.ylabel("Classification accuracy")
    plt.title(f"{model_name}: verdict classification accuracy per skipped layer "
              f"(n={n_samples})")
    plt.legend(loc='lower right')
    return fig


# ── Intervention helpers ──────────────────────────────────────────────

def _seq_dim(t):
    return t.dim() - 2


def _align_dims(a, b):
    while a.dim() < b.dim():
        a = a.unsqueeze(0)
    while a.dim() > b.dim():
        a = a.squeeze(0)
    return a


def merge_io(intervened, orig, t=None, no_skip_front=1):
    intervened = _align_dims(intervened, orig)
    sd = _seq_dim(orig)

    def sl(start, stop):
        s = [slice(None)] * orig.dim()
        s[sd] = slice(start, stop)
        return tuple(s)

    parts = [orig[sl(None, no_skip_front)]]
    if t is not None:
        parts.append(intervened[sl(no_skip_front, t)].to(orig.device))
        parts.append(orig[sl(t, None)])
    else:
        parts.append(intervened[sl(no_skip_front, None)].to(orig.device))
    return torch.cat(parts, dim=sd)


def apply_intervention(layer, t, part, no_skip_front, input_attr, precomputed_mlp=None):
    if input_attr == "input":
        layer_in = layer.input[0]
    else:
        layer_in = layer.inputs[0][0]

    if part == "layer":
        raw_output = layer.output
        if isinstance(raw_output, tuple):
            hidden = raw_output[0]
            new_hidden = merge_io(layer_in, hidden, t, no_skip_front)
            layer.output[0][:] = new_hidden
        else:
            new_hidden = merge_io(layer_in, raw_output, t, no_skip_front)
            layer.output[:] = new_hidden

    elif part == "mlp":
        mlp_out = layer.mlp.output
        if isinstance(mlp_out, tuple):
            m = mlp_out[0]
            new_m = merge_io(torch.zeros_like(m), m, t, no_skip_front)
            layer.mlp.output[0][:] = new_m
        else:
            new_m = merge_io(torch.zeros_like(mlp_out), mlp_out, t, no_skip_front)
            layer.mlp.output[:] = new_m

    elif part == "attention":
        assert precomputed_mlp is not None
        raw_output = layer.output
        if isinstance(raw_output, tuple):
            hidden = raw_output[0]
            li_aligned  = _align_dims(layer_in,        hidden)
            mlp_aligned = _align_dims(precomputed_mlp, hidden)
            no_attn = li_aligned + mlp_aligned
            new_hidden = merge_io(no_attn, hidden, t, no_skip_front)
            layer.output[0][:] = new_hidden
        else:
            li_aligned  = _align_dims(layer_in,        raw_output)
            mlp_aligned = _align_dims(precomputed_mlp, raw_output)
            no_attn = li_aligned + mlp_aligned
            new_hidden = merge_io(no_attn, raw_output, t, no_skip_front)
            layer.output[:] = new_hidden

    else:
        raise ValueError(f"Invalid part: {part}")


def get_future(data, t):
    if t is not None:
        return data[:, t:]
    return data


# ── Trace functions ───────────────────────────────────────────────────

def _realize(x):
    return x.value if hasattr(x, 'value') else x


def _residuals_to_diffs(saved_inputs, n_layers):
    inputs = [_realize(t).detach() for t in saved_inputs]
    residual_diffs = []
    for li in range(n_layers):
        h_in  = inputs[li]
        h_out = inputs[li + 1]
        if h_in.dim() == 2:
            h_in = h_in.unsqueeze(0)
        if h_out.dim() == 2:
            h_out = h_out.unsqueeze(0)
        residual_diffs.append((h_out - h_in).float().cpu())
    return torch.cat(residual_diffs, dim=0)


def trace_baseline(llm, prompt, input_attr):
    layers   = _layers
    norm     = _norm
    n_layers = _n_layers

    saved_inputs = []
    saved_logits = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for layer in layers:
                if input_attr == "input":
                    saved_inputs.append(layer.input[0].save())
                else:
                    saved_inputs.append(layer.inputs[0][0].save())
            saved_inputs.append(norm.input[0].save())
            saved_logits = llm.output.logits.save()

    residual_diffs = _residuals_to_diffs(saved_inputs, n_layers)
    outputs = _realize(saved_logits).detach().float().softmax(dim=-1).cpu()
    return residual_diffs, outputs


def trace_capture_mlp_outputs(llm, prompt):
    layers = _layers

    saved_mlp = []
    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for layer in layers:
                saved_mlp.append(layer.mlp.output.save())

    out = []
    for s in saved_mlp:
        v = _realize(s).detach()
        if isinstance(v, tuple):
            v = v[0]
        out.append(v)
    return out


def trace_intervened(llm, prompt, lskip, t, part, no_skip_front, input_attr,
                     precomputed_mlp_per_layer=None):
    layers   = _layers
    norm     = _norm
    n_layers = _n_layers

    precomputed_mlp = None
    if part == "attention":
        assert precomputed_mlp_per_layer is not None
        precomputed_mlp = precomputed_mlp_per_layer[lskip]

    saved_inputs = []
    saved_logits = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for li, layer in enumerate(layers):
                if input_attr == "input":
                    saved_inputs.append(layer.input[0].save())
                else:
                    saved_inputs.append(layer.inputs[0][0].save())
                if li == lskip:
                    apply_intervention(layer, t, part, no_skip_front,
                                       input_attr, precomputed_mlp=precomputed_mlp)
            saved_inputs.append(norm.input[0].save())
            saved_logits = llm.output.logits.save()

    residual_diffs = _residuals_to_diffs(saved_inputs, n_layers)
    outputs = _realize(saved_logits).detach().float().softmax(dim=-1).cpu()
    return residual_diffs, outputs


@ndif_cache_wrapper
def test_effect(llm, prompt, positions, part, no_skip_front=1):
    n_layers = _n_layers

    residual_log, outputs = trace_baseline(llm, prompt, _input_attr)

    precomputed_mlp = None
    if part == "attention":
        print("  (capturing mlp outputs in a separate trace for attention experiment)")
        precomputed_mlp = trace_capture_mlp_outputs(llm, prompt)

    all_diffs     = []
    all_out_diffs = []

    for t in positions:
        diffs     = []
        out_diffs = []

        for lskip in range(n_layers):
            new_logs, new_outputs = trace_intervened(
                llm, prompt, lskip, t, part, no_skip_front, _input_attr,
                precomputed_mlp_per_layer=precomputed_mlp
            )

            relative_diffs = (
                (get_future(residual_log, t) - get_future(new_logs, t)).norm(dim=-1)
                / get_future(residual_log, t).norm(dim=-1).clamp(min=1e-6)
            )
            diffs.append(relative_diffs.max(dim=-1).values)
            out_diffs.append(
                (get_future(new_outputs, t) - get_future(outputs, t)).norm(dim=-1).max(dim=-1).values
            )

        all_diffs.append(torch.stack(diffs, dim=0))
        all_out_diffs.append(torch.stack(out_diffs, dim=0))

    dall     = torch.stack(all_diffs,     dim=0).max(dim=0).values
    dall_out = torch.stack(all_out_diffs, dim=0).max(dim=0).values
    return dall, dall_out


def test_future_max_effect(llm, prompt, N_CHUNKS=4, part="layer"):
    _, tokens = tokenize(llm, prompt)
    positions = list(range(8, len(tokens) - 4, 8))
    random.shuffle(positions)
    positions = positions[:N_CHUNKS]
    return test_effect(llm, prompt, positions, part)


def probe_input_attr(llm, layers, prompt):
    try:
        with torch.no_grad():
            with llm.trace(prompt, remote=llm.remote):
                _ = layers[0].input[0].shape.save()
        return "input"
    except Exception:
        pass
    try:
        with torch.no_grad():
            with llm.trace(prompt, remote=llm.remote):
                _ = layers[0].inputs[0][0].shape.save()
        return "inputs"
    except Exception:
        pass
    return None


# ── Generate full text completions per sample, save to JSON ───────────

def generate_completion(llm, prompt, max_new_tokens=2048,
                        stop_strings=("</VERDICT_CLASSIFICATION>",)):
    """Greedy text generation. Returns (completion, was_truncated, n_generated_tokens)."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    tokenizer = llm.tokenizer
    model     = llm._model

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    class _StopOnString(StoppingCriteria):
        def __init__(self, tokenizer, stop_strings, prompt_length):
            super().__init__()
            self.tokenizer = tokenizer
            self.stop_strings = list(stop_strings)
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            text = self.tokenizer.decode(
                input_ids[0, self.prompt_length:],
                skip_special_tokens=True
            )
            return any(s in text for s in self.stop_strings)

    stopping = StoppingCriteriaList([
        _StopOnString(tokenizer, stop_strings, input_length)
    ])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            stopping_criteria=stopping,
        )

    new_tokens   = output_ids[0, input_length:]
    completion   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_generated  = int(new_tokens.shape[0])
    was_truncated = (n_generated >= max_new_tokens
                     and not any(s in completion for s in stop_strings))
    return completion, was_truncated, n_generated


def extract_predicted_classification(completion):
    """Strict extraction: ONLY parse from inside the
    <VERDICT_CLASSIFICATION>...</VERDICT_CLASSIFICATION> tag.

    Returns:
      - Canonical English label (one of CLASSIFICATION_OPTIONS) if found.
      - None if:
          * the completion is empty,
          * the <VERDICT_CLASSIFICATION> tag is missing entirely, OR
          * the content inside the tag can't be mapped to a known label.

    Returning None signals "no valid prediction" — the evaluator will
    treat None as a wrong prediction (`is_correct = False`).

    Handles both English and Arabic labels inside the tag via
    normalize_classification().
    """
    if not completion:
        return None

    # Look for the full tagged section (closing tag present)
    m = re.search(
        r'<VERDICT_CLASSIFICATION>\s*(.+?)\s*</VERDICT_CLASSIFICATION>',
        completion, re.IGNORECASE | re.DOTALL
    )
    if not m:
        # Tag may be missing its closing form because generation was truncated
        # mid-tag. Accept content from the opening tag up to a blank line or EOS.
        m = re.search(
            r'<VERDICT_CLASSIFICATION>\s*(.+?)(?:\n\n|\Z)',
            completion, re.IGNORECASE | re.DOTALL
        )

    if not m:
        # No tag at all — refuse to guess. Treat as no prediction.
        return None

    candidate = m.group(1).strip()
    if not candidate:
        return None

    # normalize_classification returns None if the content can't be mapped.
    return normalize_classification(candidate)


def save_completions_json(llm, model_name, target_dir, max_new_tokens=2048):
    """Generate full text completions for every sample and save to JSON."""
    print("\n===== Generating full text completions (no intervention) =====")

    legal_dataset = _make_dataset()
    ground_truths = legal_dataset.get_all_verdict_classifications()
    n_samples = len(legal_dataset)
    print(f"Will generate completions for {n_samples} samples "
          f"(max_new_tokens={max_new_tokens}).")

    results = []
    n_correct = 0
    n_with_gt = 0
    n_truncated = 0

    for idx, sample in enumerate(legal_dataset.data):
        gt = ground_truths[idx]
        prompt = legal_dataset.format_sample(sample)

        print(f"  [{idx+1}/{n_samples}] generating...")
        try:
            completion, was_truncated, n_gen = generate_completion(
                llm, prompt, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"    generation failed: {type(e).__name__}: {str(e)[:200]}")
            completion = None
            was_truncated = False
            n_gen = 0

        predicted = extract_predicted_classification(completion) if completion else None
        is_correct = (predicted is not None and gt is not None and predicted == gt)

        if gt is not None:
            n_with_gt += 1
            if is_correct:
                n_correct += 1
        if was_truncated:
            n_truncated += 1

        trunc_marker = " [TRUNCATED]" if was_truncated else ""
        print(f"    gt={gt!r:14s}  pred={predicted!r:14s}  "
              f"{'✓' if is_correct else '✗'}  "
              f"({n_gen} tokens{trunc_marker})")

        results.append({
            "sample_idx": idx,
            "ground_truth_classification": gt,
            "predicted_classification_from_text": predicted,
            "is_correct": is_correct,
            "n_generated_tokens": n_gen,
            "was_truncated": was_truncated,
            "completion": completion,
        })

    out_path = os.path.join(target_dir, f"{model_name}_completions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved completions to {out_path}")
    if n_with_gt > 0:
        gen_acc = n_correct / n_with_gt
        print(f"Generation-based classification accuracy: {gen_acc:.2%} "
              f"({n_correct}/{n_with_gt})")
    if n_truncated > 0:
        print(f"WARNING: {n_truncated}/{n_samples} completions hit max_new_tokens "
              f"({max_new_tokens}) without finishing.")


# ── Per-layer full-completion experiment ──────────────────────────────

def generate_completion_with_layer_skip(llm, prompt, skip_layer=None,
                                        max_new_tokens=2048,
                                        stop_strings=("</VERDICT_CLASSIFICATION>",)):
    """Token-by-token greedy generation with one decoder layer fully ablated."""
    tokenizer = llm.tokenizer
    model     = llm._model
    layers    = _layers

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    generated_ids = []
    cur_ids = input_ids

    stop_hit = False
    for step in range(max_new_tokens):
        cur_text = tokenizer.decode(cur_ids[0], skip_special_tokens=False)

        saved_logits = None
        with torch.no_grad():
            with llm.trace(cur_text, remote=llm.remote):
                for li, layer in enumerate(layers):
                    if li == skip_layer:
                        apply_intervention(
                            layer, t=None, part="layer", no_skip_front=0,
                            input_attr=_input_attr, precomputed_mlp=None
                        )
                saved_logits = llm.output.logits.save()

        logits = _last_position_logits(saved_logits)
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)
        cur_ids = torch.cat(
            [cur_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

        if eos_token_id is not None and next_id == eos_token_id:
            break

        decoded_new = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if any(s in decoded_new for s in stop_strings):
            stop_hit = True
            break

    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_generated = len(generated_ids)
    was_truncated = (n_generated >= max_new_tokens and not stop_hit
                     and (eos_token_id is None or generated_ids[-1] != eos_token_id))
    return completion, was_truncated, n_generated


def run_layer_skip_completions(llm, model_name, target_dir,
                                max_new_tokens=2048,
                                skip_layers=None):
    """For each (sample, layer-to-skip), generate the full model response."""
    print("\n===== Per-layer full-completion experiment =====")

    legal_dataset = _make_dataset()
    ground_truths = legal_dataset.get_all_verdict_classifications()
    n_samples = len(legal_dataset)

    if skip_layers is None:
        skip_layers = list(range(_n_layers))
    skip_layers = list(skip_layers)
    print(f"Samples: {n_samples}, layers to ablate: {len(skip_layers)} "
          f"(plus baseline). max_new_tokens={max_new_tokens}.")
    print("This will take a while — each generation runs one trace per token.")

    prompts = [legal_dataset.format_sample(legal_dataset.data[i])
               for i in range(n_samples)]

    records = []

    print("\nGenerating baseline (no skip)...")
    for idx in range(n_samples):
        gt = ground_truths[idx]
        print(f"  [baseline] sample {idx+1}/{n_samples}...", flush=True)
        try:
            completion, was_truncated, n_gen = generate_completion_with_layer_skip(
                llm, prompts[idx], skip_layer=None, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {str(e)[:200]}")
            completion = None; was_truncated = False; n_gen = 0

        records.append({
            "skip_layer": None,
            "sample_idx": idx,
            "ground_truth_classification": gt,
            "completion": completion,
            "n_generated_tokens": n_gen,
            "was_truncated": was_truncated,
        })
        _save_records(records, model_name, target_dir)

    for li_pos, lskip in enumerate(skip_layers):
        print(f"\nGenerating with layer {lskip} skipped "
              f"({li_pos+1}/{len(skip_layers)})...")
        for idx in range(n_samples):
            gt = ground_truths[idx]
            print(f"  [skip={lskip}] sample {idx+1}/{n_samples}...", flush=True)
            try:
                completion, was_truncated, n_gen = generate_completion_with_layer_skip(
                    llm, prompts[idx], skip_layer=lskip,
                    max_new_tokens=max_new_tokens
                )
            except Exception as e:
                print(f"    failed: {type(e).__name__}: {str(e)[:200]}")
                completion = None; was_truncated = False; n_gen = 0

            records.append({
                "skip_layer": lskip,
                "sample_idx": idx,
                "ground_truth_classification": gt,
                "completion": completion,
                "n_generated_tokens": n_gen,
                "was_truncated": was_truncated,
            })
            _save_records(records, model_name, target_dir)

    print(f"\nDone. Total records: {len(records)}")
    return records


def _save_records(records, model_name, target_dir):
    out_path = os.path.join(target_dir, f"{model_name}_layer_skip_completions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ── Build accuracy plot from saved completions JSON ───────────────────

def build_accuracy_plot_from_completions(model_name, target_dir):
    """Read per-layer completions JSON, extract classifications, plot accuracy."""
    print("\n===== Building accuracy plot from saved completions =====")

    json_path = os.path.join(target_dir, f"{model_name}_layer_skip_completions.json")
    if not os.path.exists(json_path):
        print(f"No completions JSON at {json_path} — skipping plot.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    by_layer = {}
    for r in records:
        skip = r["skip_layer"]
        gt   = r["ground_truth_classification"]
        comp = r.get("completion")
        pred = extract_predicted_classification(comp) if comp else None
        is_correct = (pred is not None and gt is not None and pred == gt)
        r["predicted_classification_from_text"] = pred
        r["is_correct"] = is_correct
        by_layer.setdefault(skip, []).append((gt, pred, is_correct))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    def acc_for(skip_key):
        bucket = by_layer.get(skip_key, [])
        bucket = [b for b in bucket if b[0] is not None]
        if not bucket:
            return None, 0
        n_correct = sum(1 for _, _, c in bucket if c)
        return n_correct / len(bucket), len(bucket)

    baseline_acc, baseline_n = acc_for(None)
    if baseline_acc is None:
        print("No baseline records with GT — cannot plot.")
        return
    print(f"Baseline (no skip): {baseline_acc:.2%} (n={baseline_n})")

    n_layers = _n_layers
    accuracies = []
    sample_counts = []
    for lskip in range(n_layers):
        acc, n = acc_for(lskip)
        if acc is None:
            accuracies.append(0.0)
            sample_counts.append(0)
        else:
            accuracies.append(acc)
            sample_counts.append(n)
        print(f"  Layer {lskip:3d} skipped: accuracy = {accuracies[-1]:.2%}  "
              f"(n={sample_counts[-1]})")

    fig = plt.figure(figsize=(8, 3))
    plt.bar(list(range(n_layers)), accuracies)
    plt.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Baseline (no skip): {baseline_acc:.2f}')
    plt.xlim(-0.5, n_layers - 0.5)
    plt.ylim(0, 1.05)
    plt.xlabel("Layer skipped")
    plt.ylabel("Classification accuracy (from generated text)")
    plt.title(f"{model_name}: verdict classification accuracy per skipped layer "
              f"(n={baseline_n}, from generated completions)")
    plt.legend(loc='lower right')
    out_path = os.path.join(target_dir, f"{model_name}_classification_accuracy.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ── Probe-based classification accuracy (legacy, unused) ──────────────

def _get_option_first_tokens(llm):
    tokenizer = llm.tokenizer
    first_tokens = {}
    for opt in CLASSIFICATION_OPTIONS:
        ids = tokenizer.encode(opt, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Tokenizer returned empty for {opt!r}")
        first_tokens[opt] = ids[0]
    return first_tokens


def _last_position_logits(saved_logits):
    raw = _realize(saved_logits).detach().float().cpu()
    if raw.dim() == 3:
        return raw[0, -1, :]
    if raw.dim() == 2:
        return raw[-1, :]
    return raw


# ── Main run ──────────────────────────────────────────────────────────

def run(llm, model_name, parts, do_completions, do_accuracy,
        n_examples=16, max_new_tokens=2048, output_dir="out/future_effects"):
    random.seed(123)

    target_dir = output_dir
    os.makedirs(target_dir, exist_ok=True)

    legal_dataset = _make_dataset()
    n_prompts = min(n_examples, len(legal_dataset))

    # ── Sublayer ablation experiments ──────────────────────────────
    if parts:
        print(f"\nWill use {n_prompts} legal prompts for sublayer experiments.")
        for what in parts:
            print(f"\n===== Experiment: {what} =====")
            d_max   = torch.zeros([1])
            dout_max = torch.zeros([1])

            for idx, prompt in enumerate(legal_dataset):
                if idx >= n_prompts:
                    break
                print(f"[{what}] [{idx+1}/{n_prompts}]")
                diff_now, diff_out = test_future_max_effect(llm, prompt, part=what)
                d_max    = torch.max(d_max,    diff_now)
                dout_max = torch.max(dout_max, diff_out)

            fig = plot_layer_diffs(d_max)
            fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_{what}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)

            fig = plot_logit_diffs(dout_max)
            fig.savefig(os.path.join(target_dir, f"{model_name}_future_max_effect_out_{what}.pdf"),
                        bbox_inches="tight")
            plt.close(fig)
    else:
        print("\n[Skipping] all sublayer ablation experiments")

    # ── Baseline full text completions ─────────────────────────────
    if do_completions:
        save_completions_json(llm, model_name, target_dir,
                              max_new_tokens=max_new_tokens)
    else:
        print("\n[Skipping] baseline full text completion generation")

    # ── Per-layer full completions + accuracy plot ─────────────────
    if do_accuracy:
        run_layer_skip_completions(
            llm, model_name, target_dir,
            max_new_tokens=max_new_tokens,
            skip_layers=None,
        )
        build_accuracy_plot_from_completions(model_name, target_dir)
    else:
        print("\n[Skipping] per-layer completion generation + accuracy plot")

    print(f"\nAll outputs saved to {target_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze layer-skipping effects on LLM internal computation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (default: english)
  python analyze_future_effects.py qwen3_8b

  # Arabic system prompt + custom data file
  python analyze_future_effects.py qwen3_8b \\
      --language arabic \\
      --data-path /home/sabeasm/llm_effective_depth/data/arabic_cases.json \\
      --output-dir out/arabic_full

  # Only the 'layer' sublayer experiment, English, custom file
  python analyze_future_effects.py qwen3_8b \\
      --parts layer --no-completions --no-accuracy \\
      --data-path /home/sabeasm/llm_effective_depth/data/english_cases.json

  # Replot accuracy bar chart from a saved JSON (no GPU work)
  python analyze_future_effects.py qwen3_8b --replot-only --output-dir out/arabic_full
"""
    )
    parser.add_argument(
        "model_name",
        help="Model name from lib.models (e.g. qwen3_8b, gemma_4_e4b)"
    )
    parser.add_argument(
        "--parts", nargs="*",
        default=["layer", "mlp", "attention"],
        choices=["layer", "mlp", "attention"],
        metavar="PART",
        help="Which sublayer ablation experiments to run. "
             "Default: all three. Pass --parts with no value to skip all."
    )
    parser.add_argument(
        "--language", choices=["english", "arabic"], default="english",
        help="System prompt language: 'english' (default) or 'arabic'. "
             "Determines which system prompt LegalDataset uses."
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to the legal cases JSON file. "
             "If omitted, LegalDataset's default path is used."
    )
    parser.add_argument(
        "--no-completions", action="store_true",
        help="Skip generating full text completions (the JSON file)"
    )
    parser.add_argument(
        "--no-accuracy", action="store_true",
        help="Skip the per-layer completion generation + accuracy plot"
    )
    parser.add_argument(
        "--n-examples", type=int, default=10,
        help="Max number of samples for sublayer effect experiments (default: 10)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Max new tokens per generated completion (default: 2048)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="out/future_effects",
        help="Directory to save all plots and JSON outputs (default: out/future_effects)"
    )
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Skip ALL generation/trace work. Just re-read the existing "
             "<model>_layer_skip_completions.json and rebuild the accuracy plot."
    )
    args = parser.parse_args()

    # Set globals BEFORE any branch references them.
    global _layers, _norm, _lm_head, _n_layers, _input_attr
    global _language, _data_path
    _language  = args.language
    _data_path = args.data_path

    # ── Short-circuit: just rebuild the plot from saved JSON ──────
    if args.replot_only:
        target_dir = args.output_dir
        llm = create_model(args.model_name, force_local=False)
        set_eval(llm)
        _layers   = get_layers(llm)
        _norm     = get_norm(llm)
        _lm_head  = get_lm_head(llm)
        _n_layers = len(_layers)
        _input_attr = "input"
        build_accuracy_plot_from_completions(args.model_name, target_dir)
        return

    llm = create_model(args.model_name, force_local=False)
    set_eval(llm)

    _layers   = get_layers(llm)
    _norm     = get_norm(llm)
    _lm_head  = get_lm_head(llm)
    _n_layers = len(_layers)

    test_prompt = next(iter(_make_dataset()))
    _input_attr = probe_input_attr(llm, _layers, test_prompt)
    if _input_attr is None:
        raise RuntimeError("Cannot determine nnsight input access pattern.")

    # ── Print configuration summary ────────────────────────────────
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"  Model:           {args.model_name}")
    print(f"  N text layers:   {_n_layers}")
    print(f"  Input attr:      {_input_attr}")
    print(f"  Language:        {args.language}")
    print(f"  Data path:       {args.data_path or '(default in legal.py)'}")
    print(f"  Sublayer parts:  {args.parts if args.parts else '(none — skip all)'}")
    print(f"  Completions:     {'skip' if args.no_completions else 'run'}")
    print(f"  Accuracy probe:  {'skip' if args.no_accuracy else 'run'}")
    print(f"  N examples:      {args.n_examples}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 60)

    # Show how many samples got loaded — to make the "1 completion" mystery obvious
    n_loaded = len(_make_dataset())
    print(f"\nLoaded {n_loaded} samples from "
          f"{args.data_path or '(default path)'}")
    if n_loaded == 1:
        print("  ↑ NOTE: only 1 sample was loaded. If you expected more, "
              "check that --data-path points to a file with multiple cases.")
    print()

    run(llm, args.model_name,
        parts=args.parts,
        do_completions=not args.no_completions,
        do_accuracy=not args.no_accuracy,
        n_examples=args.n_examples,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir)


if __name__ == "__main__":
    main()