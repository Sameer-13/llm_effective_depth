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
import shutil
import random
from typing import Optional

import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.datasets import LegalDataset
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, get_lm_head, set_eval


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


# ── Intervention helpers ──────────────────────────────────────────────

def _seq_dim(t):
    """seq is the second-to-last dim. Works for 2D [seq, hidden] and 3D [batch, seq, hidden]."""
    return t.dim() - 2


def _align_dims(a, b):
    """Make a and b have the same number of dims (matching b's dim count)."""
    while a.dim() < b.dim():
        a = a.unsqueeze(0)
    while a.dim() > b.dim():
        a = a.squeeze(0)
    return a


def merge_io(intervened, orig, t=None, no_skip_front=1):
    """
    Slice along the seq dim and concatenate. Works for 2D or 3D tensors.
    `intervened` is auto-aligned to match `orig`'s dim count.
    """
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
    """Modify layer output to simulate skipping a sublayer.
    Must be called at the natural execution point of this layer.

    For part='attention', `precomputed_mlp` must be a real tensor (the
    mlp output captured from a previous trace), because reading both
    layer.output and layer.mlp.output in the same trace triggers nnsight
    0.7's hook-collision bug.
    """
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
        assert precomputed_mlp is not None, \
            "attention intervention needs precomputed_mlp tensor"
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
    """Baseline forward pass. Save layer inputs (in execution order) + logits."""
    layers   = _layers
    norm     = _norm
    n_layers = _n_layers

    saved_inputs = []
    saved_logits = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            # Access layers IN EXECUTION ORDER — save each as we go
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
    """Capture mlp.output for every layer in one trace.
    Used as a precomputation step for the 'attention' intervention,
    since we can't read layer.output and mlp.output in the same trace
    (nnsight 0.7 hook collision bug)."""
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
    """Intervention pass. Apply intervention AT THE NATURAL EXECUTION POINT
    of layer `lskip`, in-order (nnsight 0.7 requires this).

    For part='attention', `precomputed_mlp_per_layer` must be a list of
    pre-captured mlp outputs (one per layer); we'll use the one for layer
    `lskip` to avoid the hook-collision bug."""
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

    # For the attention intervention, we need mlp.output values but can't
    # read them in the same trace as layer.output. Capture them once here.
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


def run(llm, model_name):
    N_EXAMPLES = 10
    random.seed(123)

    target_dir = "out/future_effects"
    os.makedirs(target_dir, exist_ok=True)

    legal_dataset = LegalDataset()
    n_prompts = min(N_EXAMPLES, len(legal_dataset))
    print(f"Will use {n_prompts} legal prompts.")

    for what in ["layer", "mlp", "attention"]:
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

    print(f"\nAll plots saved to {target_dir}/")


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise ValueError("Please provide a model name")

    llm = create_model(model_name, force_local=False)
    set_eval(llm)

    global _layers, _norm, _lm_head, _n_layers, _input_attr
    _layers   = get_layers(llm)
    _norm     = get_norm(llm)
    _lm_head  = get_lm_head(llm)
    _n_layers = len(_layers)

    test_prompt = next(iter(LegalDataset()))
    _input_attr = probe_input_attr(llm, _layers, test_prompt)
    if _input_attr is None:
        raise RuntimeError("Cannot determine nnsight input access pattern.")
    print(f"Using input access pattern: {_input_attr}")

    run(llm, model_name)


if __name__ == "__main__":
    main()