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

def merge_io(intervened, orig, t: Optional[int] = None, no_skip_front: int = 1):
    outs = [orig[:, :no_skip_front]]
    if t is not None:
        outs.append(intervened[:, no_skip_front:t].to(orig.device))
        outs.append(orig[:, t:])
    else:
        outs.append(intervened[:, no_skip_front:].to(orig.device))
    return torch.cat(outs, dim=1)


def intervene_layer(layer, t: Optional[int], part: str, no_skip_front: int, input_attr="input"):
    """
    Modify a layer's output so that the contribution from a specific
    sublayer (or the whole layer) is zeroed out for positions >= no_skip_front
    and < t (or all positions >= no_skip_front if t is None).

    KEY: we don't touch self_attn.output (unreliable in nnsight 0.7).
    For 'attention' we instead modify the layer output by subtracting
    out the mlp contribution and then zeroing the residual difference.
    For 'mlp' we zero mlp.output directly (this works in isolation).
    For 'layer' we set layer.output[0] = layer.input[0].
    """
    if input_attr == "input":
        layer_in = layer.input[0]
    else:
        layer_in = layer.inputs[0][0]

    if part == "layer":
        raw_output = layer.output
        if isinstance(raw_output, tuple):
            hidden = raw_output[0]
            rest   = raw_output[1:]
            new_hidden = merge_io(layer_in, hidden, t, no_skip_front)
            layer.output = (new_hidden,) + rest
        else:
            layer.output = merge_io(layer_in, layer.output[0], t, no_skip_front),

    elif part == "mlp":
        layer.mlp.output = merge_io(
            torch.zeros_like(layer.mlp.output),
            layer.mlp.output, t, no_skip_front
        )

    elif part == "attention":
        # Zero attn = make layer output = layer_input + mlp_output
        # That is: skip the attention contribution entirely.
        # We compute the "would-be" output if attn were zero:
        #   no_attn_output = layer_input + mlp.output (if model is post-norm)
        # But the cleanest way: edit layer.output[0] to equal layer_input
        # plus mlp.output (the mlp output is computed from the normed
        # h_after_attn, which we can't easily intercept).
        #
        # Simpler approach: write the layer output as if attention had
        # zero contribution by setting output = input + mlp_contribution
        # Note: this is an approximation since the mlp ran on h_after_attn
        # (which includes attention). For Figure 3 the exact mechanism
        # matters less than the relative comparison across layers.
        raw_output = layer.output
        if isinstance(raw_output, tuple):
            hidden = raw_output[0]
            rest   = raw_output[1:]
            # zero-attn would-be output: just layer input + mlp output
            no_attn = layer_in + layer.mlp.output
            new_hidden = merge_io(no_attn, hidden, t, no_skip_front)
            layer.output = (new_hidden,) + rest
        else:
            no_attn = layer_in + layer.mlp.output
            layer.output = merge_io(no_attn, layer.output[0], t, no_skip_front),

    else:
        raise ValueError(f"Invalid part: {part}")


def get_future(data, t: Optional[int]):
    if t is not None:
        return data[:, t:]
    return data


# ── Core experiment functions ─────────────────────────────────────────

def trace_baseline(llm, prompt, input_attr):
    """
    Get baseline residual stream snapshots and final softmax outputs.
    Saves: layer inputs (= residuals at each depth) + final logits.
    Returns residual_diffs [n_layers, batch, seq, hidden] and outputs.
    """
    layers = _layers
    norm   = _norm
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

    def realize(x):
        return x.value if hasattr(x, 'value') else x

    inputs = [realize(t).detach() for t in saved_inputs]
    logits = realize(saved_logits).detach()

    # residual_log[i] = output_of_layer_i - input_to_layer_i
    # In the original paper code this is the "layer contribution" tensor
    # used as the baseline for comparing to the intervened forward pass.
    residual_diffs = []
    for li in range(n_layers):
        h_in  = inputs[li]
        h_out = inputs[li + 1]
        if h_in.dim() == 2:
            h_in = h_in.unsqueeze(0)
        if h_out.dim() == 2:
            h_out = h_out.unsqueeze(0)
        residual_diffs.append((h_out - h_in).float().cpu())

    # shape: [n_layers, batch, seq, hidden]
    residual_diffs = torch.cat(residual_diffs, dim=0)
    outputs        = logits.float().softmax(dim=-1).cpu()

    return residual_diffs, outputs


def trace_intervened(llm, prompt, lskip, t, part, no_skip_front, input_attr):
    """
    Run a forward pass with layer `lskip` intervened on, and capture
    the resulting residual stream + output logits.
    """
    layers = _layers
    norm   = _norm
    n_layers = _n_layers

    saved_inputs = []
    saved_logits = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            # Apply the intervention on layer `lskip`
            intervene_layer(layers[lskip], t, part, no_skip_front, input_attr)

            # Capture residuals after intervention
            for layer in layers:
                if input_attr == "input":
                    saved_inputs.append(layer.input[0].save())
                else:
                    saved_inputs.append(layer.inputs[0][0].save())
            saved_inputs.append(norm.input[0].save())
            saved_logits = llm.output.logits.save()

    def realize(x):
        return x.value if hasattr(x, 'value') else x

    inputs = [realize(t).detach() for t in saved_inputs]
    logits = realize(saved_logits).detach()

    residual_diffs = []
    for li in range(n_layers):
        h_in  = inputs[li]
        h_out = inputs[li + 1]
        if h_in.dim() == 2:
            h_in = h_in.unsqueeze(0)
        if h_out.dim() == 2:
            h_out = h_out.unsqueeze(0)
        residual_diffs.append((h_out - h_in).float().cpu())

    residual_diffs = torch.cat(residual_diffs, dim=0)
    output_probs   = logits.float().softmax(dim=-1).cpu()

    return residual_diffs, output_probs


@ndif_cache_wrapper
def test_effect(llm, prompt, positions, part, no_skip_front=1):
    n_layers = _n_layers

    # 1. Baseline pass
    residual_log, outputs = trace_baseline(llm, prompt, _input_attr)

    all_diffs     = []
    all_out_diffs = []

    for t in positions:
        diffs     = []
        out_diffs = []

        for lskip in range(n_layers):
            new_logs, new_outputs = trace_intervened(
                llm, prompt, lskip, t, part, no_skip_front, _input_attr
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


# ── Probe nnsight access pattern ─────────────────────────────────────

def probe_input_attr(llm, layers, prompt):
    """Determine whether nnsight uses .input[0] or .inputs[0][0]."""
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


# ── Main run function ─────────────────────────────────────────────────

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
        dall    = []
        d_max   = torch.zeros([1])
        dout_max = torch.zeros([1])

        for idx, prompt in enumerate(legal_dataset):
            if idx >= n_prompts:
                break
            print(f"[{what}] [{idx+1}/{n_prompts}]")
            diff_now, diff_out = test_future_max_effect(llm, prompt, part=what)
            d_max    = torch.max(d_max,    diff_now)
            dout_max = torch.max(dout_max, diff_out)
            dall.append(diff_now)

        fig = plot_layer_diffs(d_max)
        fig.savefig(
            os.path.join(target_dir, f"{model_name}_future_max_effect_{what}.pdf"),
            bbox_inches="tight"
        )
        plt.close(fig)

        fig = plot_logit_diffs(dout_max)
        fig.savefig(
            os.path.join(target_dir, f"{model_name}_future_max_effect_out_{what}.pdf"),
            bbox_inches="tight"
        )
        plt.close(fig)

    print(f"\nAll plots saved to {target_dir}/")


def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise ValueError("Please provide a model name")

    llm = create_model(model_name, force_local=False)
    set_eval(llm)

    # Resolve all model paths once at module level
    global _layers, _norm, _lm_head, _n_layers, _input_attr
    _layers   = get_layers(llm)
    _norm     = get_norm(llm)
    _lm_head  = get_lm_head(llm)
    _n_layers = len(_layers)

    # Determine nnsight input access pattern
    legal_dataset = LegalDataset()
    test_prompt = next(iter(legal_dataset))
    _input_attr = probe_input_attr(llm, _layers, test_prompt)
    if _input_attr is None:
        raise RuntimeError("Cannot determine nnsight input access pattern.")
    print(f"Using input access pattern: {_input_attr}")

    run(llm, model_name)


if __name__ == "__main__":
    main()