import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['text.usetex'] = False

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

import os
import sys
import shutil
import random

import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
import torch.nn.functional as F

from lib.models import create_model
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens, set_eval
from lib.datasets import LegalDataset


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <model_name>")
    exit(1)

model_name = sys.argv[1]
target_dir = "out/logitlens"
random.seed(123123)

llm = create_model(model_name)
set_eval(llm)
os.makedirs(target_dir, exist_ok=True)

# ── Resolve model paths BEFORE any trace block ───────────────────────
_layers   = get_layers(llm)
_norm     = get_norm(llm)
_lm_head  = get_lm_head(llm)
_embed    = get_embed_tokens(llm)
_n_layers = len(_layers)
# ─────────────────────────────────────────────────────────────────────


def trace_logitlens(llm, prompt):
    """
    Capture, for each layer:
      - layer.input[0]   = residual stream entering this layer
    And once at the end:
      - the final logits from llm.output.logits

    Then OUTSIDE the trace, apply the model's final norm + lm_head to
    every residual stream snapshot. This avoids the nnsight 0.7 hook
    collision issues and keeps everything simple.
    """
    layers = _layers
    n_layers = _n_layers

    saved_residuals = []   # list of layer inputs (= residual stream at each depth)
    saved_logits    = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for layer in layers:
                saved_residuals.append(layer.input[0].save())
            # final residual = output of last layer = input to final norm
            saved_residuals.append(_norm.input[0].save())
            saved_logits = llm.output.logits.save()

    def realize(x):
        return x.value if hasattr(x, 'value') else x

    residuals = [realize(t).detach() for t in saved_residuals]
    logits    = realize(saved_logits).detach()
    return residuals, logits


def to_3d(t):
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t


@ndif_cache_wrapper
def run_logitlens(llm, prompts, K=5):
    """
    Logitlens: project the residual stream at each layer through
    final_norm and lm_head to get per-layer "predictions". Compute:
      - KL divergence from each layer's prediction to the final output
      - Top-K token overlap between each layer's prediction and the final

    We avoid all nnsight pitfalls by:
      1. Only saving raw residuals + final logits inside the trace
      2. Doing the norm + lm_head projection OUTSIDE the trace in pure PyTorch
    """
    n_layers = _n_layers

    # Pre-resolve the actual PyTorch modules for use outside the trace
    norm_module    = _norm._module       # the nn.Module wrapped by nnsight
    lm_head_module = _lm_head._module

    all_kl_per_layer  = []   # list of [n_layers] tensors, one per prompt
    all_overlap_per_layer = []   # same shape

    for pi, prompt in enumerate(prompts):
        print(f"[{pi+1}/{len(prompts)}]")
        residuals, logits = trace_logitlens(llm, prompt)

        # logits shape: [batch=1, seq, vocab]
        out_log_probs = logits.float().log_softmax(-1).cpu()   # CPU float32
        out_topk      = out_log_probs.topk(K, dim=-1).indices  # [1, seq, K]

        kl_per_layer      = []
        overlap_per_layer = []

        # Move norm and lm_head to CPU temporarily — or apply on GPU
        # We'll apply on the device the residual is currently on.
        for li in range(n_layers):
            r = to_3d(residuals[li])

            # Apply final norm + lm_head to this layer's residual
            with torch.no_grad():
                # Match the dtype the modules expect
                r_for_norm = r.to(dtype=next(norm_module.parameters()).dtype,
                                   device=next(norm_module.parameters()).device)
                layer_logits = lm_head_module(norm_module(r_for_norm))

            layer_log_probs = layer_logits.float().log_softmax(-1).cpu()

            # KL(layer || final) — measures how different this layer's
            # prediction is from the final prediction.
            # Actually the original code computes KL(layer || final) as
            #   sum(p_layer * (log p_layer - log p_final))
            kl = (layer_log_probs.exp() * (layer_log_probs - out_log_probs)).sum(-1).mean()
            kl_per_layer.append(kl)

            # Top-K overlap
            layer_topk = layer_log_probs.topk(K, dim=-1).indices  # [1, seq, K]
            # For each position, count how many of the top-K agree
            # Use one-hot intersection trick
            vocab_size = layer_log_probs.shape[-1]
            real_oh   = F.one_hot(out_topk,   vocab_size).sum(-2)  # [1, seq, V]
            layer_oh  = F.one_hot(layer_topk, vocab_size).sum(-2)  # [1, seq, V]
            overlap = (real_oh.float() * layer_oh.float()).sum(-1) / K  # [1, seq]
            overlap_per_layer.append(overlap.mean())

        all_kl_per_layer.append(torch.stack(kl_per_layer, dim=0))
        all_overlap_per_layer.append(torch.stack(overlap_per_layer, dim=0))

    # Average over all prompts
    avg_kl      = torch.stack(all_kl_per_layer,      dim=0).mean(dim=0)
    avg_overlap = torch.stack(all_overlap_per_layer, dim=0).mean(dim=0)

    return avg_kl.cpu(), avg_overlap.cpu()


# ── Load dataset ──────────────────────────────────────────────────────
legal_dataset = LegalDataset()
N_EXAMPLES = len(legal_dataset)
print(f"Loaded {N_EXAMPLES} legal prompts.")

# ── Clear stale cache ─────────────────────────────────────────────────
cache_dir = "cache/ndif_cache/run_logitlens"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared old cache.")

# ── Run ───────────────────────────────────────────────────────────────
prompts = list(legal_dataset)
avg_kl, avg_overlap = run_logitlens(llm, prompts)

# ── Plot ──────────────────────────────────────────────────────────────
plt.figure(figsize=(5, 2))
plt.bar(range(_n_layers), avg_kl.cpu().numpy())
plt.ylabel("KL Divergence")
plt.xlabel("Layer")
plt.savefig(os.path.join(target_dir, f"{model_name}_logitlens_kl_div.pdf"),
            bbox_inches="tight")
plt.close()

plt.figure(figsize=(5, 2))
plt.bar(range(_n_layers), avg_overlap.cpu().numpy())
plt.ylabel("Top-K Overlap")
plt.xlabel("Layer")
plt.savefig(os.path.join(target_dir, f"{model_name}_logitlens_topk_overlaps.pdf"),
            bbox_inches="tight")
plt.close()

print(f"All plots saved to {target_dir}/")