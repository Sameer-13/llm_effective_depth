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
_n_layers = len(_layers)
# ─────────────────────────────────────────────────────────────────────


def trace_logitlens(llm, prompt):
    """
    Inside the trace, project EACH layer's residual stream through the
    model's own final-norm and lm_head — using nnsight's wrapped modules
    (NOT raw _module objects, which are on meta device with dispatch=False).

    Save the resulting logits per layer + the actual final logits.
    """
    layers = _layers
    norm   = _norm        # nnsight Envoy
    head   = _lm_head     # nnsight Envoy
    n_layers = _n_layers

    saved_layer_logits = []   # per-layer logitlens projections
    saved_final_logits = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for layer in layers:
                tap = layer.input[0]
                # Apply norm + lm_head INSIDE the trace.
                # nnsight will invoke the real modules with proper weights.
                projected = head(norm(tap))
                saved_layer_logits.append(projected.save())
            saved_final_logits = llm.output.logits.save()

    def realize(x):
        return x.value if hasattr(x, 'value') else x

    layer_logits = [realize(t).detach() for t in saved_layer_logits]
    final_logits = realize(saved_final_logits).detach()
    return layer_logits, final_logits


def to_3d(t):
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t


@ndif_cache_wrapper
def run_logitlens(llm, prompts, K=5):
    n_layers = _n_layers

    all_kl_per_layer      = []
    all_overlap_per_layer = []

    for pi, prompt in enumerate(prompts):
        print(f"[{pi+1}/{len(prompts)}]")
        layer_logits, final_logits = trace_logitlens(llm, prompt)

        # Move everything to CPU float32 for KL math
        final_lp = to_3d(final_logits.float().cpu()).log_softmax(-1)   # [1, seq, V]
        out_topk = final_lp.topk(K, dim=-1).indices                    # [1, seq, K]
        vocab_size = final_lp.shape[-1]

        kl_per_layer      = []
        overlap_per_layer = []

        for li in range(n_layers):
            layer_lp = to_3d(layer_logits[li].float().cpu()).log_softmax(-1)

            # KL(layer || final) per token, averaged over sequence
            kl = (layer_lp.exp() * (layer_lp - final_lp)).sum(-1).mean()
            kl_per_layer.append(kl)

            # Top-K overlap
            layer_topk = layer_lp.topk(K, dim=-1).indices
            real_oh  = F.one_hot(out_topk,   vocab_size).sum(-2).float()
            layer_oh = F.one_hot(layer_topk, vocab_size).sum(-2).float()
            overlap  = (real_oh * layer_oh).sum(-1) / K
            overlap_per_layer.append(overlap.mean())

        all_kl_per_layer.append(torch.stack(kl_per_layer, dim=0))
        all_overlap_per_layer.append(torch.stack(overlap_per_layer, dim=0))

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