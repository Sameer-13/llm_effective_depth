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
import argparse
import gc

import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch

from lib.models import create_model
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens, set_eval
from lib.datasets import LegalDataset


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Logitlens: project each layer's residual stream through "
                    "final-norm + lm_head and measure closeness to final output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python logitlens.py qwen3_8b
  python logitlens.py qwen3_8b --language arabic \\
      --data-path /home/.../arabic_cases.json \\
      --output-dir output/arabic/10cases
"""
    )
    parser.add_argument("model_name", help="Model name from lib.models")
    parser.add_argument("--language", choices=["english", "arabic"],
                        default="english")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="out/logitlens")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear run_logitlens cache before running")
    return parser.parse_args()


args = parse_args()
model_name = args.model_name
target_dir = args.output_dir
random.seed(123123)


def make_dataset():
    kwargs = {"language": args.language}
    if args.data_path is not None:
        kwargs["json_path"] = args.data_path
    return LegalDataset(**kwargs)


# ── Model setup ───────────────────────────────────────────────────────

llm = create_model(model_name)
set_eval(llm)
os.makedirs(target_dir, exist_ok=True)

_layers   = get_layers(llm)
_norm     = get_norm(llm)
_lm_head  = get_lm_head(llm)
_n_layers = len(_layers)


# ── Configuration printout ────────────────────────────────────────────

print("=" * 60)
print("Configuration")
print("=" * 60)
print(f"  Model:        {model_name}")
print(f"  N layers:     {_n_layers}")
print(f"  Language:     {args.language}")
print(f"  Data path:    {args.data_path or '(default in legal.py)'}")
print(f"  Output dir:   {target_dir}")
print(f"  Clear cache:  {args.clear_cache}")
print("=" * 60)


# ── Helpers ───────────────────────────────────────────────────────────

def _realize(x):
    return x.value if hasattr(x, 'value') else x


def to_3d(t):
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t


def _free_cuda():
    """Aggressive cleanup between traces. Doing both gc and empty_cache
    matters because some references are only released after a gc cycle."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Per-layer logitlens trace (memory-safe) ───────────────────────────
# Each trace materializes ONE big [B,S,V] projection, then we move it to
# CPU and free GPU memory before the next trace. Avoids the previous
# OOM where 36 projections (≈32 GB on Qwen3-8B) were held simultaneously.

def trace_get_final_logits(llm, prompt):
    """One trace: just save the final-layer logits, move to CPU."""
    saved = None
    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            saved = llm.output.logits.save()
    out = _realize(saved).detach().float().cpu()
    del saved
    _free_cuda()
    return out


def trace_get_layer_projection(llm, prompt, layer_idx):
    """One trace: compute head(norm(layer_input[i])), save, move to CPU."""
    layer = _layers[layer_idx]
    saved = None
    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            tap = layer.input[0]
            projected = _lm_head(_norm(tap))
            saved = projected.save()
    out = _realize(saved).detach().float().cpu()
    del saved
    _free_cuda()
    return out


def _kl_and_overlap(layer_logits_cpu, final_logits_cpu, K=5):
    """KL(layer || final) and top-K overlap. CPU-only, memory-light.

    Avoids the previous one-hot blow-up: instead of allocating
    [B, S, K, vocab] one-hots (~10 GB for Qwen3 vocab), do K vs K
    set-intersection via broadcasting — [B, S, K, K] (tiny)."""
    layer_lp = to_3d(layer_logits_cpu).log_softmax(-1)   # [B, S, V]
    final_lp = to_3d(final_logits_cpu).log_softmax(-1)   # [B, S, V]

    # KL per token, then mean over (batch, seq)
    kl = (layer_lp.exp() * (layer_lp - final_lp)).sum(-1).mean()

    layer_topk = layer_lp.topk(K, dim=-1).indices        # [B, S, K]
    final_topk = final_lp.topk(K, dim=-1).indices        # [B, S, K]
    # For each of layer's top-K, is it among final's top-K?
    matches = (layer_topk[..., :, None] == final_topk[..., None, :]).any(-1)  # [B,S,K]
    overlap = matches.float().sum(-1).mean() / K

    return kl, overlap


@ndif_cache_wrapper
def run_logitlens(llm, prompts, K=5):
    n_layers = _n_layers

    all_kl_per_layer      = []
    all_overlap_per_layer = []

    for pi, prompt in enumerate(prompts):
        print(f"[prompt {pi+1}/{len(prompts)}]")

        # 1) Final-layer logits, once per prompt
        final_logits = trace_get_final_logits(llm, prompt)

        # 2) Per-layer projection, one trace at a time
        kl_per_layer      = []
        overlap_per_layer = []
        for li in range(n_layers):
            layer_logits = trace_get_layer_projection(llm, prompt, li)
            kl, overlap = _kl_and_overlap(layer_logits, final_logits, K=K)
            kl_per_layer.append(kl)
            overlap_per_layer.append(overlap)
            del layer_logits
            if (li + 1) % 6 == 0:
                print(f"  layer {li+1}/{n_layers}  KL={kl.item():.3f}  "
                      f"top{K}={overlap.item():.3f}")

        del final_logits
        _free_cuda()

        all_kl_per_layer.append(torch.stack(kl_per_layer, dim=0))
        all_overlap_per_layer.append(torch.stack(overlap_per_layer, dim=0))

    avg_kl      = torch.stack(all_kl_per_layer,      dim=0).mean(dim=0)
    avg_overlap = torch.stack(all_overlap_per_layer, dim=0).mean(dim=0)
    return avg_kl.cpu(), avg_overlap.cpu()


# ── Load dataset ──────────────────────────────────────────────────────

legal_dataset = make_dataset()
N_EXAMPLES = len(legal_dataset)
print(f"\nLoaded {N_EXAMPLES} legal prompts from "
      f"{args.data_path or '(default path)'}")
if N_EXAMPLES == 1:
    print("  ↑ NOTE: only 1 sample. Pass --data-path for a multi-case file.")

# ── Clear stale cache if requested ────────────────────────────────────

cache_dir = "cache/ndif_cache/run_logitlens"
if args.clear_cache and os.path.exists(cache_dir):
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

print(f"\nAll plots saved to {target_dir}/")