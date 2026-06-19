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

import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
import torch.nn.functional as F

from lib.models import create_model
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens, set_eval
from lib.datasets import LegalDataset


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Logitlens analysis: project each layer's residual stream "
                    "through final-norm + lm_head and measure how close each "
                    "layer's 'guess' is to the final output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default English, default data path, default output dir
  python logitlens.py qwen3_8b

  # Arabic prompt with a specific data file and output dir
  python logitlens.py qwen3_8b \\
      --language arabic \\
      --data-path /home/sabeasm/llm_effective_depth/data/arabic_cases.json \\
      --output-dir output/arabic/10cases

  # English with 1-case file
  python logitlens.py qwen3_8b \\
      --data-path /home/sabeasm/llm_effective_depth/data/single_english_case.json \\
      --output-dir output/english/single
"""
    )
    parser.add_argument(
        "model_name",
        help="Model name from lib.models (e.g. qwen3_8b)"
    )
    parser.add_argument(
        "--language", choices=["english", "arabic"], default="english",
        help="System prompt language for LegalDataset (default: english)"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to legal cases JSON file. If omitted, LegalDataset's default is used."
    )
    parser.add_argument(
        "--output-dir", type=str, default="out/logitlens",
        help="Directory to save plots (default: out/logitlens)"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear the run_logitlens cache before running"
    )
    return parser.parse_args()


args = parse_args()
model_name = args.model_name
target_dir = args.output_dir
random.seed(123123)


# ── Helper: build dataset with CLI overrides ──────────────────────────

def make_dataset():
    kwargs = {"language": args.language}
    if args.data_path is not None:
        kwargs["json_path"] = args.data_path
    return LegalDataset(**kwargs)


# ── Model setup ───────────────────────────────────────────────────────

llm = create_model(model_name)
set_eval(llm)
os.makedirs(target_dir, exist_ok=True)

# Resolve model paths BEFORE any trace block
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


# ── Logitlens trace ───────────────────────────────────────────────────

def trace_logitlens(llm, prompt):
    """Inside the trace, project EACH layer's residual stream through the
    model's own final-norm and lm_head — using nnsight's wrapped modules
    (NOT raw _module objects, which are on meta device with dispatch=False).

    Save the resulting logits per layer + the actual final logits."""
    layers = _layers
    norm   = _norm
    head   = _lm_head
    n_layers = _n_layers

    saved_layer_logits = []
    saved_final_logits = None

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for layer in layers:
                tap = layer.input[0]
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

        final_lp = to_3d(final_logits.float().cpu()).log_softmax(-1)
        out_topk = final_lp.topk(K, dim=-1).indices
        vocab_size = final_lp.shape[-1]

        kl_per_layer      = []
        overlap_per_layer = []

        for li in range(n_layers):
            layer_lp = to_3d(layer_logits[li].float().cpu()).log_softmax(-1)

            kl = (layer_lp.exp() * (layer_lp - final_lp)).sum(-1).mean()
            kl_per_layer.append(kl)

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

legal_dataset = make_dataset()
N_EXAMPLES = len(legal_dataset)
print(f"\nLoaded {N_EXAMPLES} legal prompts from "
      f"{args.data_path or '(default path)'}")
if N_EXAMPLES == 1:
    print("  ↑ NOTE: only 1 sample. Pass --data-path to use a multi-case file.")

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