import matplotlib.pyplot as plt

import os
import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
import random

from lib.models import create_model
from lib.ndif_cache import ndif_cache_wrapper
import torch.nn.functional as F
from lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens, set_eval
from lib.datasets import LegalDataset

import sys

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} model")
    exit(1)
else:
    model_name = sys.argv[1]

N_EXAMPLES = 10
target_dir = "out/logitlens"

random.seed(123123)

llm = create_model(model_name)
set_eval(llm)

os.makedirs(target_dir, exist_ok=True)

# ── Resolve model paths BEFORE any session/trace block ───────────────
_layers   = get_layers(llm)
_norm     = get_norm(llm)
_lm_head  = get_lm_head(llm)
_embed    = get_embed_tokens(llm)
_n_layers = len(_layers)
# ─────────────────────────────────────────────────────────────────────


@ndif_cache_wrapper
def run_logitlens(llm, prompts, K=5):
    # Use pre-resolved variables — never call get_*() inside session
    layers  = _layers
    norm    = _norm
    head    = _lm_head
    embed   = _embed
    n_layers = _n_layers

    res_kl_divs  = []
    res_topks    = []

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            for prompt in prompts:
                kl_divs    = []
                topks      = []
                layer_logs = []

                with llm.trace(prompt, remote=llm.remote):
                    for l in range(n_layers):
                        tap = layers[l].inputs[0][0]
                        layer_logs.append(head(norm(tap)).detach().float())
                    out_logits = llm.output.logits

                lout  = out_logits.float().log_softmax(-1)
                otopl = lout.topk(K, dim=-1).indices

                for l in range(n_layers):
                    llayer = layer_logs[l].log_softmax(-1)
                    kl_divs.append(
                        (llayer.exp() * (llayer - lout)).sum(-1).mean().detach()
                    )
                    itopl = llayer.topk(K, dim=-1).indices
                    topks.append(itopl.save())

                topks.append(otopl.save())
                res_kl_divs.append(torch.stack(kl_divs, dim=0).save())
                res_topks.append(topks)

    # Compute top-k overlaps outside the session
    vocab_size = embed._module.weight.shape[0]

    res_topk_overlaps = []
    for topks in res_topks:
        real_topk   = topks[-1]
        other_topks = topks[:-1]

        real_oh  = F.one_hot(real_topk, vocab_size).sum(-2)
        overlaps = [
            (
                F.one_hot(ot.to(real_oh.device), vocab_size)
                .sum(-2).unsqueeze(-2).float()
                @ real_oh.unsqueeze(-1).float()
                / K
            ).mean()
            for ot in other_topks
        ]
        overlaps = torch.stack(overlaps, dim=0)
        res_topk_overlaps.append(overlaps)

    return [d.cpu() for d in res_kl_divs], res_topk_overlaps


# ── Load dataset ──────────────────────────────────────────────────────
legal_dataset = LegalDataset(
    json_path="samples.json",
    max_samples=N_EXAMPLES,
    include_steps=False,
    use_chat_template=True,
)
N_EXAMPLES = len(legal_dataset)
print(f"Loaded {N_EXAMPLES} legal prompts.")

# ── Run logitlens over each prompt ────────────────────────────────────
accu_kl_div       = 0
accu_topk_overlaps = 0

for i, prompt in enumerate(legal_dataset):
    print(f"[{i+1}/{N_EXAMPLES}]")
    kl_divs, topk_overlaps = run_logitlens(llm, [prompt])
    accu_kl_div        = accu_kl_div        + kl_divs[0]
    accu_topk_overlaps = accu_topk_overlaps + topk_overlaps[0]

accu_kl_div        = accu_kl_div        / N_EXAMPLES
accu_topk_overlaps = accu_topk_overlaps / N_EXAMPLES

# ── Plots ─────────────────────────────────────────────────────────────
plt.figure(figsize=(5, 2))
plt.bar(range(_n_layers), accu_kl_div.cpu().numpy())
plt.ylabel("KL Divergence")
plt.xlabel("Layer")
plt.savefig(os.path.join(target_dir, f"{model_name}_logitlens_kl_div.pdf"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(5, 2))
plt.bar(range(_n_layers), accu_topk_overlaps.cpu().numpy())
plt.ylabel("Overlap")
plt.xlabel("Layer")
plt.savefig(os.path.join(target_dir, f"{model_name}_logitlens_topk_overlaps.pdf"), bbox_inches="tight")
plt.close()

print(f"Plots saved to {target_dir}/")