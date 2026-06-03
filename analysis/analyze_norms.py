import matplotlib
matplotlib.use("Agg")              # no display backend
matplotlib.rcParams['text.usetex'] = False   # don't require latex binary

from lib.matplotlib_config import sort_zorder
import matplotlib.pyplot as plt

# Force usetex off and use default fonts (override anything matplotlib_config set)
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

import os
import sys
import shutil
import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
import torch.nn.functional as F

from lib.models import create_model
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, set_eval
from lib.datasets import LegalDataset

N_EXAMPLES = 10

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Please provide a model name")

llm = create_model(model_name)
target_dir = "out/norms"
set_eval(llm)
os.makedirs(target_dir, exist_ok=True)

# ── Resolve model paths BEFORE any trace block ───────────────────────
_layers   = get_layers(llm)
_norm     = get_norm(llm)
_n_layers = len(_layers)
# ─────────────────────────────────────────────────────────────────────


def get_mlp_module(layer):
    """Return the MLP sub-module, handling naming differences."""
    raw = layer._module
    for candidate in ["mlp", "feed_forward", "ffn"]:
        if hasattr(raw, candidate):
            return getattr(layer, candidate)
    raise AttributeError(
        f"No MLP sub-module found in {type(raw).__name__}. "
        f"Children: {[n for n, _ in raw.named_children()]}"
    )


def trace_one_prompt(llm, prompt):
    """
    Run ONE trace and save:
      - For each layer i:   layer[i].input[0]   (residual stream into layer i)
      - For each layer i:   layer[i].mlp.output (MLP contribution)
      - For the final norm: norm.input[0]       (residual stream after last layer)

    KEY TRICK: we never touch `layer.output` directly because in nnsight 0.7
    that hook silently cancels the child `mlp.output` hook. Instead we get
    each layer's output from the NEXT layer's input — they're the same tensor.

    Returns three lists:
      layer_in_t[i]   = h going into layer i        (length: n_layers + 1)
      mlp_out_t[i]    = MLP contribution at layer i (length: n_layers)
    The last element of layer_in_t is the final residual after all layers.
    """
    layers = _layers
    norm   = _norm
    n_layers = _n_layers

    saved_layer_in = []   # h_in for each layer + final residual
    saved_mlp_out  = []   # MLP output for each layer

    with torch.no_grad():
        with llm.trace(prompt, remote=llm.remote):
            for li, layer in enumerate(layers):
                saved_layer_in.append(layer.input[0].save())
                mlp = get_mlp_module(layer)
                saved_mlp_out.append(mlp.output.save())

            # The final norm's input is the output of the last decoder layer
            saved_layer_in.append(norm.input[0].save())

    # After trace exits, access .value (or the proxy itself, for some versions)
    def realize(x):
        if hasattr(x, 'value'):
            return x.value
        return x

    layer_in_t = [realize(t).detach() for t in saved_layer_in]
    mlp_out_t  = [realize(t).detach() for t in saved_mlp_out]

    return layer_in_t, mlp_out_t


def to_3d(t):
    """Ensure tensor is [batch, seq, hidden]."""
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t


@ndif_cache_wrapper
def analyze_norms(llm, prompts):
    """
    Strategy: derive everything from layer inputs (residual stream) and
    MLP outputs only — we never touch layer.output or self_attn.output.

    For layer i:
        h_in           = layer_in[i]
        h_out          = layer_in[i+1]            (== layer[i].output[0])
        mlp_contrib    = mlp_out[i]
        h_after_attn   = h_out - mlp_contrib
        attn_contrib   = h_after_attn - h_in
        layer_contrib  = h_out - h_in
    """
    n_layers = _n_layers

    res_norms_acc = torch.zeros(n_layers + 1)
    att_norms_acc = torch.zeros(n_layers)
    mlp_norms_acc = torch.zeros(n_layers)
    cnt = 0

    att_cos_acc      = torch.zeros(n_layers)
    mlp_cos_acc      = torch.zeros(n_layers)
    layer_cos_acc    = torch.zeros(n_layers)
    layer_io_cos_acc = torch.zeros(n_layers)

    max_res_norms = torch.zeros(n_layers + 1)
    max_att_norms = torch.zeros(n_layers)
    max_mlp_norms = torch.zeros(n_layers)

    rel_att_acc   = torch.zeros(n_layers)
    rel_mlp_acc   = torch.zeros(n_layers)
    rel_layer_acc = torch.zeros(n_layers)

    max_rel_att   = torch.zeros(n_layers)
    max_rel_mlp   = torch.zeros(n_layers)
    max_rel_layer = torch.zeros(n_layers)

    for pi, prompt in enumerate(prompts):
        print(f"[{pi+1}/{len(prompts)}]")

        layer_in_t, mlp_out_t = trace_one_prompt(llm, prompt)

        per_res_norms   = []  # [n_layers+1]
        per_att_norms   = []  # [n_layers]
        per_mlp_norms   = []  # [n_layers]
        per_att_cos     = []
        per_mlp_cos     = []
        per_layer_cos   = []
        per_io_cos      = []
        per_rel_att     = []
        per_rel_mlp     = []
        per_rel_layer   = []

        for li in range(n_layers):
            r_in  = to_3d(layer_in_t[li].cpu().float())
            r_out = to_3d(layer_in_t[li + 1].cpu().float())
            m_out = to_3d(mlp_out_t[li].cpu().float())

            h_after_attn = r_out - m_out
            a_out        = h_after_attn - r_in
            layer_diff   = r_out - r_in

            r_in_norm  = r_in.norm(dim=-1).clamp(min=1e-6)
            r_out_norm = r_out.norm(dim=-1).clamp(min=1e-6)
            a_norm     = a_out.norm(dim=-1)
            m_norm     = m_out.norm(dim=-1)
            l_norm     = layer_diff.norm(dim=-1)
            h_aa_norm  = h_after_attn.norm(dim=-1).clamp(min=1e-6)

            if li == 0:
                per_res_norms.append(r_in_norm)
            per_res_norms.append(r_out_norm)

            per_att_norms.append(a_norm)
            per_mlp_norms.append(m_norm)

            per_rel_att.append((a_norm / r_in_norm).squeeze(0))
            per_rel_mlp.append((m_norm / h_aa_norm).squeeze(0))
            per_rel_layer.append((l_norm / r_in_norm).squeeze(0))

            per_att_cos.append(F.cosine_similarity(a_out, r_in, dim=-1).squeeze(0))
            per_mlp_cos.append(F.cosine_similarity(m_out, h_after_attn, dim=-1).squeeze(0))
            per_layer_cos.append(F.cosine_similarity(layer_diff, r_in, dim=-1).squeeze(0))
            per_io_cos.append(F.cosine_similarity(r_out, r_in, dim=-1).squeeze(0))

        # Stack across layers; squeeze batch dim
        res_norms_stack = torch.stack(per_res_norms, dim=0).squeeze(1)  # [L+1, seq]
        att_norms_stack = torch.stack(per_att_norms, dim=0).squeeze(1)  # [L, seq]
        mlp_norms_stack = torch.stack(per_mlp_norms, dim=0).squeeze(1)  # [L, seq]

        seq_len = res_norms_stack.shape[1]
        res_norms_acc += res_norms_stack.sum(dim=1)
        att_norms_acc += att_norms_stack.sum(dim=1)
        mlp_norms_acc += mlp_norms_stack.sum(dim=1)
        cnt += seq_len

        max_res_norms = torch.maximum(max_res_norms, res_norms_stack.max(dim=1).values)
        max_att_norms = torch.maximum(max_att_norms, att_norms_stack.max(dim=1).values)
        max_mlp_norms = torch.maximum(max_mlp_norms, mlp_norms_stack.max(dim=1).values)

        rel_att_stack   = torch.stack(per_rel_att,   dim=0)
        rel_mlp_stack   = torch.stack(per_rel_mlp,   dim=0)
        rel_layer_stack = torch.stack(per_rel_layer, dim=0)

        rel_att_acc   += rel_att_stack.sum(dim=1)
        rel_mlp_acc   += rel_mlp_stack.sum(dim=1)
        rel_layer_acc += rel_layer_stack.sum(dim=1)

        max_rel_att   = torch.maximum(max_rel_att,   rel_att_stack.max(dim=1).values)
        max_rel_mlp   = torch.maximum(max_rel_mlp,   rel_mlp_stack.max(dim=1).values)
        max_rel_layer = torch.maximum(max_rel_layer, rel_layer_stack.max(dim=1).values)

        att_cos_stack   = torch.stack(per_att_cos,   dim=0)
        mlp_cos_stack   = torch.stack(per_mlp_cos,   dim=0)
        layer_cos_stack = torch.stack(per_layer_cos, dim=0)
        io_cos_stack    = torch.stack(per_io_cos,    dim=0)

        att_cos_acc      += att_cos_stack.sum(dim=1)
        mlp_cos_acc      += mlp_cos_stack.sum(dim=1)
        layer_cos_acc    += layer_cos_stack.sum(dim=1)
        layer_io_cos_acc += io_cos_stack.sum(dim=1)

    res_norms = res_norms_acc / cnt
    att_norms = att_norms_acc / cnt
    mlp_norms = mlp_norms_acc / cnt

    att_cos_all      = att_cos_acc      / cnt
    mlp_cos_all      = mlp_cos_acc      / cnt
    layer_cos_all    = layer_cos_acc    / cnt
    layer_io_cos_all = layer_io_cos_acc / cnt

    mean_rel_att   = rel_att_acc   / cnt
    mean_rel_mlp   = rel_mlp_acc   / cnt
    mean_rel_layer = rel_layer_acc / cnt

    return (
        att_norms, mlp_norms, res_norms,
        max_att_norms, max_mlp_norms, max_res_norms,
        mean_rel_att, mean_rel_mlp, mean_rel_layer,
        max_rel_att, max_rel_mlp,
        layer_cos_all, att_cos_all, mlp_cos_all, layer_io_cos_all
    )


# ── Load prompts ──────────────────────────────────────────────────────
prompts = list(LegalDataset())
print(f"Loaded {len(prompts)} legal prompts.")

# ── Clear stale cache ────────────────────────────────────────────────
cache_dir = "cache/ndif_cache/analyze_norms"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared old cache.")

# ── Run ──────────────────────────────────────────────────────────────
(
    att_norms, mlp_norms, res_norms,
    max_att_norms, max_mlp_norms, max_res_norms,
    mean_relative_contribution_att,
    mean_relative_contribution_mlp,
    mean_relative_contribution_layer,
    max_relative_contribution_att,
    max_relative_contribution_mlp,
    layer_cos_all, att_cos_all,
    mlp_cos_all, layer_io_cos_all
) = analyze_norms(llm, prompts)

# ── Plotting ──────────────────────────────────────────────────────────
W_BAR   = 1.1
W, H    = 6, 3
x_range = list(range(_n_layers))


def set_xlim(l):
    plt.xlim(-0.5, l - 0.5)


plt.figure(figsize=(W, H))
bars = []
bars.append(plt.bar(x_range, att_norms.float().cpu().numpy(),
                    label=r"Attention: $\|a_l\|_2$", width=W_BAR))
bars.append(plt.bar(x_range, mlp_norms.float().cpu().numpy(),
                    label=r"MLP: $\|m_l\|_2$", width=W_BAR))
bars.append(plt.bar(x_range, res_norms[:-1].float().cpu().numpy(),
                    label=r"Residual: $\|h_l\|_2$", width=W_BAR))
plt.xlabel("Layer index ($l$)")
plt.ylabel("Mean Norm")
plt.legend()
sort_zorder(bars)
set_xlim(_n_layers)
plt.savefig(os.path.join(target_dir, f"{model_name}_mean_norms.pdf"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(W, H))
bars = []
bars.append(plt.bar(x_range, max_att_norms.float().cpu().numpy(),
                    label=r"Attention $a_l$", width=W_BAR))
bars.append(plt.bar(x_range, max_mlp_norms.float().cpu().numpy(),
                    label=r"MLP $m_l$", width=W_BAR))
bars.append(plt.bar(x_range, max_res_norms[:-1].float().cpu().numpy(),
                    label=r"Residual $h_l$", width=W_BAR))
plt.xlabel("Layer index ($l$)")
plt.ylabel("Max Norm")
plt.legend()
sort_zorder(bars)
set_xlim(_n_layers)
plt.savefig(os.path.join(target_dir, f"{model_name}_max_norms.pdf"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(W, H))
bars = []
bars.append(plt.bar(x_range, mean_relative_contribution_att.float().cpu().numpy(),
                    label=r"Attention: $\|a_l\|_2/\|h_l\|_2$", width=W_BAR))
bars.append(plt.bar(x_range, mean_relative_contribution_mlp.float().cpu().numpy(),
                    label=r"MLP: $\|m_l\|_2/\|h_l + a_l\|_2$", width=W_BAR))
bars.append(plt.bar(x_range, mean_relative_contribution_layer.float().cpu().numpy(),
                    label=r"Attention + MLP: $\|a_l + m_l\|_2/\|h_l\|_2$",
                    width=W_BAR))
plt.legend()
sort_zorder(bars)
set_xlim(_n_layers)
if max(
    mean_relative_contribution_att.max().item(),
    mean_relative_contribution_mlp.max().item(),
    mean_relative_contribution_layer.max().item()
) > 1.5:
    plt.ylim(0, 1.5)
plt.xlabel("Layer index ($l$)")
plt.ylabel("Mean Relative Contribution")
plt.savefig(os.path.join(target_dir, f"{model_name}_mean_relative_contribution.pdf"),
            bbox_inches="tight")
plt.close()

plt.figure(figsize=(W, H))
bars = []
bars.append(plt.bar(x_range, max_relative_contribution_att.float().cpu().numpy(),
                    label=r"Attention $a_l$", width=W_BAR))
bars.append(plt.bar(x_range, max_relative_contribution_mlp.float().cpu().numpy(),
                    label=r"MLP $m_l$", width=W_BAR))
plt.ylim(0, 2)
plt.xlabel("Layer index ($l$)")
plt.ylabel("Max Relative Contribution")
plt.legend()
sort_zorder(bars)
set_xlim(_n_layers)
plt.savefig(os.path.join(target_dir, f"{model_name}_max_relative_contribution.pdf"),
            bbox_inches="tight")
plt.close()

plt.figure(figsize=(W, H))
bars = []
bars.append(plt.bar(x_range, att_cos_all.float().cpu().numpy(),
                    label=r"Attention: $\mathrm{cossim}(a_l, h_l)$", width=W_BAR))
bars.append(plt.bar(x_range, mlp_cos_all.float().cpu().numpy(),
                    label=r"MLP: $\mathrm{cossim}(m_l, h_l + a_l)$", width=W_BAR))
bars.append(plt.bar(x_range, layer_cos_all.float().cpu().numpy(),
                    label=r"Attention + MLP: $\mathrm{cossim}(a_l + m_l, h_l)$",
                    width=W_BAR))
plt.xlabel("Layer index ($l$)")
plt.ylabel("Cosine similarity")
plt.legend()
sort_zorder(bars)
set_xlim(_n_layers)
plt.savefig(os.path.join(target_dir, f"{model_name}_avg_cossims.pdf"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(W, H))
plt.bar(x_range, layer_io_cos_all.float().cpu().numpy(),
        label=r"Attention + MLP $a_l + m_l$")
plt.xlabel("Layer index ($l$)")
plt.ylabel("Cosine similarity")
set_xlim(_n_layers)
plt.savefig(os.path.join(target_dir, f"{model_name}_avg_io_cossims.pdf"), bbox_inches="tight")
plt.close()

print(f"All plots saved to {target_dir}/")