from lib.matplotlib_config import sort_zorder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sys
import nnsight
nnsight.CONFIG.API.APIKEY = os.environ["NDIF_TOKEN"]
import torch
import torch.nn.functional as F
from nnsight import LanguageModel

from lib.models import create_model
from lib.nnsight_tokenize import tokenize
from lib.ndif_cache import ndif_cache_wrapper
from lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens, set_eval
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

# ── Resolve model paths BEFORE any session/trace block ───────────────
_layers   = get_layers(llm)
_n_layers = len(_layers)
# ─────────────────────────────────────────────────────────────────────


def get_attn_module(layer):
    """
    Return the attention sub-module. Handles naming across model families:
      Llama / Qwen / Gemma all use self_attn, but we probe to be safe.
    """
    raw = layer._module
    for candidate in ["self_attn", "attn", "attention"]:
        if hasattr(raw, candidate):
            return getattr(layer, candidate)
    raise AttributeError(
        f"Cannot find attention sub-module in {type(raw).__name__}. "
        f"Children: {[n for n, _ in raw.named_children()]}"
    )


def get_mlp_module(layer):
    """Return the MLP sub-module."""
    raw = layer._module
    for candidate in ["mlp", "feed_forward", "ffn"]:
        if hasattr(raw, candidate):
            return getattr(layer, candidate)
    raise AttributeError(
        f"Cannot find MLP sub-module in {type(raw).__name__}. "
        f"Children: {[n for n, _ in raw.named_children()]}"
    )


def get_attn_output_tensor(attn_module):
    """
    Return the attention hidden-state output as a plain tensor.
    Some models return a tuple (hidden, cache, ...), others a plain tensor.
    """
    out = attn_module.output
    if isinstance(out, tuple):
        return out[0]
    return out


def probe_layer_attrs(llm, layers, prompt):
    """Quick trace to confirm attribute names work before the full run."""
    print("Probing layer attribute names...")
    try:
        with torch.no_grad():
            with llm.trace(prompt, remote=llm.remote):
                l0   = layers[0]
                attn = get_attn_module(l0)
                mlp  = get_mlp_module(l0)
                a_shape = get_attn_output_tensor(attn).shape.save()
                m_shape = mlp.output.shape.save()
                r_in_s  = l0.inputs[0][0].shape.save()
                r_out_s = l0.output[0].shape.save()
        print(f"  layer input  shape : {r_in_s}")
        print(f"  layer output shape : {r_out_s}")
        print(f"  attn  output shape : {a_shape}")
        print(f"  mlp   output shape : {m_shape}")
        print("  ✓ Attribute probe OK")
        return True
    except Exception as e:
        print(f"  ✗ Probe failed: {type(e).__name__}: {str(e)[:300]}")
        return False


@ndif_cache_wrapper
def analyze_norms(llm, prompts):
    layers   = _layers
    n_layers = _n_layers

    with llm.session(remote=llm.remote) as session:
        with torch.no_grad():
            res_norms_all = 0
            att_norms_all = 0
            mlp_norms_all = 0
            cnt = 0

            att_cos_all      = 0
            mlp_cos_all      = 0
            layer_cos_all    = 0
            layer_io_cos_all = 0

            max_res_norms = torch.zeros(1)
            max_att_norms = torch.zeros(1)
            max_mlp_norms = torch.zeros(1)

            mean_relative_contribution_att   = 0
            mean_relative_contribution_mlp   = 0
            mean_relative_contribution_layer = 0

            max_relative_contribution_att   = torch.zeros(1)
            max_relative_contribution_mlp   = torch.zeros(1)
            max_relative_contribution_layer = torch.zeros(1)

            for pi, prompt in enumerate(prompts):
                print(f"[{pi+1}/{len(prompts)}]")
                with llm.trace(prompt, remote=llm.remote):
                    residual_log = []
                    att_outputs  = []
                    mlp_outputs  = []

                    att_cos      = []
                    mlp_cos      = []
                    layer_cos    = []
                    layer_io_cos = []

                    relative_contribution_att   = []
                    relative_contribution_mlp   = []
                    relative_contribution_layer = []

                    for li, layer in enumerate(layers):
                        attn = get_attn_module(layer)
                        mlp  = get_mlp_module(layer)

                        r_in  = layer.inputs[0][0].detach()
                        r_out = layer.output[0].detach()
                        a_out = get_attn_output_tensor(attn).detach()
                        m_out = mlp.output.detach()

                        if li == 0:
                            residual_log.clear()
                            residual_log.append(r_in)

                        residual_log.append(r_out)
                        att_outputs.append(a_out)
                        mlp_outputs.append(m_out)

                        relative_contribution_att.append(
                            a_out.norm(dim=-1).cpu().float()
                            / r_in.norm(dim=-1).clamp(min=1e-6).cpu().float()
                        )

                        mlp_input = (a_out + r_in).detach()

                        relative_contribution_mlp.append(
                            m_out.norm(dim=-1).cpu().float()
                            / mlp_input.norm(dim=-1).clamp(min=1e-6).cpu().float()
                        )

                        layer_diff = (r_out - r_in).detach()

                        relative_contribution_layer.append(
                            layer_diff.norm(dim=-1).cpu().float()
                            / r_in.norm(dim=-1).clamp(min=1e-6).cpu().float()
                        )

                        att_cos.append(
                            F.cosine_similarity(a_out, r_in, dim=-1).sum(1).cpu().float()
                        )
                        mlp_cos.append(
                            F.cosine_similarity(m_out, mlp_input, dim=-1).sum(1).cpu().float()
                        )
                        layer_cos.append(
                            F.cosine_similarity(layer_diff, r_in, dim=-1).sum(1).cpu().float()
                        )
                        layer_io_cos.append(
                            F.cosine_similarity(r_out, r_in, dim=-1).sum(1).cpu().float()
                        )

                    r = torch.cat(residual_log, dim=0).norm(dim=-1).cpu().float()
                    a = torch.cat(att_outputs,  dim=0).norm(dim=-1).cpu().float()
                    m = torch.cat(mlp_outputs,  dim=0).norm(dim=-1).cpu().float()

                    res_norms = r.sum(dim=1) + res_norms_all
                    att_norms = a.sum(dim=1) + att_norms_all
                    mlp_norms = m.sum(dim=1) + mlp_norms_all
                    cnt += r.shape[1]

                    max_res_norms = torch.maximum(max_res_norms, r.max(dim=1).values)
                    max_att_norms = torch.maximum(max_att_norms, a.max(dim=1).values)
                    max_mlp_norms = torch.maximum(max_mlp_norms, m.max(dim=1).values)

                    relative_contribution_att   = torch.cat(relative_contribution_att,   dim=0)
                    relative_contribution_mlp   = torch.cat(relative_contribution_mlp,   dim=0)
                    relative_contribution_layer = torch.cat(relative_contribution_layer, dim=0)

                    mean_relative_contribution_att   += relative_contribution_att.sum(dim=1)
                    mean_relative_contribution_mlp   += relative_contribution_mlp.sum(dim=1)
                    mean_relative_contribution_layer += relative_contribution_layer.sum(dim=1)

                    max_relative_contribution_att = torch.maximum(
                        max_relative_contribution_att,
                        relative_contribution_att.max(dim=1).values
                    )
                    max_relative_contribution_mlp = torch.maximum(
                        max_relative_contribution_mlp,
                        relative_contribution_mlp.max(dim=1).values
                    )
                    max_relative_contribution_layer = torch.maximum(
                        max_relative_contribution_layer,
                        relative_contribution_layer.max(dim=1).values
                    )

                    att_cos_all      += torch.cat(att_cos,      dim=0)
                    mlp_cos_all      += torch.cat(mlp_cos,      dim=0)
                    layer_cos_all    += torch.cat(layer_cos,    dim=0)
                    layer_io_cos_all += torch.cat(layer_io_cos, dim=0)

            res_norms = (res_norms / cnt).save()
            att_norms = (att_norms / cnt).save()
            mlp_norms = (mlp_norms / cnt).save()

            att_cos_all      = (att_cos_all      / cnt).save()
            mlp_cos_all      = (mlp_cos_all      / cnt).save()
            layer_cos_all    = (layer_cos_all    / cnt).save()
            layer_io_cos_all = (layer_io_cos_all / cnt).save()

            mean_relative_contribution_att   = (mean_relative_contribution_att   / cnt).save()
            mean_relative_contribution_mlp   = (mean_relative_contribution_mlp   / cnt).save()
            mean_relative_contribution_layer = (mean_relative_contribution_layer / cnt).save()

            max_att_norms = max_att_norms.save()
            max_mlp_norms = max_mlp_norms.save()
            max_res_norms = max_res_norms.save()

            max_relative_contribution_att   = max_relative_contribution_att.save()
            max_relative_contribution_mlp   = max_relative_contribution_mlp.save()

    return (
        att_norms.cpu(), mlp_norms.cpu(), res_norms.cpu(),
        max_att_norms.cpu(), max_mlp_norms.cpu(), max_res_norms.cpu(),
        mean_relative_contribution_att.cpu(),
        mean_relative_contribution_mlp.cpu(),
        mean_relative_contribution_layer.cpu(),
        max_relative_contribution_att.cpu(),
        max_relative_contribution_mlp.cpu(),
        layer_cos_all.cpu(), att_cos_all.cpu(),
        mlp_cos_all.cpu(), layer_io_cos_all.cpu()
    )


# ── Load prompts ──────────────────────────────────────────────────────
prompts = list(LegalDataset(
    json_path="samples.json",
    max_samples=N_EXAMPLES,
    include_steps=False,
    use_chat_template=True,
))
print(f"Loaded {len(prompts)} legal prompts.")

# ── Probe to confirm attribute names before expensive run ─────────────
probe_ok = probe_layer_attrs(llm, _layers, prompts[0])
if not probe_ok:
    print("WARNING: probe failed — the run may still crash. Check attribute names.")

# ── Run ───────────────────────────────────────────────────────────────
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
                    label="Attention: $||\\bm{a}_l||_2$", width=W_BAR))
bars.append(plt.bar(x_range, mlp_norms.float().cpu().numpy(),
                    label="MLP: $||\\bm{m}_l||_2$", width=W_BAR))
bars.append(plt.bar(x_range, res_norms[:-1].float().cpu().numpy(),
                    label="Residual: $||\\bm{h}_{l}||_2$", width=W_BAR))
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
                    label="Attention $\\bm{a}_l$", width=W_BAR))
bars.append(plt.bar(x_range, max_mlp_norms.float().cpu().numpy(),
                    label="MLP $\\bm{m}_l$", width=W_BAR))
bars.append(plt.bar(x_range, max_res_norms[:-1].float().cpu().numpy(),
                    label="Residual $\\bm{h}_{l}$", width=W_BAR))
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
                    label="Attention: $||\\bm{a}_l||_2/||\\bm{h}_l||_2$", width=W_BAR))
bars.append(plt.bar(x_range, mean_relative_contribution_mlp.float().cpu().numpy(),
                    label="MLP: $||\\bm{m}_l||_2/||\\bm{h}_l + \\bm{a}_l||_2$", width=W_BAR))
bars.append(plt.bar(x_range, mean_relative_contribution_layer.float().cpu().numpy(),
                    label="Attention + MLP: $||\\bm{a}_l + \\bm{m}_l||_2/||\\bm{h}_{l}||_2$",
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
                    label="Attention $\\bm{a}_l$", width=W_BAR))
bars.append(plt.bar(x_range, max_relative_contribution_mlp.float().cpu().numpy(),
                    label="MLP $\\bm{m}_l$", width=W_BAR))
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
                    label="Attention: $\\text{cossim}(\\bm{a}_l, \\bm{h}_l)$", width=W_BAR))
bars.append(plt.bar(x_range, mlp_cos_all.float().cpu().numpy(),
                    label="MLP: $\\text{cossim}(\\bm{m}_l, \\bm{h}_l + \\bm{a}_l)$", width=W_BAR))
bars.append(plt.bar(x_range, layer_cos_all.float().cpu().numpy(),
                    label="Attention + MLP: $\\text{cossim}(\\bm{a}_l + \\bm{m}_l, \\bm{h}_l)$",
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
        label="Attention + MLP $\\bm{a}_l + \\bm{m}_l$")
plt.xlabel("Layer index ($l$)")
plt.ylabel("Cosine similarity")
set_xlim(_n_layers)
plt.savefig(os.path.join(target_dir, f"{model_name}_avg_io_cossims.pdf"), bbox_inches="tight")
plt.close()

print(f"All plots saved to {target_dir}/")