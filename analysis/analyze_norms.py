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


def probe_layer_attrs(llm, layers, prompt):
    """
    Probe what nnsight can actually intercept for this model.
    Tests both .input and .inputs to find which one works.
    Reports results without crashing.
    """
    print("Probing layer attribute names and nnsight access patterns...")

    # Test 1: can we read layer input via .input[0]?
    try:
        with torch.no_grad():
            with llm.trace(prompt, remote=llm.remote):
                val = layers[0].input[0].shape.save()
        print(f"  layer.input[0]     shape : {val}  ✓")
        input_attr = "input[0]"
    except Exception as e:
        print(f"  layer.input[0]     → FAILED: {str(e)[:80]}")
        input_attr = None

    # Test 2: can we read layer input via .inputs[0][0]?
    if input_attr is None:
        try:
            with torch.no_grad():
                with llm.trace(prompt, remote=llm.remote):
                    val = layers[0].inputs[0][0].shape.save()
            print(f"  layer.inputs[0][0] shape : {val}  ✓")
            input_attr = "inputs[0][0]"
        except Exception as e:
            print(f"  layer.inputs[0][0] → FAILED: {str(e)[:80]}")

    # Test 3: can we read layer output via .output[0]?
    try:
        with torch.no_grad():
            with llm.trace(prompt, remote=llm.remote):
                val = layers[0].output[0].shape.save()
        print(f"  layer.output[0]    shape : {val}  ✓")
    except Exception as e:
        print(f"  layer.output[0]    → FAILED: {str(e)[:80]}")

    # Test 4: check mlp sub-module
    for mlp_name in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layers[0]._module, mlp_name):
            try:
                mlp_mod = getattr(layers[0], mlp_name)
                with torch.no_grad():
                    with llm.trace(prompt, remote=llm.remote):
                        val = mlp_mod.output.shape.save()
                print(f"  layer.{mlp_name}.output shape : {val}  ✓")
                break
            except Exception as e:
                print(f"  layer.{mlp_name}.output → FAILED: {str(e)[:80]}")

    print(f"  Detected input access pattern: {input_attr}")
    return input_attr


def get_layer_input(layer, input_attr):
    """Get the residual stream coming INTO this layer."""
    if input_attr == "input[0]":
        return layer.input[0]
    else:
        return layer.inputs[0][0]


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


@ndif_cache_wrapper
def analyze_norms(llm, prompts, input_attr):
    """
    Compute residual stream and sublayer contribution norms.

    Strategy: derive attention contribution from residual differences
    rather than hooking into self_attn.output directly. This is more
    reliable across model families.

    For a pre-LN transformer:
        h_after_attn  = layer_output_after_attention_residual_add
        attn_contrib  = h_after_attn - h_in
        mlp_contrib   = layer_output - h_after_attn
        layer_contrib = layer_output - h_in

    nnsight exposes intermediate states via sub-module hooks.
    We hook the MLP input (= h_after_attn) to get the split point.
    """
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
                    residual_log  = []   # h_0, h_1, ..., h_L  (layer outputs)
                    mlp_input_log = []   # h_after_attn for each layer
                    mlp_output_log = []  # mlp output for each layer

                    att_cos      = []
                    mlp_cos      = []
                    layer_cos    = []
                    layer_io_cos = []

                    relative_contribution_att   = []
                    relative_contribution_mlp   = []
                    relative_contribution_layer = []

                    for li, layer in enumerate(layers):
                        mlp = get_mlp_module(layer)

                        # h coming into this layer
                        r_in = get_layer_input(layer, input_attr).detach()

                        # h after the full layer (attn residual + mlp residual)
                        r_out = layer.output[0].detach()

                        # h after attn residual add = mlp's input
                        # In pre-LN: mlp takes the normed version, but the
                        # residual add happens BEFORE norm, so mlp.input[0]
                        # is the normed state. We want the un-normed residual,
                        # which we can get as r_out - mlp.output
                        m_out = mlp.output.detach()

                        # h_after_attn = r_out - mlp_contribution
                        h_after_attn = (r_out - m_out).detach()

                        # contributions
                        a_out      = (h_after_attn - r_in).detach()   # attn contribution
                        layer_diff = (r_out - r_in).detach()           # full layer contribution

                        if li == 0:
                            residual_log.clear()
                            residual_log.append(r_in)

                        residual_log.append(r_out)
                        mlp_input_log.append(h_after_attn)
                        mlp_output_log.append(m_out)

                        relative_contribution_att.append(
                            a_out.norm(dim=-1).cpu().float()
                            / r_in.norm(dim=-1).clamp(min=1e-6).cpu().float()
                        )
                        relative_contribution_mlp.append(
                            m_out.norm(dim=-1).cpu().float()
                            / h_after_attn.norm(dim=-1).clamp(min=1e-6).cpu().float()
                        )
                        relative_contribution_layer.append(
                            layer_diff.norm(dim=-1).cpu().float()
                            / r_in.norm(dim=-1).clamp(min=1e-6).cpu().float()
                        )

                        att_cos.append(
                            F.cosine_similarity(a_out, r_in, dim=-1).sum(1).cpu().float()
                        )
                        mlp_cos.append(
                            F.cosine_similarity(m_out, h_after_attn, dim=-1).sum(1).cpu().float()
                        )
                        layer_cos.append(
                            F.cosine_similarity(layer_diff, r_in, dim=-1).sum(1).cpu().float()
                        )
                        layer_io_cos.append(
                            F.cosine_similarity(r_out, r_in, dim=-1).sum(1).cpu().float()
                        )

                    r = torch.cat(residual_log,   dim=0).norm(dim=-1).cpu().float()
                    a = torch.cat(mlp_input_log,  dim=0).norm(dim=-1).cpu().float()
                    m = torch.cat(mlp_output_log, dim=0).norm(dim=-1).cpu().float()

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
prompts = list(LegalDataset())
print(f"Loaded {len(prompts)} legal prompts.")

# ── Probe to find correct nnsight input access pattern ────────────────
# Delete old cache so we don't reuse a broken result
import shutil
cache_dir = "cache/ndif_cache/analyze_norms"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared old cache.")

input_attr = probe_layer_attrs(llm, _layers, prompts[0])
if input_attr is None:
    raise RuntimeError(
        "Could not determine how to access layer inputs for this model. "
        "Check the probe output above."
    )

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
) = analyze_norms(llm, prompts, input_attr)

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