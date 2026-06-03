# python3 verify.py

import sys
import torch

# Increase recursion limit as a safety measure
sys.setrecursionlimit(10000)

from lib.models import create_model
from lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens

for model_name in ["qwen3_8b", "gemma_4_e4b"]:
    print(f"\n{'='*60}")
    print(f"Checking: {model_name}")
    print(f"{'='*60}")

    try:
        llm = create_model(model_name, force_local=True)
        llm.eval()

        # Access the underlying torch module directly to avoid
        # nnsight's broken __repr__ on large models
        raw_model = llm._model   # the actual nn.Module, not the nnsight wrapper

        layers       = get_layers(llm)
        norm         = get_norm(llm)
        lm_head      = get_lm_head(llm)
        embed_tokens = get_embed_tokens(llm)

        # Use type().__name__ only — never print the objects themselves
        print(f"  Number of layers : {len(layers)}")
        print(f"  Layer[0] type    : {type(layers[0]._module).__name__}")
        print(f"  Norm type        : {type(norm._module).__name__}")
        print(f"  LM head type     : {type(lm_head._module).__name__}")
        print(f"  Embed tokens     : {type(embed_tokens._module).__name__}")
        print(f"  is_nested        : {getattr(llm, 'is_nested', False)}")

        # Quick forward pass sanity check
        print(f"  Running quick forward pass...")
        with torch.no_grad():
            with llm.trace("Hello world", remote=False):
                # just grab the output of layer 0 to confirm hooks work
                layer0_out = layers[0].output.save()

        # layer0_out is now a tensor (or tuple)
        if isinstance(layer0_out, tuple):
            shape = layer0_out[0].shape
        else:
            shape = layer0_out.shape

        print(f"  Layer 0 output shape: {shape}")
        print(f"  ✓ All paths OK for {model_name}")

        # Explicitly delete to free memory before loading next model
        del llm
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()