# python3 verify_models.py

import torch
from analysis.lib.models import create_model
from analysis.lib.model_compat import get_layers, get_norm, get_lm_head, get_embed_tokens

for model_name in ["qwen3_8b", "gemma_4_e4b"]:
    print(f"\n{'='*60}")
    print(f"Checking: {model_name}")
    print(f"{'='*60}")

    llm = create_model(model_name, force_local=True)
    llm.eval()

    layers      = get_layers(llm)
    norm        = get_norm(llm)
    lm_head     = get_lm_head(llm)
    embed_tokens = get_embed_tokens(llm)

    print(f"  Number of layers : {len(layers)}")
    print(f"  Layer type       : {type(layers[0]).__name__}")
    print(f"  Norm type        : {type(norm).__name__}")
    print(f"  LM head type     : {type(lm_head).__name__}")
    print(f"  Embed tokens     : {type(embed_tokens).__name__}")
    print(f"  is_nested        : {getattr(llm, 'is_nested', False)}")
    print(f"  ✓ All paths OK")