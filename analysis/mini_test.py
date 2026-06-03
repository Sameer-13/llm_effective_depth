import sys
import torch
from lib.models import create_model
from lib.model_compat import get_layers, set_eval

model_name = sys.argv[1]
llm = create_model(model_name)
set_eval(llm)

layers = get_layers(llm)
layer0 = layers[0]
mlp = layer0.mlp

prompt = "The capital of France is"

print("\n=== Test 1: probe mlp.output via .shape.save() ===")
try:
    with torch.no_grad():
        with llm.trace(prompt, remote=False):
            x = mlp.output.shape.save()
    print(f"  ✓ Got shape: {x}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:200]}")

print("\n=== Test 2: save mlp.output directly ===")
try:
    with torch.no_grad():
        with llm.trace(prompt, remote=False):
            x = mlp.output.save()
    print(f"  ✓ Got tensor of shape: {x.shape if hasattr(x, 'shape') else 'no shape'}")
    print(f"  value attribute: {x.value.shape if hasattr(x, 'value') else 'no .value'}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:300]}")

print("\n=== Test 3: save layer.output[0] + mlp.output together ===")
try:
    with torch.no_grad():
        with llm.trace(prompt, remote=False):
            a = layer0.output[0].save()
            b = mlp.output.save()
    print(f"  ✓ layer.output[0] shape: {a.shape if hasattr(a, 'shape') else 'unknown'}")
    print(f"  ✓ mlp.output    shape: {b.shape if hasattr(b, 'shape') else 'unknown'}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:300]}")

print("\n=== Test 4: loop over ALL layers, save mlp.output for each ===")
try:
    saved = []
    with torch.no_grad():
        with llm.trace(prompt, remote=False):
            for li, layer in enumerate(layers):
                m = layer.mlp.output.save()
                saved.append(m)
    print(f"  ✓ Saved {len(saved)} mlp outputs")
    print(f"  First shape: {saved[0].shape if hasattr(saved[0], 'shape') else 'unknown'}")
    print(f"  Last  shape: {saved[-1].shape if hasattr(saved[-1], 'shape') else 'unknown'}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:300]}")

print("\n=== Test 5: loop over all layers, save layer.input[0], layer.output[0], mlp.output ===")
try:
    saved_in, saved_out, saved_mlp = [], [], []
    with torch.no_grad():
        with llm.trace(prompt, remote=False):
            for li, layer in enumerate(layers):
                saved_in.append(layer.input[0].save())
                saved_out.append(layer.output[0].save())
                saved_mlp.append(layer.mlp.output.save())
    print(f"  ✓ Saved {len(saved_in)} layer inputs")
    print(f"  ✓ Saved {len(saved_out)} layer outputs")
    print(f"  ✓ Saved {len(saved_mlp)} mlp outputs")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:300]}")