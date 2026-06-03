from nnsight import LanguageModel
import os
import pickle
import hashlib
import nnsight
import torch
import numpy as np


def md5_of_string(s: str) -> str:
    b = s.encode('utf-8')
    m = hashlib.md5()
    m.update(b)
    return m.hexdigest()


def _is_nnsight_proxy(obj):
    """
    Check if an object is an nnsight InterventionProxy, robust across
    nnsight versions where the module path has changed:
      - 0.4.x:  nnsight.intervention.graph.proxy.InterventionProxy
      - 0.7.x:  nnsight.intervention.InterventionProxy
    """
    # Walk the MRO of the object's type and check class names
    try:
        for cls in type(obj).__mro__:
            qualname = f"{cls.__module__}.{cls.__name__}"
            if "InterventionProxy" in cls.__name__:
                return True
            if "nnsight" in cls.__module__ and "Proxy" in cls.__name__:
                return True
    except Exception:
        pass
    return False


def ndif_cache_wrapper(func):
    ndif_cache_dir = "cache/ndif_cache"

    def format_arg(arg):
        if isinstance(arg, str):
            return md5_of_string(arg)
        elif isinstance(arg, (int, float)):
            return str(arg)
        elif isinstance(arg, LanguageModel):
            return arg.config.name_or_path.replace("/", "_") + "_" + str(getattr(arg, "remote", True))
        elif isinstance(arg, list):
            return "[" + (",".join([format_arg(item) for item in arg])) + "]"
        elif isinstance(arg, tuple):
            return "(" + (",".join([format_arg(item) for item in arg])) + ")"
        elif arg is None:
            return "None"
        else:
            raise ValueError(f"Unsupported argument type: {type(arg)}")

    def fix_result(results):
        # Containers first
        if isinstance(results, list):
            return list([fix_result(r) for r in results])
        elif isinstance(results, tuple):
            return tuple(fix_result(r) for r in results)
        elif isinstance(results, dict):
            return {k: fix_result(v) for k, v in results.items()}
        # Plain tensors and scalars pass through
        elif torch.is_tensor(results):
            return results
        elif isinstance(results, (np.ndarray, int, float, str)):
            return results
        elif results is None:
            return None
        # nnsight proxy — convert to a real tensor
        elif _is_nnsight_proxy(results):
            # In newer nnsight, proxies expose .value after the trace.
            # Fall back to attribute probing.
            if hasattr(results, "value"):
                val = results.value
                if torch.is_tensor(val):
                    return val.cpu()
                return val
            if hasattr(results, "cpu"):
                return results.cpu()
            return results
        else:
            # Last-ditch: if it quacks like a tensor (has shape and detach), keep it
            if hasattr(results, "shape") and hasattr(results, "detach"):
                try:
                    return results.detach().cpu() if hasattr(results, "cpu") else results.detach()
                except Exception:
                    pass
            raise ValueError(f"Unsupported result type: {type(results)}")

    def wrapper(*args, **kwargs):
        name = ""
        for arg in args:
            name += f"_{format_arg(arg)}"
        for k in sorted(kwargs.keys()):
            name += f"_{k}={format_arg(kwargs[k])}"

        if kwargs.get("cache_name") is not None:
            name = kwargs["cache_name"] + "_" + name
            del kwargs["cache_name"]

        if len(name) > 250:
            name = md5_of_string(name)

        cache_name = os.path.join(ndif_cache_dir, func.__name__, f"{name}.pkl")
        os.makedirs(os.path.dirname(cache_name), exist_ok=True)
        print(f"Registering cache for {func.__name__}: {cache_name}")

        if os.path.exists(cache_name):
            print(f"Loading cache for {func.__name__}: {cache_name}")
            with open(cache_name, "rb") as f:
                res = pickle.load(f)
            print("  Loaded cache.")
            return res
        else:
            print(f"Running {func.__name__}: {cache_name}")
            result = func(*args, **kwargs)
            result = fix_result(result)
            with open(cache_name, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            return result

    return wrapper