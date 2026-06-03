from nnsight import LanguageModel
import torch


remote_model_table = {
    "llama_3.1_8b":  "meta-llama/Meta-Llama-3.1-8B",
    "llama_3.1_70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama_3.1_405b":"meta-llama/Meta-Llama-3.1-405B",
}

local_model_table = {
    "qwen2.5_14b":            ("Qwen/Qwen2.5-14B",              True),
    "qwen2.5_7b":             ("Qwen/Qwen2.5-7B",               True),
    "qwen2.5_1.5b":           ("Qwen/Qwen2.5-1.5B",             True),
    "qwen3_8b":               ("Qwen/Qwen3-8B",                 False),
    "qwen3_14b":              ("Qwen/Qwen3-14B",                True),
    "qwen3_32b":              ("Qwen/Qwen3-32B",                True),
    "llama_3.1_8b":           ("meta-llama/Meta-Llama-3.1-8B",  True),
    "llama_3.1_405b":         ("meta-llama/Meta-Llama-3.1-405B",True),
    "llama_3.1_8b_instruct":  ("meta-llama/Meta-Llama-3.1-8B-Instruct",  True),
    "llama_3.1_70b_instruct": ("meta-llama/Meta-Llama-3.1-70B-Instruct", True),

    # Gemma 4 — False because it's already natively quantized (E4B)
    "gemma_4_e4b":            ("google/gemma-4-E4B-it",         False),
}

# Models that need path remapping because their text layers
# are not at model.layers but nested deeper.
# value = the attribute chain to reach the language model sub-module
nested_language_model_table = {
    "gemma_4_e4b": "language_model",   # model.language_model.layers
}


def get_model(model_name):
    return remote_model_table[model_name]


def create_model(model_name, force_local=False):
    if (not force_local) and (model_name in remote_model_table):
        model = LanguageModel(remote_model_table[model_name], device_map="auto")
        model.remote = True

    elif model_name in local_model_table:
        model_path, use_quantization = local_model_table[model_name]

        if use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        model = LanguageModel(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            dispatch=False
        )
        model.remote = False

        # If this model has its text layers nested under a sub-module,
        # attach a convenience .text_model attribute so all analysis
        # scripts can use a single consistent access pattern.
        if model_name in nested_language_model_table:
            sub_attr = nested_language_model_table[model_name]
            # model.model.language_model → attach as model.model.text_model
            # so analysis scripts can always use llm.model.text_model.layers
            sub_module = getattr(model.model, sub_attr)
            model.model.text_model = sub_module
            model.is_nested = True
        else:
            # For Llama/Qwen the text model IS model.model directly
            model.model.text_model = model.model
            model.is_nested = False

    else:
        raise ValueError(f"Model {model_name} not found")

    return model