from nnsight import LanguageModel, VisionLanguageModel
import torch


remote_model_table = {
    "llama_3.1_8b":   "meta-llama/Meta-Llama-3.1-8B",
    "llama_3.1_70b":  "meta-llama/Meta-Llama-3.1-70B",
    "llama_3.1_405b": "meta-llama/Meta-Llama-3.1-405B",
}

local_model_table = {
    "qwen2.5_14b":            ("Qwen/Qwen2.5-14B",               True,  "language"),
    "qwen2.5_7b":             ("Qwen/Qwen2.5-7B",                True,  "language"),
    "qwen2.5_1.5b":           ("Qwen/Qwen2.5-1.5B",              True,  "language"),
    "qwen3_8b":               ("Qwen/Qwen3-8B",                  False, "language"),
    "qwen3_14b":              ("Qwen/Qwen3-14B",                 True,  "language"),
    "qwen3_32b":              ("Qwen/Qwen3-32B",                 True,  "language"),
    "llama_3.1_8b":           ("meta-llama/Meta-Llama-3.1-8B",   False,  "language"),
    "llama_3.1_405b":         ("meta-llama/Meta-Llama-3.1-405B", True,  "language"),
    "llama_3_1_8b_instruct":  ("meta-llama/Meta-Llama-3.1-8B-Instruct",  False, "language"),
    "llama_3.1_70b_instruct": ("meta-llama/Meta-Llama-3.1-70B-Instruct", True, "language"),
    "allam_7b_instruct": ("humain-ai/ALLaM-7B-Instruct-preview", False, "language"),
    "gemma_4_e4b": ("google/gemma-4-E4B-it", False, "vision"),
}

text_submodule_table = {
    "gemma_4_e4b": "language_model",
}


def get_model(model_name):
    return remote_model_table[model_name]


# Models whose tokenizer is SentencePiece-based (tokenizer.model file)
# and conflicts with the transformers 5.x tiktoken auto-converter.
# These need use_fast=False to load via the legacy Python tokenizer.
_NEEDS_SLOW_TOKENIZER = {"allam_7b_instruct"}


def create_model(model_name, force_local=False):
    if (not force_local) and (model_name in remote_model_table):
        model = LanguageModel(remote_model_table[model_name], device_map="auto")
        model.remote = True

    elif model_name in local_model_table:
        model_path, use_quantization, model_type = local_model_table[model_name]

        if use_quantization:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        # IMPORTANT: dispatch=True loads weights to real devices immediately.
        # With dispatch=False, calling modules inside trace (norm/lm_head for
        # logit-lens) leaves them on meta device and saved tensors are meta.
        load_kwargs = dict(
            device_map="auto",
            dtype=torch.bfloat16,
            dispatch=True,
        )
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        # ── SentencePiece-tokenizer special case ────────────────────────
        # Models like ALLaM ship a tokenizer.model file that transformers
        # 5.x's tiktoken auto-converter can't parse, producing:
        #   ValueError: Error parsing line b'\x0e' in tokenizer.model
        # Forcing use_fast=False loads via the slow Python tokenizer that
        # actually understands SentencePiece. Requires `pip install sentencepiece`.
        if model_name in _NEEDS_SLOW_TOKENIZER:
            load_kwargs["tokenizer_kwargs"] = {"use_fast": False}

        if model_type == "vision":
            model = VisionLanguageModel(model_path, **load_kwargs)
        else:
            model = LanguageModel(model_path, **load_kwargs)

        model.remote = False

        sub_attr = text_submodule_table.get(model_name, None)
        if sub_attr is not None:
            model.model.text_model = getattr(model.model, sub_attr)
            model.is_nested = True
        else:
            model.model.text_model = model.model
            model.is_nested = False

    else:
        raise ValueError(f"Model {model_name} not found")

    return model