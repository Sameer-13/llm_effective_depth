"""
model_compat.py
"""


def get_text_model(llm):
    """
    Returns the sub-module containing .layers / .norm / .embed_tokens.
    Works for flat models (Llama, Qwen) and nested models (Gemma 4).
    """
    return llm.model.text_model


def get_layers(llm):
    return get_text_model(llm).layers


def get_norm(llm):
    return get_text_model(llm).norm


def get_embed_tokens(llm):
    return get_text_model(llm).embed_tokens


def get_lm_head(llm):
    return llm.lm_head


def set_eval(llm):
    """
    Call eval() on the underlying PyTorch model directly.
    Never call llm.eval() — nnsight's __getattr__ triggers __repr__
    which causes infinite recursion on large models.
    """
    llm._model.eval()


def set_train(llm):
    """Same issue — use this instead of llm.train()."""
    llm._model.train()