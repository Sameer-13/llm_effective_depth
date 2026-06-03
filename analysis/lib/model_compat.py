"""
model_compat.py

Provides get_text_model(llm) which returns a unified view of the
language model sub-module regardless of whether the model is:
  - Llama / Qwen  → layers live at llm.model.layers
  - Gemma 4       → layers live at llm.model.language_model.layers

All analysis scripts should call get_text_model(llm) instead of
accessing llm.model directly when they need:
    .layers       — the list of transformer decoder layers
    .norm         — the final RMS norm before the lm_head
    .embed_tokens — the token embedding table
"""


def get_text_model(llm):
    """
    Returns the sub-module that contains .layers, .norm, .embed_tokens.
    Works for both flat models (Llama, Qwen) and nested models (Gemma 4).
    """
    return llm.model.text_model


def get_layers(llm):
    """Shortcut: returns the list of transformer decoder layers."""
    return get_text_model(llm).layers


def get_norm(llm):
    """Shortcut: returns the final layer norm."""
    return get_text_model(llm).norm


def get_embed_tokens(llm):
    """Shortcut: returns the token embedding table."""
    return get_text_model(llm).embed_tokens


def get_lm_head(llm):
    """
    Returns the language model head (logit projection).
    For all models tested this is always at llm.lm_head.
    """
    return llm.lm_head