"""Preserve pre-rendered pipeline prompts across supported Outlines versions."""

import outlines


def from_transformers(*args, **kwargs):
    model = outlines.from_transformers(*args, **kwargs)
    # Our dialog_to_prompt methods already apply the tokenizer's chat template.
    # Outlines 1.3 also templates string inputs unless this adapter flag is off.
    adapter = getattr(model, 'type_adapter', None)
    if hasattr(adapter, 'has_chat_template'):
        adapter.has_chat_template = False
    return model
