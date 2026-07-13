from __future__ import annotations

import json

import ollama

from align_system.algorithms.abstracts import StructuredInferenceEngine
from align_system.utils import logging

log = logging.getLogger(__name__)


class OllamaInferenceEngine(StructuredInferenceEngine):
    """
    StructuredInferenceEngine backed by a local Ollama model.

    Uses Ollama's native structured output support
    (https://ollama.com/blog/structured-outputs): the JSON schema is
    passed as the `format` parameter so the server constrains the
    output to match the schema.
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        temperature: float = 0.0,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx

    def dialog_to_prompt(self, dialog) -> str:
        """
        Flatten a dialog list into a plain-text prompt for Ollama.

        System messages are prepended as an unlabelled block so they
        land at the top; user/assistant turns follow with role labels.
        """
        system_parts = []
        turn_parts = []

        for elem in dialog:
            role = elem.role if hasattr(elem, "role") else elem["role"]
            content = elem.content if hasattr(elem, "content") else elem["content"]

            if role == "system":
                system_parts.append(content)
            else:
                turn_parts.append(f"[{role.upper()}]\n{content}")

        parts = []
        if system_parts:
            parts.append("\n\n".join(system_parts))
        parts.extend(turn_parts)
        return "\n\n".join(parts)

    def run_inference(self, prompts, schema: str, temperature: float = None):
        """
        Run inference for each prompt string and return parsed JSON dicts.

        `schema` is a JSON Schema string passed to Ollama's `format`
        parameter for server-side constrained generation.
        """
        format_schema = json.loads(schema)
        effective_temperature = self.temperature if temperature is None else temperature

        if single_prompt := isinstance(prompts, str):
            prompts = [prompts]

        results = []
        for prompt in prompts:
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                format=format_schema,
                options={"temperature": effective_temperature, "num_ctx": self.num_ctx},
            )
            text = resp["response"]
            log.debug(f"[OllamaInferenceEngine] raw response:\n{text}")

            results.append(json.loads(text))

        return results[0] if single_prompt else results

    def cache_repr(self) -> str:
        return (
            f"OllamaInferenceEngine(model={self.model}, "
            f"temperature={self.temperature}, num_ctx={self.num_ctx})"
        )
