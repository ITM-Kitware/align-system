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
        model: str = "gemma4:12b",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 4096,
        max_retries: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.max_retries = max_retries

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
            for attempt in range(self.max_retries + 1):
                # On retries, sample with some temperature so a greedy
                # engine doesn't just reproduce the same bad output
                retry_temperature = (effective_temperature if attempt == 0
                                     else max(effective_temperature, 0.2))
                resp = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    format=format_schema,
                    options={"temperature": retry_temperature,
                             "num_ctx": self.num_ctx,
                             "num_predict": self.num_predict},
                )
                text = resp["response"]
                log.debug(f"[OllamaInferenceEngine] raw response:\n{text}")

                try:
                    results.append(self._parse_json_response(text))
                    break
                except (json.JSONDecodeError, RuntimeError) as e:
                    if attempt == self.max_retries:
                        raise
                    log.warning(f"[OllamaInferenceEngine] failed to parse "
                                f"response (attempt {attempt + 1} of "
                                f"{self.max_retries + 1}): {e}; retrying")

        return results[0] if single_prompt else results

    @staticmethod
    def _parse_json_response(text: str):
        """
        Parse the first JSON value in the response, tolerating trailing
        garbage (some models emit extra text after the schema-constrained
        JSON despite the `format` parameter).
        """
        stripped = text.strip()
        if not stripped:
            raise RuntimeError(
                "Ollama returned an empty response; the model may not "
                "support structured output via the `format` parameter")

        obj, end = json.JSONDecoder().raw_decode(stripped)
        trailing = stripped[end:].strip()
        if trailing:
            log.warning(f"[OllamaInferenceEngine] ignoring trailing data "
                        f"after JSON response: {trailing[:100]!r}")
        return obj

    def cache_repr(self) -> str:
        return (
            f"OllamaInferenceEngine(model={self.model}, "
            f"temperature={self.temperature}, num_ctx={self.num_ctx})"
        )
