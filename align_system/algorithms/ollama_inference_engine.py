from __future__ import annotations

import json

import ollama

from align_system.algorithms.abstracts import StructuredInferenceEngine
# from align_system.algorithms.planner_adm.llm_ollama import _extract_json_object, _repair_json
from align_system.utils import logging

log = logging.getLogger(__name__)

def _extract_json_object(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        raise ValueError("No JSON object found")
    return s[i : j + 1]

def _loads_json(s: str) -> JSON:
    return json.loads(_extract_json_object(s))


def _repair_json(model: str, bad_text: str, schema_hint: str, num_ctx: int) -> JSON:
    prompt = (
        "You output invalid JSON. Fix it.\n"
        "Return ONLY valid JSON, no prose.\n"
        f"Schema hint:\n{schema_hint}\n\n"
        f"Bad output:\n{bad_text}\n"
    )
    resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0, "num_ctx": num_ctx})
    return _loads_json(resp["response"])



class OllamaInferenceEngine(StructuredInferenceEngine):
    """
    StructuredInferenceEngine backed by a local Ollama model.

    Unlike the Outlines engine, output is not grammar-constrained — the
    schema is appended to the prompt as an instruction and the response
    is parsed as JSON with a repair fallback.
    """

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        json_repair_attempts: int = 2,
    ):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.json_repair_attempts = json_repair_attempts

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

    def run_inference(self, prompts, schema: str, temperature: float = None) -> list[dict]:
        """
        Run inference for each prompt string and return parsed JSON dicts.

        `schema` is a JSON Schema string appended to each prompt as an
        instruction. Malformed responses trigger up to
        `json_repair_attempts` self-repair calls.
        """
        schema_instruction = (
            "\n\nRespond with ONLY valid JSON that matches this schema "
            "(no prose, no markdown fences):\n" + schema
        )
        effective_temperature = self.temperature if temperature is None else temperature

        results = []
        for prompt in prompts:
            full_prompt = prompt + schema_instruction
            resp = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={"temperature": effective_temperature, "num_ctx": self.num_ctx},
            )
            text = resp["response"]
            log.debug(f"[OllamaInferenceEngine] raw response:\n{text}")

            data = None
            try:
                data = json.loads(_extract_json_object(text))
            except Exception:
                for _ in range(self.json_repair_attempts):
                    try:
                        data = _repair_json(self.model, text, schema, self.num_ctx)
                        break
                    except Exception:
                        data = None

            if data is None:
                log.warning("[OllamaInferenceEngine] JSON parse failed; returning empty dict")
                data = {}

            results.append(data)

        return results

    def cache_repr(self) -> str:
        return (
            f"OllamaInferenceEngine(model={self.model}, "
            f"temperature={self.temperature}, num_ctx={self.num_ctx})"
        )
