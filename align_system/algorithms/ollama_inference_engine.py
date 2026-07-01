import json
import time
from textwrap import dedent
from typing import Union, Optional, List, Dict

import ollama as _ollama

from align_system.algorithms.abstracts import StructuredInferenceEngine
from align_system.utils import logging
from align_system.utils import compute_stats as _compute_stats

log = logging.getLogger(__name__)


class OllamaInferenceEngine(StructuredInferenceEngine):
    """StructuredInferenceEngine using the native Ollama Python client.

    Uses the native /api/chat endpoint, which correctly honors per-request
    ``num_ctx`` — unlike Ollama's OpenAI-compatible /v1/chat/completions
    endpoint, which ignores ``options.num_ctx`` and always uses the default
    4096-token context window.
    """

    def __init__(self,
                 model_name: str,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_tokens: int = 8192,
                 host: str = 'http://localhost:11434',
                 num_ctx: Optional[int] = None,
                 max_retries: int = 3):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.client = _ollama.Client(host=host)

    def cache_repr(self):
        """
        Return a string representation of this object for caching;
        .i.e. if the return value of this function is the same for two
        object instances, it's assumed that inference output will be
        the same
        """
        return dedent(f"""
                       {self.__class__.__module__}.{self.__class__.__name__}(
                       model_name="{self.model_name}",
                       temperature={self.temperature},
                       top_p={self.top_p},
                       max_tokens={self.max_tokens},
                       num_ctx={self.num_ctx},
                       )""").strip()

    def dialog_to_prompt(self, dialog) -> str:
        return json.dumps([dict(d) for d in dialog])

    def run_inference(self, prompts: Union[str, List[str]], schema: str) -> List[Dict]:
        return self._run_inference(prompts, schema=schema)

    def run_inference_unstructured(self, prompts: Union[str, List[str]]) -> List[Dict]:
        return self._run_inference(prompts, schema=None)

    def _run_inference(self, prompts, schema=None):
        if isinstance(prompts, str):
            message_lists = [json.loads(prompts)]
        else:
            message_lists = [json.loads(p) for p in prompts]

        return [self._run_single(msgs, schema) for msgs in message_lists]

    def _run_single(self, messages: List[Dict], schema: Optional[str]) -> Dict:
        options = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'num_predict': self.max_tokens,
        }
        if self.num_ctx is not None:
            options['num_ctx'] = self.num_ctx

        format_arg = json.loads(schema) if schema else None

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                format=format_arg,
            )
            elapsed = time.perf_counter() - t0

            content = response.message.content
            done_reason = response.done_reason
            in_tok = response.prompt_eval_count
            out_tok = response.eval_count

            log.warning(
                "Raw response: %r | done_reason: %s | in: %s tokens | out: %s tokens",
                content, done_reason, in_tok, out_tok,
            )

            if not content:
                if attempt < self.max_retries:
                    log.warning(
                        "Empty response (attempt %d/%d, done_reason=%s). Retrying.",
                        attempt + 1, self.max_retries + 1, done_reason,
                    )
                    continue
                raise ValueError(
                    f"Empty response after {self.max_retries + 1} attempts "
                    f"(done_reason={done_reason})."
                )

            _compute_stats.record({
                "model_name": self.model_name,
                "n_prompts": 1,
                "elapsed_s": elapsed,
                "elapsed_per_prompt_s": elapsed,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            })
            log.info("[%s] %.2fs, in: %s tokens, out: %s tokens",
                     self.model_name, elapsed, in_tok, out_tok)

            if schema is None:
                return {"response": content}

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                if attempt < self.max_retries:
                    log.warning(
                        "JSON parse failed (attempt %d/%d): %s. Retrying with correction.",
                        attempt + 1, self.max_retries + 1, e,
                    )
                    correction = (
                        f"Your previous response was invalid: {e}. "
                        "Please respond with ONLY a valid JSON object matching the schema"
                        f":\n{schema}"
                    )
                    messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": correction},
                    ]
                    continue
                raise

        raise RuntimeError("Unreachable")
