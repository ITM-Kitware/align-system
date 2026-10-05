import json
import random
import time

from typing import Union, Optional, List, Dict
from textwrap import dedent

import anthropic
from anthropic import APIStatusError
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from align_system.algorithms.abstracts import StructuredInferenceEngine
from align_system.utils import logging

log = logging.getLogger(__name__)


class ClaudeInferenceEngine(StructuredInferenceEngine):
    """StructuredInferenceEngine implementation using the Anthropic Messages API.

    Supports two execution modes:
      - **Synchronous** (default for single prompts): calls the Messages
        API directly for each prompt.
      - **Batch** (default for multiple prompts): submits prompts to the
        Message Batches API for asynchronous processing at 50% cost.
    """

    _RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 501, 502, 503, 504, 529}

    def __init__(self,
                 model_name: str,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 max_tokens: int = 4096,
                 api_key: Optional[str] = None,
                 use_batch_api: bool = True,
                 batch_poll_interval: int = 60,
                 max_retries: int = 20,
                 retry_base_delay: float = 1.0):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.use_batch_api = use_batch_api
        self.batch_poll_interval = batch_poll_interval
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        self.client = anthropic.Anthropic(api_key=api_key)

        self._cache_repr = dedent(f"""
                        {self.__class__.__module__}.{self.__class__.__name__}(
                        model_name="{model_name}",
                        temperature="{temperature}",
                        top_p="{top_p}",
                        max_tokens="{max_tokens}",
                        use_batch_api="{use_batch_api}"
                       )""").strip()

    def dialog_to_prompt(self, dialog: list[dict]) -> str:
        """Serialize a dialog into a JSON string for use with run_inference.

        Separates system messages from user/assistant messages, since
        the Anthropic API takes a top-level `system` parameter rather
        than a system role in the messages list.

        Returns:
            JSON string encoding a dict with "system" and "messages" keys.
        """
        system_parts = []
        messages = []
        for element in dialog:
            d = dict(element)
            if d["role"] == "system":
                system_parts.append(d["content"])
            else:
                messages.append({"role": d["role"], "content": d["content"]})

        prompt_obj = {
            "system": "\n".join(system_parts) if system_parts else None,
            "messages": messages,
        }
        return json.dumps(prompt_obj)

    def run_inference(self, prompts: Union[str, List[str]], schema: str) -> Union[Dict, List[Dict]]:
        return self._run_inference(prompts, schema)

    def run_inference_unstructured(self, prompts: Union[str, list[str]]) -> Union[str, List[str]]:
        return self._run_inference(prompts)

    def _run_inference(self, prompts: Union[str, list[str]], schema: Optional[str] = None):
        parsed_prompts = _deserialize_prompts(prompts)
        output_config = _build_output_config(schema)

        if len(parsed_prompts) > 1 and self.use_batch_api:
            return self._create_batches(parsed_prompts, output_config)
        else:
            return self._create_messages(parsed_prompts, output_config)

    def _build_message_params(self, prompt: dict, output_config: dict) -> dict:
        """Build the kwargs dict for a single messages.create call."""
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": prompt["messages"],
            "temperature": self.temperature,
        }
        if prompt.get("system"):
            params["system"] = prompt["system"]
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if output_config:
            params["output_config"] = output_config
        return params

    def _retry_api_call(self, func, *args, **kwargs):
        """Call *func* with retry on 408, 425, 429, and 5xx responses.

        Uses the Retry-After header when present, otherwise exponential
        backoff with jitter (capped at 60 s).
        """
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except APIStatusError as exc:
                retryable = exc.status_code in self._RETRYABLE_STATUS_CODES
                if not retryable or attempt == self.max_retries:
                    raise

                retry_after = exc.response.headers.get("retry-after")
                if retry_after is not None:
                    delay = float(retry_after)
                else:
                    delay = min(self.retry_base_delay * (2 ** attempt), 60.0)
                    delay += random.uniform(0, delay * 0.25)

                log.warning(
                    f"Retryable API error (HTTP {exc.status_code}), "
                    f"attempt {attempt + 1}/{self.max_retries}, "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)

    def _create_messages(self, prompts: List[dict], output_config: dict) -> List[Dict]:
        """Run prompts synchronously via the Messages API."""
        results = []
        for prompt in prompts:
            params = self._build_message_params(prompt, output_config)
            message = self._retry_api_call(self.client.messages.create, **params)
            parsed = _extract_message_content(message, has_schema=bool(output_config))
            results.append(parsed)
        return results

    def _create_batches(self, prompts: List[dict], output_config: dict) -> List[Dict]:
        """Run prompts via the Message Batches API for 50% cost reduction.

        Creates a batch, polls until processing ends, then streams
        results and returns them in original prompt order.
        """
        requests = []
        for idx, prompt in enumerate(prompts):
            params = self._build_message_params(prompt, output_config)
            requests.append(
                Request(
                    custom_id=f"request-{idx}",
                    params=MessageCreateParamsNonStreaming(**params),
                )
            )

        message_batch = self._retry_api_call(
            self.client.messages.batches.create, requests=requests
        )
        log.info(f"Batch {message_batch.id} created, polling for completion...")

        while message_batch.processing_status != "ended":
            time.sleep(self.batch_poll_interval)
            message_batch = self._retry_api_call(
                self.client.messages.batches.retrieve, message_batch.id
            )
            log.debug(
                f"Batch {message_batch.id} status: {message_batch.processing_status}, "
                f"succeeded: {message_batch.request_counts.succeeded}, "
                f"errored: {message_batch.request_counts.errored}"
            )

        log.info(
            f"Batch {message_batch.id} ended. "
            f"Succeeded: {message_batch.request_counts.succeeded}, "
            f"Errored: {message_batch.request_counts.errored}, "
            f"Expired: {message_batch.request_counts.expired}, "
            f"Canceled: {message_batch.request_counts.canceled}"
        )

        results_dict = {}
        for result in self.client.messages.batches.results(message_batch.id):
            idx = int(result.custom_id.split("-")[1])
            if result.result.type == "succeeded":
                parsed = _extract_message_content(
                    result.result.message,
                    has_schema=bool(output_config),
                )
                results_dict[idx] = parsed
            elif result.result.type == "errored":
                log.error(f"Request {result.custom_id} errored: {result.result.error}")
                results_dict[idx] = None
            elif result.result.type == "expired":
                log.error(f"Request {result.custom_id} expired")
                results_dict[idx] = None
            elif result.result.type == "canceled":
                log.error(f"Request {result.custom_id} was canceled")
                results_dict[idx] = None

        return [results_dict.get(i) for i in range(len(prompts))]

    def cache_repr(self):
        return self._cache_repr


def _deserialize_prompts(prompts: Union[str, List[str]]) -> List[dict]:
    """Normalize prompt input into a list of parsed prompt dicts.

    Each parsed dict has "system" (str or None) and "messages" (list) keys,
    as produced by dialog_to_prompt.
    """
    if isinstance(prompts, str):
        return [json.loads(prompts)]
    elif isinstance(prompts, list):
        return [json.loads(p) for p in prompts]
    else:
        raise TypeError(f"Unexpected prompts type: {type(prompts)}")


def _build_output_config(schema: Optional[str]) -> dict:
    """Build the output_config parameter for structured JSON output.

    Uses the Anthropic structured output format with json_schema type.
    Strips schema keywords unsupported by the Anthropic API.
    """
    if not schema:
        return {}
    schema_dict = json.loads(schema)
    _strip_unsupported_keywords(schema_dict)
    _enforce_additional_properties_false(schema_dict)
    return {
        "format": {
            "type": "json_schema",
            "schema": schema_dict,
        }
    }
def _enforce_additional_properties_false(schema: dict) -> dict:
    """Recursively set additionalProperties: false on all object types.

    OpenAI's strict structured output mode requires this on every object
    in the schema. Schemas defined for local outlines inference often omit
    it, so we normalize here rather than patching each template.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        for value in schema.values():
            if isinstance(value, dict):
                _enforce_additional_properties_false(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _enforce_additional_properties_false(item)
    return schema

# Keywords that Outlines/local models support but the Anthropic
# structured output API rejects.
_UNSUPPORTED_KEYWORDS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength",
    "pattern",
    "minItems", "maxItems", "uniqueItems",
    "multipleOf",
}


def _strip_unsupported_keywords(schema):
    """Recursively remove JSON Schema validation keywords unsupported by Claude."""
    if isinstance(schema, dict):
        for keyword in _UNSUPPORTED_KEYWORDS:
            schema.pop(keyword, None)
        for value in schema.values():
            if isinstance(value, dict):
                _strip_unsupported_keywords(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _strip_unsupported_keywords(item)


def _extract_message_content(message, has_schema: bool) -> dict:
    """Extract text content from an Anthropic Message response.

    Args:
        message: An anthropic Message object (from messages.create or
            batch results).
        has_schema: If True, parse the text as JSON. Otherwise wrap in
            {"response": text}.

    Returns:
        Parsed content dict.
    """
    if not message.content:
        raise ValueError("Response from Claude contained no content blocks.")

    text_block = None
    for block in message.content:
        if block.type == "text":
            text_block = block
            break

    if text_block is None:
        raise ValueError("Response from Claude contained no text content block.")

    if has_schema:
        return json.loads(text_block.text)
    else:
        return {"response": text_block.text}
