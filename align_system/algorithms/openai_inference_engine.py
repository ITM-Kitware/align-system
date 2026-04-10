import json
import os
import random
import re
import time
import tempfile

from typing import Union, Optional, Literal, List, Dict
from openai import OpenAI, APIStatusError, not_given
from textwrap import dedent

from align_system.algorithms.abstracts import StructuredInferenceEngine
from align_system.utils import logging

log = logging.getLogger(__name__)

class OpenAIInferenceEngine(StructuredInferenceEngine):
    """StructuredInferenceEngine implementation that interfaces with the OpenAI Responses API.

    Supports two execution modes:
      - **Synchronous** (default for single prompts): calls the Responses
        API directly for each prompt.
      - **Batch** (default for multiple prompts): uploads prompts as JSONL
        to the OpenAI Batch API and polls for completion.

    Also works against vLLM-compatible servers when ``base_url`` is provided,
    in which case the "developer" role rename is skipped.
    """

    _RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 501, 502, 503, 504}

    def __init__(self,
                 model_name: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 _strict_response_validation: bool = False,
                 timeout: Union[float, None, Literal["NOT_GIVEN"]] = "NOT_GIVEN",
                 use_batch_api: bool = True,
                 batch_poll_interval: int = 60,
                 max_retries: int = 20,
                 retry_base_delay: float = 1.0):
        """Initialize the OpenAI inference engine.

        Args:
            model_name: OpenAI model identifier (e.g. "gpt-5.2").
            temperature: Sampling temperature (currently disabled server-side).
            top_p: Nucleus sampling parameter (currently disabled server-side).
            max_tokens: Maximum output tokens per response.
            base_url: Optional URL for a vLLM-compatible server. When set,
                the "system" role is kept instead of being renamed to
                "developer".
            api_key: OpenAI API key. Defaults to the ``OPENAI_API_KEY``
                environment variable.
            _strict_response_validation: When True, raise on unexpected
                fields in API responses.
            timeout: Request timeout in seconds, or None for no timeout.
            use_batch_api: When True (default), use the Batch API for
                multi-prompt requests.
            batch_poll_interval: Seconds between batch status polls.
            max_retries: Maximum number of retry attempts for API calls.
            retry_base_delay: Base delay in seconds for exponential backoff.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.use_batch_api = use_batch_api
        self.batch_poll_interval = batch_poll_interval
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.using_vllm = base_url is not None

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=not_given if timeout == "NOT_GIVEN" else timeout,
            _strict_response_validation=_strict_response_validation
        )

        self._cache_repr = dedent(f"""
                        {self.__class__.__module__}.{self.__class__.__name__}(
                        model_name="{model_name}",
                        temperature="{temperature}",
                        top_p="{top_p}",
                        max_tokens="{max_tokens}",
                        base_url="{base_url}",
                        use_batch_api="{use_batch_api}",
                        batch_poll_interval="{batch_poll_interval}"
                       )""").strip()

    def _build_request_params(self, prompt, text_kwargs, *, include_service_tier=True) -> dict:
        """Build the kwargs dict for a single responses.create or batch request.

        Args:
            prompt: The input prompt (serialized string for sync, list for batch).
            text_kwargs: Dict with ``text`` key for structured output format,
                or empty dict for unstructured.
            include_service_tier: Include ``service_tier`` (sync only; the
                Batch API does not support it).
        """
        params = {
            "model": self.model_name,
            "reasoning": {"effort": "medium", "summary": "auto"},
            "max_output_tokens": self.max_tokens,
            "store": True,
            "input": prompt,
            # Temporarily disabled
            # "temperature": self.temperature,
            # "top_p": self.top_p,
            **text_kwargs,
        }
        if include_service_tier:
            params["service_tier"] = "flex"
        return params

    def dialog_to_prompt(self, dialog: list[dict]) -> str:
        """Serialize a dialog into a JSON string for use with ``run_inference``.

        Converts each element to a plain dict. When targeting the native
        OpenAI API (not vLLM), "system" roles are renamed to "developer"
        via ``_conform_to_openai_prompt_roles``.

        Args:
            dialog: List of DialogElements or message dicts with "role" and
                "content" keys.

        Returns:
            JSON string encoding the list of message dicts, suitable for
            passing directly to ``run_inference`` or ``run_inference_unstructured``.
        """
        # Convert DialogElement objects to dicts if needed
        prompts = [dict(d) for d in dialog]
        if not self.using_vllm:
            prompts = _conform_to_openai_prompt_roles(prompts)
        return json.dumps(prompts)

    def run_inference(self, prompts: Union[str, List[str]], schema: str) -> Union[Dict, List[Dict]]:
        """Run inference with structured (JSON schema) output.

        Args:
            prompts: A single JSON-serialized prompt string or a list of them.
            schema: JSON schema string that the model output must conform to.

        Returns:
            A list of parsed response dicts (one per prompt).
        """
        return self._run_inference(prompts, schema)

    def run_inference_unstructured(self, prompts: Union[str, list[str]]) -> Union[str, List[str]]:
        """Run inference without structured output constraints.

        Args:
            prompts: A single JSON-serialized prompt string or a list of them.

        Returns:
            A list of dicts with a "response" key containing the raw text.
        """
        return self._run_inference(prompts)

    def _run_inference(self, prompts: Union[str, list[str]], schema: Optional[str] = None) -> Dict | List[Dict] | str | List[str] :
        """Internal dispatch: deserializes prompts and routes to sync or batch execution.

        Uses the Batch API when multiple prompts are provided and
        ``use_batch_api`` is True; otherwise processes synchronously.

        Args:
            prompts: A single JSON-serialized prompt string or a list of them.
            schema: Optional JSON schema string for structured output.

        Returns:
            A list of parsed response dicts.
        """
        json_prompts = _deserialize_prompts(prompts)
        text_kwargs = _response_format_field(schema)

        # Use batch API if enabled and we have multiple prompts
        if len(json_prompts) > 1 and self.use_batch_api:
            return self._create_batches(json_prompts, text_kwargs)
        else:
            return self._create_responses(json_prompts, text_kwargs)
  

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

    def _create_responses(self, prompts: List[List[Dict]], text_kwargs):
        """Run prompts synchronously via the Responses API.

        Iterates over each prompt, calls ``responses.create``, and
        extracts the parsed content from each response.

        Args:
            prompts: List of deserialized message lists.
            text_kwargs: Dict with ``text`` key for structured output
                format, or empty dict for unstructured.

        Returns:
            List of parsed response dicts (one per prompt).
        """
        results = []
        for p in prompts:
            for parse_attempt in range(self.max_retries + 1):
                params = self._build_request_params(
                    json.dumps(p), text_kwargs, include_service_tier=True
                )
                response = self._retry_api_call(
                    self.client.responses.create, **params
                )
                # Convert response object to dict for processing
                response_dict = response.model_dump()
                try:
                    parsed_result = _extract_response_content(response_dict, has_schema=bool(text_kwargs))
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    if parse_attempt < self.max_retries:
                        log.warning(
                            "Failed to parse response (attempt %d/%d): %s. "
                            "Re-requesting from API.",
                            parse_attempt + 1, self.max_retries + 1, e
                        )
                    else:
                        raise
            results.append(parsed_result)
        return results

    def _create_batches(self, prompts: List[List[Dict]], text_kwarg: dict) -> List[Dict]:
        """
        Run inference using OpenAI Batch API for async processing.
        Docs: https://platform.openai.com/docs/guides/batch

        Args:
            prompts: List of List of 2 JSON dictionaries each
            text_kwarg: Dictionary containing text format configuration

        Returns:
            List of parsed response dictionaries
        """
        # Prepare batch requests in JSONL format
        batch_requests = []
        for idx, prompt in enumerate(prompts):
            request_params = self._build_request_params(
                prompt, text_kwarg, include_service_tier=False
            )

            # Create batch request in the format expected by the Batch API
            batch_request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": request_params
            }
            batch_requests.append(batch_request)

        # Write batch requests to a temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
            batch_file_path = f.name

        try:
            # Upload the batch file
            with open(batch_file_path, 'rb') as f:
                batch_input_file = self._retry_api_call(
                    self.client.files.create, file=f, purpose="batch"
                )

            # Create the batch
            batch = self._retry_api_call(
                self.client.batches.create,
                input_file_id=batch_input_file.id,
                endpoint="/v1/responses",
                completion_window="24h"
            )

            # Poll for batch completion
            log.info(f"Batch {batch.id} created, polling for completion...")
            while batch.status in ["validating", "in_progress", "finalizing"]:
                time.sleep(self.batch_poll_interval)
                batch = self._retry_api_call(self.client.batches.retrieve, batch.id)
                counts = batch.request_counts
                log.debug(
                    f"Batch {batch.id} status: {batch.status}, "
                    f"completed: {counts.completed}/{counts.total}, "
                    f"failed: {counts.failed}"
                )

            counts = batch.request_counts
            log.info(
                f"Batch {batch.id} finished with status: {batch.status}. "
                f"Completed: {counts.completed}, Failed: {counts.failed}, "
                f"Total: {counts.total}"
            )

            if batch.status == "completed":
                # Download and parse results
                result_file_id = batch.output_file_id

                # Retry a few times if output_file_id is None (API may need time to populate it)
                retry_count = 0
                max_file_retries = 3
                while result_file_id is None and retry_count < max_file_retries:
                    log.warning(f"output_file_id is None, retrying ({retry_count + 1}/{max_file_retries})...")
                    time.sleep(2)
                    batch = self.client.batches.retrieve(batch.id)
                    result_file_id = batch.output_file_id
                    retry_count += 1

                if result_file_id is None:
                    log.error(f"Batch {batch.id} completed but output_file_id is None after {max_file_retries} retries")
                    raise RuntimeError(
                        f"Batch {batch.id} completed but output_file_id was not populated after {max_file_retries} retries. "
                        f"Check batch error details: {getattr(batch, 'errors', 'No error info available')}"
                    )

                result_content = self.client.files.content(result_file_id)
                result_lines = result_content.text.strip().split('\n')

                # Parse results and match to original order by custom_id
                results_dict = {}
                for line in result_lines:
                    result = json.loads(line)
                    custom_id = result['custom_id']
                    idx = int(custom_id.split('-')[1])

                    response_body = result.get('response', {})
                    status_code = response_body.get('status_code')

                    if status_code == 200:
                        response_data = response_body.get('body', {})
                        try:
                            parsed_result = _extract_response_content(
                                response_data,
                                has_schema=bool(text_kwarg)
                            )
                            results_dict[idx] = parsed_result
                        except (json.JSONDecodeError, ValueError) as e:
                            log.error(f"Request {custom_id} returned 200 but content extraction failed: {e}")
                            results_dict[idx] = None
                    elif result.get('error'):
                        log.error(f"Request {custom_id} errored: {result['error']}")
                        results_dict[idx] = None
                    else:
                        log.error(
                            f"Request {custom_id} returned unexpected status "
                            f"{status_code}: {response_body}"
                        )
                        results_dict[idx] = None

                return [results_dict.get(i) for i in range(len(prompts))]

            elif batch.status == "failed":
                raise RuntimeError(f"Batch processing failed: {batch}")
            elif batch.status == "expired":
                raise RuntimeError(f"Batch processing expired: {batch}")
            else:
                raise RuntimeError(f"Unexpected batch status: {batch.status}")

        finally:
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

    def cache_repr(self):
        """Return a string representation of this engine for caching purposes."""
        return self._cache_repr

def _conform_to_openai_prompt_roles(prompts: list[Dict]) -> list[Dict]:
    """Convert a dialog to a JSON-serialized prompt string.

    Converts DialogElement objects to plain dicts and, for the native
    OpenAI API (non-vLLM), renames the "system" role to "developer"
    per OpenAI's message role conventions.

    Sources:
    1) https://platform.openai.com/docs/guides/prompt-engineering#message-roles-and-instruction-following
    2) https://model-spec.openai.com/2025-02-12.html#definitions

    Args:
        dialog: List of DialogElements or message dicts with "role" and "content" keys.

    Returns:
        JSON-serialized prompt string consumable by ``run_inference``.
    """
    for p in prompts:
        if p["role"] == "system":
            p["role"] = "developer"
    return prompts

def _response_format_field(schema: Optional[str]) -> Dict:
    """
    Create response_format parameter for structured outputs.
    Supported by: gpt-4o-mini, gpt-4o-2024-08-06 and later.
    For older models like gpt-4, this will fail - use gpt-4o or gpt-4-turbo instead.
    """
    if not schema:
        return {}
    else:
        schema_dict = json.loads(schema)
        _strip_unsupported_keywords(schema_dict)
        _enforce_additional_properties_false(schema_dict)
        text = {
            "format": {
                "type": "json_schema",
                "name": "ITM_Schema",
                "schema": schema_dict,
                "strict": True
            }
        }
        return {"text": text}


def _deserialize_prompts(prompts: Union[str, List[str]]) -> List[List[Dict]]:
    """Normalize prompt input into a list of parsed message lists.

    Args:
        prompts: Raw prompt(s) as returned by ``dialog_to_prompt``.

    Returns:
        A list where each element is a deserialized message list
        (list of role/content dicts).
    """
    if isinstance(prompts, str):
        return [json.loads(prompts)]
    elif isinstance(prompts, list):
        return [json.loads(p) for p in prompts]
    else:
        raise TypeError(f"Unexpected prompts type: {type(prompts)}")


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


# Keywords that Outlines/local models support but the OpenAI
# structured output API rejects when strict mode is enabled.
_UNSUPPORTED_KEYWORDS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength",
    "pattern",
    "minItems", "maxItems", "uniqueItems",
    "multipleOf",
}


def _strip_unsupported_keywords(schema):
    """Recursively remove JSON Schema validation keywords unsupported by OpenAI."""
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


def _extract_response_content(response_data: dict, has_schema: bool) -> Dict:
    """Extract the text payload from a Responses API result dict.

    Searches ``response_data["output"]`` for the first item with
    ``type == "message"``, then reads its first content block's ``"text"``
    field. Reasoning output items are ignored.

    Args:
        response_data: A response body dict as returned by the Responses API
            (or the ``body`` field of a Batch API result line).
        has_schema: If True, parses the text as JSON and returns the resulting
            dict. If False, wraps the raw text in ``{"response": text}``.

    Returns:
        Parsed content dict.

    Raises:
        ValueError: If ``output`` is empty, contains no message item, or the
            message has no content.
    """
    output = response_data.get('output', [])
    if not output:
        raise ValueError("Response from server was empty. Could not extract response data from an empty message.")

    # Find the message output (not the reasoning)
    message_output = None
    for output_item in output:
        if isinstance(output_item, dict) and output_item.get('type') == 'message':
            message_output = output_item
            break

    if not message_output:
        raise ValueError("Response from server contained no message.")
    if not message_output.get('content'):
        raise ValueError("Response message from server contained no content.")

    # Get the text from the first content item
    first_content = message_output['content'][0]
    text_content = first_content.get('text')

    if text_content:
        if has_schema:
            # Parse as JSON for structured outputs
            try:
                return json.loads(text_content)
            except json.JSONDecodeError as e:
                log.warning("Initial JSON parse failed at char %d: %s. "
                            "Attempting repair.", e.pos, e.msg)
                repaired = _repair_json(text_content)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    log.error("JSON repair failed. Raw text (first 500 chars): %s",
                              text_content[:500])
                    raise
        else:
            # Return as plain text for unstructured outputs
            return {"response": text_content}


def _repair_json(text: str) -> str:
    """Attempt to fix common JSON issues produced by LLMs.

    Handles:
      - Unescaped control characters inside string values
      - Trailing commas before closing braces/brackets
    """
    # Escape unescaped control characters inside strings
    repaired = re.sub(
        r'(?<=": ")((?:[^"\\]|\\.)*)(?=")',
        lambda m: m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'),
        text
    )
    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
    return repaired