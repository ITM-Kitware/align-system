import json
import time
import tempfile

from typing import Union, Optional, Literal, List, Dict
from openai import OpenAI, not_given
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

    def __init__(self,
                 model_name: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 _strict_response_validation: bool = False,
                 timeout: Union[float, None, Literal["NOT_GIVEN"]] = "NOT_GIVEN",
                 use_batch_api: bool = True
                ):
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
        """

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=not_given if timeout == "NOT_GIVEN" else timeout,
            _strict_response_validation=_strict_response_validation
        )

        self.using_vllm = base_url is not None
        self.use_batch_api = use_batch_api

        self.static_responses_kwargs = {
            "model": model_name,
            "reasoning": {"effort": "medium",  "summary": "auto"},
            "max_output_tokens": max_tokens,
            # Temporarily disabled.
            # Server is telling me these are not supported, but docs say they are.
            # "temperature": temperature,
            # "top_p": top_p,
            "service_tier":"flex",
            "store":True
        }

        self.static_batches_kwargs = {
            "model": model_name,
            "reasoning": {"effort": "medium",  "summary": "auto"},
            "max_output_tokens": max_tokens,
            # Temporarily disabled.
            # Server is telling me these are not supported, but docs say they are.
            # "temperature": temperature,
            # "top_p": top_p,
            "store":True
        }

        self._cache_repr: str = dedent(f"""
                        {self.__class__.__module__}.{self.__class__.__name__}(
                        model_name="{model_name}",
                        temperature="{temperature}",
                        top_p="{top_p}",
                        max_tokens="{max_tokens}",
                        base_url="{base_url}",
                        api_key="{api_key}",
                        _strict_response_validation="{_strict_response_validation}",
                        timeout="{timeout}"
                       )""").strip()

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

    def run_inference_unstructured(self, prompts: Union[str, list[str]]) -> Union[Dict, List[Dict]]:
        """Run inference without structured output constraints.

        Args:
            prompts: A single JSON-serialized prompt string or a list of them.

        Returns:
            A list of dicts with a "response" key containing the raw text.
        """
        return self._run_inference(prompts)

    def _run_inference(self, prompts: Union[str, list[str]], schema: Optional[str] = None) -> Union[Dict, List[Dict]]:
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
            response = self.client.responses.create(input=json.dumps(p), **text_kwargs, **self.static_responses_kwargs)
            # Convert response object to dict for processing
            response_dict = response.model_dump()
            parsed_result = _extract_response_content(response_dict, has_schema=bool(text_kwargs))
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
            # Build the request parameters
            request_params = {
                **self.static_batches_kwargs,
                **text_kwarg,
                "input": prompt
            }

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
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            # Create the batch
            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/responses",
                completion_window="24h"
            )

            # Poll for batch completion
            log.info(f"Batch {batch.id} created, polling for completion...")
            while batch.status in ["validating", "in_progress", "finalizing"]:
                time.sleep(5)  # Poll every 5 seconds
                batch = self.client.batches.retrieve(batch.id)
                log.debug(f"Batch status: {batch.status}")

            if batch.status == "completed":
                # Download and parse results
                result_file_id = batch.output_file_id

                # Retry a few times if output_file_id is None (API may need time to populate it)
                retry_count = 0
                max_retries = 3
                while result_file_id is None and retry_count < max_retries:
                    log.warning(f"Warning: output_file_id is None, retrying ({retry_count + 1}/{max_retries})...")
                    time.sleep(2)  # Wait 2 seconds
                    batch = self.client.batches.retrieve(batch.id)
                    result_file_id = batch.output_file_id
                    retry_count += 1

                # If still None after retries, raise error
                if result_file_id is None:
                    log.error(f"ERROR: Batch {batch.id} marked as completed but output_file_id is None after {max_retries} retries")
                    log.error(f"Batch details: {batch}")
                    raise RuntimeError(
                        f"Batch {batch.id} completed but output_file_id was not populated after {max_retries} retries. "
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

                    # Extract the response
                    if result.get('response'):
                        response_data = result['response']['body']
                        # Parse the response similar to sync path
                        parsed_result = _extract_response_content(
                            response_data,
                            has_schema=bool(text_kwarg)
                        )
                        results_dict[idx] = parsed_result
                    else:
                        # Handle errors
                        results_dict[idx] = None

                # Return results in original order
                return [results_dict.get(i) for i in range(len(prompts))]

            elif batch.status == "failed":
                raise RuntimeError(f"Batch processing failed: {batch}")
            elif batch.status == "expired":
                raise RuntimeError(f"Batch processing expired: {batch}")
            else:
                raise RuntimeError(f"Unexpected batch status: {batch.status}")

        finally:
            # Clean up the temporary file
            import os
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
    """Normalize prompt input into a uniform list-of-message-lists format.

    Handles three input shapes:
        1. A single JSON string encoding one developer-user prompt pair. I.e. a DialogElement.
        2. A list of JSON strings encoding multiple prompt pairs. I.e. A Dialog with a multiple DialogElements.
        3. A single JSON string wrapped in a length-1 list. I.e. A Dialog with a single DialogElement.

    Args:
        prompts: Raw prompt(s) as returned by ``dialog_to_prompt``.

    Returns:
        A list where each element is a deserialized message list
        (list of role/content dicts).

    Raises:
        AssertionError: If ``prompts`` doesn't match any expected shape.
        TypeError: If ``prompts`` has an entirely unexpected type.
    """
    is_single_json_string = isinstance(prompts, str)
    is_multiple_json_strings = isinstance(prompts, list) and len(prompts) > 1 and all([isinstance(p,str) for p in prompts])
    is_wrapped_single_json_string = isinstance(prompts, list) and len(prompts) == 1 and isinstance(prompts[0], str)

    assert is_single_json_string or is_multiple_json_strings or is_wrapped_single_json_string

    if is_single_json_string:
        # Single developer-user prompt-pair; wrap it for uniformity
        return [json.loads(prompts)]
    elif is_multiple_json_strings or is_wrapped_single_json_string:
        return [json.loads(p) for p in prompts]
    else:
        raise TypeError("Argument prompts had an unexpected type")


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
            return json.loads(text_content)
        else:
            # Return as plain text for unstructured outputs
            return {"response": text_content}