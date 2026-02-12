import json
import time
import tempfile

from typing import Union, Optional, Literal, Iterable, List, Dict, Tuple
from openai import OpenAI, not_given
from textwrap import dedent
from functools import partial

from align_system.algorithms.abstracts import StructuredInferenceEngine

class OpenAIInferenceEngine(StructuredInferenceEngine):
    def __init__(self,
                 model_name: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 project: Optional[str] = None,
                 webhook_secret: Optional[str] = None,
                 _strict_response_validation: bool = False,
                 timeout: Union[float, None, Literal["NOT_GIVEN"]] = "NOT_GIVEN",
                 use_batch_api: bool = True
                ):

        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            webhook_secret=webhook_secret,
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
                        organization="{organization}",
                        project="{project}",
                        webhook_secret="{webhook_secret}",
                        _strict_response_validation="{_strict_response_validation}",
                        timeout="{timeout}"
                       )""").strip()

    def dialog_to_prompt(self, dialog: list[dict]) -> str:
        # Convert DialogElement objects to dicts if needed
        prompt = [dict(d) for d in dialog]

        # OpenAI uses "developer" prompt instread of "system" (which is not exposed to the caller)
        # https://platform.openai.com/docs/guides/prompt-engineering#message-roles-and-instruction-following
        # https://model-spec.openai.com/2025-02-12.html#definitions
        if not self.using_vllm:
            for p in prompt:
                if p["role"] == "system":
                    p["role"] = "developer"
        return json.dumps(prompt)

    def run_inference(self, prompts: Union[str, List[str]], schema: str) -> Union[Dict, List[Dict]]:
        return self._run_inference(prompts, schema)                                

    def run_inference_unstructured(self, prompts: Union[str, list[str]]) -> Union[Dict, List[Dict]]:
        return self._run_inference(prompts)                            

    def _run_inference(self, prompts: Union[str, list[str]], schema: Optional[str] = None) -> Union[Dict, List[Dict]]:
        json_prompts = self._deserialize_prompts(prompts)
        text_kwargs = {"text": self.response_format_field(schema)} if schema else {}

        # Use batch API if enabled and we have multiple prompts
        if len(json_prompts) > 1 and self.use_batch_api:
            return self._create_batches(json_prompts, text_kwargs)
        else:
            return self._create_responses(json_prompts, text_kwargs)
        
    @staticmethod
    def _deserialize_prompts(prompts: Union[str, List[str]]) -> List[List[Dict]]:
        # We assume that we are either receiving:
        # 1: A json string containing a single developer-user prompt-pair (List[Dict]). 
        # 2: An iterable of json strings containing multiple developer-user prompt-pairs (List[List[Dict]])
        # 3: An iterable of a single json string containing a single developer-user prompt-pair wrapped in an iterable container (List[List[Dict]]) but outer List has len == 1.
        # Hence, the need for this check.
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

    def _create_responses(self, prompts: List[List[Dict]], text_kwargs):
            # Synchronous processing (default)
            create_responses = partial(self.client.responses.create, **text_kwargs, **self.static_responses_kwargs)
            results = []
            for p in prompts:
                response = create_responses(input=p)
                # Convert response object to dict for processing
                response_dict = response.model_dump()
                parsed_result = self._extract_response_content(response_dict, has_schema=bool(text_kwargs))
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
            print(prompt)
            print("----")
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
            print(f"Batch {batch.id} created, polling for completion...")
            while batch.status in ["validating", "in_progress", "finalizing"]:
                time.sleep(5)  # Poll every 5 seconds
                batch = self.client.batches.retrieve(batch.id)
                print(f"Batch status: {batch.status}")

            if batch.status == "completed":
                # Download and parse results
                result_file_id = batch.output_file_id

                # Retry a few times if output_file_id is None (API may need time to populate it)
                retry_count = 0
                max_retries = 3
                while result_file_id is None and retry_count < max_retries:
                    print(f"Warning: output_file_id is None, retrying ({retry_count + 1}/{max_retries})...")
                    time.sleep(2)  # Wait 2 seconds
                    batch = self.client.batches.retrieve(batch.id)
                    result_file_id = batch.output_file_id
                    retry_count += 1

                # If still None after retries, raise error
                if result_file_id is None:
                    print(f"ERROR: Batch {batch.id} marked as completed but output_file_id is None after {max_retries} retries")
                    print(f"Batch details: {batch}")
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
                        parsed_result = self._extract_response_content(
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

    def _extract_response_content(self, response_data: dict, has_schema: bool) -> Optional[Dict]:
        """
        Extract and parse content from a response object.

        Args:
            response_data: The response data dictionary
            has_schema: Whether structured output was requested

        Returns:
            Parsed content dictionary or None
        """
        output = response_data.get('output', [])
        if not output:
            return None

        # Find the message output (not the reasoning)
        message_output = None
        for output_item in output:
            if isinstance(output_item, dict) and output_item.get('type') == 'message':
                message_output = output_item
                break

        if not message_output or not message_output.get('content'):
            return None

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

        return None

    def cache_repr(self):
        return self._cache_repr

    @staticmethod
    def response_format_field(schema: str) -> dict:
        """
        Create response_format parameter for structured outputs.
        Supported by: gpt-4o-mini, gpt-4o-2024-08-06 and later.
        For older models like gpt-4, this will fail - use gpt-4o or gpt-4-turbo instead.
        """
        return {
            "format": {
                "type": "json_schema",
                "name": "ITM_Schema",
                "schema": json.loads(schema),
                "strict": True
            }
        }
        



