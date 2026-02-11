import json

from typing import Union, Optional, Literal, Iterable, Callable, List, Dict
from copy import deepcopy
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
                 timeout: Union[float, None, Literal["NOT_GIVEN"]] = "NOT_GIVEN"
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

        print(self.static_responses_kwargs)

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
        if self.using_vllm:
            print("We are targetting the OpenAI service")
            for p in prompt:
                if p["role"] == "system":
                    p["role"] = "developer"
        print(prompt)
        return json.dumps(prompt)

    def run_inference(self, prompts: Union[str, List[str]], schema: str) -> Union[Dict, List[Dict]]:
        return self._run_inference(prompts, schema)                                

    def run_inference_unstructured(self, prompts: Union[str, list[str]]) -> Union[Dict, List[Dict]]:
        return self._run_inference(prompts)                            

    def _run_inference(self, prompts: Union[str, list[str]], schema: Optional[str] = None) -> Union[Dict, List[Dict]]:

        run_async = False if isinstance(prompts, str) else True
        run_async = False  # DELETE ME
        prompts = [prompts] if isinstance(prompts, str) else prompts

        text_kwarg = {"text": self.response_format_field(schema)} if schema else {}

        if isinstance(prompts, Iterable) and run_async:
            # https://developers.openai.com/api/docs/guides/batch?lang=python
            pass
        elif isinstance(prompts, Iterable) and not run_async:
            create_responses = partial(self.client.responses.create, **text_kwarg, **self.static_responses_kwargs)
            # return [create_responses(input=json.loads(p)).model_dump() for p in prompts]
            results = []
            for p in prompts:
                response = create_responses(input=json.loads(p))

                # Extract the parsed JSON content from the response
                # response.output is a list that can contain:
                # - ResponseReasoningItem (type='reasoning') - the internal reasoning
                # - ResponseOutputMessage (type='message') - the actual output
                if response.output and len(response.output) > 0:
                    # Find the message output (not the reasoning)
                    message_output = None
                    for output_item in response.output:
                        if hasattr(output_item, 'type') and output_item.type == 'message':
                            message_output = output_item
                            break

                    if message_output and message_output.content and len(message_output.content) > 0:
                        # Get the text from the first content item (handle different content types)
                        first_content = message_output.content[0]
                        text_content = getattr(first_content, 'text', None)
                        if text_content:
                            if schema:
                                # Parse as JSON for structured outputs
                                results.append(json.loads(text_content))
                            else:
                                # Return as plain text for unstructured outputs
                                results.append({"response": text_content})
                        else:
                            # Handle refusal or other content types
                            results.append(None)
                    else:
                        results.append(None)
                else:
                    results.append(None)
            return results
        else:
            raise TypeError("Don't know how to run inference on provided "
                            "`prompts` object")  
    
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
        



