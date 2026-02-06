import json

from typing import Union, Optional, Literal, Iterable
from copy import deepcopy
from openai import OpenAI, not_given
from textwrap import dedent

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

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            webhook_secret=webhook_secret,
            base_url=base_url,
            timeout=not_given if timeout == "NOT_GIVEN" else timeout,
            _strict_response_validation=_strict_response_validation
        )

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
        # OpenAI uses "developer" prompt instread of "system" (which is not exposed to the caller)
        # https://platform.openai.com/docs/guides/prompt-engineering#message-roles-and-instruction-following
        # https://model-spec.openai.com/2025-02-12.html#definitions
        if not self.client.base_url:
            # We are targetting the OpenAI service
            prompt = deepcopy(dialog)
            for p in prompt:
                if p["role"] == "system":
                    p["role"] = "developer"
        else:
            # TOOD verify with local VLLM model if we need-to/should-be doing the above remapping regardless
            prompt = dialog
        return json.dumps(prompt)

    def run_inference(self, prompts: Union[str, list[str]], schema: str) -> Union[dict, list[dict]]:
        prompts = [prompts] if isinstance(prompts, str) else prompts
        text = self.text_field(schema)
        if isinstance(prompts, Iterable):
            return [
                    self.client.responses.create(
                        input=json.loads(p),
                        text=text,
                        **self.responses_kwargs()) 
                    for p in prompts
                    ]
        else:
            raise TypeError("Don't know how to run inference on provided "
                            "`prompts` object")                                  

    def cache_repr(self):
        return self._cache_repr

    def responses_kwargs(self) -> dict:
        return {
            "model": self.model_name,
            "reasoning": {"effort": "medium",  "summary": "auto"},
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "service_tier":"flex",
            "store":True
        }

    @staticmethod
    def text_field(schema: str) -> dict:
        return {
            "format": {
                "type": "json_schema",
                "name": "ITM Schema",
                "schema": json.loads(schema),
                "strict": True
            }
        }




