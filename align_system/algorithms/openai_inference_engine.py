from typing import Union, Optional, Literal
from functools import partial
from copy import deepcopy
from openai import OpenAI


from align_system.algorithms.abstracts import StructuredInferenceEngine


class VLLMInferenceEngine(StructuredInferenceEngine):

# TODO Either create a second class VLLMInferenceEngine or two different params classes
class OpenAIInferenceEngine(StructuredInferenceEngine):
    def __init__(self,
                 model_name: str,
                 temperature: float,
                 top_p: float,
                 max_tokens: int,
                 inference_batch_size: int
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
        self.top_p = top_p
        self.inference_batch_size = inference_batch_size

        # Delete if VLLM does not care about the presence of an api key
        # _api_key = os.environ.get("OPENAI_API_KEY") if not (base_url or api_key) else api_key

        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            webhook_secret=webhook_secret,
            base_url=base_url,
            timeout=timeout,
            _strict_response_validation=_strict_response_validation
        )

        self.responses_kwargs = {
            "model": self.model,
            "reasoning": {"effort": "medium",  "summary": "auto"},
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "service_tier":"flex",
            "store":True
        }

    def dialog_to_prompt(self, dialog: list[dict]) -> str:
        # OpenAI uses "developer" prompt instread of "system" (which is not exposed to the caller)
        # https://platform.openai.com/docs/guides/prompt-engineering#message-roles-and-instruction-following
        if not self.base_url:
            # We are targetting the OpenAI service
            prompt = deepcopy(dialog)
            for p in prompt:
                if p["role"] == "system":
                    p["role"] = "developer"
        return prompt   


    def run_inference(self, prompts: Union[str, list[str]], schema: str) -> Union[dict, list[dict]]:
        return self.client.responses.create(
                                                input=prompts,
                                                text=self.text_field(schema)
                                                **self.responses_kwargs
                                            )

    @staticmethod
    def text_field(schema):
        return {
            "format": {
                "type": "json_schema",
                "name": "ITM Schema",
                "schema": schema,
                "strict": True
            }
        }




