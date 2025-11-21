from typing import Union
import json

import jinja2
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from align_system.algorithms.abstracts import StructuredInferenceEngine

# Sometimes the internal default for VLLM is 50,
# leading to very short (and often invalid JSON) outputs.  Setting a
# somewhat generous default.
DEFAULT_MAX_TOKENS = 8192

class VLLMInferenceEngine(StructuredInferenceEngine):
    def __init__(self,
                 model_name,
                 sampling_params=None):
        self.llm = LLM(model=model_name)

        self.sampling_params = sampling_params
        if self.sampling_params is None:
            self.sampling_params = {}

        if 'max_tokens' not in self.sampling_params:
            self.sampling_params['max_tokens'] = DEFAULT_MAX_TOKENS

    def dialog_to_prompt(self, dialog: list[dict]) -> str:
        tokenizer = self.llm.get_tokenizer()

        try:
            encoded_dialog = tokenizer.apply_chat_template(dialog)
        except jinja2.exceptions.TemplateError:
            # Assume that the tokenizer chat template doesn't accept
            # system messages; combine system message first user
            # message
            # Ensure each dialog element is a dict
            system_msg, user_msg, *rest = [dict(d) for d in dialog]

            assert user_msg['role'] == 'user'

            updated_content = system_msg['content'] + '\n' + user_msg['content']

            dialog = [{'role': 'user', 'content': updated_content}, *rest]

            encoded_dialog = tokenizer.apply_chat_template(dialog)

        return tokenizer.decode(encoded_dialog)

    def run_inference(self,
                      prompts: Union[str, list[str]],
                      schema: str) -> Union[dict, list[dict]]:
        json_schema = json.loads(schema)
        schema_params = StructuredOutputsParams(json=json_schema)

        structured_sampling_params = SamplingParams(
            **self.sampling_params,
            structured_outputs=schema_params)

        outputs = self.llm.generate(
            prompts,
            sampling_params=structured_sampling_params)

        parsed_outputs = [json.loads(o.outputs[0].text) for o in outputs]

        if isinstance(prompts, str):
            # Return single instance if single prompt provided as a string
            return parsed_outputs[0]
        else:
            return parsed_outputs
