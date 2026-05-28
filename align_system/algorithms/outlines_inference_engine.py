import itertools
import json
from collections.abc import Iterable
from textwrap import dedent

import jinja2
import outlines
import torch
import transformers
from outlines.types import JsonSchema

from align_system.algorithms.abstracts import StructuredInferenceEngine

# Sometimes the internal default for outlines/transformers is 20,
# leading to very short (and often invalid JSON) outputs.  Setting a
# somewhat generous default.
DEFAULT_MAX_GENERATOR_TOKENS = 8192


class OutlinesTransformersInferenceEngine(StructuredInferenceEngine):
    def __init__(
        self,
        model_name,
        device="auto",
        precision="full",
        max_generator_tokens=DEFAULT_MAX_GENERATOR_TOKENS,
        inference_batch_size=5,
        generation_kwargs=None,
        model_kwargs=None,
        tokenizer_kwargs=None,
    ):
        self.model_name = model_name
        self.precision = precision
        self.inference_batch_size = inference_batch_size

        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        self.tokenizer_kwargs = tokenizer_kwargs

        if generation_kwargs is None:
            generation_kwargs = {}
        self.generation_kwargs = generation_kwargs

        self.max_generator_tokens = max_generator_tokens

        if self.precision == "half":
            torch_dtype = torch.float16
        elif self.precision == "full":
            torch_dtype = torch.float32
        else:
            raise RuntimeError(
                f"Unexpected value for 'precision' ({precision})"
                ", expecting either 'half' or 'full'"
            )

        self.model_kwargs["dtype"] = torch_dtype

        self.model = outlines.from_transformers(
            transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **self.model_kwargs, device_map=device
            ),
            transformers.AutoTokenizer.from_pretrained(
                model_name, **self.tokenizer_kwargs
            ),
            device_dtype=torch_dtype,
        )

        # https://github.com/dottxt-ai/outlines/pull/1817
        # newer verion of outlines fixes this issue, but we are blocked with the vllm dependency
        self.model.tokenizer.is_llama = True

        # If generation_kwargs includes temperature, enable sampling in the model's
        # generation_config so transformers doesn't warn that temperature is invalid.
        if self.generation_kwargs.get("temperature", 0.0) > 0:
            self.model.model.generation_config.do_sample = True

    def dialog_to_prompt(self, dialog):
        tokenizer = self.model.tokenizer.tokenizer

        try:
            encoded_dialog = tokenizer.apply_chat_template(dialog)
        except jinja2.exceptions.TemplateError:
            # Assume that the tokenizer chat template doesn't accept
            # system messages; combine system message first user
            # message
            # Ensure each dialog element is a dict
            system_msg, user_msg, *rest = [dict(d) for d in dialog]

            assert user_msg["role"] == "user"

            updated_content = system_msg["content"] + "\n" + user_msg["content"]

            dialog = [{"role": "user", "content": updated_content}, *rest]

            encoded_dialog = tokenizer.apply_chat_template(dialog)

        return tokenizer.decode(encoded_dialog)

    # Function borrowed from
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # (since itertools.batched is only available in Python 3.12 or newer):
    @classmethod
    def batched(cls, iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            yield batch

    @classmethod
    def run_in_batches(
        cls,
        inference_function,
        inputs,
        batch_size,
        max_generator_tokens=DEFAULT_MAX_GENERATOR_TOKENS,
        **generation_kwargs,
    ):
        """Batch inference to avoid out of memory error"""
        outputs = []
        for batch in cls.batched(inputs, batch_size):
            output = inference_function(
                list(batch), max_new_tokens=max_generator_tokens, **generation_kwargs
            )
            if not isinstance(output, list):
                output = [output]
            outputs.extend(output)
        return outputs

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1].strip()
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j > i:
            text = text[i:j + 1]
        return json.loads(text)

    def _prompt_based_inference(self, prompts, schema) -> list[dict]:
        """Fallback: append schema as a prompt hint and parse free-text output."""
        schema_instruction = (
            "\n\nRespond with ONLY valid JSON matching this example "
            "(no prose, no markdown fences):\n" + schema
        )
        if isinstance(prompts, str):
            prompts = [prompts]
        generator = outlines.Generator(self.model)
        results = []
        for prompt in prompts:
            raw = generator(
                prompt + schema_instruction,
                max_new_tokens=self.max_generator_tokens,
                **self.generation_kwargs,
            )
            try:
                results.append(self._parse_json(raw))
            except Exception:
                results.append({})
        return results

    def run_inference(self, prompts, schema, temperature: float = None):
        json_schema = JsonSchema(schema, whitespace_pattern=r"[ ]?")
        try:
            generator = outlines.Generator(self.model, json_schema)
        except (ValueError, Exception):
            # schema is a JSON example/template rather than a proper JSON
            # Schema — fall back to prompt-based generation + parsing
            return self._prompt_based_inference(prompts, schema)

        gen_kwargs = dict(self.generation_kwargs)
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"] = True

        if isinstance(prompts, str):
            output = generator(
                prompts,
                max_new_tokens=self.max_generator_tokens,
                **gen_kwargs,
            )
            try:
                return json.loads(output)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse structured generation output as JSON "
                    f"(output may be truncated; consider increasing "
                    f"max_generator_tokens above {self.max_generator_tokens}). "
                    f"Raw output: {output!r}. Original error: {e}"
                ) from e
        elif isinstance(prompts, Iterable):
            output = self.run_in_batches(
                generator.batch,
                prompts,
                self.inference_batch_size,
                self.max_generator_tokens,
                **gen_kwargs,
            )
            try:
                return [json.loads(r) for r in output]
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse structured generation output as JSON "
                    f"(output may be truncated; consider increasing "
                    f"max_generator_tokens above {self.max_generator_tokens}). "
                    f"Raw output: {output!r}. Original error: {e}"
                ) from e
        else:
            raise TypeError(
                "Don't know how to run inference on provided `prompts` object"
            )

    def run_inference_unstructured(self, prompts):
        generator = outlines.Generator(self.model)

        if isinstance(prompts, str):
            return generator(prompts, max_new_tokens=self.max_generator_tokens, **self.generation_kwargs)
        elif isinstance(prompts, Iterable):
            return self.run_in_batches(
                generator, prompts, self.inference_batch_size, self.max_generator_tokens
            )
        else:
            raise TypeError(
                "Don't know how to run inference on provided `prompts` object"
            )

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
                       precision="{self.precision}",
                       inference_batch_size={self.inference_batch_size},
                       model_kwargs={self.model_kwargs},
                       tokenizer_kwargs={self.tokenizer_kwargs},
                       generation_kwargs={self.generation_kwargs},
                       )""").strip()


class SpectrumTunedInferenceEngine(OutlinesTransformersInferenceEngine):
    def dialog_to_prompt(self, dialog):
        tokenizer = self.model.tokenizer.tokenizer

        # Use roles spectrum tuned models expect
        # https://github.com/tsor13/spectrum/blob/main/README.md
        for element in dialog:
            if element.role == "system":
                element.role = "description"
            elif element.role == "user":
                element.role = "input"
            elif element.role == "assistant":
                element.role = "output"
            else:
                raise RuntimeError(f"{element.role} dialog element unrecognized.")

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
