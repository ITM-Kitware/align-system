"""Local, GPU-only structured classification for live CAGE policy selection."""

import json
from functools import partial
from time import perf_counter

import outlines
from outlines.samplers import GreedySampler
from jsonschema.validators import validator_for
import torch

from align_system.algorithms.outlines_inference_engine import OutlinesTransformersInferenceEngine


class CAGELocalInferenceEngine(OutlinesTransformersInferenceEngine):
    def __init__(self, model_name, device='cuda:0', precision='half', max_tokens=128,
                 model_kwargs=None, tokenizer_kwargs=None):
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Local CIA classification requires a visible CUDA GPU. '
                               'Run inside your GPU allocation with CUDA-enabled torch.')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        self.max_tokens = max_tokens
        self.request_history = []
        self._generators = {}
        super().__init__(model_name=model_name, device=device, precision=precision,
                         sampler=GreedySampler(), inference_batch_size=1,
                         model_kwargs=dict(model_kwargs or {}),
                         tokenizer_kwargs=dict(tokenizer_kwargs or {}))
        self.model.model.eval()
        # apply_chat_template already includes BOS/EOS markers. The Cisco fast
        # tokenizer otherwise prepends another BOS during Outlines' encoding.
        self.model.tokenizer.encode = partial(self.model.tokenizer.encode,
                                               add_special_tokens=False)

    def dialog_to_prompt(self, dialog):
        # Cisco's Llama chat template needs the assistant generation header.
        return self.model.tokenizer.tokenizer.apply_chat_template(
            [dict(message) for message in dialog], tokenize=False,
            add_generation_prompt=True)

    def run_inference(self, prompts, schema):
        if isinstance(prompts, list):
            return [self.run_inference(prompt, schema) for prompt in prompts]
        if not isinstance(prompts, str):
            raise TypeError('Live CIA classification expects one trajectory per call')
        started = perf_counter()
        record = {'backend': 'local_gpu', 'model': self.model_name,
                  'precision': self.precision, 'max_tokens': self.max_tokens}
        try:
            # The schema is constant across steps. Compile its token constraints
            # once per worker instead of rebuilding them for every trajectory.
            if schema not in self._generators:
                self._generators[schema] = outlines.generate.json(
                    self.model, schema, sampler=self.sampler, whitespace_pattern=r'[ ]?')
            with torch.inference_mode():
                result = self._generators[schema](prompts, max_tokens=self.max_tokens)
            schema_dict = json.loads(schema)
            validator_for(schema_dict)(schema_dict).validate(result)
            record['response'] = result
            return result
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            # No hosted requests or provider token accounting. The shared runner
            # persists these local invocation records in request_history.jsonl.
            record['elapsed_s'] = perf_counter() - started
            self.request_history.append(record)
