"""Direct, tensor-parallel vLLM inference for live CIA policy selection."""

import json
import os
from time import perf_counter

from jsonschema.validators import validator_for

from align_system.algorithms.abstracts import StructuredInferenceEngine


class CAGEVLLMInferenceEngine(StructuredInferenceEngine):
    def __init__(self, model_name, tensor_parallel_size=2, dtype='bfloat16',
                 max_model_len=8192, gpu_memory_utilization=0.90,
                 max_tokens=128, seed=0):
        # Match the GPU launcher when this adapter is invoked directly from Python.
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        import torch
        from vllm import LLM

        if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
            raise ValueError('tensor_parallel_size must be a positive integer')
        if not torch.cuda.is_available() or torch.cuda.device_count() < tensor_parallel_size:
            raise RuntimeError(f'Direct vLLM inference requires {tensor_parallel_size} visible CUDA GPUs. '
                               'Run inside the requested GPU allocation.')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.seed = seed
        self.request_history = []
        self._sampling_params = {}
        self.model = LLM(
            model=model_name, tensor_parallel_size=tensor_parallel_size, dtype=dtype,
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=1, max_num_batched_tokens=4096,
            language_model_only=True, generation_config='vllm', seed=seed,
            distributed_executor_backend='mp', enforce_eager=True)

    def dialog_to_prompt(self, dialog):
        return json.dumps([{'role': message['role'], 'content': message['content']}
                           for message in map(dict, dialog)])

    def run_inference(self, prompts, schema):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if not isinstance(prompts, str):
            raise TypeError('Live CIA classification expects one trajectory per call')
        schema_dict = json.loads(schema)
        validator = validator_for(schema_dict)(schema_dict)
        if schema not in self._sampling_params:
            validator.check_schema(schema_dict)
            self._sampling_params[schema] = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=self.max_tokens, seed=self.seed,
                structured_outputs=StructuredOutputsParams(json=schema_dict))
        messages = json.loads(prompts)
        # Preserve the hosted experiment's explicit schema instruction.
        messages.append({'role': 'user', 'content': (
            'Return only valid JSON matching this JSON Schema. '
            'Do not use Markdown fences or surrounding commentary:\n' + json.dumps(schema_dict))})
        started = perf_counter()
        record = {'backend': 'local_vllm', 'model': self.model_name, 'precision': self.dtype,
                  'tensor_parallel_size': self.tensor_parallel_size, 'max_tokens': self.max_tokens,
                  'seed': self.seed, 'temperature': 0.0}
        try:
            outputs = self.model.chat(
                messages, sampling_params=self._sampling_params[schema], use_tqdm=False,
                chat_template_kwargs={'enable_thinking': False})
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise ValueError('Expected exactly one vLLM completion')
            request, completion = outputs[0], outputs[0].outputs[0]
            record.update(response_text=completion.text, finish_reason=completion.finish_reason,
                          usage={'prompt_tokens': len(request.prompt_token_ids),
                                 'completion_tokens': len(completion.token_ids)})
            if completion.finish_reason != 'stop':
                raise ValueError(f'vLLM completion ended with {completion.finish_reason!r}')
            result = json.loads(completion.text)
            validator.validate(result)
            record['status'] = 'completed'
            return result
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            record['latency_s'] = perf_counter() - started
            self.request_history.append(record)
