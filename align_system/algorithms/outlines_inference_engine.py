import itertools
from collections.abc import Iterable
from textwrap import dedent

import outlines
from outlines.samplers import MultinomialSampler
import jinja2
import torch

from align_system.algorithms.abstracts import StructuredInferenceEngine


class OutlinesTransformersInferenceEngine(StructuredInferenceEngine):
    def __init__(self,
                 model_name,
                 device='auto',
                 precision='full',
                 sampler=MultinomialSampler(),
                 inference_batch_size=5,
                 model_kwargs={},
                 tokenizer_kwargs={}):
        self.model_name = model_name
        self.precision = precision
        self.inference_batch_size = inference_batch_size
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs

        if self.precision == 'half':
            torch_dtype = torch.float16
        elif self.precision == 'full':
            torch_dtype = torch.float32
        else:
            raise RuntimeError(
                f"Unexpected value for 'precision' ({precision})"
                ", expecting either 'half' or 'full'")

        self.model_kwargs['torch_dtype'] = torch_dtype

        self.model = outlines.models.transformers(
            self.model_name,
            device=device,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs)
        # NOTE: In cases where we want multiple samples, we're passing
        # in a list of prompts (this allows us to shuffle answers in
        # each prompt), rather than setting the number of samples in
        # the sampler itself (which defaults to 1); setting the number
        # of samples in the sampler may result in unexpected behavior
        self.sampler = sampler

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

            assert user_msg['role'] == 'user'

            updated_content = system_msg['content'] + '\n' + user_msg['content']

            dialog = [{'role': 'user', 'content': updated_content}, *rest]

            encoded_dialog = tokenizer.apply_chat_template(dialog)

        return tokenizer.decode(encoded_dialog)

    # Function borrowed from
    # https://docs.python.org/3/library/itertools.html#itertools.batched
    # (since itertools.batched is only available in Python 3.12 or newer):
    @classmethod
    def batched(cls, iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(itertools.islice(iterator, n)):
            yield batch

    @classmethod
    def run_in_batches(cls, inference_function, inputs, batch_size):
        ''' Batch inference to avoid out of memory error'''
        outputs = []
        for batch in cls.batched(inputs, batch_size):
            output = inference_function(list(batch))
            if not isinstance(output, list):
                output = [output]
            outputs.extend(output)
        return outputs

    def run_inference(self, prompts, schema):
        generator = outlines.generate.json(
            self.model,
            schema,
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        if isinstance(prompts, str):
            return generator(prompts)
        elif isinstance(prompts, Iterable):
            return self.run_in_batches(
                generator, prompts, self.inference_batch_size)
        else:
            raise TypeError("Don't know how to run inference on provided "
                            "`prompts` object")

    def cache_repr(self):
        '''
        Return a string representation of this object for caching;
        .i.e. if the return value of this function is the same for two
        object instances, it's assumed that inference output will be
        the same
        '''
        def _sampler_repr(sampler):
            return "{}.{}({})".format(
                sampler.__class__.__module__,
                sampler.__class__.__name__,
                ", ".join([f"{k}={v}" for k, v in vars(sampler).items()]))

        return dedent(f"""
                       {self.__class__.__module__}.{self.__class__.__name__}(
                       model_name="{self.model_name}",
                       precision="{self.precision}",
                       sampler={_sampler_repr(self.sampler)},
                       inference_batch_size={self.inference_batch_size},
                       model_kwargs={self.model_kwargs},
                       tokenizer_kwargs={self.tokenizer_kwargs},
                       )""").strip()
