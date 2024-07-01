import json
import re
import random
import os
import pathlib
import numpy as np
import torch

from jinja2.exceptions import TemplateError

from rich.highlighter import JSONHighlighter
from transformers import AutoModelForCausalLM, AutoTokenizer

from align_system.utils import logging
from align_system.algorithms.abstracts import AlignedDecisionMaker
from align_system.similarity_measures import build_force_choice_func


log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


kdmas = {
    'basic_knowledge',
    'fairness',
    'protocol_focus',
    'time_pressure',
    'risk_aversion',
    'utilitarianism',
    'mission',
    'denial',
    'moral_deservingness',
    'lives_saved',
    'continuation_of_care',
    'maximization'
}

kdma_remapping = {
    'basicknowledge': 'basic_knowledge',
    'protocolfocus': 'protocol_focus',
    'riskaversion': 'risk_aversion',
    'moraldeservingness': 'moral_deservingness',
    'continuationofcare': 'continuation_of_care',
    'livesaved': 'lives_saved',
    'timepressure': 'time_pressure',
}


default_system_messages_path=os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/single_kdma_adm_system_messges')

chat_template_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/chat_templates')


def load_system_message(alignment=None,
                        system_messages_path=default_system_messages_path):
    if alignment is None:
        file_name = 'baseline.txt'
    else:
        sorted_kdmas = sorted(alignment.keys())

        alignment_string = '-'.join(
            '{}-{}'.format(alignment[k], kdma_remapping.get(k, k))
            for k in sorted_kdmas)

        file_name = f'{alignment_string}.txt'

    with open(os.path.join(system_messages_path, file_name), 'r') as f:
        system_message = f.read()
    return system_message


STANDARD_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>}\\n"

TREATMENT_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>, \"Location\": \"<Specific location on the body where the treatment should be applied, one of: 'right forearm', 'left forearm', 'right calf', 'left calf', 'right thigh', 'left thigh', 'right stomach', 'left stomach', 'right bicep', 'left bicep', 'right shoulder', 'left shoulder', 'right side', 'left side', 'right chest', 'left chest', 'right wrist', 'left wrist', 'left face', 'right face', 'left neck', 'right neck', 'unspecified'>\"}\\n"

TAGGING_MULTIPLE_CHOICE_JSON_FORMAT = "{\"Reasoning\": \"<Provide a reasoned explanation here>\", \"Answer\": <Integer index corresponding to your final answer>, \"Tag\": \"<Specific medical triage tag to apply, one of: 'MINIMAL', 'DELAYED', 'IMMEDIATE', 'EXPECTANT'>\"}\\n"


class CodeActAgentADM(AlignedDecisionMaker):

    def __init__(self, 
                 device : str = 'cuda',
                 hf_model : str = 'xingyaoww/CodeActAgent-Mistral-7b-v0.1',
                 precision : str = 'full',
                 temperature : float = 0.7,
                 do_sample : bool = True,
                 **kwargs):
        
        self.device = device
        self.hf_model = hf_model
        self.temperature = temperature
        self.do_sample = do_sample
        self.chat_template = kwargs.get('chat_template', None)

        assert precision in ['full', 'half'], "precision must be either 'full' or 'half'."
        self.precision = torch.float32 if precision == 'full' else torch.float16

        self.model = None
        self.tokenizer = None


    def load_model(self, model=None, tokenizer=None):
        assert (model is None) == (tokenizer is None), "model and tokenizer must both be None or both be not None."
        if model is not None:
            print('Loading model and tokenizer from provided objects.')
            self.model = model
            self.tokenizer = tokenizer
        else:
            print('Loading model:', self.hf_model)
            if self.device == 'auto':
                self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, torch_dtype=self.precision, device_map='auto')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.hf_model, torch_dtype=self.precision)
                self.model = self.model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)

            if self.chat_template is not None:
                with open(os.path.join(chat_template_path, self.chat_template), 'r') as f:
                    self.tokenizer.chat_template = f.read().replace('    ', '').replace('\n', '')


    def __call__(self, sample, target_kdma_values, **kwargs):

        # TODO: Refactor the following code to fit CodeAct Agent
        # Refer to 'align-system/align_system/algorithms/llama_2_single_kdma_adm.py' for related methods and detailed implementations

        prompt = sample['scenario']
        if sample['state'] is not None:
            prompt += f'\n{sample["state"]}'

        if 'retriever' in kwargs:
            # retriever_prompt = "How would you treat the following injuries: {}".format(prompt)
            retriever_prompt = "{}  {}".format(prompt, sample['probe'])

            retriever = kwargs['retriever']
            retrieved_nodes = retriever.retrieve(retriever_prompt)

            if 'summarizer' in kwargs:
                summarizer = kwargs['summarizer']
                summary = summarizer.synthesize(retriever_prompt, nodes=retrieved_nodes)

                log.explain("[bold] ** Retrieval Summary ** [/bold]",
                            extra={"markup": True})
                log.explain(summary)

                prompt += "\n#############\n{}\n#############".format(summary)

            else:
                prompt += "\n#############\n{}\n#############".format(
                    "\n#############\n".join((n.text for n in retrieved_nodes)))

            prompt += f'\nGiven the scenario and documentation above.. {sample["probe"]}'
        else:
            prompt += f'\n{sample["probe"]}'

        choices = sample['choices']

        labels = kwargs.get('labels', {})

        alignment_target = None
        if target_kdma_values is not None and len(target_kdma_values) > 0:
            target_kdma = next(iter(next(iter(filter(lambda x: len(x) > 0, labels))))) # get the frist key of the first label that is not empty

            for label in labels:
                assert len(label) == 0 or (target_kdma in label and len(label) == 1), f'All labels must have the same KDMA: labels={labels}'

            alignment_target = {
                target_kdma: target_kdma_values[target_kdma]
            }

        reasoning, answer_idx, responses, inference_pairs = self.run_aligned_decision_maker_with_voting(
            prompt,
            choices,
            alignment_target,
            n_positive_samples=kwargs.get('n_positive_samples', 5),
            n_negative_samples=kwargs.get('n_negative_samples', 5),
            baseline=kwargs.get('baseline', False),
            shuffle=kwargs.get('shuffle', False)
        )

        raw_data = {
            'params': {
                'model': self.hf_model,
                'temperature': self.temperature,
                'n_positive_samples': kwargs.get('n_positive_samples', 5),
                'n_negative_samples': kwargs.get('n_negative_samples', 5),
                'baseline': kwargs.get('baseline', False),
                'shuffle': kwargs.get('shuffle', False),
            },
            'inference_pairs': inference_pairs
        }

        return {
            'choice': int(answer_idx),
            'info': {
                'reasoning': reasoning,
                'responses': responses,
                'raw_data': raw_data,
            }
        }


