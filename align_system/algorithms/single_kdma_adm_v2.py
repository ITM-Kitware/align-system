import json
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union, List, Dict, Tuple, Optional, TextIO
from align_system.algorithms.lib.chat.chat_language_model import ChatLanguageModel
from align_system.algorithms.lib.aligned_decision_maker import AlignedDecisionMaker
from align_system.algorithms.lib.util import read_template

from rich.highlighter import JSONHighlighter
from align_system.utils import logging

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()



class SingleKDMAADM(ChatLanguageModel, AlignedDecisionMaker):
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, system_messages_dir: str, domain_documents_dir: str=None, in_context_example_set=None) -> None:
        super().__init__(model, tokenizer)
        self.system_messages_dir = system_messages_dir
        self.domain_documents_dir = domain_documents_dir
        self.in_context_example_set = in_context_example_set
    
    
    def load_system_message(self, target_kdma=None, align_high=True):
        if target_kdma is None:
            
            file_name = 'baseline.txt'
        else:
            file_name = f'{"high" if align_high else "low"}-{target_kdma}.txt'

        with open(os.path.join(self.system_messages_dir, file_name), 'r') as f:
            system_message = f.read()
        
        return system_message


    def build_dialog(self, system_message, sample):
        if self.domain_documents_dir is not None:
            raise NotImplementedError('SingleKDMAADM does not support domain documents yet')

        if self.in_context_example_set is not None:
            raise NotImplementedError('SingleKDMAADM does not support in-context examples yet')

        question = sample['scenario']
        if sample['state'] is not None:
            question += f'\n{sample["state"]}'

        question += f'\n{sample["probe"]}'

        formatted_options = [f'({i}) {option}' for i, option in enumerate(sample['choices'])]

        content = f'{question} {formatted_options}'

        dialog = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": content
            }
        ]

        return dialog
        
    
    def __call__(self, sample, target_kdma_values, n_positive_samples, n_negative_samples, **kwargs):
        if target_kdma_values is None:
            positive_system_message = self.load_system_message()
            negative_system_message = None
        else:
            assert len(target_kdma_values) == 1, 'SingleKDMAADM only supports one target KDMA value'
            target_kdma = next(iter(target_kdma_values))
            align_high = target_kdma_values[target_kdma] >= 5
            positive_system_message = self.load_system_message(target_kdma, align_high)
        
        positive_dialog = self.build_dialog(positive_system_message, sample)
        negative_dialog = None
        if negative_system_message is not None:
            negative_dialog = self.build_dialog(negative_system_message, sample)
        
        positive_responses = self.generate_responses([positive_dialog] * n_positive_samples, **kwargs)

        
        
        
        
        
        
    
    