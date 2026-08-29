import json
from typing import Sequence
from rich.highlighter import JSONHighlighter
from swagger_client.models import KDMAValue
from enum import Enum
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement, Dialog

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

class HistoryMode(Enum):
    in_dialog = "IN_DIALOG"
    in_prompt = "IN_PROMPT"
    in_prompt_with_reasoning = "IN_PROMPT_WITH_REASONING"
    off="OFF"

class BasicOpenWorldDialogADMComponent(ADMComponent):
    '''
    IMPORTANT: This ADM is not compatible with batch mode LLM calls.
    '''
    def __init__(self,
                 structured_inference_engine,
                 scenario_description_template,
                 prompt_template,
                 output_schema_template,
                 system_prompt_template,
                 history_mode
                 ):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template
        self.system_prompt_template = system_prompt_template
        self.history_mode = HistoryMode(history_mode)

        self.dialog: Dialog = []

    def get_kdma(self, alignment_target):
        kdma_values = alignment_target.kdma_values
        if len(kdma_values) != 1:
            raise RuntimeError("This ADM assumes a single KDMA target, aborting!")
        kdma_value = kdma_values[0]
        if isinstance(kdma_value, KDMAValue):
            kdma_value = kdma_value.to_dict()

        kdma = kdma_value['kdma']
        value = kdma_value['value']
        return kdma, value
    
    def create_system_prompt(self, *, alignment_target, choices, scenario_description):
        if self.system_prompt_template is not None:
            kdma, value = self.get_kdma(alignment_target) if alignment_target else [None, None]
            system_prompt = call_with_coerced_args(
                self.system_prompt_template,
                {'target_kdma': kdma,
                 'target_value': value,
                 "choices": choices,
                 "scenario_description": scenario_description
                 })
            return str(system_prompt)

    def create_user_prompt(self, scenario_state, choices):
        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {'scenario_state': scenario_state})

        # import web_pdb; web_pdb.set_trace()
        def first_if_list(x): 
            return x[0] if isinstance(x, list) else x 
        actions = {
            HistoryMode.in_dialog: {},
            HistoryMode.in_prompt: {'actions': [first_if_list(json.loads(x.content))["action_choice"] for x in self.dialog if x.role == 'assistant']},
            HistoryMode.in_prompt_with_reasoning: {'actions': [f'{first_if_list(json.loads(x.content))["action_choice"]}. Reasoning: {first_if_list(json.loads(x.content))["detailed_reasoning"]}' for x in self.dialog if x.role == 'assistant']},
            HistoryMode.off: {}
        }[self.history_mode]
        
        user_prompt = call_with_coerced_args(
            self.prompt_template,
            {'scenario_state': scenario_state,
            'scenario_description': scenario_description,
            'choices': choices,
            **actions},
            partial=False)
        
        return str(user_prompt)

    def run_returns(self):
        return ('chosen_choice', 'justification')

    def run_sanity_check(self, scenario_state,
            choices,
            alignment_target):

        # Add System Prompt First Time Only
        # if len(self.dialog) == 0:
        self.dialog.clear()

        system_prompt = self.create_system_prompt(alignment_target=alignment_target, choices=choices, scenario_description=scenario_state.unstructured)
        if system_prompt:
            self.dialog.append(DialogElement(role='system', content=system_prompt))

        user_prompt = self.create_user_prompt(scenario_state, choices)
        self.dialog.append(DialogElement(role="user", content=str(user_prompt)))

        response = self.structured_inference_engine.run_inference(
            prompts=self.structured_inference_engine.dialog_to_prompt(self.dialog),
            schema=call_with_coerced_args(self.output_schema_template,{'choices': choices})
        )

        self.dialog.append(DialogElement(role="assistant", content=json.dumps(response)))
        
        if isinstance(response, Sequence):
            response = response[0]

        chosen_choice = response['action_choice']
        justification = response['detailed_reasoning']
        return chosen_choice, justification, self.dialog


    def run(self, scenario_state,
                choices,
                alignment_target):

            # Add System Prompt First Time Only
            if len(self.dialog) == 0:
                system_prompt = self.create_system_prompt(alignment_target=alignment_target, choices=choices, scenario_description=scenario_state.unstructured)
                if system_prompt:
                    self.dialog.append(DialogElement(role='system', content=system_prompt))
            
            user_prompt = self.create_user_prompt(scenario_state, choices)
            self.dialog.append(DialogElement(role="user", content=str(user_prompt)))

            _system_and_latest_prompt = [self.dialog[0], self.dialog[-1]]
            mode_adjusted_dialog = {
                HistoryMode.in_dialog: self.dialog,
                HistoryMode.in_prompt: _system_and_latest_prompt,
                HistoryMode.in_prompt_with_reasoning: _system_and_latest_prompt,
                HistoryMode.off: _system_and_latest_prompt
            }[self.history_mode]
            
            response = self.structured_inference_engine.run_inference(
                prompts=self.structured_inference_engine.dialog_to_prompt(mode_adjusted_dialog),
                schema=call_with_coerced_args(self.output_schema_template,{'choices': choices})
            )

            self.dialog.append(DialogElement(role="assistant", content=json.dumps(response)))
            
            if isinstance(response, Sequence):
                response = response[0]
            
            chosen_choice = response['action_choice']
            justification = response['detailed_reasoning']
            return chosen_choice, justification
    
    def reset_history(self):
        super().reset_history()
        self.dialog.clear()