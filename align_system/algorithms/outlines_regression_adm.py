import json
import random
import os 
import pathlib
import yaml
import itertools

import outlines
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum,
    kdma_value
)

from align_system.utils import logging
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    detailed_unstructured_treatment_action_text,
    scenario_state_description_1,
    outcomes_system_prompt,
    outcome_prediction_prompt,
    outcome_prediction_json_schema,
    kdma_score_prediction_system_prompt,
    kdma_score_prediction_prompt,
    kdma_score_prediction_json_schema,
    regression_alignment_system_prompt,
    action_selection_prompt
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

# TODO - make this configurable 
KDMA_DESCRIPTIONS_FILE_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/kdma_descriptions.yml')

def run_in_batches(inference_function, inputs, batch_size):
    ''' Batch inference to avoid out of memory error'''
    outputs = []
    for batch in itertools.batched(inputs, batch_size):
        outputs.extend(inference_function(list(batch)))
    return outputs

class OutlinesTransformersRegressionADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 **kwargs):
        self.baseline = baseline
        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=kwargs.get('model_kwargs', {}),
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))


    def sample_outcome_predictions(self, 
                                   scenario_description, 
                                   choices, 
                                   num_samples=1,
                                   batch_size=5):
        '''
        Samples prediction of what the outcome would be if choices were to be selected
        Returns a list of samples where each sample is a list of predicted outcomes
        '''
        outcome_dialogs = []
        outcomes_sys_prompt = outcomes_system_prompt()

        for _ in range(num_samples):
            for choice in choices:
                predict_outcome_prompt = outcome_prediction_prompt(scenario_description, choice)
                outcome_dialogs.append([{'role': 'system', 'content': outcomes_sys_prompt},
                                        {'role': 'user', 'content': predict_outcome_prompt}])

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        outcome_generator = outlines.generate.json(
            self.model,
            outcome_prediction_json_schema(),
            whitespace_pattern=r"[ ]?")

        outcome_dialog_texts = [self.dialog_to_prompt(d) for d in outcome_dialogs]

        log.info("[bold]*OUTCOMES PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(outcome_dialog_texts[0])

        # List of {predicted_outcomes:} with length = num_samples * len(choices)
        predicted_outcomes = run_in_batches(outcome_generator, outcome_dialog_texts, batch_size)
        # Reshape to matrix of num_samples x len(choices)
        predicted_outcomes = [predicted_outcomes[i:i+len(choices)] for i in range(0,len(predicted_outcomes),len(choices))]

        log.info("[bold]*OUTCOME PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(predicted_outcomes, extra={"highlighter": JSON_HIGHLIGHTER})

        return predicted_outcomes


    def sample_kdma_score_predictions(self, 
                                      scenario_description, 
                                      choices, 
                                      target_kdmas, 
                                      predicted_outcomes=None, 
                                      num_samples=1,
                                      batch_size=6):
        '''
        Samples predictions of kdma scores associated with each choice
        Outputs a list of ADM responses and a corresponding keys:
        - kdma_score_responses = [{score:int, reasoning:str}, ...]
        - reponse_keys = [{kdma:str, choice:str}, ...]
        '''
        kdma_dialogs = []
        response_keys = []
        # loop over samples
        for sample_idx in range(num_samples):
            # loop over target kdmas
            for target_kdma in target_kdmas: 
                kdma_score_sys_prompt = kdma_score_prediction_system_prompt(target_kdma['name'], target_kdma['description'])
                # loop over choices
                for choice_idx in range(len(choices)):
                    choice = choices[choice_idx]
                    if predicted_outcomes:
                        outcome = predicted_outcomes[sample_idx][choice_idx]['predicted_outcome']
                    else:
                        outcome = None
                    predict_kdma_prompt = kdma_score_prediction_prompt(scenario_description, choice, outcome, target_kdma['name'])
                    kdma_dialogs.append([{'role': 'system', 'content': kdma_score_sys_prompt},
                                         {'role': 'user', 'content': predict_kdma_prompt}])
                    response_keys.append({'kdma':target_kdma['kdma'], 'choice':choice})

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        kdma_score_generator = outlines.generate.json(
            self.model,
            kdma_score_prediction_json_schema(),
            whitespace_pattern=r"[ ]?")

        kdma_dialog_texts = [self.dialog_to_prompt(d) for d in kdma_dialogs]

        log.info("[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(kdma_dialog_texts[0])

        # List of {score:int, reasoning:str} with length = num_samples*len(choices)*len(target_kdmas)
        kdma_score_responses = run_in_batches(kdma_score_generator, kdma_dialog_texts, batch_size)
        # # Reshape to matrix of num_samples x (len(choices)*len(target_kdmas))
        # sample_size = len(choices)*len(target_kdmas)
        # kdma_score_responses = [kdma_score_responses[i:i+sample_size] for i in range(0,len(kdma_score_responses),sample_size)]
        # response_keys = [response_keys[i:i+sample_size] for i in range(0,len(response_keys),sample_size)]

        log.info("[bold]*KDMA SCORE PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(kdma_score_responses, extra={"highlighter": JSON_HIGHLIGHTER})

        return kdma_score_responses, response_keys


    # TODO - create a separate class for distribution matching approaches
    # (each with a __call__ method) so we can specify the class target and 
    # initialize in our hydra configs. 

    def average_distribution_matching(self, predicted_kdma_values, target_kdmas):
        '''
        Selects a choice by first averaging score across samples,
        then selecting the one with minimal MSE to the target.
        Returns the selected choice and reasoning.
        '''
        # Get average of predicted scores
        average_predictions_for_each_choice = []
        choices = []
        for choice in list(predicted_kdma_values.keys()):
            choices.append(choice)
            average_predictions = {}
            for target_kdma in target_kdmas:
                kdma = target_kdma['kdma']
                samples = predicted_kdma_values[choice][kdma]['scores']
                average_predictions[kdma] = (sum(samples) / len(samples)) / 10 # /10 because predictions are 0-10, but targets are 0-1
            average_predictions_for_each_choice.append(average_predictions)

        log.explain("[bold]*AVERAGE PREDICTED SCORES*[/bold]",
                    extra={"markup": True})
        log.explain(average_predictions_for_each_choice, extra={"highlighter": JSON_HIGHLIGHTER})

        # get target kdma values - currently assumed to be float value 0-1
        target_kdma_values = {}
        for target_kdma in target_kdmas:
            target_kdma_values[target_kdma['kdma']]=target_kdma['value']

        # mean square error
        def mse(target_kdma_values, predicted_kdma_values):
            kdmas = set(target_kdma_values.keys()) & set(predicted_kdma_values.keys())
            if len(kdmas) == 0:
                return 0
            return sum([(target_kdma_values[kdma] - predicted_kdma_values[kdma])**2 for kdma in kdmas]) / len(kdmas)

        # find index of min mse and get selected choice
        choice_idx = 0
        min_mse = float('inf')
        for i in range(len(choices)):
            mse_ = mse(target_kdma_values, average_predictions_for_each_choice[i])
            if mse_ < min_mse:
                min_mse = mse_
                choice_idx = i
        selected_choice = choices[choice_idx]
        
        # If outcomes were predicted, add to reasoning
        if predicted_kdma_values[selected_choice]['outcomes']:
            reasoning = 'The predicted outcome for choice ' + selected_choice + ' was: '
            reasoning += predicted_kdma_values[selected_choice]['outcomes'][0]
        else:
            reasoning = ''
        # Add average predicted KDMA acores to reasoning
        for target_kdma in target_kdmas:
            reasoning += ' The average predcited score for ' + target_kdma['name'] + ' was ' + \
                            str(average_predictions_for_each_choice[choice_idx][target_kdma['kdma']]) + '.'
        # TODO - could improve returned reasoning

        return selected_choice, reasoning


    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                predict_outcomes=False,
                                distribution_matching='average',
                                generator_batch_size=5,
                                **kwargs):

        scenario_description = scenario_state_description_1(scenario_state)

        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = [a.unstructured for a in available_actions]

        if len(set(choices)) != len(choices):
            log.warning("Unstructured text for available actions is not "
                        "unique, appending action parameters to choices")

            character_id_to_name = {c.id: c.name for c in scenario_state.characters}
            # Important that the choices stay in the same order as the
            # available actions as we'll use the selected index later to
            # map to the corresponding action
            choices = []
            for a in available_actions:
                if(a.action_type == ActionTypeEnum.APPLY_TREATMENT
                   and a.parameters is not None and len(a.parameters) > 0):
                    choices.append(detailed_unstructured_treatment_action_text(a, character_id_to_name))
                elif(a.action_type == ActionTypeEnum.TAG_CHARACTER
                     and a.parameters is not None and len(a.parameters) > 0):
                    choices.append(detailed_unstructured_tagging_action_text(a, character_id_to_name))
                else:
                    # Not covering every possible case here, may need
                    # to add more dedicated detailed prompts
                    choices.append(a.unstructured)

        target_kdmas = alignment_target.kdma_values

        # Get kdma names and descriptions 
        with open(KDMA_DESCRIPTIONS_FILE_PATH, 'r') as f:
            kdma_descriptions = yaml.load(f, Loader=yaml.FullLoader)
        # Add names and descriptions to target_kdmas
        for kdma_idx in range(len(target_kdmas)):
            if not isinstance(target_kdmas[kdma_idx], dict):
                if isinstance(target_kdmas[kdma_idx], kdma_value.KDMAValue):
                    target_kdmas[kdma_idx] = target_kdmas[kdma_idx].to_dict()
                else:
                    target_kdmas[kdma_idx] = dict(target_kdmas[kdma_idx])
            kdma = target_kdmas[kdma_idx]['kdma']
            if kdma not in kdma_descriptions:
                raise RuntimeError("Missing target kdma description.")
            else:
                target_kdmas[kdma_idx]['name'] = kdma_descriptions[kdma]['name']
                target_kdmas[kdma_idx]['description'] = kdma_descriptions[kdma]['description']

        predicted_kdma_values = {}      # Dictionary to save output for each choice to
        for choice in choices:
            predicted_kdma_values[choice] = {
                'outcomes':[]           # List to add sampled outcome predictions to
            }
            for target_kdma in target_kdmas:
                predicted_kdma_values[choice][target_kdma['kdma']] = {
                    'scores':[],        # List to add sampled kdma scores to
                    'reasonings':[]     # List to add sampled kdma score reasonings to
                }

        # Predict outcome of selecting each choice - optional
        if predict_outcomes:
            predicted_outcomes = self.sample_outcome_predictions(scenario_description, choices, \
                                                                 num_samples, generator_batch_size)
            # Save outcome predictions
            for sample in predicted_outcomes:
                for idx in range(len(choices)):
                    predicted_kdma_values[choices[idx]]['outcomes'].append(sample[idx]['predicted_outcome'])
        else:
            predicted_outcomes = None

        # Predict kdma values
        kdma_score_responses, response_keys = self.sample_kdma_score_predictions(scenario_description, choices, \
                                                                                target_kdmas, predicted_outcomes, \
                                                                                num_samples, generator_batch_size)
        # Save predictions
        for idx in range(len(kdma_score_responses)):
            response = kdma_score_responses[idx]
            kdma = response_keys[idx]['kdma']
            choice = response_keys[idx]['choice']
            predicted_kdma_values[choice][kdma]['scores'].append(response['score'])
            predicted_kdma_values[choice][kdma]['reasonings'].append(response['reasoning'])

        # Regress best choice
        if distribution_matching == 'average':
            # Averages over predicted score samples and selects choice with minimum MSE to target
            selected_choice, first_justification = self.average_distribution_matching(predicted_kdma_values, target_kdmas)
            # Currently returning the reasoning associated with the first sample for the selected choice
        else:
            raise RuntimeError("Distribution matching function not recognized.")

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(selected_choice, extra={"highlighter": JSON_HIGHLIGHTER})

        selected_choice_idx = choices.index(selected_choice)
        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = first_justification

        # Set up simple diaolg to return for follow-ups
        alignment_system_prompt = regression_alignment_system_prompt(target_kdmas)
        prompt = action_selection_prompt(scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog