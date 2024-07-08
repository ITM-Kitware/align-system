import json
import random
import itertools
import os 
import pathlib
import yaml

import outlines
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum
)

from align_system.utils import logging
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    scenario_state_description_1,
    outcomes_system_prompt,
    outcome_prediction_prompt,
    outcome_prediction_json_schema,
    kdma_score_prediction_system_prompt,
    kdma_score_prediction_prompt,
    kdma_score_prediction_json_schema
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

# TODO - make this configurable? 
KDMA_DESCRIPTIONS_FILE_PATH = os.path.join(
    pathlib.Path(__file__).parent.absolute(), '..',
    'prompt_engineering/kdma_descriptions.yml')


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


    # Returns a list of predicted outcomes corresponding to the list of choices
    def predict_outcomes(self, scenario_description, choices):
        outcome_dialogs = []
        outcomes_sys_prompt = outcomes_system_prompt()

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

        outcome_dialog_texts = [self.dialog_to_prompt(d) for d in
            itertools.chain(outcome_dialogs)]

        # List of {predicted_outcomes:}, one of each choice in order of choices
        predicted_outcomes = outcome_generator(outcome_dialog_texts)

        return predicted_outcomes

    # Predicts kdma scores associated with each choice
    # Outputs a list of ADM responses and a corresponding list of response keys:
    #   kdma_score_responses = [{score:int, reasoning:str}, ...]
    #   reponse_keys = [{kdma:str, choice:str}, ...]
    def predict_kdma_scores(self, scenario_description, choices, target_kdmas, predicted_outcomes=None):
        kdma_dialogs = []
        response_keys = []
        # loop over target kdmas
        for target_kdma in target_kdmas:
            kdma_score_sys_prompt = kdma_score_prediction_system_prompt(target_kdma['name'], target_kdma['description'])
            # loop over choices
            for choice_idx in range(len(choices)):
                choice = choices[choice_idx]
                if predicted_outcomes:
                    outcome = predicted_outcomes[choice_idx]['predicted_outcome']
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

        kdma_dialog_texts = [self.dialog_to_prompt(d) for d in
            itertools.chain(kdma_dialogs)]

        # List of {score:int, reasoning:str}
        kdma_score_responses = kdma_score_generator(kdma_dialog_texts)

        return kdma_score_responses, response_keys


    # Select choice by first averaging score across samples,
    # then selecting the one with minimal MSE to the target
    def average_distribution_matching(self, predicted_kdma_values, target_kdmas):
        # Get average of predicted scores
        average_predictions_for_each_choice = []
        choices = []
        for choice in list(predicted_kdma_values.keys()):
            choices.append(choice)
            average_predictions = {}
            for target_kdma in target_kdmas:
                kdma = target_kdma['kdma']
                samples = predicted_kdma_values[choice][kdma]['scores']
                average_predictions[kdma] = sum(samples) / len(samples) 
            average_predictions_for_each_choice.append(average_predictions)

        # get target kdma values - assumed to be int, TODO 
        target_kdma_values = {}
        for target_kdma in target_kdmas:
            target_kdma_values[target_kdma['kdma']]=target_kdma['value']

        # mean square error
        def mse(target_kdma_values, predicted_kdma_values):
            kdmas = set(target_kdma_values.keys()) & set(predicted_kdma_values.keys())
            if len(kdmas) == 0:
                return 0
            return sum([(target_kdma_values[kdma] - predicted_kdma_values[kdma])**2 for kdma in kdmas]) / len(kdmas)

        # find index of min mse
        choice_idx = 0
        min_mse = float('inf')
        for i in range(len(choices)):
            mse_ = mse(target_kdma_values, average_predictions_for_each_choice[i])
            if mse_ < min_mse:
                min_mse = mse_
                choice_idx = i
        selected_choice = choices[choice_idx]

        # For now return reasoning of first sample, TODO improve this
        reasoning = ''
        for kdma in list(target_kdma_values.keys()):
            reasoning += predicted_kdma_values[selected_choice][kdma]['reasonings'][0] + ' '

        return selected_choice, reasoning


    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                shuffle_choices=False,
                                predict_outcomes=False,
                                distribution_matching='average',
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
            kdma = target_kdmas[kdma_idx]['kdma']
            target_kdmas[kdma_idx]['name'] = kdma_descriptions[kdma]['name']
            target_kdmas[kdma_idx]['description'] = kdma_descriptions[kdma]['description'] # TODO error if description is missing

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

        # Sample multiple predictions
        for _ in range(num_samples):
            # Shuffle
            if shuffle_choices:
                shuffled_choices = random.sample(choices, len(choices))
            else:
                shuffled_choices = choices

            # Predict outcome of selecting each choice - optional
            if predict_outcomes:
                predicted_outcomes = self.predict_outcomes(scenario_description, shuffled_choices)
                # Save outcome predictions
                for idx in range(len(choices)):
                    predicted_kdma_values[shuffled_choices[idx]]['outcomes'].append(predicted_outcomes[idx]['predicted_outcome'])
            else:
                predicted_outcomes = None

            # Predict kdma values
            kdma_score_responses, response_keys = self.predict_kdma_scores(scenario_description, shuffled_choices, \
                                                                            target_kdmas, predicted_outcomes)
            # Save predictions
            for idx in range(len(kdma_score_responses)):
                response = kdma_score_responses[idx]
                kdma = response_keys[idx]['kdma']
                choice = response_keys[idx]['choice']
                predicted_kdma_values[choice][kdma]['scores'].append(response['score'])
                predicted_kdma_values[choice][kdma]['reasonings'].append(response['reasoning'])

            # TODO Logging

        # Regress best choice
        if distribution_matching == 'average':
            selected_choice, random_justification = self.average_distribution_matching(predicted_kdma_values, target_kdmas)
        else:
            raise RuntimeError("Distribution matching function not recognized.")

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(selected_choice, extra={"highlighter": JSON_HIGHLIGHTER})

        selected_choice_idx = choices.index(selected_choice)
        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = random_justification

        return action_to_take, [] # TODO