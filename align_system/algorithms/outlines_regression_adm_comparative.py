import json
import random
import os
import pathlib
import yaml
import itertools
import torch
from collections import defaultdict

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter
from swagger_client.models import (
    ActionTypeEnum,
    kdma_value
)

from align_system.utils import logging
from align_system.utils.hydrate_state import hydrate_scenario_state
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    detailed_unstructured_treatment_action_text,
    scenario_state_description_1,
    comparative_outcomes_system_prompt,
    comparative_outcome_prediction_prompt,
    comparative_outcome_prediction_json_schema,
    comparative_kdma_score_prediction_system_prompt,
    comparative_kdma_score_prediction_prompt,
    comparative_kdma_score_prediction_json_schema,
    regression_alignment_system_prompt,
    action_selection_prompt
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

# Data strucutre reformat helper function 
def merge_samples(dict_list):
    # Initialize the result dictionary
    result = {}

    # Iterate over each dictionary in the list
    for d in dict_list:
        for outer_key, inner_dict in d.items():
            if outer_key not in result:
                result[outer_key] = {}

            for inner_key, value in inner_dict.items():
                if isinstance(value, dict):
                    if inner_key not in result[outer_key]:
                        result[outer_key][inner_key] = {}
                    
                    for sub_key, sub_value in value.items():
                        if sub_key not in result[outer_key][inner_key]:
                            result[outer_key][inner_key][sub_key] = []
                        result[outer_key][inner_key][sub_key].append(sub_value)
                else:
                    if inner_key not in result[outer_key]:
                        result[outer_key][inner_key] = []
                    result[outer_key][inner_key].append(value)
    
    return result

class OutlinesTransformersComparativeRegressionADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 **kwargs):
        self.baseline = baseline
        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=kwargs.get('model_kwargs', {}),
            tokenizer_kwargs=kwargs.get('tokenizer_kwargs', {}))
        # NOTE: In cases where we want multiple samples, we're passing
        # in a list of prompts (this allows us to shuffle answers in
        # each prompt), rather than setting the number of samples in
        # the sampler itself (which defaults to 1); setting the number
        # of samples in the sampler may result in unexpected behavior
        self.sampler = sampler


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
        outcomes_sys_prompt = comparative_outcomes_system_prompt()

        for _ in range(num_samples):
            predict_outcome_prompt = comparative_outcome_prediction_prompt(scenario_description, choices)
            outcome_dialogs.append([{'role': 'system', 'content': outcomes_sys_prompt},
                                    {'role': 'user', 'content': predict_outcome_prompt}])


        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        outcome_generator = outlines.generate.json(
            self.model,
            comparative_outcome_prediction_json_schema(choices),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        outcome_dialog_texts = [self.dialog_to_prompt(d) for d in outcome_dialogs]

        log.info("[bold]*OUTCOMES PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(outcome_dialog_texts[0])

        # List of {choice: {predicted_outcomes:str}, ...} with length = num_samples
        predicted_outcomes = self.run_in_batches(outcome_generator, outcome_dialog_texts, batch_size)

        log.info("[bold]*OUTCOME PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(predicted_outcomes, extra={"highlighter": JSON_HIGHLIGHTER})

        return predicted_outcomes

    def _format_single_incontext_prompt(self, prompt, target_kdma, expected_value):
        # Predicted scores are 0-10, KDMA values are 0-1
        expected_value = int(expected_value * 10)

        # TODO: Add chain of thought reasoning?
        target_wording_mapping = {
            "MoralDesert": "moral deservingness",
            "maximization": "maximization"
        }
        reasoning_text = f"Per the principle of {target_wording_mapping.get(target_kdma, target_kdma)}, " \
                         f"this response scores a {expected_value}"

        # TODO: Validate against schema?
        answer = f'{{"reasoning": "{reasoning_text}", "score": {expected_value}}}'

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]

    def sample_kdma_score_predictions(self,
                                      scenario_description,
                                      choices,
                                      target_kdmas,
                                      predictions,
                                      num_samples=1,
                                      batch_size=6,
                                      incontext_settings={}):
        '''
        Samples predictions of kdma scores associated with each choice
        Outputs a list of ADM responses and a corresponding keys:
        - kdma_score_responses = [{score:int, reasoning:str}, ...]
        - reponse_keys = [{kdma:str, choice:str}, ...]
        '''
        use_icl = False
        icl_datasets = {}
        if "number" in incontext_settings and incontext_settings["number"] > 0:
            use_icl = True
            n_icl_examples = incontext_settings["number"]

            # Read dataset(s)
            for dset_kdma, dset_f in incontext_settings["datasets"].items():
                with open(dset_f) as f:
                    dset = json.load(f)

                icl_datasets[dset_kdma] = []
                for icl_sample in dset:
                    state, actions = hydrate_scenario_state(icl_sample["input"])
                    icl_choices = self.format_choices(
                        [a.unstructured for a in actions],
                        actions,
                        state
                    )
                    for icl_choice, label in zip(icl_choices, icl_sample["label"]):
                        if dset_kdma not in label:
                            continue

                        icl_scenario_description = scenario_state_description_1(state)

                        # TODO: Include outcome in ICL example?
                        icl_prompt = kdma_score_prediction_prompt(
                            icl_scenario_description, icl_choice, None, dset_kdma
                        )

                        icl_datasets[dset_kdma].append({
                            "prompt": icl_prompt,
                            "expected_value": label[dset_kdma],
                        })

        kdma_dialogs = []
        # loop over samples
        for sample_idx in range(num_samples):
            # loop over target kdmas
            for target_kdma in target_kdmas:
                kdma_sys_name = target_kdma['kdma']
                target_kdma_name = target_kdma['name']
                kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt(target_kdma_name, target_kdma['description'])


                icl_examples = []
                if use_icl:
                    if kdma_sys_name not in icl_datasets:
                        raise RuntimeError(f"No incontext samples for targeted kdma: {kdma_sys_name}")
                    possible_icl_examples = icl_datasets[kdma_sys_name]
                    if len(possible_icl_examples) < n_icl_examples:
                        raise RuntimeError(f"Not enough possible incontext samples to learn from. Only "
                                        f"{len(possible_icl_examples)} samples available while asking for "
                                        f"{n_icl_examples} incontext samples.")

                    # Downselect to n_icl_examples via given method
                    icl_strategy = incontext_settings["method"]
                    if icl_strategy == "random":
                        selected_icl_examples = random.sample(possible_icl_examples, n_icl_examples)
                    elif icl_strategy == "bert_similarity":
                        # TODO: Include outcome prediction for ICL examples?
                        no_outcome_prompt = kdma_score_prediction_prompt(scenario_description, choice, None, target_kdma_name)

                        possible_icl_prompts = [icl_sample["prompt"] for icl_sample in possible_icl_examples]

                        # Create similarity scores between the ICL samples and find top-k indices
                        from bert_score import score
                        _, _, F1 = score([no_outcome_prompt]*len(possible_icl_prompts), possible_icl_prompts, lang="en")
                        _, indices = torch.topk(F1, n_icl_examples)

                        selected_icl_examples = [possible_icl_examples[i] for i in indices]
                    else:
                        raise ValueError(f'"{icl_strategy}" is not a valid incontext method. Please use "random" or '
                                         '"bert_similarity"')

                    for icl_sample in selected_icl_examples:
                        icl_examples.extend(
                            self._format_single_incontext_prompt(icl_sample["prompt"], target_kdma_name, icl_sample["expected_value"])
                        )

                predict_kdma_prompt = comparative_kdma_score_prediction_prompt(scenario_description, predictions[sample_idx], target_kdma_name)
                dialog = [{'role': 'system', 'content': kdma_score_sys_prompt}]
                # dialog.extend(icl_examples) # TODO
                dialog.append({'role': 'user', 'content': predict_kdma_prompt})
                kdma_dialogs.append(dialog)

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        kdma_score_generator = outlines.generate.json(
            self.model,
            comparative_kdma_score_prediction_json_schema(choices),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        kdma_dialog_texts = [self.dialog_to_prompt(d) for d in kdma_dialogs]

        log.info("[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(kdma_dialog_texts[0])

        # List of {choice: {score:int, reasoning:str}, ...} with length = num_samples*len(target_kdmas)
        kdma_score_responses = self.run_in_batches(kdma_score_generator, kdma_dialog_texts, batch_size)
        # Reshape to matrix of num_samples x len(target_kdmas)
        kdma_score_responses = [kdma_score_responses[i:i+len(target_kdmas)] for i in range(0,len(kdma_score_responses),len(target_kdmas))]

        log.info("[bold]*KDMA SCORE PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(kdma_score_responses, extra={"highlighter": JSON_HIGHLIGHTER})

        # Add responses to predictions
        for sample_idx in range(num_samples):
            for kdma_idx in range(len(target_kdmas)):
                kdma_prediction = kdma_score_responses[sample_idx][kdma_idx]
                kdma_key = target_kdmas[kdma_idx]['kdma']
                for choice in choices:
                    predictions[sample_idx][choice][kdma_key] = kdma_prediction[choice]

        return predictions


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
                samples = predicted_kdma_values[choice][kdma]['score']
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
        if predicted_kdma_values[selected_choice]['predicted_outcome'][0] is not None:
            reasoning = 'The predicted outcome for choice ' + selected_choice + ' was: '
            reasoning += predicted_kdma_values[selected_choice]['predicted_outcome'][0]
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
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                **kwargs):

        scenario_description = scenario_state_description_1(scenario_state)

        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = self.format_choices(
            [a.unstructured for a in available_actions],
            available_actions,
            scenario_state
        )

        target_kdmas = alignment_target.kdma_values

        # Get kdma names and descriptions
        with open(kdma_descriptions_map, 'r') as f:
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

        # Predict outcome of selecting each choice - optional
        if predict_outcomes:
            predictions = self.sample_outcome_predictions(scenario_description, choices, \
                                                                 num_samples, generator_batch_size)
        else:
            # set predicted outcomes to none
            predictions = []
            for _ in range(num_samples):
                sample = {}
                for choice in choices:
                    sample[choice] = {}
                    sample[choice]['predicted_outcome'] = None
                predictions.append(sample)


        # Predict kdma values
        predictions = self.sample_kdma_score_predictions(scenario_description, choices, \
                                                        target_kdmas, predictions, \
                                                        num_samples, generator_batch_size, \
                                                        incontext_settings=kwargs.get("incontext", {}))

        # Reformat predictions from a list of sampled dictionaries 
        # to a single dictionary with values that are a list of samples
        predicted_kdma_values = merge_samples(predictions)

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
