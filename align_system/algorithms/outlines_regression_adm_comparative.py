import json
import random
import yaml
import torch
from copy import deepcopy

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter
from swagger_client.models import kdma_value

from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.utils import outlines_prompts_utils
from align_system.utils import alignment_utils
from align_system.utils import incontext_utils
from align_system.utils.hydrate_state import hydrate_scenario_state
from align_system.algorithms.outlines_adm import OutlinesTransformersADM
from align_system.prompt_engineering.outlines_prompts import (
    scenario_state_description_with_relevant_char_info,
    comparative_outcomes_system_prompt,
    comparative_outcome_prediction_prompt,
    comparative_outcome_prediction_json_schema,
    comparative_kdma_score_prediction_system_prompt,
    comparative_kdma_score_prediction_system_prompt_with_examples,
    comparative_kdma_score_prediction_prompt,
    comparative_kdma_score_prediction_json_schema,
    baseline_system_prompt,
    action_selection_prompt
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesTransformersComparativeRegressionADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 **kwargs):
        self.baseline = baseline

        model_kwargs = kwargs.get('model_kwargs', {})
        if 'precision' in kwargs:
            if kwargs['precision'] == 'half':
                torch_dtype = torch.float16
            elif kwargs['precision'] == 'full':
                torch_dtype = torch.float32
            else:
                raise RuntimeError(
                    f"Unexpected value for 'precision' ({kwargs['precision']})"
                    ", expecting either 'half' or 'full'")

            model_kwargs['torch_dtype'] = torch_dtype

        self.model = outlines.models.transformers(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
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

    def sample_kdma_score_predictions(self,
                                      scenario_description,
                                      choices,
                                      target_kdmas,
                                      outcome_predictions,
                                      num_samples=1,
                                      batch_size=6,
                                      kdma_score_examples=False,
                                      incontext_settings={}):
        '''
        Samples predictions of kdma scores associated with each choice
        Returns ADM score predictions and reasoning for each choice and KDMA
        - predictions: {choice1:{kdma1:[score1(int), ...], ...}, ...}
        - reasonings: {choice1:{kdma1:[reasoning1(str), ...], ...}, ...}
        '''
        use_icl = False
        if "number" in incontext_settings and incontext_settings["number"] > 0:
            use_icl = True
            icl_example_generator = incontext_utils.ComparativeRegressionIncontextExampleGenerator(incontext_settings,
                                                                                                   target_kdmas)

        kdma_dialogs = []
        # loop over samples
        for sample_idx in range(num_samples):
            # loop over target kdmas
            for target_kdma in target_kdmas:
                if kdma_score_examples:
                    kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt_with_examples(target_kdma['name'],
                                                                                                          target_kdma['description'],
                                                                                                          target_kdma['score_examples'])
                else:
                    kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt(target_kdma['name'],
                                                                                            target_kdma['description'])

                icl_examples = []
                if use_icl:
                    # Exclude outcome prediction in prompt_to_match because ICL examples don't have outcomes
                    no_outcome_predictions = {}
                    for choice in choices:
                        no_outcome_predictions[choice] = {}
                        no_outcome_predictions[choice]['predicted_outcome'] = None
                    prompt_to_match = comparative_kdma_score_prediction_prompt(scenario_description,
                                                                                    no_outcome_predictions,
                                                                                    target_kdma['name'])
                    selected_icl_examples = icl_example_generator.select_icl_examples(target_kdma['kdma'], prompt_to_match)
                    for icl_sample in selected_icl_examples:
                        icl_examples.extend([
                            {"role": "user", "content": icl_sample['prompt']},
                            {"role": "assistant", "content": f'{icl_sample["response"]}'}
                        ])

                predict_kdma_prompt = comparative_kdma_score_prediction_prompt(scenario_description,
                                                                               outcome_predictions[sample_idx],
                                                                               target_kdma['name'])
                dialog = [{'role': 'system', 'content': kdma_score_sys_prompt}]
                dialog.extend(icl_examples)
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
        kdma_score_responses = [kdma_score_responses[i:i+len(target_kdmas)] \
                                for i in range(0,len(kdma_score_responses),len(target_kdmas))]

        log.info("[bold]*KDMA SCORE PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(kdma_score_responses, extra={"highlighter": JSON_HIGHLIGHTER})

        # Initialize output dictionaries
        predictions = {}
        reasonings = {}
        for choice in choices:
            predictions[choice] = {}
            reasonings[choice] = {}
            for target_kdma in target_kdmas:
                predictions[choice][target_kdma['kdma']] = []
                reasonings[choice][target_kdma['kdma']] = []

        # Add responses to output dictionaries
        for sample_idx in range(num_samples):
            for kdma_idx in range(len(target_kdmas)):
                kdma_prediction = kdma_score_responses[sample_idx][kdma_idx]
                kdma_key = target_kdmas[kdma_idx]['kdma']
                for choice in choices:
                    reasonings[choice][kdma_key].append(kdma_prediction[choice]['reasoning'])
                    # Scale score to be between 0 and 1 instead of 0 and 10 to match targets
                    predictions[choice][kdma_key].append(kdma_prediction[choice]['score']/10)

        return predictions, reasonings

    # Returns the outcome prediction (if there was one) and score reasoning for the best sample of the selected choice
    def get_selected_choice_reasoning(self, selected_choice, best_sample_index, outcome_predictions, reasonings):
        # If outcomes were predicted, add the best sample outcome prediction reasoning
        if outcome_predictions[best_sample_index][selected_choice]['predicted_outcome'] is not None:
            reasoning = f'{outcome_predictions[best_sample_index][selected_choice]["predicted_outcome"]} '
        else:
            reasoning = ''
        # Add the score prediction reasoning for each KDMA
        for target_kdma in list(reasonings[selected_choice].keys()):
            reasoning += f'{reasonings[selected_choice][target_kdma][best_sample_index]}'
        return reasoning


    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                predict_outcomes=False,
                                distribution_matching='sample',
                                kde_norm='globalnorm',
                                generator_batch_size=5,
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                kdma_score_examples=False,
                                **kwargs):

        character_info = outlines_prompts_utils.get_relevant_structured_character_info(scenario_state.characters)
        scenario_description = scenario_state_description_with_relevant_char_info(scenario_state, character_info)

        # Important that the choices stay in the same order as the
        # available actions as we'll use the selected index later to
        # map to the corresponding action
        choices = adm_utils.format_choices(
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
                raise RuntimeError(f'Missing target kdma description for {kdma}')
            else:
                target_kdmas[kdma_idx]['name'] = kdma_descriptions[kdma]['name']
                target_kdmas[kdma_idx]['description'] = kdma_descriptions[kdma]['description']
                target_kdmas[kdma_idx]['score_examples'] = kdma_descriptions[kdma]['score_examples']


        # Predict outcome of selecting each choice - optional
        if predict_outcomes:
            outcome_predictions = self.sample_outcome_predictions(scenario_description, choices,
                                                                 num_samples, generator_batch_size)
        else:
            # set predicted outcomes to none
            outcome_predictions = []
            for _ in range(num_samples):
                sample = {}
                for choice in choices:
                    sample[choice] = {}
                    sample[choice]['predicted_outcome'] = None
                outcome_predictions.append(sample)


        # Predict kdma values
        predicted_kdma_values, reasonings = self.sample_kdma_score_predictions(scenario_description,
                                                        choices, target_kdmas, outcome_predictions,
                                                        num_samples, generator_batch_size,
                                                        kdma_score_examples,
                                                        incontext_settings=kwargs.get("incontext", {}))

        # Get type of targets
        all_scalar_targets = True
        all_kde_targets = True
        for target_kdma in target_kdmas:
            if 'value' not in target_kdma or target_kdma["value"] is None:
                all_scalar_targets = False
            if 'kdes' not in target_kdma or target_kdma["kdes"] is None:
                all_kde_targets = False

        # Select aligned choice
        if all_scalar_targets:
            alignment_function = alignment_utils.AvgDistScalarAlignment()
            selected_choice, probs = alignment_function(predicted_kdma_values, target_kdmas)
            best_sample_index = alignment_function.get_best_sample_index(predicted_kdma_values, target_kdmas, selected_choice)
        elif all_kde_targets:
            if distribution_matching == 'sample':
                alignment_function = alignment_utils.MinDistToRandomSampleKdeAlignment()
            elif distribution_matching == 'max_likelihood':
                alignment_function = alignment_utils.MaxLikelihoodKdeAlignment()
            elif distribution_matching == 'js_divergence':
                alignment_function = alignment_utils.JsDivergenceKdeAlignment()
            else:
                raise RuntimeError(distribution_matching, "distribution matching function unrecognized.")
            selected_choice, probs = alignment_function(predicted_kdma_values, target_kdmas, kde_norm=kde_norm)
            best_sample_index = alignment_function.get_best_sample_index(predicted_kdma_values, target_kdmas, selected_choice, kde_norm=kde_norm)

        else:
            # TODO: Currently we assume all targets either have scalar values or KDES,
            #       Down the line, we should extend to handling multiple targets of mixed types
            raise ValueError("ADM does not currently support a mix of scalar and KDE targets.")

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(selected_choice, extra={"highlighter": JSON_HIGHLIGHTER})

        selected_choice_idx = choices.index(selected_choice)
        action_to_take = available_actions[selected_choice_idx]
        action_to_take.justification = self.get_selected_choice_reasoning(selected_choice, best_sample_index,
                                                                          outcome_predictions, reasonings)

        # Set up simple diaolg to return for follow-ups
        alignment_system_prompt = baseline_system_prompt()
        prompt = action_selection_prompt(scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog
