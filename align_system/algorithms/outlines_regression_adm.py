import json
import random
import os
import pathlib
import yaml
import itertools
import torch

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
    scenario_state_description_dre,
    outcomes_system_prompt,
    outcome_prediction_prompt,
    outcome_prediction_json_schema,
    kdma_score_prediction_system_prompt,
    kdma_score_prediction_system_prompt_with_examples,
    kdma_score_prediction_prompt,
    kdma_score_prediction_json_schema,
    baseline_system_prompt,
    action_selection_prompt
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()

# Incontext helper function
def get_chain_of_thought_reasoning(target_kdma, action, state, choice, label):
    # Predicted scores are 0-10, KDMA values are 0-1
    expected_value = int(label * 10)
    cot_reasoning = ""

    # get character associated with action
    action_character = None
    if hasattr(action, 'character_id') and action.character_id is not None:
        for character in state.characters:
            if character.id == action.character_id:
                action_character = character
                break

    # Moral derivngess or moral judgement
    # Example: Because Alex acted with no intent and did not cause of harm,
    #          choosing Treat Alex would score a 5 for the the principle of moral judgement.
    if target_kdma['kdma'] == 'MoralDesert' or  target_kdma['kdma'] == 'Moral judgement':
        if action_character is not None:
            # add intent to reasoning
            if not hasattr(action_character, 'intent') or action_character.intent is None or action_character.intent == 'no intent':
                cot_reasoning += f"Because {action_character.name} acted with no intent"
            else:
                cot_reasoning += f"Because {action_character.name} acted to {action_character.intent}"
            # add directness of causality to reasoning
            if not hasattr(action_character, 'directness_of_causality') or action_character.directness_of_causality is None or action_character.directness_of_causality == 'none':
                    cot_reasoning += f" and did not cause of harm"
            else:
                cot_reasoning += f" and had a {action_character.directness_of_causality} cause of harm"

    # Ingroup bias
    # Example: Patient A is a soldier in your unit and is in group Allied US,
    #           choosing to treat patient A would score a 6 for ingroup bias.
    elif target_kdma['kdma'] == 'Ingroup Bias':
        if action_character is not None:
            # add unstructured charcter description to reasoning
            if hasattr(action_character, 'unstructured') and action_character.unstructured is not None:
                cot_reasoning += f"Because {action_character.name} is {action_character.unstructured.rstrip()}"
                # add military disposition
                if action_character.demographics['military_disposition'] is not None:
                    cot_reasoning += f" and is in group {action_character.demographics['military_disposition']}"

    # Quality of life
    # Example
        # Because Casualty X would have better quality of life post-treatment, choosing Treat Casualty X with a tourniquet would score a 9 for the the principle of quality of life.
        # Because Casualty X would have better quality of life post-treatment but it is not the best treatment approach, choosing Treat Casualty X with hemostatic gauze would score a 7 for the the principle of quality of life.
        # Because Casualty Y would have worse quality of life post-treatment but the supplies needed to treat the other casualty would still be available, choosing Treat Casualty Y with hemostatic gauze would score a 3 for the the principle of quality of life.
        # Because Casualty Y would have worse quality of life post-treatment, choosing Treat Casualty Y with a tourniquet would score a 1 for the the principle of quality of life.
    elif target_kdma['kdma'] == 'QualityOfLife':
        if action_character is not None:
            # Better quality of life patient
            if expected_value > 5:
                cot_reasoning += f"Because {action_character.name} would have better quality of life post-treatment"
                # Sub-optimal treatment
                if expected_value == 7:
                    cot_reasoning += f" but it is not the best treatment approach"
            # Worse quality of life patient
            elif expected_value < 5:
                cot_reasoning += f"Because {action_character.name} would have worse quality of life post-treatment"
                # Sub-optimal treatment
                if expected_value == 3:
                    cot_reasoning += " but the supplies needed to treat the other casualty would still be available"

    # Percieved quanity of lives saved
    # Example:
        # Because Casualty G would likely survive treatment, choosing Treat Casualty G with hemostatic gauze would score a 9 for the the principle of perceived quantity of lives saved.
        # Because Casualty G would likely survive treatment but it is not the best treatment approach, choosing Treat Casualty G with pressure bandage would score a 7 for the the principle of perceived quantity of lives saved.
        # Because Casualty H would likely not survive treatment but the supplies needed to treat the other casualty would still be available, choosing Treat Casualty H with pressure bandage would score a 3 for the the principle of perceived quantity of lives saved.
        # Because Casualty H would likely not survive treatment, choosing Treat Casualty H with hemostatic gauze would score a 1 for the the principle of perceived quantity of lives saved.
    elif target_kdma['kdma'] == 'PerceivedQuantityOfLivesSaved':
        if action_character is not None:
            # Better vol patient
            if expected_value > 5:
                cot_reasoning += f"Because {action_character.name} would likely survive treatment"
                # Sub-optimal treatment
                if expected_value == 7:
                    cot_reasoning += f" but it is not the best treatment approach"
            # Worse quality of life patient
            elif expected_value < 5:
                cot_reasoning += f"Because {action_character.name} would likely not survive treatment"
                # Sub-optimal treatment
                if expected_value == 3:
                    cot_reasoning += " but the supplies needed to treat the other casualty would still be available"

    # if a "because" statement has not been added
    if cot_reasoning == '':
        cot_reasoning = f"Choosing "
    # else the reasoning will have form "Because ..., choosing ..."
    else:
        cot_reasoning += ", choosing "
    cot_reasoning += f"{choice} would score a {expected_value} for the the principle of {target_kdma['name']}."

    return cot_reasoning


class OutlinesTransformersRegressionADM(OutlinesTransformersADM):
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
        outcomes_sys_prompt = outcomes_system_prompt()

        for _ in range(num_samples):
            for choice in choices:
                predict_outcome_prompt = outcome_prediction_prompt(scenario_description, choices, choice)
                outcome_dialogs.append([{'role': 'system', 'content': outcomes_sys_prompt},
                                        {'role': 'user', 'content': predict_outcome_prompt}])

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        outcome_generator = outlines.generate.json(
            self.model,
            outcome_prediction_json_schema(),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        outcome_dialog_texts = [self.dialog_to_prompt(d) for d in outcome_dialogs]

        log.info("[bold]*OUTCOMES PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(outcome_dialog_texts[0])

        # List of {predicted_outcomes:} with length = num_samples * len(choices)
        predicted_outcomes = self.run_in_batches(outcome_generator, outcome_dialog_texts, batch_size)
        # Reshape to matrix of num_samples x len(choices)
        predicted_outcomes = [predicted_outcomes[i:i+len(choices)] for i in range(0,len(predicted_outcomes),len(choices))]

        log.info("[bold]*OUTCOME PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(predicted_outcomes, extra={"highlighter": JSON_HIGHLIGHTER})

        return predicted_outcomes

    def _format_single_incontext_prompt(self, icl_example, target_kdma):
        # Predicted scores are 0-10, KDMA values are 0-1
        expected_value = int(icl_example['expected_value'] * 10)

        answer = f'{{"reasoning": "{icl_example["reasoning"]}", "score": {expected_value}}}'

        return [
            {"role": "user", "content": icl_example['prompt']},
            {"role": "assistant", "content": answer}
        ]

    def get_icl_datasets(self, incontext_settings, target_kdmas):
        icl_datasets = {}
        # Read dataset(s)
        for target_kdma in target_kdmas:
            dset_kdma = target_kdma['kdma']
            dset_f = incontext_settings["datasets"][dset_kdma]
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
                for action, icl_choice, label in zip(actions, icl_choices, icl_sample["label"]):
                    if dset_kdma not in label:
                        continue

                    icl_scenario_description = scenario_state_description_dre(state)

                    # TODO: Include outcome in ICL example?
                    icl_prompt = kdma_score_prediction_prompt(
                        icl_scenario_description, icl_choices, icl_choice, None, dset_kdma
                    )

                    icl_datasets[dset_kdma].append({
                        "prompt": icl_prompt,
                        "expected_value": label[dset_kdma],
                        "reasoning": get_chain_of_thought_reasoning(target_kdma, action, state, icl_choice, label[dset_kdma])
                    })
        return icl_datasets


    def sample_kdma_score_predictions(self,
                                      scenario_description,
                                      choices,
                                      target_kdmas,
                                      predicted_outcomes=None,
                                      num_samples=1,
                                      batch_size=6,
                                      kdma_score_examples=False,
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
            icl_datasets = self.get_icl_datasets(incontext_settings, target_kdmas)

        kdma_dialogs = []
        response_keys = []
        # loop over samples
        for sample_idx in range(num_samples):
            # loop over target kdmas
            for target_kdma in target_kdmas:
                if kdma_score_examples:
                    kdma_score_sys_prompt = kdma_score_prediction_system_prompt_with_examples(target_kdma['name'], target_kdma['description'], target_kdma['score_examples'])
                else:
                    kdma_score_sys_prompt = kdma_score_prediction_system_prompt(target_kdma['name'], target_kdma['description'])

                # loop over choices
                for choice_idx in range(len(choices)):
                    choice = choices[choice_idx]
                    if predicted_outcomes:
                        outcome = predicted_outcomes[sample_idx][choice_idx]['predicted_outcome']
                    else:
                        outcome = None

                    icl_examples = []
                    if use_icl:
                        if target_kdma['kdma'] not in icl_datasets:
                            raise RuntimeError(f"No incontext samples for targeted kdma: {target_kdma['name']}")
                        possible_icl_examples = icl_datasets[target_kdma['kdma']]
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
                            no_outcome_prompt = kdma_score_prediction_prompt(scenario_description, choices, choice, None, target_kdma['name'])

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
                                self._format_single_incontext_prompt(icl_sample, target_kdma)
                            )

                    predict_kdma_prompt = kdma_score_prediction_prompt(scenario_description, choices, choice, outcome, target_kdma['name'])
                    dialog = [{'role': 'system', 'content': kdma_score_sys_prompt}]
                    dialog.extend(icl_examples)
                    dialog.append({'role': 'user', 'content': predict_kdma_prompt})
                    kdma_dialogs.append(dialog)
                    response_keys.append({'kdma':target_kdma['kdma'], 'choice':choice})

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        kdma_score_generator = outlines.generate.json(
            self.model,
            kdma_score_prediction_json_schema(),
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        kdma_dialog_texts = [self.dialog_to_prompt(d) for d in kdma_dialogs]

        log.info("[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(kdma_dialog_texts[0])

        # List of {score:int, reasoning:str} with length = num_samples*len(choices)*len(target_kdmas)
        kdma_score_responses = self.run_in_batches(kdma_score_generator, kdma_dialog_texts, batch_size)

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
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                kdma_score_examples=False,
                                **kwargs):

        scenario_description = scenario_state_description_dre(scenario_state)

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
                target_kdmas[kdma_idx]['score_examples'] = kdma_descriptions[kdma]['score_examples']

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
                                                                                num_samples, generator_batch_size, \
                                                                                kdma_score_examples, \
                                                                                incontext_settings=kwargs.get("incontext", {}))
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
        alignment_system_prompt = baseline_system_prompt()
        prompt = action_selection_prompt(scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog
