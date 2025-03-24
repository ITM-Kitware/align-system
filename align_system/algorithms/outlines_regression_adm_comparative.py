import yaml
import torch
import jinja2
import json
import numpy as np
import copy

import outlines
from outlines.samplers import MultinomialSampler
from rich.highlighter import JSONHighlighter
from swagger_client.models import kdma_value
import ubelt as ub

from align_system.utils import logging
from align_system.utils import adm_utils
from align_system.utils import outlines_prompts_utils
from align_system.utils import alignment_utils
from align_system.utils import incontext_utils
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
    enum_comparative_kdma_score_prediction_json_schema,
    baseline_system_prompt,
    action_selection_prompt,
    relevance_classification_system_prompt,
    relevance_classification_prompt,
    relevance_classification_json_schema
)

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class OutlinesTransformersComparativeRegressionADM(OutlinesTransformersADM):
    def __init__(self,
                 model_name,
                 device='auto',
                 baseline=False,
                 sampler=MultinomialSampler(),
                 probabilistic=False,
                 **kwargs):
        self.baseline = baseline
        self.probabilistic = probabilistic
        self.choice_history = {} # Used for cumulative KDE alignment
        self.environment = jinja2.Environment()

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

    def reset_history(self):
        self.choice_history = {}

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

    def sample_relevance_predictions(self,
                                     scenario_state,
                                     scenario_description,
                                     choices,
                                     target_kdmas,
                                     available_actions,
                                     batch_size=5,
                                     incontext_settings={}):
        '''
        Samples prediction of the relevance of each response to each KDMA
        '''
        scenario_state_copy = copy.deepcopy(scenario_state)
        # Don't consider the elapsed_time of the state when caching
        scenario_state_copy.elapsed_time = 0
        depends = '\n'.join((
            repr(self.model.model),
            repr(scenario_state_copy),
            repr(scenario_description),
            repr(choices),
            repr([t['kdma'] for t in target_kdmas]),
            repr(available_actions),
            repr(incontext_settings)))

        cacher = ub.Cacher('comp_reg_relevance', depends, verbose=0)
        log.debug(f'cacher.fpath={cacher.fpath}')

        cached_output = cacher.tryload()
        if cached_output is not None:
            log.info("Cache hit for `sample_relevance_predictions` returning cached output")
            return cached_output
        else:
            log.info("Cache miss for `sample_relevance_predictions` ..")

        use_icl = False
        if "number" in incontext_settings and incontext_settings["number"] > 0:
            use_icl = True
            icl_example_generator = incontext_utils.RelevanceIncontextExampleGenerator(incontext_settings,
                                                                                       target_kdmas)
        icl_example_responses = {}
        for target_kdma in target_kdmas:
            icl_example_responses[target_kdma['name']] = []
        relevance_dialogs = []
        # loop over target kdmas
        for target_kdma in target_kdmas:
            relevance_sys_prompt = relevance_classification_system_prompt(target_kdma['name'],
                                                            target_kdma['description'],
                                                            target_kdma['factor'])

            icl_examples = []
            if use_icl:
                prompt_to_match = relevance_classification_prompt(scenario_description,
                                                                choices,
                                                                target_kdma['name'])
                selected_icl_examples = icl_example_generator.select_icl_examples(
                    sys_kdma_name=target_kdma['kdma'],
                    scenario_description_to_match=scenario_description,
                    prompt_to_match=prompt_to_match,
                    state_comparison=scenario_state,
                    actions=available_actions
                )
                for icl_sample in selected_icl_examples:
                    icl_examples.extend([
                        {"role": "user", "content": icl_sample['prompt']},
                        {"role": "assistant", "content": f'{icl_sample["response"]}'}
                    ])
                    icl_example_responses[target_kdma['name']].append(icl_sample["response"])

            predict_relevance_prompt = relevance_classification_prompt(scenario_description,
                                                                    choices,
                                                                    target_kdma['name'])
            dialog = [{'role': 'system', 'content': relevance_sys_prompt}]
            dialog.extend(icl_examples)
            dialog.append({'role': 'user', 'content': predict_relevance_prompt})
            relevance_dialogs.append(dialog)

        # Need to set the whitespace_pattern to prevent the state
        # machine from looping indefinitely in some cases, see:
        # https://github.com/outlines-dev/outlines/issues/690#issuecomment-2102291934
        relevance_schema = relevance_classification_json_schema(choices, target_kdma['factor'])
        relevance_generator = outlines.generate.json(
            self.model,
            relevance_schema,
            sampler=self.sampler,
            whitespace_pattern=r"[ ]?")

        relevance_dialog_texts = [self.dialog_to_prompt(d) for d in relevance_dialogs]

        log.info("[bold]*KDMA SCORE PREDICTION DIALOG PROMPT*[/bold]",
                 extra={"markup": True})
        log.info(relevance_dialog_texts[0])

        # List of {choice: {score:int, reasoning:str}, ...} with length = num_samples*len(target_kdmas)
        relevance_score_responses = self.run_in_batches(relevance_generator, relevance_dialog_texts, batch_size)
        # Reshape to matrix of num_samples x len(target_kdmas)
        relevance_responses = [relevance_score_responses[i:i+len(target_kdmas)] \
                                for i in range(0,len(relevance_score_responses),len(target_kdmas))]

        log.info("[bold]*RELEVANCE PREDICTION RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(relevance_responses, extra={"highlighter": JSON_HIGHLIGHTER})

        # Initialize output dictionaries
        predictions = {}
        reasonings = {}
        for choice in choices:
            predictions[choice] = {}
            reasonings[choice] = {}

        # Add responses to output dictionaries
        for kdma_idx in range(len(target_kdmas)):
            rel_prediction = relevance_responses[0][kdma_idx]
            kdma_key = target_kdmas[kdma_idx]['kdma']
            for choice in choices:
                reasonings[choice][kdma_key] = rel_prediction[choice]['reasoning']
                if rel_prediction[choice]['relevant'] == 'yes':
                    predictions[choice][kdma_key] = 1
                else:
                    predictions[choice][kdma_key] = 0

        outputs = (predictions, reasonings, icl_example_responses)
        cacher.save(outputs)

        return outputs

    def sample_kdma_score_predictions(self,
                                      scenario_state,
                                      scenario_description,
                                      choices,
                                      target_kdmas,
                                      available_actions,
                                      outcome_predictions,
                                      num_samples=1,
                                      batch_size=6,
                                      kdma_score_examples=False,
                                      enum_scores=False,
                                      incontext_settings={}):
        '''
        Samples predictions of kdma scores associated with each choice
        Returns ADM score predictions and reasoning for each choice and KDMA
        - predictions: {choice1:{kdma1:[score1(int), ...], ...}, ...}
        - reasonings: {choice1:{kdma1:[reasoning1(str), ...], ...}, ...}
        '''
        scenario_state_copy = copy.deepcopy(scenario_state)
        # Don't consider the elapsed_time of the state when caching
        scenario_state_copy.elapsed_time = 0
        depends = '\n'.join((
            repr(self.model.model),
            repr(scenario_state_copy),
            repr(choices),
            repr(available_actions),
            repr(outcome_predictions),
            repr(kdma_score_examples),
            repr(enum_scores),
            repr(incontext_settings)))

        cacher = ub.Cacher('comp_reg_kdma_estimation', depends, verbose=0)
        log.debug(f'cacher.fpath={cacher.fpath}')

        cached_output = cacher.tryload()
        if cached_output is not None:
            log.info("Cache hit for `sample_kdma_score_predictions` returning cached output")
            return cached_output
        else:
            log.info("Cache miss for `sample_kdma_score_predictions` ..")

        use_icl = False
        if "number" in incontext_settings and incontext_settings["number"] > 0:
            use_icl = True
            icl_example_generator = incontext_utils.ComparativeRegressionIncontextExampleGenerator(incontext_settings,
                                                                                                   target_kdmas)

        icl_example_responses = {}
        for target_kdma in target_kdmas:
            icl_example_responses[target_kdma['kdma']] = []
        kdma_dialogs = []
        # loop over samples
        for sample_idx in range(num_samples):
            # loop over target kdmas
            for target_kdma in target_kdmas:
                if kdma_score_examples:
                    template = self.environment.from_string(target_kdma['score_examples'])
                    score_examples = template.render(kdma_scale_factor=target_kdma['factor'])
                    kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt_with_examples(target_kdma['name'],
                                                                                                          target_kdma['description'],
                                                                                                          score_examples,
                                                                                                          target_kdma['factor'])
                else:
                    kdma_score_sys_prompt = comparative_kdma_score_prediction_system_prompt(target_kdma['name'],
                                                                                            target_kdma['description'],
                                                                                            target_kdma['factor'])

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
                    selected_icl_examples = icl_example_generator.select_icl_examples(
                        sys_kdma_name=target_kdma['kdma'],
                        scenario_description_to_match=scenario_description,
                        prompt_to_match=prompt_to_match,
                        state_comparison=scenario_state,
                        actions=available_actions
                    )
                    for icl_sample in selected_icl_examples:
                        icl_examples.extend([
                            {"role": "user", "content": icl_sample['prompt']},
                            {"role": "assistant", "content": f'{icl_sample["response"]}'}
                        ])
                        icl_example_responses[target_kdma['kdma']].append(icl_sample["response"])


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

        if enum_scores:
            score_schema = enum_comparative_kdma_score_prediction_json_schema(choices, target_kdma['valid_scores'])
        else:
            score_schema = comparative_kdma_score_prediction_json_schema(choices, target_kdma['factor'])
        kdma_score_generator = outlines.generate.json(
            self.model,
            score_schema,
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
                kdma_factor = target_kdmas[kdma_idx]['factor']
                for choice in choices:
                    reasonings[choice][kdma_key].append(kdma_prediction[choice]['reasoning'])
                    # Scale score to be between 0 and 1 to match targets
                    predictions[choice][kdma_key].append(kdma_prediction[choice]['score'] / kdma_factor)

        outputs = (predictions, reasonings, icl_example_responses)
        cacher.save(outputs)

        return outputs

    # Returns the outcome prediction (if there was one) and score reasoning for the best sample of the selected choice
    def get_selected_choice_reasoning(self, selected_choice, best_sample_index, outcome_predictions, reasonings, relevance_reasonings=None):
        # If outcomes were predicted, add the best sample outcome prediction reasoning
        if outcome_predictions[best_sample_index][selected_choice]['predicted_outcome'] is not None:
            reasoning = f'{outcome_predictions[best_sample_index][selected_choice]["predicted_outcome"]} '
        else:
            reasoning = ''
        # Add the score prediction reasoning for each KDMA
        for target_kdma in list(reasonings[selected_choice].keys()):
            # If relevance was predicted add relevance reasoning
            if relevance_reasonings:
                reasoning += f'{relevance_reasonings[selected_choice][target_kdma]} '
            reasoning += f'{reasonings[selected_choice][target_kdma][best_sample_index]} '
        return reasoning


    def top_level_choose_action(self,
                                scenario_state,
                                available_actions,
                                alignment_target,
                                num_samples=1,
                                predict_outcomes=False,
                                predict_relevance=False,
                                distribution_matching='sample',
                                kde_norm='globalnorm',
                                generator_batch_size=5,
                                kdma_descriptions_map='align_system/prompt_engineering/kdma_descriptions.yml',
                                kdma_score_examples=False,
                                enum_scores=False,
                                priornorm_factor=0.5,
                                **kwargs):

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
                if "values" in kdma_descriptions[kdma]['valid_scores']:
                    kdma_descriptions[kdma]['valid_scores'] = kdma_descriptions[kdma]['valid_scores']["values"]
                elif "range" in kdma_descriptions[kdma]['valid_scores']:
                    r_params = kdma_descriptions[kdma]['valid_scores']['range']
                    kdma_descriptions[kdma]['valid_scores'] = list(range(
                        r_params['min'], r_params['max'] + r_params['step'], r_params['step']))
                else:
                    raise RuntimeError("Unknown valid scores option, expecting 'values' or 'range'")
                target_kdmas[kdma_idx].update(kdma_descriptions[kdma])

        relevant_fields = []
        for target_kdma in target_kdmas:
            relevant_fields.extend(target_kdma['relevant_structured_character_info'])
        if 'all_unique' in relevant_fields:
            character_info = outlines_prompts_utils.get_unique_structured_character_info(scenario_state.characters)
        else:
            character_info = outlines_prompts_utils.get_relevant_structured_character_info(scenario_state.characters, target_kdmas)
        scenario_description = scenario_state_description_with_relevant_char_info(scenario_state, character_info)

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

        # Predict relevance of each KDMA to each choice - optional
        if predict_relevance:
            predicted_relevance, relevance_reasonings, relevance_icl_responses = self.sample_relevance_predictions(
                scenario_state, scenario_description, choices, target_kdmas, available_actions,
                generator_batch_size, incontext_settings=kwargs.get("incontext", {})
            )

        # Predict kdma values
        predicted_kdma_values, reasonings, icl_example_responses = self.sample_kdma_score_predictions(
            scenario_state, scenario_description, choices, target_kdmas, available_actions, outcome_predictions,
            num_samples, generator_batch_size, kdma_score_examples, enum_scores,
            incontext_settings=kwargs.get("incontext", {})
        )

        # Log true kdma values if present
        true_kdma_values = {}
        for choice_idx in range(len(available_actions)):
            true_kdma_values[choices[choice_idx]] = available_actions[choice_idx].kdma_association
        log.info("True KDMA Values:")
        log.info(json.dumps(true_kdma_values))

        # Log predicted kdma values
        log.info("Predicted KDMA Values:")
        log.info(json.dumps(predicted_kdma_values))

        choice_info = {'true_kdma_values':true_kdma_values, 'predicted_kdma_values':predicted_kdma_values, 'icl_example_responses':icl_example_responses}

        if predict_relevance:
            # Log true relevance if present
            true_relevance = {}
            for choice_idx in range(len(available_actions)):
                true_relevance[choices[choice_idx]] = {
                    kdma['kdma']: 1 if available_actions[choice_idx].kdma_association is not None and kdma['kdma'] in available_actions[choice_idx].kdma_association else 0
                    for kdma in target_kdmas
                }
            log.info("True Relevance")
            log.info(json.dumps(true_relevance))
            choice_info['true_relevance'] = true_relevance

            # Log predicted relevance
            log.info("Predicted Relevance Values:")
            log.info(json.dumps(predicted_relevance))
            choice_info['predicted_relevance'] = predicted_relevance
            choice_info['relevance_icl_example_responses'] = relevance_icl_responses

        # Get type of targets
        all_scalar_targets = True
        all_kde_targets = True
        for target_kdma in target_kdmas:
            if 'value' not in target_kdma or target_kdma["value"] is None:
                all_scalar_targets = False
            if 'kdes' not in target_kdma or target_kdma["kdes"] is None:
                all_kde_targets = False

        # Use relevance in alignment function
        if predict_relevance:
            if all_scalar_targets:
                if distribution_matching == 'relevance_average':
                    alignment_function = alignment_utils.RelevanceAvgDistScalarAlignment()
                    selected_choice, probs = alignment_function(predicted_kdma_values, predicted_relevance,
                                                                target_kdmas, probabilistic=self.probabilistic)
                else:
                    raise RuntimeError(distribution_matching, "distribution matching function unimplemented for scalar targets with relevance.")
                best_sample_index = alignment_function.get_best_sample_index(predicted_kdma_values,
                                                                                target_kdmas, selected_choice)
            elif all_kde_targets:
                if distribution_matching == 'relevance_cumulative_kde':
                    alignment_function = alignment_utils.RelevanceCumulativeJsDivergenceKdeAlignment()
                    selected_choice, probs = alignment_function(predicted_kdma_values, predicted_relevance,
                                                                target_kdmas, self.choice_history,
                                                                kde_norm=kde_norm, priornorm_factor=priornorm_factor,
                                                                probabilistic=self.probabilistic)
                else:
                    raise RuntimeError(distribution_matching, "distribution matching function unimplemented for KDE targets with relevance.")
                best_sample_index = alignment_function.get_best_sample_index(predicted_kdma_values,
                                                                             target_kdmas, selected_choice,
                                                                             kde_norm=kde_norm)
            else:
                # TODO: Currently we assume all targets either have scalar values or KDES,
                #       Down the line, we should extend to handling multiple targets of mixed types
                raise ValueError("ADM does not currently support a mix of scalar and KDE targets with relevance.")

        # Align without relevance
        else:
            if all_scalar_targets:
                alignment_function = alignment_utils.AvgDistScalarAlignment()
                selected_choice, probs = alignment_function(predicted_kdma_values, target_kdmas, probabilistic=self.probabilistic)
                best_sample_index = alignment_function.get_best_sample_index(predicted_kdma_values, target_kdmas, selected_choice)
            elif all_kde_targets:
                if distribution_matching == 'cumulative_kde':
                    alignment_function = alignment_utils.CumulativeJsDivergenceKdeAlignment()
                    selected_choice, probs = alignment_function(
                        predicted_kdma_values, target_kdmas, self.choice_history, kde_norm=kde_norm,
                        priornorm_factor=priornorm_factor, probabilistic=self.probabilistic
                    )
                else:
                    if distribution_matching == 'sample':
                        alignment_function = alignment_utils.MinDistToRandomSampleKdeAlignment()
                    elif distribution_matching == 'max_likelihood':
                        alignment_function = alignment_utils.MaxLikelihoodKdeAlignment()
                    elif distribution_matching == 'js_divergence':
                        alignment_function = alignment_utils.JsDivergenceKdeAlignment()
                    else:
                        raise RuntimeError(distribution_matching, "distribution matching function unrecognized.")
                    selected_choice, probs = alignment_function(
                        predicted_kdma_values, target_kdmas, kde_norm=kde_norm, probabilistic=self.probabilistic
                    )
                best_sample_index = alignment_function.get_best_sample_index(
                    predicted_kdma_values, target_kdmas, selected_choice, kde_norm=kde_norm
                )
            else:
                # TODO: Currently we assume all targets either have scalar values or KDES,
                #       Down the line, we should extend to handling multiple targets of mixed types
                raise ValueError("ADM does not currently support a mix of scalar and KDE targets.")

        # Update chocie history
        for target_kdma in target_kdmas:
            kdma = target_kdma['kdma']
            if kdma not in self.choice_history:
                self.choice_history[kdma] = []
            self.choice_history[kdma].append(np.mean(predicted_kdma_values[selected_choice][kdma]))

        log.info("[bold]*STRUCTURED RESPONSE*[/bold]",
                 extra={"markup": True})
        log.info(selected_choice, extra={"highlighter": JSON_HIGHLIGHTER})

        selected_choice_idx = choices.index(selected_choice)
        action_to_take = available_actions[selected_choice_idx]
        if predict_relevance:
            action_to_take.justification = self.get_selected_choice_reasoning(selected_choice, best_sample_index,
                                                                              outcome_predictions, reasonings,
                                                                              relevance_reasonings)
        else:
            action_to_take.justification = self.get_selected_choice_reasoning(selected_choice, best_sample_index,
                                                                              outcome_predictions, reasonings)

        # Set up simple diaolg to return for follow-ups
        alignment_system_prompt = baseline_system_prompt()
        prompt = action_selection_prompt(scenario_description, choices)
        dialog = [{'role': 'system', 'content': alignment_system_prompt},
                  {'role': 'user', 'content': prompt}]

        return action_to_take, dialog, choice_info


    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        action_to_take, dialog, choice_info = self.top_level_choose_action(
            scenario_state,
            available_actions,
            alignment_target,
            **kwargs)

        action_to_take, dialog = self.populate_action_parameters(
            scenario_state,
            action_to_take,
            dialog)

        return action_to_take, choice_info
