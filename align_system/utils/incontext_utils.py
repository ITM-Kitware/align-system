import json
from jsonschema import validate
import torch
import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import namedtuple

from align_system.utils import adm_utils
from align_system.utils import outlines_prompts_utils
from align_system.utils import alignment_utils
from align_system.prompt_engineering.outlines_prompts import (
    action_choice_json_schema,
    scenario_state_description_1,
    action_selection_prompt,
    scenario_state_description_with_relevant_char_info,
    comparative_kdma_score_prediction_prompt,
    comparative_kdma_score_prediction_json_schema,
    relevance_classification_prompt,
    relevance_classification_json_schema,
    phase2_scenario_state_description,
    comparative_regression_prompt,
    comparative_regression_json_schema
)


ICLSelectionResult = namedtuple('ICLSelectionResult', ['candidates', 'scores', 'indices'])


def add_similarity_scores(examples, scores):
    """Add similarity scores to ICL examples using list comprehension"""
    return [
        {**ex.copy(), 'similarity_score': float(score) if score is not None else None}
        for ex, score in zip(examples, scores)
    ]


def select_random_strategy(possible_examples, n_examples, **kwargs):
    """Random selection strategy for ICL examples"""
    selected_samples = random.sample(possible_examples, n_examples)
    scores = [None] * len(selected_samples)
    indices = list(range(len(selected_samples)))
    return ICLSelectionResult(selected_samples, scores, indices)


def select_scenario_bert_similarity_strategy(possible_examples, n_examples, scenario_to_match, least_similar, **kwargs):
    """Scenario-based BERT similarity selection strategy"""
    scenario_description_set = set()
    final_candidates = []
    for icl_sample in possible_examples:
        if icl_sample["scenario_description"] not in scenario_description_set:
            final_candidates.append(icl_sample)
            scenario_description_set.add(icl_sample["scenario_description"])
    
    possible_scenarios = [icl_sample["scenario_description"] for icl_sample in final_candidates]
    
    from bert_score import score
    _, _, F1 = score([scenario_to_match] * len(possible_scenarios), possible_scenarios, lang="en")
    _, indices = torch.topk(F1, n_examples, largest=(not least_similar))
    
    selected_candidates = [final_candidates[i] for i in indices]
    selected_scores = [F1[i] for i in indices]
    
    return ICLSelectionResult(selected_candidates, selected_scores, indices)


def select_prompt_bert_similarity_strategy(possible_examples, n_examples, prompt_to_match, least_similar, **kwargs):
    """Prompt-based BERT similarity selection strategy"""
    prompt_set = set()
    final_candidates = []
    for icl_sample in possible_examples:
        if icl_sample["prompt"] not in prompt_set:
            final_candidates.append(icl_sample)
            prompt_set.add(icl_sample["prompt"])
    
    possible_prompts = [icl_sample["prompt"] for icl_sample in final_candidates]
    
    from bert_score import score
    _, _, F1 = score([prompt_to_match] * len(possible_prompts), possible_prompts, lang="en")
    _, indices = torch.topk(F1, n_examples, largest=(not least_similar))
    
    selected_candidates = [final_candidates[i] for i in indices]
    selected_scores = [F1[i] for i in indices]
    
    return ICLSelectionResult(selected_candidates, selected_scores, indices)


def select_matching_actions_strategy(possible_examples, n_examples, prompt_to_match, actions, least_similar, **kwargs):
    """Action-matching with BERT similarity selection strategy"""
    action_types = set([action.action_type for action in actions])
    possible_prompts = [icl_sample["prompt"] for icl_sample in possible_examples]
    possible_actions = [set([action.action_type for action in icl_sample['actions']]) for icl_sample in possible_examples]
    
    from bert_score import score
    _, _, scores = score([prompt_to_match] * len(possible_prompts), possible_prompts, lang="en")
    
    for i in range(len(scores)):
        if action_types.issubset(possible_actions[i]):
            scores[i] += 1
    
    _, indices = torch.topk(scores, n_examples, largest=(not least_similar))
    
    selected_candidates = [possible_examples[i] for i in indices]
    selected_scores = [scores[i] for i in indices]
    
    return ICLSelectionResult(selected_candidates, selected_scores, indices)


def select_matching_characters_strategy(possible_examples, n_examples, prompt_to_match, actions, least_similar, **kwargs):
    """Character-matching with BERT similarity selection strategy"""
    action_chars = set([action.character_id for action in actions])
    possible_prompts = [icl_sample["prompt"] for icl_sample in possible_examples]
    possible_chars = [set([action.character_id for action in icl_sample['actions']]) for icl_sample in possible_examples]
    
    from bert_score import score
    _, _, scores = score([prompt_to_match] * len(possible_prompts), possible_prompts, lang="en")
    
    for i in range(len(scores)):
        if action_chars.issubset(possible_chars[i]):
            scores[i] += 1
    
    _, indices = torch.topk(scores, n_examples, largest=(not least_similar))
    
    selected_candidates = [possible_examples[i] for i in indices]
    selected_scores = [scores[i] for i in indices]
    
    return ICLSelectionResult(selected_candidates, selected_scores, indices)


ICL_SELECTION_STRATEGIES = {
    'random': select_random_strategy,
    'scenario_bert_similarity': select_scenario_bert_similarity_strategy,
    'prompt_bert_similarity': select_prompt_bert_similarity_strategy,
    'matching_actions': select_matching_actions_strategy,
    'matching_characters': select_matching_characters_strategy
}


class IncontextExampleGenerator(object, metaclass=ABCMeta):
    '''
    Abstract class for incontext example generator
    Instances of this class have unique set_icl_datasets() functions for formatting prompt and reponses
    '''
    def __init__(
        self,
        incontext_settings,
        target_kdmas,
        state_hydration_domain=None,
    ):
        self.incontext_settings = incontext_settings
        self.target_kdmas = []
        for target_kdma in target_kdmas:
            if not isinstance(target_kdma, dict):
                self.target_kdmas.append(dict(target_kdma))
            else:
                self.target_kdmas.append(target_kdma)

        # TODO: Replace this logic with a callable to be instantiated via Hydra
        if state_hydration_domain is None or state_hydration_domain == "p1":
            from align_system.utils.hydrate_state import hydrate_scenario_state
            self.state_hydration_fn = hydrate_scenario_state
        elif state_hydration_domain == "p2triage":
            from align_system.utils.hydrate_state import p2triage_hydrate_scenario_state
            self.state_hydration_fn = p2triage_hydrate_scenario_state
        else:
            raise RuntimeError(f"Unknown state_hydration_domain: {state_hydration_domain}")

        self.set_icl_datasets()

    @abstractmethod
    def set_icl_datasets(self):
        '''
        Sets self.icl_datasets which contains all the ICL examples
        This is specific to each instance of the class because the prompt and response will vary
        _read_icl_dataset_files() is a generic helper method for this step

        The keys of self.icl_datasets are the 'kdma' keys from self.target_kdmas,
        the values are a list of all ICL examples for that kdma,
        each ICL example is a dictionary with keys: 'scenario_description', 'prompt', 'response'
        For example: {kdma: [{'scenario_description':str, 'prompt':str, 'response':json}, ...], ...}
        '''
        self.icl_datasets = {}
        pass

    def _read_icl_dataset_files(self):
        '''
        Helper function for set_icl_datasets() - reads dataset files and gets examples for target_kdmas
        Returns incontext_data dictionary with format:
            {kdma:[{state, actions, choices, kdma_values}, ...], ...}
        '''
        incontext_data = {}
        # For each kdma
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            # Check if we have dataset files for the target KDMA
            if sys_kdma_name not in self.incontext_settings["datasets"]:
                raise RuntimeError(f"No incontext datasets are provided for targeted kdma: {sys_kdma_name}")
            # Add examples for each dataset file
            dset_files = self.incontext_settings["datasets"][sys_kdma_name]
            # If there is only one, make it a list for the following loop
            if not isinstance(dset_files, list):
                dset_files = [dset_files]

            incontext_data[sys_kdma_name] = []
            # For each dataset file
            for dset_f in dset_files:
                with open(dset_f) as f:
                    dset = json.load(f)
                # Load each example in the dataset file
                for icl_sample in dset:
                    # Get state and actions
                    state, actions = self.state_hydration_fn(icl_sample["input"])
                    labels = icl_sample["label"]
                    if self.incontext_settings.sort_actions:
                        # Impose a fixed ordering of available actions and labels to help with determinism
                        combined = list(zip(actions, labels))
                        combined_sorted = sorted(combined, key=lambda x: x[0].unstructured)
                        actions, labels = zip(*combined_sorted)
                    # Get choices
                    choices = adm_utils.format_choices(
                        [a.unstructured for a in actions],
                        actions,
                        state
                    )
                    # Get KDMA_values
                    kdma_values = []
                    for label in labels:
                        if sys_kdma_name not in label:
                            kdma_values.append(None)
                        else:
                            kdma_values.append(label[sys_kdma_name])
                    example = {'state':state, 'actions': actions, 'choices':choices, 'kdma_values':kdma_values}
                    incontext_data[sys_kdma_name].append(example)

            # Normalize ground truth KDMA values
            if 'normalization' in self.incontext_settings:
                if self.incontext_settings['normalization'] is not None and self.incontext_settings['normalization'] != 'rawscores':
                    if self.incontext_settings['normalization'] == 'globalnorm':
                        incontext_data = self._global_normalization(incontext_data)
                    elif self.incontext_settings['normalization'] == 'localnorm':
                        incontext_data = self._local_normalization(incontext_data)
                    else:
                        raise ValueError(f'{self.incontext_settings["normalization"]} is not a valid incontext normalization option. '
                                        'Please use "globalnorm" or "localnorm".')

        return incontext_data

    def _read_relevance_icl_dataset_files(self):
        '''
        Helper function for set_icl_datasets() - reads dataset files and gets examples for target_kdmas
        Returns incontext_data dictionary with format:
            {kdma:[{state, actions, choices, kdma_values}, ...], ...}
        Unlike _read_icl_dataset_files(), this function includes all dset_files for all kdmas so
        that examples contain thte case where a KDMA is irrelevant to all responses.
        '''
        dset_files = []
        # For each kdma
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            # Check if we have dataset files for the target KDMA
            if sys_kdma_name not in self.incontext_settings["datasets"]:
                raise RuntimeError(f"No incontext datasets are provided for targeted kdma: {sys_kdma_name}")
            # Add examples for each dataset file
            kdma_dset_files = self.incontext_settings["datasets"][sys_kdma_name]
            # Add to list
            if isinstance(kdma_dset_files, list):
                dset_files.extend(kdma_dset_files)
            else:
                dset_files.append(kdma_dset_files)
        # remove potential duplicates
        dset_files = list(set(dset_files))

        incontext_data = {}
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            incontext_data[sys_kdma_name] = []
            # For each dataset file
            for dset_f in dset_files:
                with open(dset_f) as f:
                    dset = json.load(f)
                # Load each example in the dataset file
                for icl_sample in dset:
                    # Get state and actions
                    state, actions = self.state_hydration_fn(icl_sample["input"])
                    labels = icl_sample["label"]
                    if self.incontext_settings.sort_actions:
                        # Impose a fixed ordering of available actions and labels to help with determinism
                        combined = list(zip(actions, labels))
                        combined_sorted = sorted(combined, key=lambda x: x[0].unstructured)
                        actions, labels = zip(*combined_sorted)
                    # Get choices
                    choices = adm_utils.format_choices(
                        [a.unstructured for a in actions],
                        actions,
                        state
                    )
                    # Get KDMA_values
                    kdma_values = []
                    for label in labels:
                        if sys_kdma_name not in label:
                            kdma_values.append(None)
                        else:
                            kdma_values.append(label[sys_kdma_name])

                    example = {'state':state, 'actions': actions, 'choices':choices, 'kdma_values':kdma_values}
                    incontext_data[sys_kdma_name].append(example)

            # Normalize ground truth KDMA values
            if 'normalization' in self.incontext_settings:
                if self.incontext_settings['normalization'] is not None and self.incontext_settings['normalization'] != 'rawscores':
                    if self.incontext_settings['normalization'] == 'globalnorm':
                        incontext_data = self._global_normalization(incontext_data)
                    elif self.incontext_settings['normalization'] == 'localnorm':
                        incontext_data = self._local_normalization(incontext_data)
                    else:
                        raise ValueError(f'{self.incontext_settings["normalization"]} is not a valid incontext normalization option. '
                                        'Please use "globalnorm" or "localnorm".')

        return incontext_data

    def _global_normalization(self, incontext_data):
        for kdma in list(incontext_data.keys()):
            # Get global min and max
            all_kdma_values = []
            for example in incontext_data[kdma]:
                all_kdma_values.extend(example['kdma_values'])
            all_kdma_values = [i for i in all_kdma_values if i is not None]
            global_min = min(all_kdma_values)
            global_max = max(all_kdma_values)
            # Normalize
            for example_idx in range(len(incontext_data[kdma])):
                norm_values = incontext_data[kdma][example_idx]['kdma_values']
                for value_idx in range(len(norm_values)):
                    if norm_values[value_idx] is not None:
                        norm_values[value_idx] = (norm_values[value_idx] - global_min) / (global_max - global_min)
                incontext_data[kdma][example_idx]['kdma_values'] = norm_values
        return incontext_data

    def _local_normalization(self, incontext_data):
        # Normalize per example
        for kdma in list(incontext_data.keys()):
            for example_idx in range(len(incontext_data[kdma])):
                norm_values = incontext_data[kdma][example_idx]['kdma_values']
                example_values = [i for i in norm_values if i is not None]
                local_min = np.min(example_values)
                local_max = np.max(example_values)
                for value_idx in range(len(norm_values)):
                    if norm_values[value_idx] is not None:
                        norm_values[value_idx] = (norm_values[value_idx] - local_min) / (local_max - local_min)
                incontext_data[kdma][example_idx]['kdma_values'] = norm_values
        return incontext_data

    def select_icl_examples(self, sys_kdma_name, scenario_description_to_match, prompt_to_match, state_comparison, actions):
        '''
        Selects a list of relevant ICL examples
        Input:
            sys_kdma_name - key of the target kdma in self.icl_datasets
            scenario_description_to_match - description of the scenario for similarity and/or LOO
            prompt_to_match - the prompt we are selecting ICL examples for
            state_comparison - the current state of the system to potentially use for LOO

        Output:
            selected_icl_examples - relevant subset of self.icl_datasets
        '''
        # Check that we have incontext examples for the target kdma
        if sys_kdma_name not in self.icl_datasets:
            raise RuntimeError(f"No incontext samples for targeted kdma: {sys_kdma_name}")
        n_icl_examples = self.incontext_settings["number"]
        possible_icl_examples = self.icl_datasets[sys_kdma_name]
        # Check that we have enough incontext examples for the target kdma
        if len(possible_icl_examples) < n_icl_examples:
            raise RuntimeError(f"Not enough possible incontext samples to learn from. Only "
                            f"{len(possible_icl_examples)} samples available while asking for "
                            f"{n_icl_examples} incontext samples.")
        # If using LOO, don't include example ICL with exact same scenario description
        loo_strategy = self.incontext_settings.get("leave_one_out_strategy", None)
        if loo_strategy == "scenario_description":
            possible_icl_examples = [
                icl_ex for icl_ex in possible_icl_examples
                if icl_ex["scenario_description"] != scenario_description_to_match
            ]
        elif loo_strategy == "characters":
            possible_icl_examples = [
                icl_ex for icl_ex in possible_icl_examples
                if icl_ex["state"].characters != state_comparison.characters
            ]
        elif loo_strategy is not None:
            raise ValueError(
                f"Unknown leave one out setting '{loo_strategy}'."
                "Please choose from 'scenario_description' or 'characters'"
            )

        # Downselect to n_icl_examples via given method
        icl_strategy = self.incontext_settings["method"]
        least_similar_examples = self.incontext_settings.get("least_similar_examples", False)
        
        if icl_strategy not in ICL_SELECTION_STRATEGIES:
            raise ValueError(f'"{icl_strategy}" is not a valid incontext method. Available strategies: '
                           f'{", ".join(ICL_SELECTION_STRATEGIES.keys())}')
        
        strategy_fn = ICL_SELECTION_STRATEGIES[icl_strategy]
        result = strategy_fn(
            possible_examples=possible_icl_examples,
            n_examples=n_icl_examples,
            scenario_to_match=scenario_description_to_match,
            prompt_to_match=prompt_to_match,
            actions=actions,
            least_similar=least_similar_examples
        )
        
        # Reverse indices if selecting least similar examples
        indices = reversed(result.indices) if least_similar_examples else result.indices
        
        # Reindex candidates and scores based on final indices
        candidates = [result.candidates[i] for i in indices]
        scores = [result.scores[i] for i in indices]
        
        selected_icl_examples = add_similarity_scores(candidates, scores)
        
        if self.incontext_settings.get("most_similar_first", True):
            return selected_icl_examples
        else:
            return list(reversed(selected_icl_examples))


class BaselineIncontextExampleGenerator(IncontextExampleGenerator):
    '''
    For OutlinesTransformersADM in align_system.algorithms.outlines_adm
    Given SCALAR targets - select the closest choice as correct
    Provides a simple binary (either high or low) example reasoning statement
    '''
    def set_icl_datasets(self):
        # Requires scalar targets
        alignment_utils._check_if_targets_are_scalar(self.target_kdmas)

        icl_datasets = {}
        incontext_data = self._read_icl_dataset_files()

        # Add each target to icl_datasets
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            icl_datasets[sys_kdma_name] = []
            kdma_incontext_data = incontext_data[sys_kdma_name]

            # Add each examples to icl_datasets
            for example in kdma_incontext_data:

                # Get scenario and prompt
                icl_scenario_description = scenario_state_description_1(example['state'])
                icl_prompt = action_selection_prompt(icl_scenario_description, example['choices'])

                # Get example response
                dist_to_tgt = [
                    abs(score - target_kdma['value']) if score is not None else float('inf')
                    for score in example["kdma_values"]
                ]
                correct_answer_idx = np.argmin(dist_to_tgt)
                correct_choice = example['choices'][correct_answer_idx]
                adjective = "low" if target_kdma['value'] < 0.5 else "high"
                reasoning = f"Per the principle of {adjective} {target_kdma['name']}, " \
                            f'\\"{correct_choice}\\" is the correct answer.'
                icl_response = {"detailed_reasoning": reasoning,
                                "action_choice": correct_choice}
                # Validate response against schema
                correct_schema = json.loads(action_choice_json_schema(json.dumps(example['choices'])))
                validate(instance=icl_response, schema=correct_schema)

                # Add example
                icl_datasets[sys_kdma_name].append({
                    "state": example["state"],
                    "scenario_description": icl_scenario_description,
                    "prompt": icl_prompt,
                    "response": icl_response
                    })

        self.icl_datasets = icl_datasets


class ComparativeRegressionIncontextExampleGenerator(IncontextExampleGenerator):
    def set_icl_datasets(self):
        icl_datasets = {}
        incontext_data = self._read_icl_dataset_files()

        # Add each target to icl_datasets
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            icl_datasets[sys_kdma_name] = []
            kdma_incontext_data = incontext_data[sys_kdma_name]

            # Add each examples to icl_datasets
            for example in kdma_incontext_data:

                # Get example response
                icl_response = {}
                included_choices = []
                for action, choice, kdma_value in zip(example['actions'], example['choices'], example["kdma_values"]):
                    # Only include choice if there is a ground truth KDMA value available
                    if kdma_value is None:
                        continue
                    # Groundtruth KDMA values are 0-1, but ADM may predict on a different scale
                    scaled_kdma_value = int(kdma_value * target_kdma["factor"])
                    icl_response[choice] = {}
                    icl_response[choice]['reasoning'] = self.get_chain_of_thought_reasoning(target_kdma, action,
                                                                                            example['state'], choice,
                                                                                            scaled_kdma_value)
                    icl_response[choice]['score'] = scaled_kdma_value
                    included_choices.append(choice)
                # Check if response is valid against json schema
                correct_schema = json.loads(comparative_kdma_score_prediction_json_schema(included_choices, target_kdma["factor"]))
                validate(instance=icl_response, schema=correct_schema)

                # Get example prompt
                relevant_fields = []
                for char_target_kdma in self.target_kdmas:
                    relevant_fields.extend(char_target_kdma['relevant_structured_character_info'])
                if 'all_unique' in relevant_fields:
                    character_info = outlines_prompts_utils.get_unique_structured_character_info(example['state'].characters)
                else:
                    character_info = outlines_prompts_utils.get_relevant_structured_character_info(example['state'].characters, self.target_kdmas)
                icl_scenario_description = scenario_state_description_with_relevant_char_info(example['state'], character_info)
                # Only include choices in the prompt if they are in the response
                included_icl_choices_with_outcomes = {}
                for choice in included_choices:
                    # TODO: Include outcome prediction for ICL examples?
                    included_icl_choices_with_outcomes[choice] = {'predicted_outcome':None}
                icl_prompt = comparative_kdma_score_prediction_prompt(icl_scenario_description,
                                                                    included_icl_choices_with_outcomes,
                                                                    target_kdma['name'])

                # Add example
                icl_datasets[sys_kdma_name].append({
                    "state": example["state"],
                    "scenario_description": icl_scenario_description,
                    "prompt": icl_prompt,
                    "response": icl_response,
                    "actions": example['actions']
                    })

        self.icl_datasets = icl_datasets

    def get_chain_of_thought_reasoning(self, target_kdma, action, state, choice, expected_value):
        '''
        Helper function for set_icl_datasets() - constructs example reasoning statements for responses
        '''
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
                        cot_reasoning += " and did not cause of harm"
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
                        cot_reasoning += " but it is not the best treatment approach"
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
                        cot_reasoning += " but it is not the best treatment approach"
                # Worse quality of life patient
                elif expected_value < 5:
                    cot_reasoning += f"Because {action_character.name} would likely not survive treatment"
                    # Sub-optimal treatment
                    if expected_value == 3:
                        cot_reasoning += " but the supplies needed to treat the other casualty would still be available"

        # if a "because" statement has not been added
        if cot_reasoning == '':
            cot_reasoning = "Choosing "
        # else the reasoning will have form "Because ..., choosing ..."
        else:
            cot_reasoning += ", choosing "
        cot_reasoning += f"{choice} would score a {expected_value} for the the principle of {target_kdma['name']}."

        return cot_reasoning


class RelevanceIncontextExampleGenerator(IncontextExampleGenerator):
    def set_icl_datasets(self):
        icl_datasets = {}
        incontext_data = self._read_relevance_icl_dataset_files()

        # Add each target to icl_datasets
        for target_kdma in self.target_kdmas:
            icl_datasets[target_kdma['kdma']] = []
            kdma_incontext_data = incontext_data[target_kdma['kdma']]

            # Add each examples to icl_datasets
            for example in kdma_incontext_data:
                # Get example response
                icl_response = {}
                included_choices = []
                for action, choice, kdma_value in zip(example['actions'], example['choices'], example["kdma_values"]):
                    icl_response[choice] = {}
                    if kdma_value is not None:
                        icl_response[choice]['reasoning'] = self.get_relevant_chain_of_thought_reasoning(target_kdma, choice)
                        icl_response[choice]['relevant'] = 'yes'
                    else:
                        # if action has character, use character-based irrelevant COT reasoning
                        if hasattr(action, 'character_id') and action.character_id is not None:
                            icl_response[choice]['reasoning'] = self.get_irrelevant_chain_of_thought_reasoning(target_kdma, choice)
                        # else add generic COT reasoning
                        else:
                            icl_response[choice]['reasoning'] = f'Selecting this response does not require considering {target_kdma["name"]}.'
                        icl_response[choice]['relevant'] = 'no'
                    included_choices.append(choice)
                # Check if response is valid against json schema
                correct_schema = json.loads(relevance_classification_json_schema(included_choices, target_kdma["factor"]))
                validate(instance=icl_response, schema=correct_schema)

                # Get example prompt
                relevant_fields = []
                for char_target_kdma in self.target_kdmas:
                    relevant_fields.extend(char_target_kdma['relevant_structured_character_info'])
                if 'all_unique' in relevant_fields:
                    character_info = outlines_prompts_utils.get_unique_structured_character_info(example['state'].characters)
                else:
                    character_info = outlines_prompts_utils.get_relevant_structured_character_info(example['state'].characters, self.target_kdmas)
                icl_scenario_description = scenario_state_description_with_relevant_char_info(example['state'], character_info)
                # Only include choices in the prompt if they are in the response
                included_icl_choices_with_outcomes = {}
                for choice in included_choices:
                    # TODO: Include outcome prediction for ICL examples?
                    included_icl_choices_with_outcomes[choice] = {'predicted_outcome':None}
                icl_prompt = relevance_classification_prompt(icl_scenario_description,
                                                                    included_icl_choices_with_outcomes,
                                                                    target_kdma['name'])
                # Add example
                icl_datasets[target_kdma['kdma']].append({
                    "state": example["state"],
                    "scenario_description": icl_scenario_description,
                    "prompt": icl_prompt,
                    "response": icl_response,
                    "actions": example['actions']
                    })

        self.icl_datasets = icl_datasets

    def get_relevant_chain_of_thought_reasoning(self, target_kdma, choice):
        '''
        Helper function for set_icl_datasets() - constructs example reasoning statements for relevant responses
        '''
        # Moral deserivngess or moral judgement
        if target_kdma['kdma'] == 'MoralDesert' or  target_kdma['kdma'] == 'Moral judgement':
            cot_reasoning = f'The response is relevant because the patients are not equally morally deserving, so moral judgment is a consideration in choosing {choice}'

        # Ingroup bias
        elif target_kdma['kdma'] == 'Ingroup Bias':
            cot_reasoning = f'The response is relevant because the patients are not in the same group, so ingroup bias is a consideration in choosing {choice}'

        # QoL
        elif target_kdma['kdma'] == 'QualityOfLife':
            cot_reasoning = f'The response is relevant because the patients will not have the same quality of life post treatment, so quality of life is a consideration in choosing {choice}'

        # VoL
        elif target_kdma['kdma'] == 'PerceivedQuantityOfLivesSaved':
            cot_reasoning = f'The response is relevant because the patients do not have the same likelihood of survival, so perceived quantity of lives saved is a consideration in choosing {choice}'

        else:
            raise RuntimeError(f"Relevance ICL is not implemented for {target_kdma['kdma']}")

        return cot_reasoning

    def get_irrelevant_chain_of_thought_reasoning(self, target_kdma, choice):
        '''
        Helper function for set_icl_datasets() - constructs example reasoning statements for irrelevant responses
        '''
        # Moral deserivngess or moral judgement
        if target_kdma['kdma'] == 'MoralDesert' or  target_kdma['kdma'] == 'Moral judgement':
            cot_reasoning = f'The response is irrelevant because the patients are equally morally deserving, so moral judgment is not a consideration in choosing {choice}'

        # Ingroup bias
        elif target_kdma['kdma'] == 'Ingroup Bias':
            cot_reasoning = f'The response is irrelevant because the patients are in the same group, so ingroup bias is a not consideration in choosing {choice}'

        # QoL
        elif target_kdma['kdma'] == 'QualityOfLife':
            cot_reasoning = f'The response is irrelevant because the patients will have the same quality of life post-treatment, so quality of life is a not consideration in choosing {choice}'

        # VoL
        elif target_kdma['kdma'] == 'PerceivedQuantityOfLivesSaved':
            cot_reasoning = f'The response is irrelevant because the patients have similar likelihood of survival, so perceived quantity of lives saved is not a consideration in choosing {choice}'

        else:
            raise RuntimeError(f"Relevance ICL is not implemented for {target_kdma['kdma']}")

        return cot_reasoning


class Phase2ComparativeRegressionIncontextExampleGenerator(IncontextExampleGenerator):
    def set_icl_datasets(self):
        icl_datasets = {}
        incontext_data = self._read_icl_dataset_files()

        # Add each target to icl_datasets
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            icl_datasets[sys_kdma_name] = []
            kdma_incontext_data = incontext_data[sys_kdma_name]

            # Add each examples to icl_datasets
            for example in kdma_incontext_data:

                # Get example response
                icl_response = {}
                included_choices = []
                for action, choice, kdma_value in zip(example['actions'], example['choices'], example["kdma_values"]):
                    # Only include choice if there is a ground truth KDMA value available
                    if kdma_value is None:
                        continue
                    # Groundtruth KDMA values are 0-1, but ADM may predict on a different scale
                    scaled_kdma_value = int(kdma_value * target_kdma["factor"])
                    icl_response[choice] = {}
                    icl_response[choice]['score'] = scaled_kdma_value
                    included_choices.append(choice)
                icl_response_with_reasoning={}
                icl_response_with_reasoning['reasoning'] = self.get_chain_of_thought_reasoning(target_kdma, icl_response)
                icl_response_with_reasoning.update(icl_response) # reasoning first
                # Check if response is valid against json schema
                correct_schema = json.loads(comparative_regression_json_schema(included_choices, target_kdma["factor"]))
                validate(instance=icl_response_with_reasoning, schema=correct_schema)

                # Get example prompt
                icl_scenario_description = phase2_scenario_state_description(example['state'])
                # Only include choices in the prompt if they are in the response
                included_icl_choices_with_outcomes = {}
                for choice in included_choices:
                    # TODO: Include outcome prediction for ICL examples?
                    included_icl_choices_with_outcomes[choice] = {'predicted_outcome':None}
                icl_prompt = comparative_regression_prompt(icl_scenario_description,
                                                           included_icl_choices_with_outcomes,
                                                           target_kdma['name'])

                # Add example
                icl_datasets[sys_kdma_name].append({
                    "state": example["state"],
                    "scenario_description": icl_scenario_description,
                    "prompt": icl_prompt,
                    "response": icl_response_with_reasoning,
                    "actions": example['actions']
                    })

        self.icl_datasets = icl_datasets

    def get_chain_of_thought_reasoning(self, target_kdma, scores):
        '''
        Helper function for set_icl_datasets() - constructs example reasoning statements for responses
        Assumes only two choices
        '''
        choices = list(scores.keys())
        if scores[choices[0]]['score'] >= scores[choices[1]]['score']:
            max_choice = choices[0]
            min_choice = choices[1]
        else:
            max_choice = choices[1]
            min_choice = choices[0]

        diff = abs(scores[choices[0]]['score'] - scores[choices[1]]['score'])
        adjective = ''
        if diff >= 75:
            adjective = 'much'
        elif diff <= 25:
            adjective = 'slightly'

        cot_reasoning = f"{max_choice} demonstates {adjective} more {target_kdma['name']} than {min_choice}."
        return cot_reasoning
