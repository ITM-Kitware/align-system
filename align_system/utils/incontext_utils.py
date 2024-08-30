import json
import torch
import random
from copy import deepcopy
from abc import ABCMeta, abstractmethod


from align_system.utils import adm_utils
from align_system.utils import outlines_prompts_utils
from align_system.utils.hydrate_state import hydrate_scenario_state
from align_system.prompt_engineering.outlines_prompts import (
    scenario_state_description_with_relevant_char_info,
    comparative_kdma_score_prediction_prompt
)


class IncontextExampleGenerator(object, metaclass=ABCMeta):
    '''
    Abstract class for incontext example generator
    '''
    @abstractmethod
    def __init__(self,
                 incontext_settings, 
                 target_kdmas, 
                 log,
                 **kwargs):
        self.incontext_settings = incontext_settings
        self.target_kdmas = target_kdmas
        self.log = log
        self.set_icl_datasets()

    @abstractmethod
    def set_icl_datasets(self):
        '''
        Sets self.icl_datasets which contains all the ICL examples
        This is specific to each instance of the class because the prompt and response will vary
        read_icl_dataset_files() is a generic helper method for this step

        The keys of self.icl_datasets are the 'kdma' keys from self.target_kdmas,
        the values are a list of all ICL examples for that kdma,
        each ICL example is a dictionary with keys: 'scenario_description', 'prompt', 'response'
        For example: {kdma: [{'scenario_description':str, 'prompt':str, 'response':json}, ...], ...}
        '''
        self.icl_datasets = {}
        pass

    @abstractmethod
    def select_icl_examples(self):
        '''
        Returns a subset of self.icl_datasets that are relevant examples
        '''
        pass

    def read_icl_dataset_files(self):
        '''
        Helper function, reads dataset files and gets examples for target_kdmas
        Return incontext_data with format:
            {kdma:[{state, actions, choices, kdma_values}, ...], ...}
        '''
        incontext_data = {}
        # For each kdma
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            # Add examples for each dataset file
            dset_files = self.incontext_settings["datasets"][sys_kdma_name]
            # If there is only one, make it a list for the following loop
            if not isinstance(dset_files, list):
                dset_files = [dset_files]
            
            for dset_f in dset_files:
                # Load dataset file
                with open(dset_f) as f:
                    dset = json.load(f)
                incontext_data[sys_kdma_name] = []
                # Load each example in the dataset file
                for icl_sample in dset:
                    # Get state
                    state, actions = hydrate_scenario_state(icl_sample["input"])
                    # Get choices
                    choices = adm_utils.format_choices(
                        [a.unstructured for a in actions],
                        actions,
                        state,
                        self.log
                    )
                    # Get KDMA_values
                    kdma_values = []
                    for label in icl_sample["label"]:
                        if sys_kdma_name not in label:
                            kdma_values.append(None)
                        else:
                            kdma_values.append(label[sys_kdma_name])
                    example = {'state':state, 'actions': actions, 'choices':choices, 'kdma_values':kdma_values}
                    incontext_data[sys_kdma_name].append(example)
            
            # TODO - normalize KDMA values
            
            return incontext_data


class ComparativeRegressionIncontextExampleGenerator(IncontextExampleGenerator):
    def __init__(self,
                 incontext_settings, 
                 target_kdmas, 
                 log,
                 **kwargs):
        self.incontext_settings = incontext_settings
        self.target_kdmas = target_kdmas
        self.log = log
        self.set_icl_datasets()
    
    def set_icl_datasets(self):
        icl_datasets = {}
        incontext_data = self.read_icl_dataset_files()
        
        # Add each target to icl_datasets
        for target_kdma in self.target_kdmas:
            sys_kdma_name = target_kdma['kdma']
            icl_datasets[sys_kdma_name] = []
            kdma_incontext_data = incontext_data[sys_kdma_name]
            
            # Add each examples to icl_datasets
            for example in kdma_incontext_data:
                # Get example scenario description
                character_info = outlines_prompts_utils.get_relevant_structured_character_info(example['state'].characters)
                icl_scenario_description = scenario_state_description_with_relevant_char_info(example['state'], character_info)
                # Get example prompt
                icl_choices_with_outcomes = {}
                for choice in example['choices']:
                    # TODO: Include outcome prediction for ICL examples?
                    icl_choices_with_outcomes[choice] = {'predicted_outcome':None}
                icl_prompt = comparative_kdma_score_prediction_prompt(icl_scenario_description,
                                                                    icl_choices_with_outcomes,
                                                                    sys_kdma_name)
                # Get example response
                icl_reponse = {}
                for action, choice, kdma_value in zip(example['actions'], example['choices'], example["kdma_values"]):
                    # Only include choice if there is a ground truth KDMA value available
                    if kdma_value is None:
                        continue
                    icl_reponse[choice] = {}
                    icl_reponse[choice]['reasoning'] = get_chain_of_thought_reasoning(target_kdma, action,
                                                                                            example['state'], choice,
                                                                                            kdma_value)
                    # Predicted scores are 0-10, KDMA values are 0-1
                    icl_reponse[choice]['score'] = int(kdma_value * 10)
                # Add full example
                icl_datasets[sys_kdma_name].append({
                    "scenario_description": icl_scenario_description,
                    "prompt": icl_prompt,
                    "response": icl_reponse
                    })
                
        self.icl_datasets = icl_datasets
    
    def select_icl_examples(self, sys_kdma_name, print_kdma_name, scenario_description, choices):
        n_icl_examples = self.incontext_settings["number"]
        if sys_kdma_name not in self.icl_datasets:
            raise RuntimeError(f"No incontext samples for targeted kdma: {sys_kdma_name}")
        possible_icl_examples = self.icl_datasets[sys_kdma_name]
        if self.incontext_settings.get("leave_one_out", False):
            # Don't include example ICL with exact same scenario state
            possible_icl_examples = [
                icl_ex for icl_ex in possible_icl_examples
                if icl_ex["scenario_description"] != scenario_description
            ]
        if len(possible_icl_examples) < n_icl_examples:
            raise RuntimeError(f"Not enough possible incontext samples to learn from. Only "
                            f"{len(possible_icl_examples)} samples available while asking for "
                            f"{n_icl_examples} incontext samples.")

        # Downselect to n_icl_examples via given method
        icl_strategy = self.incontext_settings["method"]
        if icl_strategy == "random":
            selected_icl_examples = random.sample(possible_icl_examples, n_icl_examples)
        elif icl_strategy == "bert_similarity":
            # TODO: Include outcome prediction for ICL examples?
            no_outcome_predictions = {}
            for choice in choices:
                no_outcome_predictions[choice] = {}
                no_outcome_predictions[choice]['predicted_outcome'] = None
            no_outcome_prompt = comparative_kdma_score_prediction_prompt(scenario_description,
                                                                            no_outcome_predictions,
                                                                            print_kdma_name)

            possible_icl_prompts = [icl_sample["prompt"] for icl_sample in possible_icl_examples]

            # Create similarity scores between the ICL samples and find top-k indices
            from bert_score import score
            _, _, F1 = score([no_outcome_prompt]*len(possible_icl_prompts), possible_icl_prompts, lang="en")
            _, indices = torch.topk(F1, n_icl_examples)

            selected_icl_examples = [possible_icl_examples[i] for i in indices]
        else:
            raise ValueError(f'"{icl_strategy}" is not a valid incontext method. Please use "random" or '
                                '"bert_similarity"')

        return selected_icl_examples


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