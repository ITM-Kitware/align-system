from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.outlines_prompts_utils import (
    get_unique_structured_character_info,
    get_relevant_structured_character_info)
from align_system.prompt_engineering.outlines_prompts import (
    comparative_kdma_score_prediction_prompt_no_outcomes,
    relevance_classification_prompt,
    scenario_state_description_with_relevant_char_info)
from align_system.data_models.dialog import DialogElement


class ICLADMComponent(ADMComponent):
    def __init__(self,
                 icl_generator,
                 scenario_description_template,
                 prompt_template,
                 attributes={},
                 system_prompt=None):
        self.icl_generator = icl_generator
        self.scenario_description_template = scenario_description_template
        self.attributes = attributes
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    def run(self,
            scenario_state,
            choice_evaluation,
            dialogs,
            alignment_target=None):
        available_actions = [ce['action'] for ce in choice_evaluation.values()]

        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names]

        icl_dialog_elements = []
        for attribute in target_attributes:
            scenario_description = self.scenario_description_template(
                scenario_state, alignment_target, {attribute.name,})

            prompt_to_match = self.prompt_template(
                scenario_state, scenario_description, choice_evaluation, {attribute.name,})

            selected_icl_examples = self.icl_generator.select_icl_examples(
                sys_kdma_name=attribute.kdma,
                scenario_description_to_match=scenario_description,
                prompt_to_match=prompt_to_match,
                state_comparison=scenario_state,
                actions=available_actions)

            for icl_sample in selected_icl_examples:
                icl_dialog_elements.append(DialogElement(role='user',
                                                         content=icl_sample['prompt'],
                                                         namespace='.icl',
                                                         tags=['icl']))
                icl_dialog_elements.append(DialogElement(role='assistant',
                                                         content=str(icl_sample['response']),
                                                         namespace='.icl',
                                                         tags=['icl']))

        for dialog in dialogs:
            dialog.extend(icl_dialog_elements)

        return choice_evaluation, dialogs
