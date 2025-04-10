from align_system.utils import call_with_coerced_args
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement


class ICLADMComponent(ADMComponent):
    def __init__(self,
                 icl_generator,
                 scenario_description_template,
                 prompt_template,
                 attributes={}):
        self.icl_generator = icl_generator
        self.scenario_description_template = scenario_description_template
        self.attributes = attributes
        self.prompt_template = prompt_template

    def run_returns(self):
        return 'icl_dialog_elements'

    def run(self,
            scenario_state,
            choices,
            actions,
            alignment_target=None):
        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names]

        icl_dialog_elements = []
        for attribute in target_attributes:
            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {'scenario_state': scenario_state,
                 'alignment_target': alignment_target,
                 'attribute': attribute.name,
                 'attributes_of_interest': {attribute.name,}})

            prompt_to_match = call_with_coerced_args(
                self.prompt_template,
                {'scenario_state': scenario_state,
                 'scenario_description': scenario_description,
                 'choices': choices,
                 'choice_outcomes': {c: None for c in choices},
                 'attribute': attribute.name})

            selected_icl_examples = self.icl_generator.select_icl_examples(
                sys_kdma_name=attribute.kdma,
                scenario_description_to_match=scenario_description,
                prompt_to_match=prompt_to_match,
                state_comparison=scenario_state,
                actions=actions)

            for icl_sample in selected_icl_examples:
                icl_dialog_elements.append(DialogElement(role='user',
                                                         content=icl_sample['prompt'],
                                                         namespace='.icl',
                                                         tags=['icl']))
                icl_dialog_elements.append(DialogElement(role='assistant',
                                                         content=str(icl_sample['response']),
                                                         namespace='.icl',
                                                         tags=['icl']))

        return icl_dialog_elements
