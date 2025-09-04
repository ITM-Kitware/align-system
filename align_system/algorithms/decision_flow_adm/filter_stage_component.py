from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)


class FilterStageComponent(ADMComponent):
    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt_template,
        prompt_template,
        output_schema_template,
        attributes=None,
        **kwargs,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template
        
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def run_returns(self):
        return "filter_analysis"

    def run(self, scenario_state, choices, attribute_analysis=None, variables=None, extraction=None, alignment_target=None, **kwargs):
        """Filter and weight attributes based on their relevance to the target bias"""

        # Handle alignment_target workflow similar to attribute_stage_component
        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        target_attributes = [self.attributes[n] for n in target_attribute_names if n in self.attributes]

        # If we have target attributes from alignment_target, use those; otherwise use all attributes
        if not target_attributes:
            target_attributes = list(self.attributes.values())

        filter_results = {}
        
        # Process each attribute's analysis individually to determine relevance and weight
        for attribute in target_attributes:
            # Get the attribute analysis results for this specific attribute
            attribute_data = attribute_analysis.get(attribute.name, []) if attribute_analysis else []
            
            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {
                    'scenario_state': scenario_state,
                    'alignment_target': alignment_target,
                    'attribute': attribute.name,
                    'attributes_of_interest': {attribute.name}
                })

            dialog = []
            if self.system_prompt_template is not None:
                system_prompt = call_with_coerced_args(
                    self.system_prompt_template,
                    {'target_attribute': attribute}
                )

                dialog.insert(0, DialogElement(role='system',
                                              content=system_prompt,
                                              tags=['decisionflow_system_prompt']))

            log.info(f"Filtering attribute: {attribute.name}")
            log.info(f"Scenario description: {scenario_description}")
            
            prompt = call_with_coerced_args(
                self.prompt_template,
                {
                    'scenario_description': scenario_description,
                    'choices': choices,
                    'attribute_information': attribute_data,
                    'target_bias': attribute
                },
            )
            log.info(f"Filter prompt: {prompt}")
            
            dialog.append(DialogElement(role='user',
                                       content=prompt,
                                       tags=['decisionflow_filter']))

            output_schema = call_with_coerced_args(
                self.output_schema_template,
                {})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
            
            response = self.structured_inference_engine.run_inference(
                dialog_prompt,
                output_schema
            )

            log.info(f"Filter analysis for {attribute.name} completed: Weight={response.get('Weight', 0)}")
            filter_results[attribute.name] = {
                'explanation': response.get('Explanation', ''),
                'weight': response.get('Weight', 0)
            }

        return filter_results