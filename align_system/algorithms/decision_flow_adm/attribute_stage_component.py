from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)


class AttributeStageComponent(ADMComponent):
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
        return "attribute_analysis"

    def run(self, scenario_state, choices, extraction=None, variables=None, alignment_target=None, **kwargs):
        """Identify and analyze key attributes relevant to decision making"""

        # Handle alignment_target workflow similar to comparative_regression_adm_component
        if alignment_target is None:
            # No alignment target - use all attributes
            target_attributes = list(self.attributes.values())
        else:
            # Alignment target provided - ONLY use attributes in the alignment target
            target_attribute_names = attributes_in_alignment_target(alignment_target)
            target_attributes = [self.attributes[n] for n in target_attribute_names if n in self.attributes]

        attribute_results = {}

        # Build a dictionary mapping KDMA names to their values from alignment_target
        kdma_value_dict = {}
        if alignment_target is not None and hasattr(alignment_target, 'kdma_values'):
            for kdma_entry in alignment_target.kdma_values:
                # Support both AlignmentTarget API (KDMAValue objects) and dict access
                if isinstance(kdma_entry, dict):
                    kdma_name = kdma_entry.get('kdma')
                    kdma_value = kdma_entry.get('value')
                else:
                    # KDMAValue object with property accessors (API-compliant)
                    kdma_name = kdma_entry.kdma
                    kdma_value = kdma_entry.value

                # Only store if both name and value are present
                if kdma_name is not None and kdma_value is not None:
                    kdma_value_dict[kdma_name] = kdma_value

        # Process each attribute individually, similar to comparative_regression_adm_component
        for attribute in target_attributes:
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

            log.info(f"Processing attribute: {attribute.name}")
            log.info(f"Scenario description: {scenario_description}")

            # Get the value for this attribute from the alignment target
            # Use attribute.kdma (not attribute.name) to match alignment_target kdma_values
            attribute_value = kdma_value_dict.get(attribute.kdma)

            # Determine high/low based on value (handles both API compliant values and None)
            if attribute_value is not None and attribute_value >= 0.5:
                attribute_high_low = "high"
            elif attribute_value is not None:
                attribute_high_low = "low"
            else:
                # No value available, default to low (conservative default)
                attribute_high_low = "low"
                attribute_value = 0.0

            prompt = call_with_coerced_args(
                self.prompt_template,
                {
                    'scenario_description': scenario_description,
                    'choices': choices,
                    'extraction': extraction,
                    'variables': variables,
                    'target_bias': f"{attribute_high_low} {attribute.name}"
                },
            )
            log.info(f"Attribute prompt: {prompt}")

            dialog.append(DialogElement(role='user',
                                       content=prompt,
                                       tags=['decisionflow_attribute']))

            output_schema = call_with_coerced_args(
                self.output_schema_template,
                {})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            response = self.structured_inference_engine.run_inference(
                dialog_prompt,
                output_schema
            )

            log.info(f"Attribute analysis for {attribute.name} completed: {response.get('Variable', [])}")
            attribute_results[attribute.name] = response.get('Variable', [])

        return attribute_results
