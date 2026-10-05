import json
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.decision_flow_adm.utils import validate_structured_response
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)


class AttributeFineGrainedStageComponent(ADMComponent):
    """
    Fine-grained attribute stage component that uses explicit numeric target values
    instead of binary high/low distinctions.

    This component extracts attributes using the Phase2FineGrainedAttributePrompt which
    provides scale anchor examples at specific value levels (0.0-1.0) for more
    precise value targeting.
    """

    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt_template,
        prompt_template,
        output_schema_template,
        max_json_retries=5,
        attributes=None,
        **kwargs,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template
        self.max_json_retries = max_json_retries

        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def run_returns(self):
        return "attribute_analysis"

    def run(self, scenario_state, choices, extraction=None, variables=None, alignment_target=None, **kwargs):
        """
        Identify and analyze key attributes relevant to decision making using
        fine-grained value targeting.

        Unlike the high/low attribute stage, this component uses explicit numeric
        values (0.0-1.0) from the alignment_target to guide attribute extraction
        at specific scale points.
        """

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
                                              content=system_prompt))

            log.info(f"Processing attribute (fine-grained): {attribute.name}")
            log.info(f"Scenario description: {scenario_description}")

            # Get the numeric value for this attribute from the alignment target
            # Use attribute.kdma (not attribute.name) to match alignment_target kdma_values
            attribute_value = kdma_value_dict.get(attribute.kdma)

            # Use the actual numeric value (0.0-1.0) for fine-grained targeting
            # Default to 0.5 (moderate) if no value is available
            # if attribute_value is None:
            #     log.warning(f"No value found for attribute {attribute.name} (kdma: {attribute.kdma}), defaulting to 0.5")
            #     attribute_value = 0.5

            log.info(f"Target value for {attribute.name}: {attribute_value}")

            # Call prompt template with target_attribute and target_value
            # instead of target_bias (which was "high/low attribute_name")
            prompt = call_with_coerced_args(
                self.prompt_template,
                {
                    'scenario_description': scenario_description,
                    'choices': choices,
                    'extraction': extraction,
                    'variables': variables,
                    'target_attribute': attribute.name,
                    'target_value': attribute_value
                },
            )
            log.info(f"Fine-grained attribute prompt for {attribute.name} (value={attribute_value})")

            dialog.append(DialogElement(role='user',
                                       content=prompt))

            output_schema = call_with_coerced_args(
                self.output_schema_template,
                {})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("=" * 80)
            log.info(f"Attribute Stage Dialog Prompt ({attribute.name})")
            log.info("=" * 80)
            log.info(dialog_prompt)
            log.info("=" * 80)

            # Retry loop for structured inference with validation
            response = None
            last_error = None
            context_str = f" for {attribute.name} (value={attribute_value})"

            for attempt in range(self.max_json_retries):
                try:
                    # Run structured inference
                    raw_response = self.structured_inference_engine.run_inference(
                        dialog_prompt,
                        output_schema
                    )

                    # Validate response
                    response = validate_structured_response(raw_response)

                    # Success - break out of retry loop
                    log.info(f"Fine-grained attribute stage inference succeeded on attempt {attempt + 1}{context_str}")
                    break

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    last_error = e
                    log.warning(
                        f"Fine-grained attribute stage JSON decode error on attempt {attempt + 1}/{self.max_json_retries}{context_str}: {e}"
                    )

                    if attempt < self.max_json_retries - 1:
                        log.info(f"Retrying fine-grained attribute stage inference{context_str}...")
                    else:
                        log.error(f"Fine-grained attribute stage failed after {self.max_json_retries} attempts{context_str}")
                        raise SceneSkipException(
                            f"Failed to generate valid JSON after {self.max_json_retries} attempts{context_str}. "
                            f"Last error: {last_error}",
                            component_name="AttributeFineGrainedStageComponent",
                            last_error=last_error
                        ) from last_error

            log.info(f"Fine-grained attribute analysis for {attribute.name} (value={attribute_value}) completed: {response.get('Variable', [])}")
            attribute_results[attribute.name] = response.get('Variable', [])

        return attribute_results
