import json
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.decision_flow_adm.utils import validate_structured_response

log = logging.getLogger(__name__)


class ObjectiveStageComponent(ADMComponent):
    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt_template,
        prompt_template,
        output_schema_template,
        max_json_retries=5,
        attributes=None,
        weight_threshold=0.3,
        **kwargs,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template
        self.weight_threshold = weight_threshold
        self.max_json_retries = max_json_retries

        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def run_returns(self):
        return "objective_function"

    def run(self, scenario_state, choices, filter_analysis=None, attribute_analysis=None, variables=None, extraction=None, alignment_target=None, **kwargs):
        """Create objective function by filtering attributes with weights above threshold"""

        # Handle alignment_target workflow similar to other stage components
        if alignment_target is None:
            # No alignment target - use all attributes
            target_attributes = list(self.attributes.values())
        else:
            # Alignment target provided - ONLY use attributes in the alignment target
            target_attribute_names = attributes_in_alignment_target(alignment_target)
            target_attributes = [self.attributes[n] for n in target_attribute_names if n in self.attributes]

        objective_components = []

        # Process filter analysis to identify high-weight attributes for objective function
        # Following the logic from decision_flow_stages.py lines 132-156
        if filter_analysis:
            for attribute in target_attributes:
                attribute_name = attribute.name
                if attribute_name in filter_analysis:
                    weight = filter_analysis[attribute_name].get('weight', 0)
                    explanation = filter_analysis[attribute_name].get('explanation', '')

                    # Apply weight threshold filter (similar to decision_flow_stages.py line 137)
                    if weight > self.weight_threshold:
                        log.info(f"Including attribute {attribute_name} with weight {weight} in objective function")

                        # Get attribute analysis data for this attribute
                        attribute_data = attribute_analysis.get(attribute_name, []) if attribute_analysis else []

                        objective_component = {
                            "Variable": attribute_name,
                            "Attribute": attribute.name,
                            "Weight": weight,
                            "Explanation": explanation,
                            "AttributeData": attribute_data
                        }
                        objective_components.append(objective_component)
                    else:
                        log.info(f"Excluding attribute {attribute_name} with weight {weight} (below threshold {self.weight_threshold})")

        # Create objective function string following decision_flow_stages.py format (lines 148-153)
        objective_function_text = "The final formula to be calculated is "
        if objective_components:
            for component in objective_components:
                variable = component["Variable"]
                weight = component["Weight"]
                attribute = component["Attribute"]
                objective_function_text += f"{weight} * ({attribute}) of ({variable}) + "
            # Remove trailing " + "
            objective_function_text = objective_function_text.rstrip(" + ")
        else:
            # Fallback if no components meet threshold (line 156)
            objective_function_text = "weight * attribute of variable"

        log.info(f"Generated objective function: {objective_function_text}")

        scenario_description = call_with_coerced_args(
            self.scenario_description_template,
            {
                'scenario_state': scenario_state,
                'alignment_target': alignment_target
            })

        dialog = []
        if self.system_prompt_template is not None:
            system_prompt = call_with_coerced_args(
                self.system_prompt_template,
                {'objective_components': objective_components}
            )

            dialog.insert(0, DialogElement(role='system',
                                          content=system_prompt))

        log.info(f"Creating objective function with {len(objective_components)} components")

        prompt = call_with_coerced_args(
            self.prompt_template,
            {
                'scenario_description': scenario_description,
                'choices': choices,
                'objective_components': objective_components,
                'objective_function_text': objective_function_text,
                'weight_threshold': self.weight_threshold
            },
        )
        log.info(f"Objective prompt: {prompt}")

        dialog.append(DialogElement(role='user',
                                   content=prompt))

        output_schema = call_with_coerced_args(
            self.output_schema_template,
            {})

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

        # Retry loop for structured inference with validation
        response = None
        last_error = None

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
                log.info(f"Objective stage inference succeeded on attempt {attempt + 1}")
                break

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                log.warning(
                    f"Objective stage JSON decode error on attempt {attempt + 1}/{self.max_json_retries}: {e}"
                )

                if attempt < self.max_json_retries - 1:
                    log.info("Retrying Objective stage inference...")
                else:
                    log.error(f"Objective stage failed after {self.max_json_retries} attempts")
                    raise RuntimeError(
                        f"Failed to generate valid JSON after {self.max_json_retries} attempts. "
                        f"Last error: {last_error}"
                    ) from last_error

        log.info(f"Objective function creation completed: {response.get('objective_function', objective_function_text)}")

        return {
            'objective_function': response.get('objective_function', objective_function_text),
            'components': objective_components,
            'weight_threshold_used': self.weight_threshold
        }
