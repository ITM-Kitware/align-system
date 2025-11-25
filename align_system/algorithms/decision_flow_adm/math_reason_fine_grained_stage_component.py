import json
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.decision_flow_adm.utils import validate_structured_response
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)


class MathReasonFineGrainedStageComponent(ADMComponent):
    """
    Fine-grained MathReason stage component using numeric target values (0.0-1.0)
    instead of binary high/low bias.

    This component provides more precise alignment by passing specific numeric
    targets for each KDMA (e.g., affiliation: 0.9, merit: 0.3) to guide the
    mathematical reasoning process.
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
        return ('chosen_choice', 'justification')

    def run(self, scenario_state, choices, actions, mathematical_model=None, attribute_analysis=None, objective_function=None, alignment_target=None, **kwargs):
        """Use fine-grained math_reason prompt with numeric target values"""

        log.info("=" * 80)
        log.info("MathReasonFineGrained Stage Starting")
        log.info("=" * 80)

        # Handle alignment_target workflow similar to other stage components
        if alignment_target is None:
            # No alignment target - use all attributes
            target_attributes = list(self.attributes.values())
            target_attribute_names = []
        else:
            # Alignment target provided - ONLY use attributes in the alignment target
            target_attribute_names = attributes_in_alignment_target(alignment_target)
            target_attributes = [self.attributes[n] for n in target_attribute_names if n in self.attributes]

        try:
            # Format structure for math_reason prompt
            if mathematical_model and isinstance(mathematical_model, dict):
                # Check if we have the nested 'mathematical_model' key (from Express stage output)
                if 'mathematical_model' in mathematical_model:
                    nested_model = mathematical_model['mathematical_model']
                    structure = {
                        "Objective Function": nested_model.get('Objective Function', 'weight * attribute of variable'),
                        "Decision Variables": nested_model.get('Decision Variables', []),
                        "Constraints": nested_model.get('Constraints', [])
                    }
                else:
                    # Direct access (for backwards compatibility)
                    structure = {
                        "Objective Function": mathematical_model.get('Objective Function', 'weight * attribute of variable'),
                        "Decision Variables": mathematical_model.get('Decision Variables', []),
                        "Constraints": mathematical_model.get('Constraints', [])
                    }
            else:
                # Fallback structure
                structure = {
                    "Objective Function": "weight * attribute of variable",
                    "Decision Variables": [],
                    "Constraints": []
                }

            # Format attribute data for math_reason prompt using filtered_pairs from objective stage
            attribute = []

            # Extract filtered_pairs from objective_function output (preferred - only pairs with weight > threshold)
            filtered_pairs = []
            if objective_function and isinstance(objective_function, dict):
                filtered_pairs = objective_function.get('filtered_pairs', [])

            if filtered_pairs:
                # Use filtered pairs (Variable-Attribute pairs that exceeded weight threshold)
                log.info(f"Using {len(filtered_pairs)} filtered pairs from Objective stage")
                for pair in filtered_pairs:
                    attribute.append({
                        "Variable": pair.get("Variable", ""),
                        "Attribute": pair.get("Attribute", ""),
                        "Value": pair.get("Value", [])
                    })
                    log.info(f"  - {pair.get('Variable', '')[:50]}... | {pair.get('Attribute', '')} | Weight: {pair.get('Weight', 'N/A')}")
            elif attribute_analysis:
                # Fallback to attribute_analysis if no filtered_pairs available
                log.info("No filtered_pairs available, falling back to attribute_analysis")
                for attr_name, variables_data in attribute_analysis.items():
                    if isinstance(variables_data, list):
                        for i, var_data in enumerate(variables_data):
                            if isinstance(var_data, dict):
                                variable_name = var_data.get("Variable", "")

                                # Check if 'Attribute' key contains nested list of attribute dicts
                                if 'Attribute' in var_data and isinstance(var_data['Attribute'], list):
                                    # Iterate through nested attributes and extract all values
                                    all_values = []
                                    for nested_attr in var_data['Attribute']:
                                        if isinstance(nested_attr, dict) and 'Value' in nested_attr:
                                            nested_values = nested_attr.get('Value', [])
                                            if isinstance(nested_values, list):
                                                all_values.extend(nested_values)
                                            else:
                                                all_values.append(nested_values)

                                    attribute.append({
                                        "Variable": variable_name,
                                        "Attribute": attr_name,
                                        "Value": all_values
                                    })
                                else:
                                    # Fallback to old behavior for backwards compatibility
                                    attribute.append({
                                        "Variable": variable_name,
                                        "Attribute": attr_name,
                                        "Value": var_data.get("Value", "")
                                    })

            # Format choices as the math_reason prompt expects
            formatted_choices = [f"({i}) {choice}" for i, choice in enumerate(choices)]

            # Build KDMA value dictionary for fine-grained target values (API-compliant)
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

            # Generate fine-grained target attributes with numeric values
            if alignment_target and target_attributes:
                target_attrs_values = {}
                for target_attr in target_attributes:
                    # Use target_attr.kdma to match alignment_target kdma_values
                    attr_value = kdma_value_dict.get(target_attr.kdma)
                    if attr_value is not None:
                        target_attrs_values[target_attr.name] = attr_value
                    # Note: Unlike high/low version, we don't use defaults - if no value, we don't include it

                # Format as a readable string for the prompt
                target_attributes_values_str = json.dumps(target_attrs_values, indent=2)
            else:
                target_attributes_values_str = "{}"

            dialog = []
            if self.system_prompt_template is not None:
                system_prompt = call_with_coerced_args(
                    self.system_prompt_template,
                    {
                        'alignment_target': alignment_target,
                        'target_attributes': target_attributes
                    }
                )
                dialog.insert(0, DialogElement(role='system',
                                              content=system_prompt))

            prompt = call_with_coerced_args(
                self.prompt_template,
                {
                    'objective': structure["Objective Function"],
                    'attribute': attribute,
                    'variable': structure["Decision Variables"],
                    'constraints': structure["Constraints"],
                    'choice': formatted_choices,
                    'target_attributes_values': target_attributes_values_str
                },
            )

            dialog.append(DialogElement(role='user',
                                       content=prompt))

            output_schema = call_with_coerced_args(
                self.output_schema_template,
                {})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            log.info("=" * 80)
            log.info("MathReasonFineGrained Dialog Prompt")
            log.info("=" * 80)
            log.info(dialog_prompt)
            log.info("=" * 80)

            # Retry loop for structured inference with validation
            response = None
            last_error = None

            import time
            inference_start = time.time()

            for attempt in range(self.max_json_retries):
                try:
                    # Run structured inference
                    log.info(f"Running inference attempt {attempt + 1}/{self.max_json_retries}...")
                    raw_response = self.structured_inference_engine.run_inference(
                        dialog_prompt,
                        output_schema
                    )

                    log.info("=" * 80)
                    log.info("MathReasonFineGrained Raw Response")
                    log.info("=" * 80)
                    log.info(f"{raw_response}")
                    log.info("=" * 80)

                    # Validate response
                    response = validate_structured_response(raw_response)

                    # Success - break out of retry loop
                    inference_elapsed = time.time() - inference_start
                    log.info(f"MathReasonFineGrained stage inference succeeded on attempt {attempt + 1} (took {inference_elapsed:.2f}s)")
                    break

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    last_error = e
                    log.warning(
                        f"MathReasonFineGrained stage JSON decode error on attempt {attempt + 1}/{self.max_json_retries}: {e}"
                    )

                    if attempt < self.max_json_retries - 1:
                        log.info("Retrying MathReasonFineGrained stage inference...")
                    else:
                        log.error(f"MathReasonFineGrained stage failed after {self.max_json_retries} attempts")
                        raise SceneSkipException(
                            f"Failed to generate valid JSON after {self.max_json_retries} attempts. "
                            f"Last error: {last_error}",
                            component_name="MathReasonFineGrainedStageComponent",
                            last_error=last_error
                        ) from last_error

            # Parse response to get chosen choice (string-based selection)
            reasoning = response.get('Reasoning', '')
            answer_text = response.get('Answer', '')

            log.info("=" * 80)
            log.info("MathReasonFineGrained Parsed Response")
            log.info("=" * 80)
            log.info(f"Answer Text: {answer_text}")
            log.info(f"Reasoning: {reasoning}")
            log.info("=" * 80)

            # Find matching choice using fuzzy matching
            chosen_choice = None

            # Try exact match first
            if answer_text in choices:
                chosen_choice = answer_text
                log.info(f"Exact match found for answer: {answer_text}")
            else:
                # Try substring match (answer contains choice or vice versa)
                for choice in choices:
                    if answer_text.strip() in choice or choice in answer_text.strip():
                        chosen_choice = choice
                        log.info(f"Substring match: '{answer_text}' matched to '{choice}'")
                        break

                # Try case-insensitive match
                if chosen_choice is None:
                    answer_lower = answer_text.lower().strip()
                    for choice in choices:
                        if answer_lower in choice.lower() or choice.lower() in answer_lower:
                            chosen_choice = choice
                            log.info(f"Case-insensitive match: '{answer_text}' matched to '{choice}'")
                            break

            # Fallback to first choice if no match found
            if chosen_choice is None:
                log.warning(f"Could not match answer '{answer_text}' to any choice, defaulting to first choice")
                chosen_choice = choices[0]

            log.info("=" * 80)
            log.info("MathReasonFineGrained Final Selection")
            log.info("=" * 80)
            log.info(f"Selected choice: {chosen_choice}")
            log.info(f"Justification: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
            log.info("=" * 80)
            log.info("MathReasonFineGrained Stage Completed Successfully")
            log.info("=" * 80)

            # Return chosen_choice and justification - EnsureChosenActionADMComponent will convert to action
            return chosen_choice, reasoning

        except Exception as e:
            log.warning(f"MathReasonFineGrainedStageComponent failed with error: {e}")
            log.warning("Falling back to first choice")

            # Fallback: return first choice string and error justification
            return choices[0], f"Math reasoning failed, selected first choice. Error: {str(e)}"
