from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)


class MathReasonStageComponent(ADMComponent):
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
        return 'chosen_action'

    def run(self, scenario_state, choices, actions, mathematical_model=None, attribute_analysis=None, alignment_target=None, **kwargs):
        """Use math_reason prompt to select optimal action based on mathematical model"""

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

            # Format attribute data for math_reason prompt
            attribute = []
            if attribute_analysis:
                for attr_name, variables_data in attribute_analysis.items():
                    if isinstance(variables_data, list):
                        for var_data in variables_data:
                            if isinstance(var_data, dict):
                                attribute.append({
                                    "Variable": var_data.get("Variable", ""),
                                    "Attribute": attr_name,
                                    "Value": var_data.get("Value", "")
                                })

            # Format choices as the math_reason prompt expects
            formatted_choices = [f"({i}) {choice}" for i, choice in enumerate(choices)]

            # Build KDMA value dictionary for target bias generation (API-compliant)
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

            # Generate target bias for prompt with high/low determination
            if alignment_target and target_attributes:
                bias_parts = []
                for attribute in target_attributes:
                    # Use attribute.kdma (not attribute.name) to match alignment_target kdma_values
                    attr_value = kdma_value_dict.get(attribute.kdma)
                    if attr_value is not None and attr_value >= 0.5:
                        bias_parts.append(f"high {attribute.name}")
                    elif attr_value is not None:
                        bias_parts.append(f"low {attribute.name}")
                    else:
                        bias_parts.append(f"low {attribute.name}")  # Conservative default
                target_bias = f"Focus on {', '.join(bias_parts)} when making decisions."
            else:
                target_bias = "Make decisions based on medical triage best practices."

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
                    'target_bias': target_bias
                },
            )

            dialog.append(DialogElement(role='user',
                                       content=prompt))

            output_schema = call_with_coerced_args(
                self.output_schema_template,
                {})

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

            response = self.structured_inference_engine.run_inference(
                dialog_prompt,
                output_schema
            )

            # Parse response to get chosen action
            reasoning = response.get('Reasoning', '')
            answer_idx = response.get('Answer', 0)

            # Validate answer index
            if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(actions):
                log.warning(f"Invalid answer index {answer_idx}, defaulting to 0")
                answer_idx = 0

            chosen_action = actions[answer_idx]

            # Add reasoning as justification if possible
            if hasattr(chosen_action, 'justification') and chosen_action.justification is None:
                chosen_action.justification = reasoning

            log.info(f"MathReason stage completed: Selected action {answer_idx}")

            return chosen_action

        except Exception as e:
            log.warning(f"MathReasonStageComponent failed with error: {e}")
            log.warning("Falling back to first action")

            # Fallback: return first action
            chosen_action = actions[0]
            if hasattr(chosen_action, 'justification') and chosen_action.justification is None:
                chosen_action.justification = f"Math reasoning failed, selected first action. Error: {str(e)}"

            return chosen_action
