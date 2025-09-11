from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import call_with_coerced_args, logging
from align_system.data_models.dialog import DialogElement
from align_system.utils.alignment_utils import attributes_in_alignment_target

log = logging.getLogger(__name__)


class MathReasonStageComponent(ADMComponent):
    def __init__(self, 
                 structured_inference_engine,
                 scenario_description_template,
                 attributes,
                 system_prompt_template=None,
                 prompt_template=None,
                 output_schema_template=None):
        super().__init__()
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template
        self.attributes = attributes

    def run_returns(self):
        return 'chosen_action'

    def run(self, scenario_state, choices, actions, mathematical_model=None, attribute_analysis=None, alignment_target=None):
        """Use math_reason prompt to select optimal action based on mathematical model"""
        
        log.info(f"MathReasonStageComponent starting with {len(choices)} choices and {len(actions)} actions")
        log.debug(f"Mathematical model: {mathematical_model}")
        log.debug(f"Attribute analysis: {attribute_analysis}")
        
        try:
            # Handle alignment_target workflow similar to other stage components
            if alignment_target is None:
                target_attribute_names = []
            else:
                target_attribute_names = attributes_in_alignment_target(alignment_target)

            target_attributes = [self.attributes[n] for n in target_attribute_names if n in self.attributes]

            # If we have target attributes from alignment_target, use those; otherwise use all attributes
            if not target_attributes:
                target_attributes = list(self.attributes.values())

            # Format structure for math_reason prompt
            # The openai_open_source_adm.py expects structure with these keys:
            # - "Objective Function" 
            # - "Decision Variables"
            # - "Constraints"
            
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
            # The prompt expects attribute data with Variable/Attribute/Value structure
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

            scenario_description = call_with_coerced_args(
                self.scenario_description_template,
                {
                    'scenario_state': scenario_state,
                    'alignment_target': alignment_target
                })

            # Build dialog for math_reason prompt
            dialog = []
            
            # Format choices as the math_reason prompt expects: "(0) choice1", "(1) choice2", etc.
            formatted_choices = [f"({i}) {choice}" for i, choice in enumerate(choices)]

            # Create system prompt for target bias (similar to openai_open_source_adm.py)
            if self.system_prompt_template is not None:
                system_prompt = call_with_coerced_args(
                    self.system_prompt_template,
                    {
                        'alignment_target': alignment_target,
                        'target_attributes': target_attributes
                    }
                )
            else:
                # Default system message for alignment target
                if alignment_target and target_attribute_names:
                    system_prompt = f"Focus on {', '.join(target_attribute_names)} when making decisions."
                else:
                    system_prompt = "Make decisions based on medical triage best practices."

            # Use math_reason template format
            if self.prompt_template is not None:
                prompt_content = call_with_coerced_args(
                    self.prompt_template,
                    {
                        'scenario_description': scenario_description,
                        'choices': formatted_choices,
                        'structure': structure,
                        'attribute': attribute,
                        'target_bias': system_prompt
                    }
                )
            else:
                # Fallback to basic math_reason format
                from align_system.prompt_engineering.decision_flow_prompts.mta_prompt import math_reason
                
                try:
                    prompt_content = math_reason.format(
                        objective=structure["Objective Function"],
                        attribute=attribute,
                        variable=structure["Decision Variables"],
                        constraints=structure["Constraints"],
                        choice=formatted_choices,
                        target_bias=system_prompt,
                    )
                except Exception as e:
                    log.warning(f"Failed to format math_reason prompt: {e}")
                    # Fallback prompt
                    prompt_content = f"""You will receive a mathematical model structure along with choices. Select the optimal choice.

Structure: {structure}
Attribute Data: {attribute}
Choices: {formatted_choices}
Target Bias: {system_prompt}

Please respond with: {{"Reasoning": "<explanation>", "Answer": <integer_index>}}"""

            dialog.append(DialogElement(role='user',
                                       content=prompt_content,
                                       tags=['math_reason']))

            # Get output schema
            if self.output_schema_template is not None:
                output_schema = call_with_coerced_args(
                    self.output_schema_template,
                    {}
                )
            else:
                # Default schema expecting Reasoning and Answer
                output_schema = {
                    "type": "object",
                    "properties": {
                        "Reasoning": {"type": "string"},
                        "Answer": {"type": "integer"}
                    },
                    "required": ["Reasoning", "Answer"]
                }

            dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
            
            log.debug(f"Dialog prompt: {dialog_prompt[:200]}...")
            log.debug(f"Output schema: {output_schema}")
            
            response = self.structured_inference_engine.run_inference(
                dialog_prompt,
                output_schema
            )
            
            log.debug(f"Structured inference response: {response}")
            
            # Check if response is actually the schema (indicates inference failure)
            if isinstance(response, dict) and 'properties' in response and 'type' in response:
                log.error("Structured inference returned schema instead of response - inference likely failed")
                raise RuntimeError(f"Structured inference failed, returned schema: {response}")

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
            
            log.info(f"MathReasonStageComponent selected action {answer_idx}: {choices[answer_idx]}")
            log.debug(f"Reasoning: {reasoning}")
            
            return chosen_action
            
        except Exception as e:
            log.warning(f"MathReasonStageComponent failed with error: {e}")
            log.warning("Falling back to first action")
            
            # Fallback: return first action
            chosen_action = actions[0]
            if hasattr(chosen_action, 'justification') and chosen_action.justification is None:
                chosen_action.justification = f"Math reasoning failed, selected first action. Error: {str(e)}"
            
            return chosen_action