from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement

log = logging.getLogger(__name__)


class ExpressStageComponent(ADMComponent):
    def __init__(
        self,
        structured_inference_engine,
        scenario_description_template,
        system_prompt_template,
        prompt_template,
        output_schema_template,
        **kwargs,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.scenario_description_template = scenario_description_template
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.output_schema_template = output_schema_template

    def run_returns(self):
        return "mathematical_model"

    def run(self, scenario_state, choices, objective_function=None, filter_analysis=None, attribute_analysis=None, variables=None, extraction=None, alignment_target=None, **kwargs):
        """Create complete mathematical optimization model following math_express template"""

        # Build structure following decision_flow_stages.py lines 158-188
        structure = {}

        # 1. Variables from variables stage
        structure["variables"] = variables if variables else []

        # 2. Objective function from objective stage
        if objective_function and isinstance(objective_function, dict):
            structure["objective_function"] = objective_function.get('objective_function', 'weight * attribute of variable')
        else:
            structure["objective_function"] = objective_function if objective_function else 'weight * attribute of variable'

        # 3. Attributes from filtered analysis (lines 162-175)
        structure["attribute"] = []
        if filter_analysis and attribute_analysis:
            for attribute_name, filter_data in filter_analysis.items():
                # Skip environment attributes (line 164)
                if attribute_name.lower() == "environment":
                    continue

                # Get attribute analysis data for this attribute
                attribute_data = attribute_analysis.get(attribute_name, [])

                # Process attribute data to create triples (variable, attribute, value)
                if isinstance(attribute_data, list):
                    for variable_info in attribute_data:
                        if isinstance(variable_info, dict) and 'Variable' in variable_info:
                            variable_name = variable_info['Variable']
                            if 'Attribute' in variable_info:
                                for attr_info in variable_info['Attribute']:
                                    if isinstance(attr_info, dict) and 'Attribute' in attr_info and 'Value' in attr_info:
                                        structure["attribute"].append({
                                            "Variable": variable_name,
                                            "Attribute": attr_info['Attribute'],
                                            "Value": attr_info['Value']
                                        })

        # 4. Constraints from extraction information (lines 177-188)
        structure["constraints"] = []
        if extraction:
            constraint_indicators = ["only", "limited", "must", "cannot", "time", "constraint", "restriction"]
            # Look for constraint-indicating phrases in extraction
            for info_item in extraction:
                if isinstance(info_item, str):
                    # Check for constraint indicators like "only", "limited", "must", "cannot"
                    if any(indicator in info_item.lower() for indicator in constraint_indicators):
                        structure["constraints"].append(info_item)

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
                {'structure': structure}
            )

            dialog.insert(0, DialogElement(role='system',
                                          content=system_prompt))

        prompt = call_with_coerced_args(
            self.prompt_template,
            {
                'scenario_description': scenario_description,
                'choices': choices,
                'structure': structure
            },
        )

        dialog.append(DialogElement(role='user',
                                   content=prompt))

        output_schema = call_with_coerced_args(
            self.output_schema_template,
            {})

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info(f"**Express stage dialog prompt**: {dialog_prompt}")

        import time
        start_time = time.time()
        response = self.structured_inference_engine.run_inference(
            dialog_prompt,
            output_schema
        )
        elapsed_time = time.time() - start_time
        log.info(f"**Express stage inference took {elapsed_time:.2f} seconds**")
        log.info(f"**Express stage response**: \n{response}")

        # Extract components
        objective_function = response.get('Objective Function', [])
        decision_variables = response.get('Decision Variables', [])
        constraints = response.get('Constraints', [])
        explanation = response.get('Explanation', '')

        result = {
            'mathematical_model': response,
            'structure': structure,
            'objective_function': objective_function,
            'decision_variables': decision_variables,
            'constraints': constraints,
            'explanation': explanation
        }

        log.info(f"Express stage completed: Generated mathematical model with {len(objective_function)} objective functions, {len(decision_variables)} decision variables, {len(constraints)} constraints")
        return result
