import json
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.decision_flow_adm.utils import validate_unstructured_response
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)


class ExpressStageUnstructuredComponent(ADMComponent):
    """
    Express stage component using unstructured inference for faster generation.

    This component uses run_inference_unstructured() instead of run_inference()
    to bypass JSON schema constraint checking during token generation. This results
    in significantly faster inference times (5-8x speedup) while maintaining output
    quality through post-processing JSON extraction.

    Expected performance: ~5-8 minutes vs ~40 minutes with structured inference.
    """

    def __init__(
        self,
        structured_inference_engine,
        system_prompt_template,
        prompt_template,
        max_json_retries=5,
        **kwargs,
    ):
        self.structured_inference_engine = structured_inference_engine
        self.system_prompt_template = system_prompt_template
        self.prompt_template = prompt_template
        self.max_json_retries = max_json_retries

    def run_returns(self):
        return "mathematical_model"

    def run(self, objective_function=None, variables=None, extraction=None, **kwargs):
        """Create complete mathematical optimization model following math_express template.

        Args:
            objective_function: Output from Objective stage containing:
                - objective_function: The formula string
                - filtered_pairs: List of Variable-Attribute-Value triplets that passed weight threshold
            variables: List of variables from Variables stage
            extraction: List of extracted information from Extraction stage
            **kwargs: Additional pipeline arguments (ignored)

        Note: Uses filtered_pairs from Objective stage (already filtered by weight > 0.3).
        Matches original DecisionFlow math_express which only takes structure.
        """

        # Build structure following decision_flow_stages.py lines 158-188
        structure = {}

        # 1. Variables from variables stage
        structure["variables"] = variables if variables else []

        # 2. Objective function from objective stage
        if objective_function and isinstance(objective_function, dict):
            structure["objective_function"] = objective_function.get('objective_function', 'weight * attribute of variable')
        else:
            structure["objective_function"] = objective_function if objective_function else 'weight * attribute of variable'

        # 3. Attributes from Objective stage's filtered pairs (already filtered by weight threshold)
        # This matches original DecisionFlow which uses filtered triplets
        structure["attribute"] = []
        if objective_function and isinstance(objective_function, dict):
            filtered_pairs = objective_function.get('filtered_pairs', [])
            for pair in filtered_pairs:
                # Skip environment attributes (check if 'environment' is in the variable name)
                variable = pair.get('Variable', '')
                if 'environment' in variable.lower():
                    continue
                structure["attribute"].append({
                    "Variable": variable,
                    "Attribute": pair.get('Attribute', ''),
                    "Value": pair.get('Value', [])
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

        dialog = []
        if self.system_prompt_template is not None:
            system_prompt = call_with_coerced_args(
                self.system_prompt_template,
                {'structure': structure}
            )

            dialog.insert(0, DialogElement(role='system',
                                          content=system_prompt))

        # Express stage only requires structure (matches original DecisionFlow math_express)
        prompt = call_with_coerced_args(
            self.prompt_template,
            {'structure': structure},
        )

        dialog.append(DialogElement(role='user',
                                   content=prompt))

        # Note: output_schema not needed for unstructured inference
        # The schema is embedded in the prompt's output format instructions

        dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)
        log.info(f"**Express stage (unstructured) dialog prompt**: {dialog_prompt}")

        import time
        start_time = time.time()

        # Use unstructured inference for much faster generation
        # This bypasses JSON schema constraint checking during token generation
        expected_keys = ["Objective Function", "Decision Variables", "Constraints", "Explanation"]

        # Retry loop for unstructured inference with JSON parsing
        response = None
        last_error = None

        for attempt in range(self.max_json_retries):
            try:
                # Run unstructured inference (no schema constraints)
                raw_response = self.structured_inference_engine.run_inference_unstructured(dialog_prompt)

                log.debug(f"Express stage (unstructured) raw response (attempt {attempt + 1}): {raw_response[:200]}...")

                # Validate response
                response = validate_unstructured_response(
                    raw_response,
                    expected_keys=expected_keys,
                    use_string_fallback=True
                )

                # Success - break out of retry loop
                log.info(f"Express stage (unstructured) inference succeeded on attempt {attempt + 1}")
                break

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                last_error = e
                log.warning(
                    f"Express stage (unstructured) JSON extraction error on attempt {attempt + 1}/{self.max_json_retries}: {e}"
                )

                if attempt < self.max_json_retries - 1:
                    log.info("Retrying Express stage (unstructured) inference...")
                else:
                    log.error(f"Express stage (unstructured) failed after {self.max_json_retries} attempts")
                    raise SceneSkipException(
                        f"Failed to extract valid JSON after {self.max_json_retries} attempts. "
                        f"Last error: {last_error}",
                        component_name="ExpressStageUnstructuredComponent",
                        last_error=last_error
                    ) from last_error

        elapsed_time = time.time() - start_time
        log.info(f"**Express stage (unstructured) inference took {elapsed_time:.2f} seconds**")
        log.info(f"**Express stage (unstructured) response**: \n{response}")

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

        log.info(f"Express stage (unstructured) completed: Generated mathematical model with {len(objective_function)} objective functions, {len(decision_variables)} decision variables, {len(constraints)} constraints")
        return result
