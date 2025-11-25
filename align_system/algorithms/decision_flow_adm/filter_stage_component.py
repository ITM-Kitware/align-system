import json
from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import attributes_in_alignment_target
from align_system.data_models.dialog import DialogElement
from align_system.algorithms.decision_flow_adm.utils import validate_structured_response
from align_system.exceptions import SceneSkipException

log = logging.getLogger(__name__)


class FilterStageComponent(ADMComponent):
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
        return "filter_analysis"

    def run(self, scenario_state, choices, attribute_analysis=None, variables=None, extraction=None, alignment_target=None, **kwargs):
        """Filter and weight attributes based on relevance to target bias.

        Following DecisionFlow reference: each Attribute is evaluated individually
        against the target bias. The weight is then applied to all Variable-Attribute
        pairs containing that attribute.
        """

        # Handle alignment_target workflow similar to attribute_stage_component
        if alignment_target is None:
            # No alignment target - use all attributes
            target_attributes = list(self.attributes.values())
        else:
            # Alignment target provided - ONLY use attributes in the alignment target
            target_attribute_names = attributes_in_alignment_target(alignment_target)
            target_attributes = [self.attributes[n] for n in target_attribute_names if n in self.attributes]

        # filter_results will contain weighted Variable-Attribute pairs organized by KDMA
        filter_results = {}

        # Process each KDMA's attribute analysis
        for target_attr in target_attributes:
            kdma_name = target_attr.name
            # Get the attribute analysis results for this KDMA
            attribute_data = attribute_analysis.get(kdma_name, []) if attribute_analysis else []

            if not attribute_data:
                log.info(f"No attribute data for {kdma_name}, skipping")
                continue

            log.info("=" * 60)
            log.info(f"Filter Stage: Processing KDMA '{kdma_name}'")
            log.info("=" * 60)

            # Collect unique attribute names across all variables
            unique_attributes = set()
            for var_entry in attribute_data:
                if not isinstance(var_entry, dict):
                    continue
                if var_entry.get("Variable") == "Environment":
                    continue
                for attr_entry in var_entry.get("Attribute", []):
                    if isinstance(attr_entry, dict) and attr_entry.get("Attribute"):
                        unique_attributes.add(attr_entry.get("Attribute"))

            log.info(f"Found {len(unique_attributes)} unique attributes to evaluate: {unique_attributes}")

            # Evaluate each unique attribute against target bias (cache weights)
            attribute_weights = {}
            for attr_name in unique_attributes:
                log.info(f"Filtering attribute: {attr_name}")

                dialog = []
                if self.system_prompt_template is not None:
                    system_prompt = call_with_coerced_args(
                        self.system_prompt_template,
                        {'target_attribute': target_attr}
                    )
                    dialog.insert(0, DialogElement(role='system', content=system_prompt))

                prompt = call_with_coerced_args(
                    self.prompt_template,
                    {
                        'attribute_name': attr_name,
                        'target_bias': target_attr
                    },
                )

                dialog.append(DialogElement(role='user', content=prompt))

                output_schema = call_with_coerced_args(
                    self.output_schema_template,
                    {})

                dialog_prompt = self.structured_inference_engine.dialog_to_prompt(dialog)

                # Retry loop for structured inference with validation
                response = None
                last_error = None

                for attempt in range(self.max_json_retries):
                    try:
                        raw_response = self.structured_inference_engine.run_inference(
                            dialog_prompt,
                            output_schema
                        )
                        response = validate_structured_response(raw_response)
                        log.info(f"Filter inference succeeded for '{attr_name}' on attempt {attempt + 1}")
                        break

                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        last_error = e
                        log.warning(
                            f"Filter JSON decode error for '{attr_name}' on attempt {attempt + 1}/{self.max_json_retries}: {e}"
                        )
                        if attempt < self.max_json_retries - 1:
                            log.info(f"Retrying filter inference for '{attr_name}'...")
                        else:
                            log.error(f"Filter failed for '{attr_name}' after {self.max_json_retries} attempts")
                            raise SceneSkipException(
                                f"Failed to generate valid JSON after {self.max_json_retries} attempts for '{attr_name}'. "
                                f"Last error: {last_error}",
                                component_name="FilterStageComponent",
                                last_error=last_error
                            ) from last_error

                weight = response.get('Weight', 0)
                explanation = response.get('Explanation', '')

                log.info(f"  -> Weight: {weight}, Explanation: {explanation[:100]}...")

                # Cache the weight for this attribute
                attribute_weights[attr_name] = {
                    'Weight': weight,
                    'Explanation': explanation
                }

            # Apply cached weights to all Variable-Attribute pairs
            weighted_pairs = []
            for var_entry in attribute_data:
                if not isinstance(var_entry, dict):
                    continue

                variable_name = var_entry.get("Variable", "")
                if not variable_name or variable_name == "Environment":
                    continue

                for attr_entry in var_entry.get("Attribute", []):
                    if not isinstance(attr_entry, dict):
                        continue

                    attr_name = attr_entry.get("Attribute", "")
                    attr_values = attr_entry.get("Value", [])

                    if attr_name and attr_name in attribute_weights:
                        weighted_pairs.append({
                            "Variable": variable_name,
                            "Attribute": attr_name,
                            "Value": attr_values,
                            "Weight": attribute_weights[attr_name]['Weight'],
                            "Explanation": attribute_weights[attr_name]['Explanation']
                        })

            # Store all weighted pairs for this KDMA
            filter_results[kdma_name] = {
                'weighted_pairs': weighted_pairs,
                'target_bias': target_attr.name
            }

            log.info(f"Filter analysis for '{kdma_name}' completed: {len(unique_attributes)} attributes evaluated, {len(weighted_pairs)} pairs created")

        return filter_results
