from align_system.utils import logging
from align_system.algorithms.abstracts import ADMComponent

log = logging.getLogger(__name__)


class ObjectiveStageComponent(ADMComponent):
    """Objective stage component that filters weighted pairs and builds objective function.

    Following DecisionFlow reference: this stage does NOT make an LLM call.
    It simply filters Variable-Attribute pairs by weight threshold and
    constructs the objective function string programmatically.
    """

    def __init__(
        self,
        weight_threshold=0.3,
        attributes=None,
        **kwargs,
    ):
        self.weight_threshold = weight_threshold

        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def run_returns(self):
        return "objective_function"

    def run(self, filter_analysis=None, **kwargs):
        """Create objective function from Variable-Attribute pairs that exceed weight threshold.

        Following DecisionFlow reference: this stage does NOT make an LLM call.
        It filters pairs by weight threshold and builds the objective function string.
        """

        log.info("=" * 60)
        log.info("Objective Stage: Building objective function from filtered pairs")
        log.info("=" * 60)

        # Collect all Variable-Attribute pairs that exceed weight threshold
        filtered_pairs = []

        if filter_analysis:
            for kdma_name, kdma_data in filter_analysis.items():
                weighted_pairs = kdma_data.get('weighted_pairs', [])

                for pair in weighted_pairs:
                    weight = pair.get('Weight', 0)

                    if weight > self.weight_threshold:
                        log.info(f"Including: {pair.get('Variable', '')[:30]}... - {pair.get('Attribute', '')} (weight={weight})")
                        filtered_pairs.append({
                            "Variable": pair.get('Variable', ''),
                            "Attribute": pair.get('Attribute', ''),
                            "Value": pair.get('Value', []),
                            "Weight": weight,
                            "Explanation": pair.get('Explanation', ''),
                            "KDMA": kdma_name
                        })
                    else:
                        log.info(f"Excluding: {pair.get('Variable', '')[:30]}... - {pair.get('Attribute', '')} (weight={weight} < {self.weight_threshold})")

        log.info(f"Total filtered pairs admitted to objective: {len(filtered_pairs)}")

        # Create objective function string from filtered pairs
        # Format: weight * (attribute) of (variable) + ...
        objective_function_text = "The final formula to be calculated is "
        if filtered_pairs:
            for pair in filtered_pairs:
                variable = pair["Variable"]
                attribute = pair["Attribute"]
                weight = pair["Weight"]
                objective_function_text += f"{weight} * ({attribute}) of ({variable}) + "
            # Remove trailing " + "
            objective_function_text = objective_function_text.rstrip(" + ")
        else:
            # Fallback if no components meet threshold
            objective_function_text = "weight * attribute of variable"

        log.info(f"Generated objective function: {objective_function_text}")
        log.info("Objective Stage Completed (no LLM call - matches DecisionFlow reference)")

        return {
            'objective_function': objective_function_text,
            'filtered_pairs': filtered_pairs,
            'weight_threshold_used': self.weight_threshold
        }
