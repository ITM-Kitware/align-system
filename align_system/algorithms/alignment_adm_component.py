from align_system.utils import call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import alignment_target_to_attribute_targets


class AlignmentADMComponent(ADMComponent):
    def __init__(self,
                 alignment_function,
                 attributes=None):
        self.alignment_function = alignment_function

        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def run_returns(self):
        return ('chosen_choice',
                'best_sample_idx')

    def run(self,
            scenario_state,
            attribute_prediction_scores,
            alignment_target,
            attribute_relevance_binary=None,
            choice_history=None):
        if alignment_target is None:
            raise RuntimeError("Assumption violated: `alignment_target` "
                               "was None")

        target_kdmas = alignment_target_to_attribute_targets(
            alignment_target,
            self.attributes)

        # Alignment function below is expecting target_kdmas to be dicts
        target_kdmas = [dict(t) for t in target_kdmas]

        selected_choice, probs = call_with_coerced_args(
            self.alignment_function,
            {'kdma_values': attribute_prediction_scores,
             'relevances': attribute_relevance_binary,
             'target_kdmas': target_kdmas,
             'choice_history': choice_history})

        best_sample_idx = self.alignment_function.get_best_sample_index(
            attribute_prediction_scores,
            target_kdmas,
            selected_choice)

        return selected_choice, best_sample_idx
