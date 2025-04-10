from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import alignment_target_to_attribute_targets


class AlignmentADMComponent(ADMComponent):
    def __init__(self,
                 alignment_function,
                 attributes={}):
        self.alignment_function = alignment_function
        self.attributes = attributes

    def run_returns(self):
        return ('chosen_choice',
                'best_sample_idx')

    def run(self,
            attribute_prediction_scores,
            alignment_target):
        if alignment_target is None:
            raise RuntimeError("Assumption violated: `alignment_target` "
                               "was None")

        target_kdmas = alignment_target_to_attribute_targets(
            alignment_target,
            self.attributes)

        # Alignment function below is expecting target_kdmas to be dicts
        target_kdmas = [dict(t) for t in target_kdmas]

        selected_choice, probs = self.alignment_function(
            attribute_prediction_scores,
            target_kdmas)

        best_sample_idx = self.alignment_function.get_best_sample_index(
            attribute_prediction_scores,
            target_kdmas,
            selected_choice)

        return selected_choice, best_sample_idx
