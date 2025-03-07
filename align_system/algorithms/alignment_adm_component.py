from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import alignment_target_to_attribute_targets


class AlignmentADMComponent(ADMComponent):
    def __init__(self,
                 alignment_function,
                 attributes={}):
        self.alignment_function = alignment_function
        self.attributes = attributes

    def run(self,
            scenario_state,
            choice_evaluation,
            dialogs,
            alignment_target=None):
        if alignment_target is None:
            raise RuntimeError("Assumption violated: `alignment_target` "
                               "was None")

        target_kdmas = alignment_target_to_attribute_targets(
            alignment_target,
            self.attributes)

        # Alignment function below is expecting target_kdmas to be dicts
        target_kdmas = [dict(t) for t in target_kdmas]

        predicted_kdma_values =\
            {k: v['kdma_prediction_scores']
             for k, v in choice_evaluation.items()}

        selected_choice, probs = self.alignment_function(
            predicted_kdma_values,
            target_kdmas)

        best_sample_idx = self.alignment_function.get_best_sample_index(
            predicted_kdma_values,
            target_kdmas,
            selected_choice)

        for choice, choice_eval in choice_evaluation.items():
            if choice == selected_choice:
                choice_eval['chosen'] = True
                choice_eval['best_sample_idx'] = best_sample_idx
            else:
                choice_eval['chosen'] = False

        return choice_evaluation, dialogs
