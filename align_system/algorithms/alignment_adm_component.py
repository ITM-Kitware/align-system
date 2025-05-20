import random

from align_system.utils import call_with_coerced_args, logging
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import alignment_target_to_attribute_targets

log = logging.getLogger(__name__)

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


class MedicalUrgencyAlignmentADMComponent(ADMComponent):
    def __init__(
        self,
        attributes=None
    ):
        if attributes is None:
            attributes = {}
        self.attributes = attributes

    def run_returns(self):
        return ('chosen_choice', 'best_sample_idx')

    def run(
        self,
        attribute_prediction_scores,
        alignment_target,
    ):
        """ Align based on medical urgency/KDMA tradeoff """

        med_urg_str  = "medical urgency"

        if alignment_target is None:
            raise RuntimeError(
                "Assumption violated: `alignment_target` was None"
            )

        target_kdmas = alignment_target_to_attribute_targets(
            alignment_target,
            self.attributes)
        target_kdmas = [dict(t) for t in target_kdmas]
        if len(target_kdmas) > 1:
            raise NotImplementedError("Multi-kdma alignment has not yet been implemented")

        def _handle_single_value(predictions):
            if not isinstance(predictions, list):
                return [predictions]
            return predictions

        choices = list(attribute_prediction_scores.keys())
        if len(choices) != 2:
            raise NotImplementedError("This alignment function has not yet been implemented for !=2 choices")

        # Compute averages of predicted values
        predictions = []
        for choice, all_kdma_predictions in attribute_prediction_scores.items():
            pred_dict = { "choice": choice }

            # Get medical urgency
            if med_urg_str not in all_kdma_predictions:
                raise RuntimeError("Medical Urgency predictions required for this alignment function")
            medical_urgency_preds = _handle_single_value(all_kdma_predictions[med_urg_str])
            pred_dict[med_urg_str] = sum(medical_urgency_preds) / len(medical_urgency_preds)

            # Get KDMA predictions relevant to target
            kdma_pred_dict = {}
            for target_kdma in target_kdmas:
                kdma = target_kdma["kdma"]
                kdma_preds = _handle_single_value(all_kdma_predictions[kdma])
                kdma_pred_dict[kdma] = sum(kdma_preds) / len(kdma_preds)
            pred_dict["kdmas"] = kdma_pred_dict

            predictions.append(pred_dict)

        # Sort by medical urgency (descending)
        predictions.sort(key=lambda pred: pred[med_urg_str], reverse=True)

        # Guaranteed to only have 2 choices at this point due to earlier checks
        opt_a, opt_b = predictions[0], predictions[1]
        medical_delta = opt_a[med_urg_str] - opt_b[med_urg_str]

        # TODO: Figure out what it means to be the best prediction for this alignment function
        best_sample_idx = 0

        # Guaranteed to only have 1 attribute due to earlier checks
        attr = target_kdmas[0]["kdma"]
        attribute_delta = opt_a["kdmas"][attr] - opt_b["kdmas"][attr]

        # Special conditions
        if medical_delta == 0:
            log.explain(
                "Patients predicted to have same medical urgency. Choosing attribute-worthy patient"
            )

            # Exact same patient, medically and attribute-wise
            if attribute_delta == 0:
                log.explain("Patients predicted to have same attribute value, randomly choosing")
                return (random.choice([opt_a["choice"], opt_b["choice"]]), best_sample_idx)

            return (opt_a["choice"] if attribute_delta > 0 else opt_b["choice"], best_sample_idx)
        elif attribute_delta >= 0:
            log.explain(
                f"Choice ({opt_a['choice']}) is higher in both medical AND attribute, ignoring midpoint equation"
            )
            return (opt_a["choice"], best_sample_idx)

        # Equation from ADEPT
        probe_midpoint = 0.5 + (medical_delta - attribute_delta)/2
        log.info(f"Probe Midpoint: {probe_midpoint}")

        attr_target = target_kdmas[0]["value"]
        if attr_target < probe_midpoint:
            log.explain("Target is below midpoint, choosing medically urgent patient")
            return (opt_a["choice"], best_sample_idx)
        elif attr_target > probe_midpoint:
            log.explain("Target is above midpoint, choosing attribute-worthy patient")
            return (opt_b["choice"], best_sample_idx)
        else:  # Midpoint == target, choose randomly
            log.explain("Target is exactly midpoint, randomly choosing")
            return (random.choice([opt_a["choice"], opt_b["choice"]]), best_sample_idx)
