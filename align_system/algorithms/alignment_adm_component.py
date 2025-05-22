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

    def _midpoint_eqn(self, medical_delta, attribute_delta, total_weight=2):
        # Adding attribute_delta instead of subtracting because we factored out the negative when summing across KDMAs
        return 0.5 + (medical_delta + attribute_delta)/total_weight

    def run(
        self,
        attribute_prediction_scores,
        alignment_target,
        attribute_relevance=None,
    ):
        """
        Align based on medical urgency/KDMA tradeoff

        attribute_prediction_scores: dict[str, dict[str, float | list[float]]]
            Dictionary of choices mapped to KMDA value predictions, including medical
            urgency prediction
        alignment_target: list of dictionaries of alignment target info
        attribute_relevance: dict[str, float | list[float]]
            Dictionary of probe level KDMA relevance predictions
        """
        med_urg_str = "medical"

        if alignment_target is None:
            raise RuntimeError(
                "Assumption violated: `alignment_target` was None"
            )

        target_kdmas = alignment_target_to_attribute_targets(
            alignment_target,
            self.attributes)
        target_kdmas = [dict(t) for t in target_kdmas]

        def _handle_single_value(predictions):
            if not isinstance(predictions, list):
                return [predictions]
            return predictions

        # Take a dictionary of predictions (with KDMAs as key) and return average values per KDMA
        def _get_avg_pred(all_predictions):
            pred_dict_out = {}
            for target_kdma in target_kdmas:
                kdma = target_kdma["kdma"]
                preds = _handle_single_value(all_predictions[kdma])
                pred_dict_out[kdma] = sum(preds) / len(preds)

            return pred_dict_out

        choices = list(attribute_prediction_scores.keys())
        if len(choices) != 2:
            raise NotImplementedError("This alignment function has not yet been implemented for !=2 choices")

        # Compute averages of predicted values
        predictions = []
        for choice, all_kdma_predictions in attribute_prediction_scores.items():
            pred_dict = {"choice": choice}

            # Get medical urgency
            if med_urg_str not in all_kdma_predictions:
                raise RuntimeError("Medical Urgency predictions required for this alignment function")
            medical_urgency_preds = _handle_single_value(all_kdma_predictions[med_urg_str])
            pred_dict[med_urg_str] = sum(medical_urgency_preds) / len(medical_urgency_preds)

            # Get KDMA predictions relevant to target
            pred_dict["kdmas"] = _get_avg_pred(all_kdma_predictions)

            predictions.append(pred_dict)

        # Get relevance predictions relevant to target
        probe_relevance = {}
        if attribute_relevance is not None:
            probe_relevance = _get_avg_pred(attribute_relevance[choice])

        # Sort by medical urgency (descending)
        predictions.sort(key=lambda pred: pred[med_urg_str], reverse=True)

        # Guaranteed to only have 2 choices at this point due to earlier checks
        opt_a, opt_b = predictions
        medical_weight = probe_relevance[med_urg_str] if med_urg_str in probe_relevance else 1.0
        medical_delta = medical_weight * (opt_a[med_urg_str] - opt_b[med_urg_str])

        # TODO: Figure out what it means to be the best prediction for this alignment function
        best_sample_idx = 0

        # Sum up attribute tradeoff (factored out negative)
        total_weight = medical_weight
        attribute_delta = 0
        for target_kdma in target_kdmas:
            kdma = target_kdma["kdma"]
            attr_weight = probe_relevance[kdma] if kdma in probe_relevance else 1.0
            total_weight += attr_weight
            attribute_delta -= attr_weight * (opt_b["kdmas"][kdma] - opt_a["kdmas"][kdma])

        # Special conditions
        if medical_delta == 0:
            log.explain(
                "Patients predicted to have same medical urgency. Choosing attribute-worthy patient"
            )

            # Exact same patient, medically and attribute-wise
            if attribute_delta == 0:
                log.explain("Patients predicted to have same attribute value, randomly choosing")
                return (random.choice([opt_a["choice"], opt_b["choice"]]), best_sample_idx)

            return (opt_b["choice"] if attribute_delta < 0 else opt_a["choice"], best_sample_idx)
        elif attribute_delta >= 0:  # Positive because we factored out the negative when summing across KDMAs
            log.explain(
                f"Choice ({opt_a['choice']}) is higher in both medical AND attribute, ignoring midpoint equation"
            )
            return (opt_a["choice"], best_sample_idx)

        # Equation from ADEPT
        probe_midpoint = self._midpoint_eqn(medical_delta, attribute_delta, total_weight)
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
