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
        # Midpoint equation from ADEPT
        return 0.5 + (medical_delta - attribute_delta)/total_weight

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
        alignment_target: alignment target info
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

        # Compute midpoint per attribute
        attribute_weights = {}
        attribute_deltas = {}
        attribute_midpoints = {}
        for target_kdma in target_kdmas:
            kdma = target_kdma["kdma"]
            attribute_weights[kdma] = probe_relevance[kdma] if kdma in probe_relevance else 1.0
            attribute_deltas[kdma] = attribute_weights[kdma] * (opt_b["kdmas"][kdma] - opt_a["kdmas"][kdma])
            attribute_midpoints[kdma] = self._midpoint_eqn(
                medical_delta, attribute_deltas[kdma], total_weight=(medical_weight + attribute_weights[kdma])
            )
            log.info(f"{kdma} midpoint: {attribute_midpoints[kdma]}")

        votes = {idx: 0 for idx in range(2)}
        for target_kdma in target_kdmas:
            kdma = target_kdma["kdma"]
            if attribute_weights[kdma] == 0:  # don't consider attributes with 0 weight
                log.info(f"{kdma}: Removing from consideration, 0 weight")
                continue

            attr_delta = attribute_deltas[kdma]
            if medical_delta == 0:
                if attr_delta == 0:  # patient is same medically and attribute-wise, don't vote
                    log.info(f"{kdma}: Patients are tied both medically and attribute-wise")
                    continue
                elif attr_delta > 0:
                    votes[1] += 1
                else:
                    votes[0] += 1
                log.info(f"{kdma}: Patients are tied medically, choosing attribute-worthy")
            elif attr_delta < 0:  # same patient is medically and attribute worthy
                log.info(f"{kdma}: Same patient is both medically and attribute-worthy")
                votes[0] += 1
            else:
                attr_target = target_kdma["value"]
                attr_midpoint = attribute_midpoints[kdma]
                if attr_target < attr_midpoint:
                    log.info(f"{kdma}: Target is less than midpoint")
                    votes[0] += 1
                elif attr_target > attr_midpoint:
                    log.info(f"{kdma}: Target is greater than midpoint")
                    votes[1] += 1
                else:  # Midpoint == target, tie
                    log.info(f"{kdma}: Target is exactly midpoint")
                    continue

        log.explain(votes)

        max_votes = max(votes.values())
        max_keys = [key for key, value in votes.items() if value == max_votes]
        log.info(f"Max vote keys: {max_keys}")

        if len(max_keys) > 1:  # tie, choose randomly
            log.explain("Patients predicted to have same attribute worthiness, randomly choosing")
            return (random.choice([predictions[key]["choice"] for key in max_keys]), best_sample_idx)
        else:
            return (predictions[max_keys[0]]["choice"], best_sample_idx)
