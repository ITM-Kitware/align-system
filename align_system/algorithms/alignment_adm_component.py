import math
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


class MedicalOnlyAlignmentADMComponent(ADMComponent):
    def run_returns(self):
        return ('chosen_choice', 'best_sample_idx')

    def run(
        self,
        attribute_prediction_scores,
    ):
        """
        Always choose the medically needy patient (random if tie)

        attribute_prediction_scores: dict[str, dict[str, float | list[float]]]
            Dictionary of choices mapped to KMDA value predictions, including medical
            urgency prediction
        """
        med_urg_str = "medical"

        def _handle_single_value(predictions):
            if not isinstance(predictions, list):
                return [predictions]
            return predictions

        med_urg = {}
        for choice, all_kdma_predictions in attribute_prediction_scores.items():
            # Get medical urgency
            if med_urg_str not in all_kdma_predictions:
                raise RuntimeError("Medical Urgency predictions required for this alignment function")
            medical_urgency_preds = _handle_single_value(all_kdma_predictions[med_urg_str])
            med_urg[choice] = sum(medical_urgency_preds) / len(medical_urgency_preds)

        # Get max urgency
        max_urg = max(med_urg.values())
        log.info(f"Max medical urgency: {max_urg}")

        max_keys = [key for key, value in med_urg.items() if math.isclose(value, max_urg)]
        log.info(f"Max medical urgency keys: {max_keys}")

        if len(max_keys) > 1:  # tie, choose randomly
            log.explain("Multiple patients predicted to have same medical need, randomly choosing")
            selected_choice = random.choice(max_keys)
        else:
            selected_choice = max_keys[0]

        def _get_best_sample_idx(average_urgency, samples):
            """ Return medical urgency prediction closest to average """
            med_urg_preds = _handle_single_value(samples[med_urg_str])

            best_idx = 0
            min_difference = abs(med_urg_preds[best_idx] - average_urgency)

            for idx, sample in enumerate(med_urg_preds):
                difference = abs(sample - average_urgency)
                if difference < min_difference:
                    min_difference = difference
                    best_idx = idx

            return best_idx

        return (selected_choice, _get_best_sample_idx(max_urg, attribute_prediction_scores[selected_choice]))


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

    def _midpoint_eqn(self, kdma, opt_a_value, medical_delta, attr_delta):
        medical_weights = {"affiliation": 2, "merit": 4}
        medical_weight = medical_weights.get(kdma, 1.0)

        # Midpoint equation from ADEPT
        if kdma == "affiliation":
            return (opt_a_value + medical_weight*medical_delta - opt_a_value*medical_delta)/2
        elif kdma == "merit":
            return (opt_a_value + medical_weight*medical_delta)/4
        else:
            return 0.5 + (medical_weight*medical_delta - attr_delta)/2

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
                if kdma not in all_predictions:
                    continue
                preds = _handle_single_value(all_predictions[kdma])
                pred_dict_out[kdma] = sum(preds) / len(preds)

            if med_urg_str in all_predictions:
                preds = _handle_single_value(all_predictions[med_urg_str])
                pred_dict_out[med_urg_str] = sum(preds) / len(preds)

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
            probe_relevance = _get_avg_pred(attribute_relevance)

        # Sort by medical urgency (descending)
        predictions.sort(key=lambda pred: pred[med_urg_str], reverse=True)

        # Guaranteed to only have 2 choices at this point due to earlier checks
        opt_a, opt_b = predictions

        medical_delta = opt_a[med_urg_str] - opt_b[med_urg_str]

        # TODO: Figure out what it means to be the best prediction for this alignment function
        best_sample_idx = 0

        # Compute midpoint per attribute
        attribute_weights = {}
        attribute_deltas = {}
        attribute_midpoints = {}
        for target_kdma in target_kdmas:
            kdma = target_kdma["kdma"]
            attribute_weights[kdma] = probe_relevance.get(kdma, 1.0)

            # May not have predictions for this KDMA if it had 0 relevance
            if math.isclose(attribute_weights[kdma], 0.):
                continue

            attribute_deltas[kdma] = opt_b["kdmas"][kdma] - opt_a["kdmas"][kdma]
            opt_a_value = opt_a["kdmas"][kdma]
            attribute_midpoints[kdma] = self._midpoint_eqn(kdma, opt_a_value, medical_delta, attribute_deltas[kdma])
            log.info(f"{kdma} midpoint: {attribute_midpoints[kdma]}")

        votes = {idx: 0 for idx in range(2)}
        for target_kdma in target_kdmas:
            kdma = target_kdma["kdma"]
            if math.isclose(attribute_weights[kdma], 0.):  # don't consider attributes with 0 weight
                log.info(f"{kdma}: Removing from consideration, 0 weight")
                continue

            vote_weight = attribute_weights[kdma]
            attr_delta = attribute_deltas[kdma]
            if math.isclose(medical_delta, 0.):
                if math.isclose(attr_delta, 0.):  # patient is same medically and attribute-wise, don't vote
                    log.info(f"{kdma}: Patients are tied both medically and attribute-wise, not voting")
                    continue
                elif attr_delta > 0:
                    votes[1] += vote_weight
                else:
                    votes[0] += vote_weight
                log.info(f"{kdma}: Patients are tied medically, voting for attribute-worthy")
            elif attr_delta < 0 or math.isclose(attr_delta, 0.):  # same patient is medically and attribute worthy
                log.info(f"{kdma}: Voting for patient that is both medically and attribute-worthy")
                votes[0] += vote_weight
            else:
                attr_target = target_kdma["value"]
                attr_midpoint = attribute_midpoints[kdma]
                if math.isclose(attr_target, attr_midpoint):  # Midpoint == target, tie
                    log.info(f"{kdma}: Target is exactly midpoint, not voting")
                    continue
                elif attr_target < attr_midpoint:
                    log.info(f"{kdma}: Target is less than midpoint, voting for medically-worthy.")
                    votes[0] += vote_weight
                else:  # attr_target > attr_midpoint
                    log.info(f"{kdma}: Target is greater than midpoint, voting for attribute-worthy")
                    votes[1] += vote_weight

        log.explain(votes)

        max_votes = max(votes.values())
        max_keys = [key for key, value in votes.items() if math.isclose(value, max_votes)]
        log.info(f"Max vote keys: {max_keys}")

        if len(max_keys) > 1:  # tie, choose randomly
            log.explain("Patients predicted to have same attribute worthiness, randomly choosing")
            return (random.choice([predictions[key]["choice"] for key in max_keys]), best_sample_idx)
        else:
            return (predictions[max_keys[0]]["choice"], best_sample_idx)
