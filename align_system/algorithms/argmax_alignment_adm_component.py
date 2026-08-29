from __future__ import annotations

from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import logging

log = logging.getLogger(__name__)


class ArgmaxAlignmentADMComponent(ADMComponent):
    """
    Alignment step that picks the choice with the highest predicted
    KDMA score, averaged across all attributes and samples.

    Replaces the ADEPT random effects model for domains (like AI2Thor)
    where no calibrated statistical model exists.  For single-attribute
    pipelines this reduces to a plain argmax over the LLM's scores.
    """

    def __init__(self, attributes=None):
        self.attributes = attributes or {}

    def run_returns(self):
        return ("chosen_choice", "best_sample_idx", "alignment_info")

    def run(self, attribute_prediction_scores, alignment_target=None):
        """
        attribute_prediction_scores: dict[choice_str, dict[kdma, list[float]]]
        """
        choice_totals: dict[str, float] = {}

        for choice, attr_scores in attribute_prediction_scores.items():
            total = 0.0
            count = 0
            for kdma, scores in attr_scores.items():
                vals = scores if isinstance(scores, list) else [scores]
                if vals:
                    total += sum(vals) / len(vals)
                    count += 1
            choice_totals[choice] = total / count if count else 0.0

        best_choice = max(choice_totals, key=choice_totals.get)

        log.info(f"[ArgmaxAlignment] scores: {choice_totals}")
        log.info(f"[ArgmaxAlignment] chosen: {best_choice}")

        alignment_info = {
            "source": type(self).__name__,
            "choice_scores": choice_totals,
        }

        return best_choice, 0, alignment_info
