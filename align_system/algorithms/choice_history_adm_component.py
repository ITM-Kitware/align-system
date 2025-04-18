import numpy as np

from align_system.utils import logging, call_with_coerced_args
from align_system.algorithms.abstracts import ADMComponent
from align_system.utils.alignment_utils import alignment_target_to_attribute_targets

log = logging.getLogger(__name__)


class ChoiceKDMAValueHistoryADMComponent(ADMComponent):
    def __init__(self,
                 attributes,
                 historical_context_fn):
        self.attributes = attributes

        self.historical_context_fn = historical_context_fn
        self.choice_history = {}

    def run_returns(self):
        return 'choice_history'

    def run(self,
            scenario_id,
            scenario_state,
            alignment_target,
            attribute_prediction_scores=None,
            chosen_choice=None):
        target_kdmas = alignment_target_to_attribute_targets(
            alignment_target,
            self.attributes)

        context = call_with_coerced_args(
            self.historical_context_fn,
            {'scenario_id': scenario_id,
             'scenario_state': scenario_state,
             'alignment_target': alignment_target})

        log.debug(f"Choice history context: {context}")

        context_history = self.choice_history.setdefault(context, {})

        for target_kdma in target_kdmas:
            context_history.setdefault(target_kdma.kdma, [])

        if chosen_choice is not None:
            # chosen_choice is optional, if not provided the history
            # is not updated, just returned
            for target_kdma in target_kdmas:
                context_history[target_kdma.kdma].append(
                    np.mean(attribute_prediction_scores[chosen_choice][target_kdma.kdma]))

        return context_history


class ScenarioIDHistoricalContextFunc:
    def __call__(self, scenario_id):
        return scenario_id
