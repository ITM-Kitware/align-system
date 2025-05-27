import math

from rich.highlighter import JSONHighlighter

from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import adm_utils, logging
from align_system.utils.alignment_utils import attributes_in_alignment_target

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()


class EnsureChosenActionADMComponent(ADMComponent):
    def run_returns(self):
        return ('chosen_action')

    def run(self,
            choices,
            actions,
            chosen_choice=None,
            chosen_action=None,
            justification=None):
        if chosen_choice is None and chosen_action is None:
            raise RuntimeError("Expecting at least one of 'chosen_action' or "
                               "'chosen_choice'")

        if chosen_action is None:
            chosen_choice_idx = choices.index(chosen_choice)
            chosen_action = actions[chosen_choice_idx]

        if (hasattr(chosen_action, 'justification')
                and chosen_action.justification is None
                and justification is not None):
            chosen_action.justification = justification

        return chosen_action


class ITMFormatChoicesADMComponent(ADMComponent):
    def run_returns(self):
        return ('choices')

    def run(self, scenario_state, actions):
        choices = adm_utils.format_choices(
            [a.unstructured for a in actions],
            actions,
            scenario_state)

        return choices


class JustificationFromReasonings(ADMComponent):
    def run_returns(self):
        return 'justification'

    def run(self,
            attribute_prediction_reasonings,
            chosen_choice,
            best_sample_idx):
        # Expecting that for `attribute_prediction_reasonings` for
        # each possible choice we have a dictionary of KDMA to
        # justification; just concatenating these together for now to
        # take into account all KDMAs
        best_sample_reasonings = []
        for attribute, reasonings in attribute_prediction_reasonings[chosen_choice].items():
            best_sample_reasonings.append(reasonings[best_sample_idx])

        return "\n".join(best_sample_reasonings)


class PopulateChoiceInfo(ADMComponent):
    def run_returns(self):
        return 'choice_info'

    def run(self,
            choices,
            actions,
            alignment_target=None,
            attribute_prediction_scores=None,
            attribute_relevance_binary=None):
        choice_info = {}

        if alignment_target is None:
            target_kdmas = []
        else:
            target_kdmas = attributes_in_alignment_target(alignment_target)

        if attribute_prediction_scores is not None:
            choice_info['predicted_kdma_values'] = attribute_prediction_scores

        if attribute_relevance_binary is not None:
            choice_info['predicted_relevance'] = attribute_relevance_binary

        true_kdma_values = {}
        true_relevance = {}
        for choice, action in zip(choices, actions):
            if action.kdma_association is not None:
                true_kdma_values[choice] = action.kdma_association

                for kdma in target_kdmas:
                    true_relevance[choice] = 1 if kdma in action.kdma_association else 0

        if len(true_kdma_values) > 0:
            choice_info['true_kdma_values'] = true_kdma_values

        if len(true_relevance) > 0:
            choice_info['true_relevance'] = true_relevance

        return choice_info


class Phase2RegressionRuleBasedCorrection(ADMComponent):
    def run_returns(self):
        return 'attribute_prediction_scores'

    def run(self,
            attribute_prediction_scores
            ):
        # Corrects `attribute_prediction_scores` based on specific rules

        log.info("[bold]*CORRECTING KDMA SCORE PREDICTION BASED ON RULES*[/bold]", extra={"markup": True})
        log.info("Predicted:{}".format(attribute_prediction_scores), extra={"highlighter": JSON_HIGHLIGHTER})

        # Rule: For Search vs. Stay attribute value for stay should be 0
        if 'Stay with the current patient' in attribute_prediction_scores:
            if 'Search or Stay' in attribute_prediction_scores['Stay with the current patient']:
                attribute_prediction_scores['Stay with the current patient']['Search or Stay'] = [0.0]
        # Rule: For Search vs. Stay the medical value for the unknown patient should be 0
        if 'Go to the next room' in attribute_prediction_scores:
            attribute_prediction_scores['Go to the next room']['medical urgency'] = [0.0]
        # Rule: For personal safety, waiting is risk 0 or safety 1
        if 'Wait' in attribute_prediction_scores:
            if 'Personal Safety' in attribute_prediction_scores['Wait']:
                attribute_prediction_scores['Wait']['Personal Safety'] = [1.0]
        # Rule: For personal safety, waiting is medical value 0
        if 'Wait' in attribute_prediction_scores:
           attribute_prediction_scores['Wait']['medical urgency'] = [0.0]

        log.info("Corrected:{}".format(attribute_prediction_scores), extra={"highlighter": JSON_HIGHLIGHTER})

        return attribute_prediction_scores


class OracleRelevance(ADMComponent):
    def run_returns(self):
        return 'attribute_relevance'

    def run(self, actions, alignment_target=None):
        ''' Returns groundtruth (binary) `attribute_relevance` '''
        if alignment_target is None:
            target_attribute_names = []
        else:
            target_attribute_names = attributes_in_alignment_target(alignment_target)

        attribute_relevance = {}
        for action in actions:
            if hasattr(action, 'kdma_association') and action.kdma_association is not None:
                for kdma in action.kdma_association.keys():
                    # If no target, just add all GT values. Otherwise only add those relevant to target
                    if alignment_target is None or kdma in target_attribute_names:
                        attribute_relevance[kdma] = 1.0

        for target_attribute in target_attribute_names:
            if target_attribute not in attribute_relevance:
                attribute_relevance[target_attribute] = 0.0

        log.info("[bold]*GROUND TRUTH KDMA RELEVANCE*[/bold]", extra={"markup": True})
        log.info("{}".format(attribute_relevance), extra={"highlighter": JSON_HIGHLIGHTER})

        return attribute_relevance


class OracleRegression(ADMComponent):
    def run_returns(self):
        return 'attribute_prediction_scores'

    def run(self,
            actions):
        # Returns ground truth `attribute_prediction_scores`
        attribute_prediction_scores = {}
        for action in actions:
            if hasattr(action, 'kdma_association') and action.kdma_association is not None:
                attribute_prediction_scores[action.unstructured] = action.kdma_association.copy()

        log.info("[bold]*GROUND TRUTH KDMA SCORES*[/bold]", extra={"markup": True})
        log.info("{}".format(attribute_prediction_scores), extra={"highlighter": JSON_HIGHLIGHTER})

        return attribute_prediction_scores


class OracleJustification(ADMComponent):
    def run_returns(self):
        return 'justification'

    def run(self):
        return "Looked at scores."


class ChoiceRelevanceToProbeRelevance(ADMComponent):
    def __init__(self, binarize=True):
        self.binarize = binarize

    def run_returns(self):
        return 'attribute_relevance'

    def run(self, relevance_prediction_scores):
        """
        Aggregate choice-level relevance values to probe-level
        relevance

        TODO (maybe): parameterize aggregation function, hardcoded to
        take the average for now
        """
        _attribute_relevance_values = {}
        for choice, relevance_preds in relevance_prediction_scores.items():
            for attribute, values in relevance_preds.items():
                _attribute_relevance_values.setdefault(attribute, []).extend(values)

        attribute_relevance = {}
        max_relevance = -math.inf
        for attribute, values in _attribute_relevance_values.items():
            agg_relevance = sum(values) / len(values)
            attribute_relevance[attribute] = agg_relevance

            if agg_relevance > max_relevance:
                max_relevance = agg_relevance

        if self.binarize:
            log.debug("[bold]*AGGREGATE RELEVANCE (BEFORE BINARIZATION)*[/bold]",
                      extra={"markup": True})
        else:
            log.debug("[bold]*AGGREGATE RELEVANCE*[/bold]",
                      extra={"markup": True})

        log.debug(attribute_relevance, extra={"highlighter": JSON_HIGHLIGHTER})

        if self.binarize:
            for attribute, relevance in attribute_relevance.items():
                if relevance == max_relevance:
                    attribute_relevance[attribute] = 1.0
                    log.debug("[bold]*MOST RELEVANT ATTRIBUTE: {}*[/bold]".format(attribute),
                              extra={"markup": True})
                else:
                    attribute_relevance[attribute] = 0.0

        return attribute_relevance
