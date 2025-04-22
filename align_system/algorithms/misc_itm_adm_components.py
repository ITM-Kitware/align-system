from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import adm_utils
from align_system.utils.alignment_utils import attributes_in_alignment_target


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
