from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import adm_utils


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
