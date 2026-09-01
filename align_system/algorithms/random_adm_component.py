import random

from swagger_client.models import ActionTypeEnum as OWActionTypeEnum

from align_system.data_models.compat.ta3_ph1_client_models import (
    ActionTypeEnum,
    InjuryLocationEnum,
    CharacterTagEnum)

from align_system.algorithms.abstracts import ADMComponent
from align_system.utils import get_swagger_class_enum_values
from align_system.utils.action_completion import complete_action_parameters


class RandomChoiceADMComponent(ADMComponent):
    def run_returns(self):
        return 'chosen_choice', 'justification'

    def run(self, choices):
        return random.choice(choices), "Random choice"


class RandomParameterCompletionADMComponent(ADMComponent):
    # Phase-1 counterpart of action_completion.complete_action_parameters,
    # kept separate because the action vocabularies differ (phase 1's
    # APPLY_TREATMENT / CHECK_* actions and aid_id parameter don't
    # exist in the open world / phase-2 enum, and vice versa)
    def run_returns(self):
        return 'chosen_action'

    def run(self,
            scenario_state,
            choices,
            actions,
            chosen_choice,
            chosen_action=None):
        if chosen_action is None:
            chosen_choice_idx = choices.index(chosen_choice)
            chosen_action = actions[chosen_choice_idx]

        # Action requires a character ID
        if chosen_action.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                         ActionTypeEnum.CHECK_ALL_VITALS,
                                         ActionTypeEnum.CHECK_PULSE,
                                         ActionTypeEnum.CHECK_RESPIRATION,
                                         ActionTypeEnum.MOVE_TO_EVAC,
                                         ActionTypeEnum.TAG_CHARACTER,
                                         ActionTypeEnum.CHECK_BLOOD_OXYGEN}:
            if chosen_action.character_id is None:
                chosen_action.character_id = random.choice([
                    c.id
                    for c in scenario_state.characters
                    if hasattr(c, "unseen") and not c.unseen
                ])

        if chosen_action.action_type == ActionTypeEnum.APPLY_TREATMENT:
            if chosen_action.parameters is None:
                chosen_action.parameters = {}

            if 'treatment' not in chosen_action.parameters:
                chosen_action.parameters['treatment'] = random.choice(
                    [s.type for s in scenario_state.supplies if s.quantity > 0])
            if 'location' not in chosen_action.parameters:
                chosen_action.parameters['location'] = random.choice(
                    get_swagger_class_enum_values(InjuryLocationEnum))

        elif chosen_action.action_type == ActionTypeEnum.TAG_CHARACTER:
            if chosen_action.parameters is None:
                chosen_action.parameters = {}

            if 'category' not in chosen_action.parameters:
                chosen_action.parameters['category'] = random.choice(
                    get_swagger_class_enum_values(CharacterTagEnum))

        # Action requires an aid ID
        elif chosen_action.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            if chosen_action.parameters is None:
                chosen_action.parameters = {}

            if "aid_id" not in chosen_action.parameters:
                chosen_action.parameters["aid_id"] = random.choice([
                    aid.id
                    for aid in scenario_state.environment.decision_environment.aid
                ])

        # Required since Dry Run Evaluation
        chosen_action.justification = "Random choice"

        return chosen_action


class OWRandomParameterCompletionADMComponent(ADMComponent):
    def run_returns(self):
        return 'chosen_action'

    def run(self,
            scenario_state,
            choices,
            actions,
            chosen_choice,
            chosen_action=None):
        if chosen_action is None:
            chosen_choice_idx = choices.index(chosen_choice)
            chosen_action = actions[chosen_choice_idx]

        # Phase-2 / open world action types (the phase-1 compat
        # ActionTypeEnum imported above has no TREAT_PATIENT)
        complete_action_parameters(
            scenario_state, chosen_action,
            character_required_actions={OWActionTypeEnum.TREAT_PATIENT,
                                        OWActionTypeEnum.MOVE_TO_EVAC,
                                        OWActionTypeEnum.TAG_CHARACTER})

        chosen_action.justification = "Random choice"

        return chosen_action
