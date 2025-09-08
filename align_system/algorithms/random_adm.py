import random

from align_system.data_models.compat.ta3_ph1_client_models import ActionTypeEnum, InjuryLocationEnum, CharacterTagEnum
from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values

log = logging.getLogger(__name__)


class RandomADM(ActionBasedADM):
    def __init__(self, **kwargs):
        pass

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        action_to_take = random.choice(available_actions)

        # Action requires a character ID
        if action_to_take.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                          ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.MOVE_TO_EVAC,
                                          ActionTypeEnum.TAG_CHARACTER,
                                          ActionTypeEnum.CHECK_BLOOD_OXYGEN}:
            if action_to_take.character_id is None:
                action_to_take.character_id = random.choice([
                    c.id
                    for c in scenario_state.characters
                    if hasattr(c, "unseen") and not c.unseen
                ])

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            if 'treatment' not in action_to_take.parameters:
                action_to_take.parameters['treatment'] = random.choice(
                    [s.type for s in scenario_state.supplies if s.quantity > 0])
            if 'location' not in action_to_take.parameters:
                action_to_take.parameters['location'] = random.choice(
                    get_swagger_class_enum_values(InjuryLocationEnum))

        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            if 'category' not in action_to_take.parameters:
                action_to_take.parameters['category'] = random.choice(
                    get_swagger_class_enum_values(CharacterTagEnum))

        # Action requires an aid ID
        elif action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            if "aid_id" not in action_to_take.parameters:
                action_to_take.parameters["aid_id"] = random.choice([
                    aid.id
                    for aid in scenario_state.environment.decision_environment.aid
                ])

        # Required since Dry Run Evaluation
        action_to_take.justification = "Random choice"

        choice_info = {}
        return action_to_take, choice_info
