import random
import numpy as np

from swagger_client.models import ActionTypeEnum, InjuryLocationEnum, CharacterTagEnum

from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values

log = logging.getLogger(__name__)


class HeuristicADM(ActionBasedADM):
    def __init__(self, **kwargs):
        self.intent_lookup = {'intend major help': 1.0,
                              'intend minor help': 0.75,
                              'no intent': 0.5,
                              'intend minor harm': 0.25,
                              'intend major harm': 0.0}

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        # Get character intent mapping
        character_intent = {}
        for character in scenario_state.characters:
            character_intent[character.id] = character.intent

        # Compute action value mapping
        action_values = []
        filtered_actions = []
        for action in available_actions:
            character_id = action.character_id
            if character_id is not None and character_intent[character_id] is not None:
                filtered_actions.append(action)
                action_values.append(self.intent_lookup[character_intent[character_id]])

        if len(filtered_actions) > 0 and alignment_target is not None:
            # Depend on alignment target
            target_kdma_assoc = {
                target['kdma']: target['value']
                for target in alignment_target.kdma_values
            }
            kdma_target = list(target_kdma_assoc.values())[0]
            if kdma_target > 0.5:
                action_to_take = filtered_actions[np.argmax(action_values)]
            else:
                action_to_take = filtered_actions[np.argmin(action_values)]
        else:
            # Take random action
            action_to_take = random.choice(available_actions)

        # Action requires a character ID
        if action_to_take.action_type in {ActionTypeEnum.APPLY_TREATMENT,
                                          ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.MOVE_TO_EVAC,
                                          ActionTypeEnum.TAG_CHARACTER}:
            if action_to_take.character_id is None:
                action_to_take.character_id = random.choice(
                    [c.id for c in scenario_state.characters])

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

        return action_to_take
