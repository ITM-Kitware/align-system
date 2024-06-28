import random
import numpy as np

from swagger_client.models import ActionTypeEnum, InjuryLocationEnum, CharacterTagEnum

from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values

log = logging.getLogger(__name__)


class HeuristicADM(ActionBasedADM):
    def __init__(self, **kwargs):
        self.action_lookup = {'SITREP': 1.0,
                              'CHECK_ALL_VITALS': 0.8,
                              'CHECK_PULSE': 0.6,
                              'TAG_CHARACTER': 0.4,
                              'APPLY_TREATMENT': 0.2,
                              'END_SCENE': 0.0}
        self.triage_lookup = {'IMMEDIATE': 1.0,
                              'DELAYED': 0.66,
                              'MINIMAL': 0.33,
                              'EXPECTANT': 0.0}

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        # Compute action value mapping
        action_values = []
        filtered_actions = []
        for action in available_actions:
            if action.action_type in self.action_lookup:
                filtered_actions.append(action)
                if action.action_type == 'TAG_CHARACTER' and action.parameters is not None:
                    action_values.append(self.triage_lookup[action.parameters["category"]])
                else:
                    action_values.append(self.action_lookup[action.action_type])

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
