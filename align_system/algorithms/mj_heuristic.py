import random
import pickle
import numpy as np

from swagger_client.models import ActionTypeEnum, InjuryLocationEnum, CharacterTagEnum

from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values

log = logging.getLogger(__name__)


class HeuristicADM(ActionBasedADM):
    def __init__(self, **kwargs):
        # Load pretrained sklearn model
        with open('bbn-mj.pkl', 'rb') as f:
            self.regressor = load(f)

        # Define required mappings
        # self.intent_key = {'intent minor help': 0,
        #                    'intent help': 1}
        # self.doc_key = {'none': 0,
        #                 'somewhat indirect': 1}
        self.kdma_lookup = {'intend minor help': 0.8,
                            'no intent': 0.675,
                            'intend minor harm': 0.34375,
                            'intend major harm': 0.275,
                            'intend minor help,indirect': 0.8,
                            'intend minor harm,somewhat indirect': 0.34375,
                            'intend major harm,direct': 0.275,
                            'no intent,none': 0.758333,
                            'no intent,somewhat indirect': 0.425}

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        # Get character intent mapping
        character_intent = {}
        character_directness = {}
        for character in scenario_state.characters:
            character_intent[character.id] = character.intent
            character_directness[character.id] = character.directness_of_causality

        # Compute action value mapping
        action_values = []
        filtered_actions = []
        for action in available_actions:
            character_id = action.character_id
            if character_id is not None:
                filtered_actions.append(action)
                # character intent is primary factor
                if character_intent[character_id] is not None:
                    key = ','.join(filter(None, [character_intent.get(character_id), character_directness.get(character_id)]))
                    if key not in self.kdma_lookup:
                        key = key.split(',')[0]
                    action_values.append(self.kdma_lookup[key])
                else:
                    # assume 0.5 value
                    action_values.append(0.5)

        if len(filtered_actions) > 0 and alignment_target is not None:
            # Depend on alignment target
            target_kdma_assoc = {
                target['kdma']: target['value']
                for target in alignment_target.kdma_values
            }
            kdma_target = list(target_kdma_assoc.values())[0]
            action_to_take = available_actions[np.argmin(np.abs(np.array(action_values) - kdma_target))]
        else:
            # Take random action
            action_to_take = random.choice(available_actions)

        logging.info('Filtered actions:\n')
        logging.info(filtered_actions)
        logging.info('Chosen action:\n')
        logging.info(action_to_take)

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

        # Required since Dry Run Evaluation
        action_to_take.justification = "Taking action based on heuristic rule"

        return action_to_take
