from copy import deepcopy
import random

from swagger_client.models import ActionTypeEnum, CharacterTagEnum

from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values
from align_system.utils.alignment_utils import attributes_in_alignment_target


class OpenWorldOracleADM(ActionBasedADM):
    def __init__(self, alignment, check_vitals=False):
        self.alignment = alignment
        self.check_vitals = check_vitals
        self.reset_history()

    def reset_history(self):
        # The Part 3 server does not always report treatment/evacuation in state.
        self.completed_actions = {
            action_type: set() for action_type in (
                ActionTypeEnum.TAG_CHARACTER,
                ActionTypeEnum.TREAT_PATIENT,
                ActionTypeEnum.MOVE_TO_EVAC,
                ActionTypeEnum.CHECK_VITALS,
            )
        }

    def choose_action(self, scenario_state, available_actions,
                      alignment_target=None, **kwargs):
        if alignment_target is None:
            raise ValueError("The Open World oracle requires an alignment target")
        attributes = [kdma for kdma in attributes_in_alignment_target(alignment_target)
                      if kdma != 'medical']
        if len(attributes) != 1:
            raise ValueError("A patient attribute_rating requires exactly one target KDMA")

        actions_by_patient = {}
        for character in scenario_state.characters:
            patient_id = character.id
            if patient_id in self.completed_actions[ActionTypeEnum.MOVE_TO_EVAC]:
                continue

            tagged = (character.tag is not None or
                      patient_id in self.completed_actions[ActionTypeEnum.TAG_CHARACTER])
            treated = (bool(character.treated) or
                       patient_id in self.completed_actions[ActionTypeEnum.TREAT_PATIENT])
            checked = patient_id in self.completed_actions[ActionTypeEnum.CHECK_VITALS]
            patient_actions = [a for a in available_actions
                               if a.character_id == patient_id]
            priorities = []
            if any(a.action_type == ActionTypeEnum.MOVE_TO_EVAC for a in patient_actions):
                priorities.append(ActionTypeEnum.MOVE_TO_EVAC)
            if self.check_vitals and not checked and not (tagged and treated):
                priorities.append(ActionTypeEnum.CHECK_VITALS)
            if not tagged:
                priorities.append(ActionTypeEnum.TAG_CHARACTER)
            if not treated:
                priorities.append(ActionTypeEnum.TREAT_PATIENT)
            if not priorities:
                continue
            if not character.nearby:
                priorities = [ActionTypeEnum.MOVE_TO]

            for action_type in priorities:
                action = next((a for a in patient_actions if a.action_type == action_type), None)
                if action is not None:
                    actions_by_patient[patient_id] = action
                    break

        scores = {}
        for character in scenario_state.characters:
            if character.id not in actions_by_patient:
                continue
            if character.medical_condition is None or character.attribute_rating is None:
                raise ValueError(f"Missing ground-truth scores for patient {character.id}")
            scores[character.id] = {
                'medical': character.medical_condition,
                attributes[0]: character.attribute_rating,
            }

        if scores:
            patient_id, _, alignment_info = self.alignment.run(scores, alignment_target)
            chosen_action = deepcopy(actions_by_patient[patient_id])
            chosen_action.justification = "Selected patient using ground-truth scores and alignment target."
        else:
            end_scene = next((a for a in available_actions
                              if a.action_type == ActionTypeEnum.END_SCENE), None)
            if end_scene is None:
                raise RuntimeError("No eligible patients or END_SCENE action available")
            chosen_action = deepcopy(end_scene)
            chosen_action.justification = "All available patient care and evacuation actions are complete."
            alignment_info = {}

        if chosen_action.action_type == ActionTypeEnum.TAG_CHARACTER:
            if chosen_action.parameters is None:
                chosen_action.parameters = {}
            if 'category' not in chosen_action.parameters:
                chosen_action.parameters['category'] = random.choice(
                    get_swagger_class_enum_values(CharacterTagEnum))
        elif chosen_action.action_type == ActionTypeEnum.TREAT_PATIENT:
            if chosen_action.parameters is None:
                chosen_action.parameters = {}
            if 'treatment' not in chosen_action.parameters:
                supplies = [s.type.value for s in scenario_state.supplies if s.quantity > 0]
                if not supplies:
                    raise RuntimeError("No supplies available for treatment")
                chosen_action.parameters['treatment'] = random.choice(supplies)

        if chosen_action.action_type in self.completed_actions:
            self.completed_actions[chosen_action.action_type].add(chosen_action.character_id)

        return chosen_action, {
            'true_kdma_values': scores,
            'alignment_info': alignment_info,
        }
