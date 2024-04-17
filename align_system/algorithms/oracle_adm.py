import math
import random
import numpy as np
import pandas as pd
from typing import List, Optional

from swagger_client.models import (
    ActionTypeEnum, CharacterTagEnum, InjuryLocationEnum, SupplyTypeEnum
)

from align_system.utils import logging
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values

log = logging.getLogger(__name__)


class OracleADM(ActionBasedADM):
    def __init__(self, probabilistic: bool=False, upweight_missing_kdmas: bool=False, **kwargs):
        self.probabilistic = probabilistic
        self.upweight_missing_kdmas = upweight_missing_kdmas

    def _inverse_distance(self, target_kdma_values, system_kdmas, min_value=0., max_value=1.):
        """
        Compute 1/(Euclidean distance between the target and system KDMAs)
        """
        if system_kdmas is None:
            system_kdmas = dict()

        distance = 0.

        # TODO: Determine if this is an appropriate measure
        for kdma, target in target_kdma_values.items():
            # Edge case #1: Target KDMA is not present in System KDMAs, assign "midpoint" distance
            # The idea here is that an unlabeled choice may be more desired than a fully misaligned one
            if kdma not in system_kdmas:
                val_range = max_value - min_value
                midpoint = val_range/2. + min_value
                if self.upweight_missing_kdmas:
                    distance += midpoint**2
                else:
                    distance +=  (midpoint + abs(target - midpoint) + 0.1)**2  # Assign maximum distance
            else:
                distance += (target - system_kdmas[kdma])**2

        #Edge case #2: System KDMA is not present in Target KDMAs => ignoring for now
        # e.g if Target: X=10; Choice A: X=3; Choice B: X=3, Y=5; then A and B are equivalent

        # TODO: Should we just return 1/distance**2?
        # Small epsilon for a perfect (0 distance) match
        return math.sqrt(1/(distance+1e-16))

    def choose_action(self, scenario_state, available_actions, alignment_target, **kwargs):
        if available_actions is None or len(available_actions) == 0:
            return None

        if alignment_target is None:
            raise ValueError("Oracle ADM needs alignment target")

        # Build out the corresponding kdma_association target dictionary
        target_kdma_assoc = {
            target['kdma']: target['value']
            for target in alignment_target.kdma_values
        }

        # Weight action choices with distance-based metric
        inv_dists = [
            self._inverse_distance(target_kdma_values=target_kdma_assoc, system_kdmas=action.kdma_association)
            for action in available_actions
        ]

        # Convert inverse distances to probabilities
        probs = [inv_dist/sum(inv_dists) for inv_dist in inv_dists]

        if self.probabilistic:
            action_to_take = np.random.choice(available_actions, p=probs)
        else:  # Always choose (one of the) max probability action
            max_prob = max(probs)
            max_actions = [idx for idx, p in enumerate(probs) if p == max_prob]
            action_to_take = available_actions[random.choice(max_actions)]

        # Log scoring results
        results = pd.DataFrame([
            (action.unstructured, prob)
            for action, prob in zip(available_actions, probs)
        ], columns=["choice", "probability"])
        results = results.sort_values(by=["probability"], ascending=False)
        log.explain(results)

        # Action requires a character ID
        if action_to_take.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.MOVE_TO_EVAC,
                                          ActionTypeEnum.TAG_CHARACTER}:
            # TODO: Is there a good heuristic for what character we should apply this to?
            if action_to_take.character_id is None:
                action_to_take.character_id = random.choice(
                    [c.id for c in scenario_state.characters])

        if action_to_take.action_type == ActionTypeEnum.APPLY_TREATMENT:
            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            if action_to_take.character_id is None:
                # Limit to characters with injuries
                # TODO: Limit to untreated injuries?
                poss_characters = [c for c in scenario_state.characters if c.injuries]
            else:
                poss_characters = [c for c in scenario_state.characters if c.id == action_to_take.character_id]

            def _get_treatment(poss_treatments: List[str], injury_name: str, injury_location: str) -> Optional[str]:
                """
                Return appropriate treatment for given injury name and location, given available supplies. If no
                treatment exists under these conditions, None will be returned.
                """
                match injury_name:
                    case 'Amputation':
                        if 'Tourniquet' in poss_treatments:
                            return 'Tourniquet'
                    case 'Burn':
                        if 'Burn Dressing' in poss_treatments:
                            return 'Burn Dressing'
                    case 'Broken Bone':
                        if 'Splint' in poss_treatments:
                            return 'Splint'
                    case 'Chest Collapse':
                        if 'Decompression Needle' in poss_treatments:
                            return 'Decompression Needle'
                    case 'Laceration':
                        if 'thigh' in injury_location:
                            if 'Tourniquet' in poss_treatments:
                                return 'Tourniquet'
                        else:
                            if 'Pressure bandage' in poss_treatments:
                                return 'Pressure bandage'
                    case 'Puncture':
                        if 'bicep' in injury_location or 'thigh' in injury_location:
                            if 'Tourniquet' in poss_treatments:
                                return 'Tourniquet'
                        else:
                            if 'Hemostatic gauze' in poss_treatments:
                                return 'Hemostatic gauze'
                    case 'Shrapnel':
                        if 'face' in injury_location:
                            if 'Nasopharyngeal airway' in poss_treatments:
                                return 'Nasopharyngeal airway'
                        else:
                            if 'Hemostatic gauze' in poss_treatments:
                                return 'Hemostatic gauze'
                    case 'Internal':
                        return 'Pain Medications'
                    case 'Ear Bleed':
                        return None
                    case 'Asthmatic':
                        return None
                    case _:
                        raise ValueError(f"Unknown injury type: {injury_name}")

                return None

            while len(poss_characters) > 0:
                # Select a random character
                selected_char = random.choice(poss_characters)

                # Identify which treatments are available to perform
                poss_treatments = [s.type for s in scenario_state.supplies if s.quantity > 0]
                poss_treatments = [t for t in poss_treatments if t in get_swagger_class_enum_values(SupplyTypeEnum)]
                if "treatment" in action_to_take.parameters:
                    poss_treatments = [action_to_take.parameters["treatment"]]

                # Identify selected character's treatable injuries
                poss_injuries = [
                    injury
                    for injury in selected_char.injuries
                    if (("location" not in action_to_take.parameters or injury.location == action_to_take.parameters["location"]) and
                        _get_treatment(poss_treatments, injury.name, injury.location) is not None)
                ]

                # Randomly selected a treatable injury (if one exists)
                if len(poss_injuries) > 0:
                    selected_injury = random.choice(poss_injuries)
                else:
                    # No treatable injuries, remove character from consideration and try again
                    poss_characters = [c for c in poss_characters if c.id != selected_char.id]
                    continue

                action_to_take.character_id = selected_char.id
                action_to_take.parameters['treatment'] = _get_treatment(
                    poss_treatments, selected_injury.name, selected_injury.location)
                action_to_take.parameters['location'] = selected_injury.location
                break
            else:  # No "possible" characters left
                log.warn("Could not identify character/treatment, randomly selecting")
                if action_to_take.character_id is None:
                    action_to_take.character_id = random.choice(
                        [c.id for c in scenario_state.characters])
                if 'treatment' not in action_to_take.parameters:
                    action_to_take.parameters['treatment'] = random.choice(
                        [s.type for s in scenario_state.supplies if s.quantity > 0])
                # TODO: Reduce available locations by treatment so that we don't end up with
                # something like tourniquet around neck?
                if 'location' not in action_to_take.parameters:
                    action_to_take.parameters['location'] = random.choice(
                        get_swagger_class_enum_values(InjuryLocationEnum))

        elif action_to_take.action_type == ActionTypeEnum.TAG_CHARACTER:
            if action_to_take.parameters is None:
                action_to_take.parameters = {}

            if 'category' not in action_to_take.parameters:
                # TODO: Implement better tagging logic
                action_to_take.parameters['category'] = random.choice(
                    get_swagger_class_enum_values(CharacterTagEnum))

        return action_to_take
