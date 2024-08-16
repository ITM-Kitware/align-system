import math
import random
import numpy as np
import pandas as pd
from typing import List, Optional

from swagger_client.models import (
    ActionTypeEnum, CharacterTagEnum, InjuryLocationEnum, SupplyTypeEnum
)

from align_system.utils import logging
from align_system.utils import alignment_utils
from align_system.algorithms.abstracts import ActionBasedADM
from align_system.utils import get_swagger_class_enum_values

log = logging.getLogger(__name__)


class OracleADM(ActionBasedADM):
    def __init__(
        self, probabilistic: bool=False, upweight_missing_kdmas: bool=False,
        filter_unlabeled_actions: bool=False, misaligned: bool=False, **kwargs
    ):
        self.probabilistic = probabilistic
        self.upweight_missing_kdmas = upweight_missing_kdmas
        self.filter_unlabeled_actions = filter_unlabeled_actions
        self.misaligned = misaligned

    def choose_action(self, scenario_state, available_actions, alignment_target,
                      distribution_matching: str='sample',
                      kde_norm: str='rawscores', **kwargs):
        if available_actions is None or len(available_actions) == 0:
            return None

        if self.filter_unlabeled_actions:
            available_actions = [
                action for action in available_actions
                if action.kdma_association is not None and len(action.kdma_association) > 0
            ]

            if len(available_actions) == 0:
                raise RuntimeError("No actions left to take after filtering unlabled actions")

        if alignment_target is None:
            raise ValueError("Oracle ADM needs alignment target")

        target_kdmas = alignment_target.kdma_values

        # Get type of targets
        all_scalar_targets = True
        all_kde_targets = True
        for target_kdma in target_kdmas:
            if not hasattr(target_kdma, 'value') or target_kdma.value is None:
                all_scalar_targets = False
            if not hasattr(target_kdma, 'kdes') or target_kdma.kdes is None:
                all_kde_targets = False

        # get ground truth kdma values
        gt_kdma_values = {}
        actions_with_kdma_values = []
        for action in available_actions:
            if hasattr(action, 'kdma_association') and action.kdma_association is not None:
                gt_kdma_values[action.action_id] = action.kdma_association.copy()
                actions_with_kdma_values.append(action)

        if all_scalar_targets:
            alignment_function = alignment_utils.AvgDistScalarAlignment()
            selected_choice_id, probs = alignment_function(gt_kdma_values, target_kdmas, misaligned=self.misaligned)

        elif all_kde_targets:
            if distribution_matching == 'sample':
                alignment_function = alignment_utils.MinDistToRandomSampleKdeAlignment()
            elif distribution_matching == 'max_likelihood':
                alignment_function = alignment_utils.MaxLikelihoodKdeAlignment()
            elif distribution_matching == 'js_divergence':
                alignment_function = alignment_utils.JsDivergenceKdeAlignment()
            else:
                raise RuntimeError(distribution_matching, "distribution matching function unrecognized.")
            selected_choice_id, probs = alignment_function(gt_kdma_values, target_kdmas, misaligned=self.misaligned, kde_norm=kde_norm)
        else:
            # TODO: Currently we assume all targets either have scalar values or KDES,
            #       Down the line, we should extend to handling multiple targets of mixed types
            raise ValueError("ADM does not currently support a mix of scalar and KDE targets.")

        for action in available_actions:
            if selected_choice_id == action.action_id:
                action_to_take = action

        # Log scoring results
        results = pd.DataFrame([
            (action.unstructured, probs[action.unstructured])
            for action in actions_with_kdma_values
        ], columns=["choice", "probability"])
        results = results.sort_values(by=["probability"], ascending=False)
        log.explain(results)

        # Action requires a character ID
        if action_to_take.action_type in {ActionTypeEnum.CHECK_ALL_VITALS,
                                          ActionTypeEnum.CHECK_PULSE,
                                          ActionTypeEnum.CHECK_RESPIRATION,
                                          ActionTypeEnum.MOVE_TO_EVAC,
                                          ActionTypeEnum.TAG_CHARACTER,
                                          ActionTypeEnum.CHECK_BLOOD_OXYGEN}:
            # TODO: Is there a good heuristic for what character we should apply this to?
            if action_to_take.character_id is None:
                action_to_take.character_id = random.choice([
                    c.id
                    for c in scenario_state.characters
                    if hasattr(c, "unseen") and not c.unseen
                ])

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
                        log.warn(f"Unknown injury type: {injury_name}. Choosing random treatment")

                return random.choice(poss_treatments)

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

        elif action_to_take.action_type == ActionTypeEnum.MOVE_TO_EVAC:
            if "aid_id" not in action_to_take.parameters:
                action_to_take.parameters["aid_id"] = random.choice([
                    aid.id
                    for aid in scenario_state.environment.decision_environment.aid
                ])

        action_to_take.justification = "Looked at scores"
        return action_to_take
