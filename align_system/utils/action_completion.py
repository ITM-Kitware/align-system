import random

from align_system.data_models.compat.ta3_ph1_client_models import (
    CharacterTagEnum, InjuryLocationEnum)
from align_system.utils import get_swagger_class_enum_values
from swagger_client.models import ActionTypeEnum


DEFAULT_TAGS = get_swagger_class_enum_values(CharacterTagEnum)
VALID_INJURY_LOCATIONS = get_swagger_class_enum_values(InjuryLocationEnum)


def complete_action_parameters(scenario_state, action,
                               character_required_actions,
                               tags=DEFAULT_TAGS):
    """Randomly fill in required-but-missing action parameters (target
    character_id, triage tag category, treatment supply/location) so
    the environment will accept the action; already-set parameters are
    left untouched.

    `character_required_actions` is the set of action types the
    environment rejects without a character_id (this varies by
    environment/domain, so callers must supply it)."""
    if (action.action_type in character_required_actions
            and action.character_id is None):
        candidate_ids = [c.id for c in scenario_state.characters
                         if not getattr(c, 'unseen', False)]
        if candidate_ids:
            action.character_id = random.choice(candidate_ids)

    if action.action_type == ActionTypeEnum.TAG_CHARACTER:
        if action.parameters is None:
            action.parameters = {}

        if 'category' not in action.parameters:
            action.parameters['category'] = random.choice(tags)

    if action.action_type == ActionTypeEnum.TREAT_PATIENT:
        # The (live) environment errors on TREAT_PATIENT without a
        # treatment supply/location; only completable when the state
        # reports supplies
        in_stock_supplies = [
            s.type for s in (getattr(scenario_state, 'supplies', None) or [])
            if s.quantity is None or s.quantity > 0]

        if in_stock_supplies:
            if action.parameters is None:
                action.parameters = {}

            if 'treatment' not in action.parameters:
                action.parameters['treatment'] = random.choice(
                    in_stock_supplies)
            if 'location' not in action.parameters:
                action.parameters['location'] = random.choice(
                    VALID_INJURY_LOCATIONS)

    return action
