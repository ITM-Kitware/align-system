import random

from align_system.data_models.compat.ta3_ph1_client_models import (
    CharacterTagEnum)
from align_system.utils import get_swagger_class_enum_values
from swagger_client.models import ActionTypeEnum


DEFAULT_TAGS = get_swagger_class_enum_values(CharacterTagEnum)


def complete_action_parameters(scenario_state, action,
                               character_required_actions,
                               tags=DEFAULT_TAGS):
    """Randomly fill in required-but-missing action parameters (target
    character_id, triage tag category) so the environment will accept
    the action; already-set parameters are left untouched.

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

    return action
