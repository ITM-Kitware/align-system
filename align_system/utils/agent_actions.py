"""Stable action identities for recovery validation."""
from collections.abc import Mapping


def action_identity(action):
    """Identify a completed action without prose, justification, or labels."""
    get = action.get if isinstance(action, Mapping) else (
        lambda name, default=None: getattr(action, name, default))
    return {
        'action_id': get('action_id'),
        'action_type': get('action_type'),
        'character_id': get('character_id'),
        'parameters': get('parameters') or {},
        'intent_action': bool(get('intent_action', False)),
    }
