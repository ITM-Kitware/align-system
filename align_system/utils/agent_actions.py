"""Action identities shared by agent listings and value assessments."""
import json
from collections.abc import Mapping
from align_system.utils.alignment_targets import alignment_target_values


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


def describe_action(action, scenario_state=None):
    """Include the environment ID and parameters, even for duplicate prose."""
    character_id = getattr(action, 'character_id', None)
    names = {c.id: c.name for c in getattr(scenario_state, 'characters', [])}
    details = {
        'action_id': getattr(action, 'action_id', None),
        'action_type': action.action_type,
        'character_id': character_id,
        'parameters': getattr(action, 'parameters', None) or {},
    }
    if character_id is not None:
        details['character_name'] = names.get(character_id, character_id)
    return f"{action.unstructured} {json.dumps(details, sort_keys=True, default=str)}"


def assessment_cache_key(session, actions):
    """Refresh an assessment if the offered actions change without an action."""
    return (session.n_actions, tuple(describe_action(a) for a in actions),
            tuple(sorted(alignment_target_values(session.alignment_target).items())))


def assessment_rows(rows):
    """Detach assessment rows from mutable environment action objects."""
    return [{**{k: v for k, v in row.items() if k != 'action'},
             'action': (row['action'].to_dict()
                        if hasattr(row['action'], 'to_dict')
                        else vars(row['action']).copy())}
            for row in rows]
