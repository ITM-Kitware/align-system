"""Blue-only P4 prompts, separate from the historical Red-action prompts.

The fixed system prompt has three seven-step Monitor examples per class from
classifier-blue-prompts. Live windows contain nine completed Blue observations.
Provenance and the exact prompt are retained alongside this module.
"""

import json
from pathlib import Path


EXAMPLES_DIR = Path(__file__).with_name('blue_observation_examples')
SYSTEM_PROMPT_PATH = EXAMPLES_DIR / 'p4_system.txt'
COLUMNS = ['Subnet', 'IP Address', 'Hostname', 'Activity', 'Compromised']


def blue_table_snapshot(env):
    """Copy the PPO wrapper's defender table, never the simulator's true table."""
    table = env.get_observation_table(is_true_table=False)
    if list(table.field_names) != COLUMNS:
        raise ValueError('Unexpected Blue table columns')
    return sorted(([str(cell) for cell in row] for row in table.rows), key=lambda row: row[2])


def _blue_action(action):
    text = action['Action']
    if 'Subnet' in action:
        text += f" on subnet {action['Subnet']}"
    elif action.get('Host') not in (None, action['Action']):
        text += f" on {action['Host']}"
    return f"{text} (success: {action['Status']})"


def format_blue_window(window):
    """Match P4's compact history using only explicitly allowed Blue fields.

    Every window starts with its own full table. Subsequent rows include both
    changes (including activity clearing) and repeated non-None detections.
    """
    if not window:
        raise ValueError('Blue observation window must not be empty')
    lines = ['Columns: ' + ', '.join(COLUMNS) + '.']
    previous = None
    first_step = window[0]['step']
    for offset, record in enumerate(window):
        if record['step'] != first_step + offset:
            raise ValueError('Blue observation steps must be contiguous')
        current = {}
        for row in record['blue_table']:
            if len(row) != 5 or row[2] in current:
                raise ValueError('Malformed or duplicate Blue table row')
            if row[3] not in ('None', 'Scan', 'Exploit') or row[4] not in ('No', 'Unknown', 'User', 'Privileged'):
                raise ValueError('Unexpected Blue table state')
            current[row[2]] = list(row)
        if not current:
            raise ValueError('Empty Blue table')
        lines.append(f"Step {record['step']}: Blue {_blue_action(record['blue_action'])}")
        if previous is None:
            selected = list(current.values())
            prefix = 'Initial visible table rows: '
        else:
            if current.keys() != previous.keys():
                raise ValueError('Host inventory changed within the Blue observation window')
            selected = [row for host, row in current.items()
                        if row != previous[host] or row[3] != 'None']
            prefix = 'Changed rows and current activity: '
        lines.append(prefix + json.dumps(selected, separators=(',', ':')))
        if previous is not None:
            lines.append('All other rows unchanged.')
        previous = current
    return '\n'.join(lines)


class CAGEBlueObservationFewshotSystemPrompt:
    def __init__(self, window_size=9):
        if window_size != 9:
            raise ValueError('The Blue-only hybrid protocol requires a 9-step live window')

    def __call__(self):
        return SYSTEM_PROMPT_PATH.read_text()


class CAGEBlueObservationClassificationPrompt:
    def __call__(self, observations, labels):
        return '# OBSERVATIONS\n' + observations + '\n\n# AVAILABLE LABELS\n' + '\n'.join(labels)
