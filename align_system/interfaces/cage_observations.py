"""Recorded CAGE observations shared by classification and replay preparation."""

import hashlib
import json


DEFENDER_FIELDS = ('blue_actions', 'defender_observations')
TRAJECTORY_FIELDS = {
    'blue_actions': 'Blue', 'red_actions': 'Red',
    'defender_observations': 'Blue observation',
}
HISTORICAL_CHECKS = (
    'fresh_process_repeat_identical', 'historical_initial_network_matches',
    'historical_metric_scores_match', 'historical_action_trace_matches',
)


def format_action(action):
    text = action['Action']
    if 'Subnet' in action:
        text += f" on subnet {action['Subnet']}"
    elif action.get('Host') not in (None, action['Action']):
        text += f" on {action['Host']}"
    return f"{text} (success: {action['Status']})"


def validate_fields(fields):
    if not fields or set(fields) - TRAJECTORY_FIELDS.keys() or len(set(fields)) != len(fields):
        raise ValueError(f'observation_fields must select unique fields from {list(TRAJECTORY_FIELDS)}')


def trajectory_length(trajectory, fields=DEFENDER_FIELDS):
    validate_fields(fields)
    if 'steps' in trajectory:
        return len(trajectory['steps'])
    missing = set(fields) - trajectory.keys()
    if missing:
        raise ValueError(f'Missing recorded observations: {sorted(missing)}; capture Blue observations first')
    lengths = {len(trajectory[field]) for field in fields}
    if len(lengths) != 1:
        raise ValueError('Recorded observation fields have different lengths')
    return lengths.pop()


def format_window(trajectory, start, end, fields=DEFENDER_FIELDS):
    length = trajectory_length(trajectory, fields)
    if not (type(start) is int and type(end) is int and 0 <= start < end <= length):
        raise ValueError('Captured observations do not cover the requested window')
    lines = []
    capture_fields = {'blue_actions': 'blue_action', 'red_actions': 'red_action',
                      'defender_observations': 'defender_observation'}
    for step in range(start, end):
        lines.append(f'Step {step}:')
        row = trajectory['steps'][step] if 'steps' in trajectory else None
        if row is not None and row['step'] != step:
            raise ValueError('Captured observations have noncontiguous step indices')
        for field in fields:
            value = row[capture_fields[field]] if row is not None else trajectory[field][step]
            if field == 'defender_observations':
                if not isinstance(value, str) or not value.strip():
                    raise ValueError('Missing defender observation table')
                lines.extend(('  Blue observation:', value))
            else:
                lines.append(f'  {TRAJECTORY_FIELDS[field]}: {format_action(value)}')
    return '\n'.join(lines)


def verified_monitor_episodes(summary):
    episodes = summary.get('episodes', [])
    if summary.get('status') != 'complete' or not episodes:
        raise ValueError('Monitor reproducibility audit must be complete and nonempty')
    for episode in episodes:
        if not all(episode.get(field) is True for field in HISTORICAL_CHECKS):
            raise ValueError(f"Historical episode verification failed: {episode['trajectory_id']}")
    return episodes


def capture_paths(directory):
    """Use the audit's primary captures, excluding repeats and metadata files."""
    summary = json.loads((directory / 'summary.json').read_text())
    if summary.get('condition') == 'active_hybrid_ppo':
        episodes = summary.get('episodes', [])
        if summary.get('status') != 'complete' or not episodes:
            raise ValueError('Active-defender capture must be complete and nonempty')
    else:
        episodes = verified_monitor_episodes(summary)
    paths = []
    for episode in episodes:
        path = directory / episode['capture_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != episode['capture_sha256']:
            raise ValueError(f'Captured evidence changed: {path}')
        if summary.get('condition') == 'active_hybrid_ppo':
            capture = json.loads(path.read_text())
            if not all(capture.get(key) is True for key in (
                'repeat_capture_identical', 'monitor_control_reproduced', 'initial_network_matches_monitor',
            )):
                raise ValueError(f'Active replay or Monitor control verification failed: {path}')
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate captures in audit summary')
    return paths
