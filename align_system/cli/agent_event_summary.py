"""Reliability denominators come from attempted episodes, not successful rows."""
from collections import Counter
import json
from pathlib import Path

from align_system.utils.agent_actions import action_identity


def load_event_report(input_path, records, events_path=None):
    path = (Path(events_path) if events_path is not None
            else Path(input_path).with_name('agent_events.jsonl'))
    if not path.exists() and events_path is None:
        return None
    lines = path.read_text().splitlines(keepends=True)
    events, truncated = [], False
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except ValueError:
            # A killed writer can leave a partial final line. Keep the valid
            # prefix, and visibly flag it; corruption elsewhere is an error.
            if index == len(lines) - 1 and not line.endswith('\n'):
                truncated = True
                break
            raise ValueError(f'Invalid event JSON at line {index + 1}: {path}') from None
        if (not isinstance(event, dict) or event.get('schema_version') != 1
                or type(event.get('sequence')) is not int or event['sequence'] != index
                or not isinstance(event.get('run_id'), str)
                or not isinstance(event.get('event'), str)):
            raise ValueError(f'Invalid event envelope at line {index + 1}: {path}')
        events.append(event)
    if not events or events[0]['event'] != 'run_started':
        raise ValueError(f'Missing run_started event: {path}')
    run_id = events[0]['run_id']
    episodes, counts, successes = {}, Counter(), {}
    run_status = 'unfinished'
    for event in events:
        if event['run_id'] != run_id or run_status != 'unfinished':
            raise ValueError(f'Mixed runs or events after run termination: {path}')
        kind, episode_id = event['event'], event.get('episode_id')
        counts[kind] += 1
        if kind == 'run_started' and event['sequence'] != 0:
            raise ValueError(f'Repeated run_started event: {path}')
        if kind in ('run_finished', 'run_aborted'):
            run_status = 'finished' if kind == 'run_finished' else 'aborted'
        if kind == 'episode_started':
            if not isinstance(episode_id, str) or episode_id in episodes:
                raise ValueError(f'Invalid/repeated episode: {path}')
            episodes[episode_id] = {
                'scenario_id': event['scenario_id'], 'outcome': 'unfinished',
                'successful_actions': 0,
                'decisions': set(), 'completed_decisions': set()}
        if episode_id is None:
            continue
        if episode_id not in episodes:
            raise ValueError(f'Event refers to an unknown episode: {path}')
        episode = episodes[episode_id]
        if kind == 'episode_aborted':
            episode['outcome'] = 'aborted'
        elif kind == 'episode_finished':
            if type(event.get('completed')) is not bool:
                raise ValueError(f'Invalid episode completion flag: {path}')
            episode['outcome'] = 'completed' if event['completed'] else 'unfinished'
        elif kind == 'decision_started':
            episode['decisions'].add(event['decision_index'])
        elif kind == 'action_succeeded':
            key = (episode_id, event['decision_index'])
            if key in successes:
                raise ValueError(f'Duplicate successful decision: {path}')
            successes[key] = action_identity(event['action'])
            episode['successful_actions'] += 1
            episode['completed_decisions'].add(event['decision_index'])
    seen = set()
    for record in records:
        info = record.get('choice_info', {})
        binding = info.get('event_log', {})
        key = (binding.get('episode_id'),
               info.get('langchain_agent', {}).get('n_actions_taken_in_scenario'))
        if (binding.get('run_id') != run_id or key in seen
                or successes.get(key) != action_identity(record['output']['action'])
                or episodes[key[0]]['scenario_id'] != record['input']['scenario_id']):
            raise ValueError(f'Action trace does not match its event journal: {path}')
        seen.add(key)
    outcomes = Counter(ep['outcome'] for ep in episodes.values())
    return {
        'run_id': run_id, 'run_status': run_status,
        'truncated_last_line': truncated,
        'episodes_started': len(episodes),
        'episode_outcomes': dict(outcomes),
        'zero_action_episodes': sum(ep['successful_actions'] == 0 for ep in episodes.values()),
        'decisions_without_executed_action': sum(
            len(ep['decisions'] - ep['completed_decisions']) for ep in episodes.values()),
        'successful_actions_missing_from_trace': len(successes) - len(records),
        'event_counts': dict(counts),
    }


def summarize_event_reports(reports):
    available = [report for report in reports if report is not None]
    if not available:
        return {'runs_with_events': 0, 'runs_missing_events': len(reports),
                'episode_completion_rate': None}
    outcomes, statuses, counts = Counter(), Counter(), Counter()
    for report in available:
        outcomes.update(report['episode_outcomes'])
        statuses[report['run_status']] += 1
        counts.update(report['event_counts'])
    attempted = sum(report['episodes_started'] for report in available)
    return {
        'runs_with_events': len(available), 'runs_missing_events': len(reports) - len(available),
        'run_outcomes': dict(statuses), 'episodes_started': attempted,
        'episode_outcomes': dict(outcomes),
        'episode_completion_rate': outcomes['completed'] / attempted if attempted else None,
        'truncated_journals': sum(r['truncated_last_line'] for r in available),
        **{key: sum(report[key] for report in available) for key in (
            'zero_action_episodes', 'decisions_without_executed_action',
            'successful_actions_missing_from_trace')},
        'event_counts': dict(counts),
        'model_call_scope': 'Actor attempts only; toolkit inference is not counted.',
    }
