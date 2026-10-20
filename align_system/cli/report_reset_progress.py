"""Report care coverage and navigation proxies from preserved agent traces.

Usage: python -m align_system.cli.report_reset_progress RUN_DIR [RUN_DIR ...]
Completion comes from the event journal, including zero-action failures.
Coverage counts distinct successful treatments in each scene, not inferred
medical recovery. Revisit counts are proxies, not judgments of clinical need.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from align_system.cli.agent_event_summary import load_event_report


def episode_metrics(records):
    scenes = []
    visits, revisions = {}, defaultdict(int)
    movement_streak = longest_streak = movements = revisits = 0
    for record in records:
        state = record['input']['full_state']
        scene_id = state['meta_info']['scene_id']
        if not scenes or scene_id != scenes[-1]['scene_id']:
            scenes.append({'scene_id': scene_id, 'patients': set(), 'treated': set(),
                           'treatment_offered': False})
            visits.clear()
            revisions.clear()
            movement_streak = 0
        scene = scenes[-1]
        scene['patients'].update(c['id'] for c in state['characters'])
        scene['treatment_offered'] |= any(
            a['action_type'] == 'TREAT_PATIENT' for a in record['input']['choices'])
        action = record['output']['action']
        # Intentions are not executed care or movement.
        if action.get('intent_action'):
            continue
        patient_id, action_type = action.get('character_id'), action['action_type']
        if action_type == 'TREAT_PATIENT':
            scene['treated'].add(patient_id)
        if action_type == 'MOVE_TO':
            movements += 1
            movement_streak += 1
            longest_streak = max(longest_streak, movement_streak)
            revisits += visits.get(patient_id) == revisions[patient_id]
            visits[patient_id] = revisions[patient_id]
        else:
            movement_streak = 0
            if patient_id and action_type in {
                    'CHECK_VITALS', 'TREAT_PATIENT', 'TAG_CHARACTER', 'MOVE_TO_EVAC'}:
                revisions[patient_id] += 1
    coverage = []
    for index, scene in enumerate(scenes):
        if not scene['treatment_offered']:
            continue
        covered = scene['patients'] & scene['treated']
        coverage.append({
            'scene_id': scene['scene_id'], 'after_scene_reset': index > 0,
            'patients': len(scene['patients']), 'treated': len(covered),
            'coverage': len(covered) / len(scene['patients']) if scene['patients'] else None,
            'without_treatment': sorted(scene['patients'] - covered),
        })
    return {'total_actions': len(records), 'movement_actions': movements,
            'longest_movement_sequence': longest_streak,
            'revisits_without_patient_action': revisits,
            'treatment_coverage_by_scene': coverage}


def report_run(directory):
    directory = Path(directory)
    input_path = directory / 'input_output.json'
    records = json.loads(input_path.read_text()) if input_path.exists() else []
    journal = load_event_report(input_path, records)
    if journal is None:
        raise ValueError(f'An event journal is required to count all attempted episodes: {directory}')
    episodes, grouped = {}, defaultdict(list)
    for line in (directory / 'agent_events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            # load_event_report already validated the only allowed truncation.
            break
        episode_id = event.get('episode_id')
        if event['event'] == 'episode_started':
            episodes[episode_id] = {
                'episode_id': episode_id, 'scenario_id': event['scenario_id'],
                'completed': False, 'outcome': 'unfinished',
                'alignment_target_id': None,
            }
        elif event['event'] == 'episode_context':
            episodes[episode_id].update({
                'alignment_target_id': event.get('alignment_target_id'),
                'reset_awareness': event.get('reset_awareness', False),
                'navigation_safeguards': event.get('navigation_safeguards', False),
            })
        elif event['event'] == 'episode_finished':
            episodes[episode_id].update(
                completed=event['completed'],
                outcome='completed' if event['completed'] else 'unfinished')
        elif event['event'] == 'episode_aborted':
            episodes[episode_id]['outcome'] = 'aborted'
    for record in records:
        grouped[record['choice_info']['event_log']['episode_id']].append(record)
    for episode_id, episode in episodes.items():
        episode.update(episode_metrics(grouped[episode_id]))
    return {
        'run_directory': str(directory.resolve()),
        'journal': journal,
        'trace_metrics_complete': journal['successful_actions_missing_from_trace'] == 0,
        'episodes_started': len(episodes),
        'completion_rate': (sum(e['completed'] for e in episodes.values()) / len(episodes)
                            if episodes else None),
        'episodes': list(episodes.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directories', nargs='+', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps({
        'definitions': {
            'coverage': 'Distinct successful treatments / patients observed in each care scene; not clinical recovery.',
            'revisits_without_patient_action': 'Repeat MOVE_TO to the same patient in a scene without an intervening successful check, treatment, tag, or evacuation for that patient. Clinical changes may justify these visits; this is a navigation proxy.',
            'completion_rate': 'Completed episodes / all started episodes, including zero-action failures.',
            'incomplete_trace': 'If trace_metrics_complete is false, action metrics are lower bounds.',
        },
        'runs': [report_run(path) for path in args.run_directories],
    }, indent=2))


if __name__ == '__main__':
    main()
