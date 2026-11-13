#!/usr/bin/env python3
"""Prepare episode-separated Blue-only prompt and history comparisons."""

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from align_system.interfaces.cage_observations import capture_paths, format_action, format_window
from run_classifier_replay import LABELS, digest, read_requests, read_results, write_json


INSTRUCTIONS = '''You classify a simulated attacker from partial defender observations.
Return exactly one available label in the required JSON schema.

There are two strategy families. B-line follows a comparatively direct sequential
route, usually finishing work on one host before moving forward. Meander explores
more broadly and may revisit hosts or interleave activity across hosts. These are
behavioral tendencies, not guaranteed signatures in a short partial history.

Within each family, the fixed objective controls eligible Impact targets:
C: Auth and Database; I: Auth and Front; A: Auth, Database, and Front.
Targets overlap. Auth alone distinguishes none of these objectives. Database is
compatible with C or A; Front is compatible with I or A. Exploiting a host does
not establish that the attacker intends to Impact it. The objective stays fixed
through the episode; target hosts can change.

You receive only Blue's selected actions and its resulting observation tables.
Activity is detected Scan, Exploit, or None, not an exact Red action transcript.
Compromised is Blue's retained estimate: No, Unknown, User, or Privileged. No
means no compromise inferred by this table, not proof the host is safe. Unknown
does not mean confirmed clean. A retained compromise label is not a new attack
at every turn. None activity does not prove Red did nothing. Impact and privilege
escalation are not directly named in these tables. Do not invent hidden actions.

Blue receives automatic network-event monitoring each turn. Monitor collects
network-event evidence. Analyse inspects files on a selected host while routine
monitoring continues. Remove and Restore can change retained compromise labels.
Host roles are indicated by their visible hostnames. Addresses are network
identifiers, not attacker identity labels.

Use chronology and visible host patterns. Evidence can be genuinely ambiguous;
choose the best-supported available class without treating missing detections as
certain absence. Examples show training labels even when their evidence overlaps.
'''

COMPACT = '''The compact history begins with every visible table row. Columns are
Subnet, IP Address, Hostname, Activity, Compromised. Later steps list every changed
row and every non-None Activity, including repeated detections. Other rows are
unchanged from the preceding table. Blue action results are included every turn.
No hidden Red actions are added by this representation.
'''

VARIANTS = {'P1': (False, 0, 7), 'P2': (True, 0, 7), 'P3': (True, 1, 7),
            'P4': (True, 3, 7), 'P5': (True, 1, 21), 'P6': (True, 2, 7)}
LABEL_BY_DIRECTORY = {label.lower().replace('b-line', 'bline').replace('-', '_'): label for label in LABELS}


def table_rows(text):
    rows = [[cell.strip() for cell in line.split('|')[1:-1]]
            for line in text.splitlines() if line.startswith('|')]
    if not rows or rows[0] != ['Subnet', 'IP Address', 'Hostname', 'Activity', 'Compromised']:
        raise ValueError('Unexpected Blue table columns')
    hosts = {}
    for row in rows[1:]:
        if len(row) != 5 or row[2] in hosts:
            raise ValueError('Malformed or duplicate Blue table row')
        if row[3] not in ('None', 'Scan', 'Exploit') or row[4] not in ('No', 'Unknown', 'User', 'Privileged'):
            raise ValueError('Unexpected Blue table state')
        hosts[row[2]] = row
    if not hosts:
        raise ValueError('Empty Blue table')
    return hosts


def compact_history(capture, start, end):
    if not 0 <= start < end <= len(capture['steps']):
        raise ValueError('History outside captured episode')
    lines = ['Columns: Subnet, IP Address, Hostname, Activity, Compromised.']
    previous = None
    for index in range(start, end):
        step = capture['steps'][index]
        if step['step'] != index:
            raise ValueError('Noncontiguous captured steps')
        current = table_rows(step['defender_observation'])
        lines.append(f"Step {index}: Blue {format_action(step['blue_action'])}")
        if previous is None:
            selected = list(current.values())
            lines.append('Initial visible table rows: ' + json.dumps(selected, separators=(',', ':')))
        else:
            if current.keys() != previous.keys():
                raise ValueError('Host inventory changed; compact representation needs explicit handling')
            selected = [row for host, row in current.items()
                        if row != previous[host] or row[3] != 'None']
            lines.append('Changed rows and current activity: ' + json.dumps(selected, separators=(',', ':')))
            lines.append('All other rows unchanged.')
        previous = current
    return '\n'.join(lines)


def render(capture, end, length, compact):
    return compact_history(capture, end - length, end) if compact else format_window(capture, end - length, end)


def make_request(base, capture, variant, partition, examples):
    compact, n_examples, length = VARIANTS[variant]
    result = copy.deepcopy(base)
    end = base['provenance']['window_end']
    start = max(0, end - length)
    actual_length = end - start
    instruction = INSTRUCTIONS + ('\n' + COMPACT if compact else '')
    for index, (label, text, _) in enumerate(examples[:6 * n_examples], 1):
        instruction += f'\nTraining example {index}:\n{text}\nCorrect training label: {label}\n'
    user = '# OBSERVATIONS\n' + render(capture, end, actual_length, compact)
    user += '\n\n# AVAILABLE LABELS\n' + '\n'.join(LABELS)
    result['messages'] = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': user}]
    result.update(variant=variant, partition=partition, prompt_protocol='blue_prompt_screen',
                  input_view='defender_table_compact_history' if compact else 'defender_table_and_blue_action',
                  condition='monitor', pair_id=base['request_id'], seed=capture['seed'],
                  requested_history_length=length, actual_history_length=actual_length)
    result['provenance'].update(window_start=start, window_size=actual_length,
        original_seven_step_request_id=base['request_id'],
        example_episode_seeds=sorted({item[2] for item in examples[:6 * n_examples]}),
        messages_sha256=hashlib.sha256(json.dumps(result['messages'], sort_keys=True).encode()).hexdigest())
    result['request_id'] = hashlib.sha256(json.dumps({
        'pair': base['request_id'], 'variant': variant, 'partition': partition,
        'messages': result['messages'], 'generation': result['generation'],
    }, sort_keys=True).encode()).hexdigest()
    return result


def prepare(args):
    baseline = read_requests(args.requests)
    predictions = read_results(args.baseline_predictions, baseline)
    if len(predictions) != len(baseline):
        raise ValueError('The matched baseline must have complete predictions')
    summary = json.loads((args.capture_dir / 'summary.json').read_text())
    captures = {}
    for episode, path in zip(summary['episodes'], capture_paths(args.capture_dir)):
        capture = json.loads(path.read_text())
        if capture['seed'] != episode['seed']:
            raise ValueError('Capture seed differs from audit')
        key = (episode['class_directory'], episode['trajectory_id'])
        captures[key] = capture
    groups = defaultdict(set)
    for (directory, _), capture in captures.items():
        groups[capture['seed']].add(LABEL_BY_DIRECTORY[directory])
    if len(groups) != 20 or any(labels != set(LABELS) for labels in groups.values()):
        raise ValueError('Expected twenty shared rollout seeds across all six classes')
    seeds = sorted(groups)
    random.Random(args.split_seed).shuffle(seeds)
    split = {'examples': seeds[:4], 'development': seeds[4:12], 'confirmation': seeds[12:]}
    if any(set(a) & set(b) for i, a in enumerate(split.values()) for b in list(split.values())[i+1:]):
        raise ValueError('Episode-seed partitions overlap')
    examples = []
    for index in range(3):
        for label in LABELS:
            matches = [capture for (directory, _), capture in captures.items()
                       if LABEL_BY_DIRECTORY[directory] == label and capture['seed'] == split['examples'][index]]
            if len(matches) != 1:
                raise ValueError('Example seed does not uniquely identify a class episode')
            end = (37, 58, 79)[index]
            examples.append((label, render(matches[0], end, 7, True), matches[0]['seed']))
    by_partition = defaultdict(list)
    for base in baseline:
        capture = captures[(base['provenance']['class_directory'], base['trajectory_id'])]
        expected_view = format_window(capture, base['provenance']['window_start'], base['provenance']['window_end'])
        if base['messages'][-1]['content'].split('\n\n# AVAILABLE LABELS\n')[0] != '# OBSERVATIONS\n' + expected_view:
            raise ValueError('Matched baseline observation differs from audited capture')
        if LABEL_BY_DIRECTORY[base['provenance']['class_directory']] != base['true_label']:
            raise ValueError('Captured class and reference label disagree')
        partition = next(name for name, selected in split.items() if capture['seed'] in selected)
        if partition == 'examples':
            continue
        end = base['provenance']['window_end']
        if partition == 'development' and end not in (37, 79):
            continue
        for variant in VARIANTS:
            by_partition[partition].append(make_request(base, capture, variant, partition, examples))
    for partition, rows in by_partition.items():
        rows.sort(key=lambda r: (r['provenance']['window_end'], r['seed'], r['variant'], LABELS.index(r['true_label'])))
        expected = 96 if partition == 'development' else 240
        if Counter(r['variant'] for r in rows) != Counter({v: expected for v in VARIANTS}):
            raise ValueError('Unexpected number of balanced windows')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for partition, rows in by_partition.items():
        p = args.output_dir / f'{partition}-requests.jsonl'
        p.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        read_requests(p)
    shutil.copyfile(args.baseline_predictions, args.output_dir / 'baseline-predictions.jsonl')
    reference_ids = {r['pair_id'] for rows in by_partition.values() for r in rows}
    baseline_rows = [r for r in baseline if r['request_id'] in reference_ids]
    (args.output_dir / 'baseline-requests.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in baseline_rows))
    write_json(args.output_dir / 'protocol.json', {
        'format_version': 1, 'split_seed': args.split_seed, 'shared_seed_partitions': split,
        'source_requests_sha256': digest(args.requests), 'capture_audit_sha256': digest(args.capture_dir/'summary.json'),
        'baseline_predictions_sha256': digest(args.baseline_predictions),
        'preparer_sha256': digest(Path(__file__)), 'variants': VARIANTS,
        'example_selection': 'First three shuffled example seeds; windows ending at 37,58,79 respectively; nested class-balanced examples.',
        'confirmation_limit': 'Reserved from prompt tuning; previously included in inspected aggregate baselines.',
        'early_long_history': 'P5 uses all available steps if fewer than 21 exist; actual length retained per request.',
        'primary_selection': 'Development joint accuracy, then shorter mean token count; confirm top two eligible variants.',
        'optional_conditions_omitted': ['family-only prompt', 'active-defense inference', 'supervised classifier'],
        'omission_reason': 'Prioritize five main prompt/history comparisons and paired confirmation within the secondary allocation.',
    })
    collisions = defaultdict(Counter)
    raw_fields = Counter()
    for base in baseline:
        capture = captures[(base['provenance']['class_directory'], base['trajectory_id'])]
        end = base['provenance']['window_end']
        collisions[render(capture, end, 7, True)][base['true_label']] += 1
    for capture in captures.values():
        for step in capture['steps']:
            for host, observation in step['defender_raw_observation'].items():
                if host != 'success' and isinstance(observation, dict):
                    raw_fields.update(observation.keys())
    write_json(args.output_dir / 'observation-audit.json', {
        'n_sampled_histories': len(baseline), 'n_distinct_rendered_histories': len(collisions),
        'histories_with_multiple_labels': sum(len(counts) > 1 for counts in collisions.values()),
        'in_sample_majority_count': sum(max(counts.values()) for counts in collisions.values()),
        'interpretation': 'Sample-specific collisions only; neither this count nor a perfect in-sample majority establishes generalization.',
        'raw_observation_host_field_counts': dict(raw_fields),
    })
    print(json.dumps({'output': str(args.output_dir), 'counts': {p: len(r) for p, r in by_partition.items()}, 'split': split}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', required=True, type=Path)
    p.add_argument('--baseline-predictions', required=True, type=Path)
    p.add_argument('--capture-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--split-seed', type=int, default=20261112)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
