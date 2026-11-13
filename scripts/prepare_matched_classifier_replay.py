#!/usr/bin/env python3
"""Select balanced, spaced windows from every episode in two matched input views."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from run_classifier_replay import LABELS, digest, read_requests, write_json


def select_requests(requests, paired_requests, windows_per_episode):
    if windows_per_episode < 2:
        raise ValueError('Select at least two windows per episode')
    paired = {row['original_request_id']: row for row in paired_requests}
    if len(paired) != len(paired_requests) or set(paired) != {row['request_id'] for row in requests}:
        raise ValueError('Both views must contain exactly the same unique original requests')
    episodes = {label: defaultdict(list) for label in LABELS}
    for row in requests:
        counterpart = paired[row['request_id']]
        for key in ('true_label', 'trajectory_id', 'generation', 'response_format'):
            if row[key] != counterpart[key]:
                raise ValueError(f'Paired {key} differs')
        for key in ('class_directory', 'window_start', 'window_end'):
            if row['provenance'][key] != counterpart['provenance'][key]:
                raise ValueError(f'Paired {key} differs')
        episodes[row['true_label']][row['trajectory_id']].append(row)
    counts = {len(group) for group in episodes.values()}
    if len(counts) != 1 or counts == {0}:
        raise ValueError('Each class must contain the same nonzero number of episodes')
    selections = {}
    for label, group in episodes.items():
        selections[label] = []
        for identity, rows in sorted(group.items()):
            rows.sort(key=lambda row: row['provenance']['window_start'])
            if len(rows) < windows_per_episode:
                raise ValueError(f'Not enough windows in {identity}')
            indices = [i * (len(rows) - 1) // (windows_per_episode - 1)
                       for i in range(windows_per_episode)]
            selected = [rows[i] for i in indices]
            if any(a['provenance']['window_end'] > b['provenance']['window_start']
                   for a, b in zip(selected, selected[1:])):
                raise ValueError(f'Selected windows overlap in {identity}; reduce windows-per-episode')
            selections[label].append(selected)
    ordered = [selections[label][episode][window]
               for window in range(windows_per_episode)
               for episode in range(next(iter(counts)))
               for label in LABELS]
    return ordered, [paired[row['request_id']] for row in ordered]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--paired-requests', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--windows-per-episode', type=int, default=5)
    args = parser.parse_args()
    original, paired = select_requests(read_requests(args.requests), read_requests(args.paired_requests),
                                       args.windows_per_episode)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, rows in (('privileged', original), ('defender', paired)):
        path = args.output_dir / f'{name}-requests.jsonl'
        with path.open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        read_requests(path)
        outputs[name] = {'path': path.name, 'sha256': digest(path)}
    write_json(args.output_dir / 'manifest.json', {
        'source_requests_sha256': digest(args.requests), 'source_paired_requests_sha256': digest(args.paired_requests),
        'preparation_script_sha256': digest(Path(__file__)),
        'shared_reader_sha256': digest(Path(__file__).with_name('run_classifier_replay.py')),
        'requests_per_view': len(original), 'episodes': len(original) // args.windows_per_episode,
        'windows_per_episode': args.windows_per_episode, 'outputs': outputs,
        'selection': 'Evenly spaced window indices within each episode, including first and last; no selected windows overlap.',
        'ordering': 'Round robin over classes within episode index within window index; every six requests are class balanced.',
        'messages_and_generation_unchanged': True,
        'interpretation': 'Exploratory matched subset covering every source episode. Windows within an episode are not independent trials.',
    })
    print(json.dumps({'requests_per_view': len(original), 'outputs': outputs}), flush=True)


if __name__ == '__main__':
    main()
