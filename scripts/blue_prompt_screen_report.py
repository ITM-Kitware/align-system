#!/usr/bin/env python3
"""Score paired Blue prompt variants with rollout-seed grouped uncertainty."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from run_classifier_replay import LABELS, read_requests, write_json


def predictions(path):
    if not path.exists():
        return {}
    rows = {}
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not line.endswith(b'\n'):
            continue
        row = json.loads(line)
        if row['request_id'] in rows:
            raise ValueError('Duplicate prediction')
        if row['predicted_label'] not in LABELS:
            raise ValueError('Invalid prediction label')
        rows[row['request_id']] = row
    return rows


def score(requests, completed):
    actual = []
    for row in requests:
        if row['request_id'] in completed:
            prediction = completed[row['request_id']]
            if prediction['true_label'] != row['true_label']:
                raise ValueError('Prediction label metadata differs from request')
            actual.append((row, prediction))
    confusion = [[0] * 6 for _ in range(6)]
    for row, prediction in actual:
        confusion[LABELS.index(row['true_label'])][LABELS.index(prediction['predicted_label'])] += 1
    n = len(actual)
    correct = sum(row['true_label'] == p['predicted_label'] for row, p in actual)
    f1 = []
    for i in range(6):
        denominator = sum(confusion[i]) + sum(r[i] for r in confusion)
        f1.append(2 * confusion[i][i] / denominator if denominator else 0)
    return {
        'status': 'complete' if n == len(requests) else 'partial', 'n': n, 'expected': len(requests),
        'joint_accuracy': correct / n if n else None,
        'family_accuracy': sum(r['true_label'].rsplit('-', 1)[0] == p['predicted_label'].rsplit('-', 1)[0] for r,p in actual)/n if n else None,
        'objective_accuracy': sum(r['true_label'][-1] == p['predicted_label'][-1] for r,p in actual)/n if n else None,
        'macro_f1': sum(f1)/6, 'labels': LABELS, 'confusion_matrix': confusion,
        'prediction_counts': dict(Counter(p['predicted_label'] for _,p in actual)),
        'mean_input_tokens': sum(p.get('input_tokens', 0) for _,p in actual)/n if n else None,
        'mean_inference_seconds': sum(p.get('inference_seconds', 0) for _,p in actual)/n if n else None,
    }


def grouped_difference(requests, completed, baseline):
    groups = defaultdict(list)
    for row in requests:
        if row['request_id'] not in completed or row['pair_id'] not in baseline:
            continue
        current = completed[row['request_id']]
        reference = baseline[row['pair_id']]
        if reference['true_label'] != row['true_label']:
            raise ValueError('Paired baseline label differs')
        groups[row['seed']].append(int(current['predicted_label'] == row['true_label']) -
                                   int(reference['predicted_label'] == row['true_label']))
    if not groups:
        return None
    values = list(groups.values())
    rng = random.Random(20261112)
    draws = []
    for _ in range(2000):
        sampled = [rng.choice(values) for _ in values]
        draws.append(sum(sum(g) for g in sampled) / sum(len(g) for g in sampled))
    draws.sort()
    return {'paired_requests': sum(map(len, values)), 'shared_seed_groups': len(values),
            'joint_accuracy_difference': sum(map(sum, values))/sum(map(len, values)),
            'bootstrap_95_percent_interval': [draws[49], draws[1949]],
            'resampling_unit': 'shared rollout seed across all six attacker classes',
            'interpretation': 'Exploratory interval; reserved seeds were previously included in aggregate baselines.'}


def build(inputs, output):
    development = read_requests((output if (output/'development-requests.jsonl').exists() else inputs)/'development-requests.jsonl')
    confirmation = read_requests((output if (output/'confirmation-requests.jsonl').exists() else inputs)/'confirmation-requests.jsonl')
    baseline = predictions(inputs/'baseline-predictions.jsonl')
    result = {'protocol': json.loads((inputs/'protocol.json').read_text()), 'partitions': {}}
    for partition, requests in [('development', development), ('confirmation', confirmation)]:
        completed = predictions(output/partition/'predictions.jsonl')
        expected_ids = {r['request_id'] for r in requests}
        if set(completed) - expected_ids:
            raise ValueError('Foreign predictions in experiment output')
        variants = {}
        baseline_requests = {}
        for row in requests:
            baseline_requests[row['pair_id']] = {**row, 'request_id': row['pair_id']}
        variants['P0'] = score(list(baseline_requests.values()), baseline)
        for variant in sorted({r['variant'] for r in requests}):
            selected = [r for r in requests if r['variant'] == variant]
            metrics = score(selected, completed)
            metrics['difference_from_P0'] = grouped_difference(selected, completed, baseline)
            metrics['by_window_end'] = {str(end): score([r for r in selected if r['provenance']['window_end']==end], completed)
                                        for end in sorted({r['provenance']['window_end'] for r in selected})}
            variants[variant] = metrics
        result['partitions'][partition] = variants
    for name in ['context-audit.json', 'budget-decision.json', 'selection.json']:
        if (output/name).exists():
            result[name.removesuffix('.json')] = json.loads((output/name).read_text())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'comparison.json', result)
    lines = ['# Blue observation prompt comparison', '',
             'Reserved confirmation episodes contributed to previously inspected aggregate baselines. '
             'Only completed balanced conditions support an overall comparison.', '',
             '| Partition | Variant | Status | Completed | Joint | Family | Objective |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for partition, variants in result['partitions'].items():
        for variant, metrics in variants.items():
            values = [f"{100*metrics[k]:.2f}%" if metrics[k] is not None else ''
                      for k in ['joint_accuracy','family_accuracy','objective_accuracy']]
            lines.append(f"| {partition} | {variant} | {metrics['status']} | {metrics['n']}/{metrics['expected']} | " + ' | '.join(values) + ' |')
    lines += ['', 'Source requests, prompts, seed partitions, exclusions, and timing decisions are retained alongside the JSON report.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    result=build(args.inputs,args.output)
    print(json.dumps({'report':str(args.output/'REPORT.md'), 'partitions':list(result['partitions'])}))


if __name__=='__main__':
    main()
