"""Compare variants on matched initial scenarios, with scenario bootstrap CIs.

Usage: python -m align_system.cli.compare_agent_alignment study.json
       [--bootstrap-samples 2000] [--seed 0]
Paths in the manifest are relative to its directory. No model calls are made.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
from statistics import mean

from align_system.cli.summarize_agent_alignment import load_run, summarize, target_key
from align_system.utils.agent_actions import action_identity
from align_system.cli.agent_event_summary import load_event_report, summarize_event_reports


METRICS = {
    'preference_agreement': 'higher',
    'mean_annotated_regret': 'lower',
    'mean_decision_time_s': 'lower',
    'fallback_rate': 'lower',
    'forced_end_rate': 'lower',
}


def _quantile(values, fraction):
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _estimate(differences, samples, seed):
    # Keep all target profiles and replicates of a scenario together. A long
    # trajectory must not contribute more independent observations than a short one.
    by_scenario = defaultdict(list)
    for scenario, difference in differences:
        by_scenario[scenario].append(difference)
    values = [mean(group) for _, group in sorted(by_scenario.items())]
    interval = None
    if len(values) >= 2:
        rng = random.Random(seed)
        bootstraps = sorted(mean(rng.choices(values, k=len(values))) for _ in range(samples))
        interval = [_quantile(bootstraps, 0.025), _quantile(bootstraps, 0.975)]
    return {
        'matched_episode_pairs': len(differences),
        'scenario_count': len(values),
        'mean_delta': mean(values) if values else None,
        'ci95': interval,
    }


class RunGroup:
    """Combine records while preserving each label's decision index."""
    def __init__(self):
        self.records = []
        self.labels = []
        self.sources = set()
        self.event_reports = []

    def append(self, record, label=None, source=None):
        if label is not None:
            self.labels.append({**deepcopy(label), 'record_index': len(self.records)})
            self.sources.add(source)
        self.records.append(record)

    def report(self):
        labels = ({'source': '; '.join(sorted(self.sources)), 'decisions': self.labels}
                  if self.sources else None)
        report = summarize(self.records, labels)
        if self.event_reports:
            report['reliability'] = summarize_event_reports(self.event_reports)
        return report


def _initial_context(records):
    first = records[0]
    state = first['input'].get('full_state')
    choices = first['input'].get('choices')
    agent = first.get('choice_info', {}).get('langchain_agent', {})
    # Partial/legacy logs cannot establish that their first row is episode start.
    if (not isinstance(state, dict) or not isinstance(choices, list)
            or not choices or agent.get('n_actions_taken_in_scenario') != 0):
        return None
    info = first.get('choice_info', {})
    target = info.get('alignment_target_entries', info.get('alignment_target_values'))
    if target is None:
        return None
    choices = sorted(json.dumps(action_identity(a), sort_keys=True) for a in choices)
    return json.dumps({'state': state, 'target': target, 'choices': choices}, sort_keys=True)


def _load_study(manifest, base_dir):
    runs = manifest.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('The study needs a nonempty runs list')
    variants, episodes, counts, seen_paths = {}, defaultdict(dict), defaultdict(int), set()
    seen_run_ids = set()
    for run in runs:
        if not isinstance(run, dict) or any(
                not isinstance(run.get(key), str) or not run[key].strip()
                for key in ('variant', 'replicate', 'input_output')):
            raise ValueError('Each run needs nonempty variant, replicate, and input_output strings')
        variant, replicate = run['variant'], run['replicate']
        path = (base_dir / run['input_output']).resolve()
        if path in seen_paths:
            raise ValueError(f'A run file is listed more than once: {path}')
        seen_paths.add(path)
        labels_path = base_dir / run['labels'] if run.get('labels') is not None else None
        records, labels = load_run(path, labels_path)
        # Validate labels once against the original file before slicing episodes.
        summarize(records, labels)
        indexed = {label['record_index']: label for label in labels['decisions']} if labels else {}
        source = labels['source'] if labels else None
        all_records = variants.setdefault(variant, RunGroup())
        events_path = base_dir / run['events'] if run.get('events') is not None else None
        event_report = load_event_report(path, records, events_path)
        if event_report is not None:
            if event_report['run_id'] in seen_run_ids:
                raise ValueError('An event journal run is listed more than once')
            seen_run_ids.add(event_report['run_id'])
        all_records.event_reports.append(event_report)
        counts[variant] += 1
        file_episodes = {}
        for i, record in enumerate(records):
            key = (record['input']['scenario_id'], target_key(record), replicate)
            if key in episodes[variant]:
                raise ValueError(f'Duplicate scenario/target/replicate for {variant}: {key}')
            group = file_episodes.setdefault(key, RunGroup())
            step = record.get('choice_info', {}).get('langchain_agent', {}).get('n_actions_taken_in_scenario')
            if group.records and step == 0:
                raise ValueError('Repeated episodes need separate run files and replicate IDs')
            group.append(record, indexed.get(i), source)
            all_records.append(record, indexed.get(i), source)
        for key, group in file_episodes.items():
            episodes[variant][key] = {
                'initial_context': _initial_context(group.records),
                'metrics': group.report()['overall'],
            }
    return variants, episodes, counts


def _compare(reference, variant, samples, seed):
    shared = sorted(reference.keys() & variant.keys())
    matched, mismatched, unverified = [], 0, 0
    for key in shared:
        left, right = reference[key], variant[key]
        if left['initial_context'] is None or right['initial_context'] is None:
            unverified += 1
        elif left['initial_context'] != right['initial_context']:
            mismatched += 1
        else:
            matched.append(key)
    metrics = {}
    for metric, better in METRICS.items():
        differences = []
        for key in matched:
            left, right = reference[key]['metrics'][metric], variant[key]['metrics'][metric]
            if left is not None and right is not None:
                differences.append((key[0], right - left))
        metrics[metric] = {
            **_estimate(differences, samples, seed), 'better': better,
            'pairs_missing_metric': len(matched) - len(differences),
        }
    return {
        'matched_initial_episodes': len(matched),
        'unmatched_reference_episodes': len(reference.keys() - variant.keys()),
        'unmatched_variant_episodes': len(variant.keys() - reference.keys()),
        'initial_context_mismatches': mismatched,
        'unverified_initial_episodes': unverified,
        'metrics': metrics,
    }


def compare_study(manifest, base_dir=Path('.'), bootstrap_samples=2000, seed=0):
    if not isinstance(manifest, dict):
        raise ValueError('The study manifest must be an object')
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError('bootstrap_samples must be a positive integer')
    variants, episodes, counts = _load_study(manifest, Path(base_dir))
    reference = manifest.get('reference')
    if not isinstance(reference, str) or reference not in variants or len(variants) < 2:
        raise ValueError('Set reference to a listed variant and include at least two variants')
    return {
        'reference': reference,
        'delta_definition': 'variant minus reference',
        'alignment_metrics_scope': ('Successful recorded actions only; zero-action episodes '
                                    'contribute to reliability, not paired alignment deltas.'),
        'weighting': 'equal scenarios; within each scenario, equal matched target/replicate episodes',
        'bootstrap': {'unit': 'scenario_id', 'samples': bootstrap_samples, 'seed': seed,
                      'interval': '95% percentile; null with fewer than two scenarios'},
        'variants': {name: {'run_files': counts[name], **group.report()}
                     for name, group in variants.items()},
        'paired_comparisons': {
            name: _compare(episodes[reference], episodes[name], bootstrap_samples, seed)
            for name in variants if name != reference},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        report = compare_study(manifest, args.manifest.parent, args.bootstrap_samples, args.seed)
        output = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == '__main__':
    main()
