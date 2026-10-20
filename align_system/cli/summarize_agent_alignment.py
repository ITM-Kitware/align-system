"""Summarize executed decisions; alignment metrics require external labels.

Usage: python -m align_system.cli.summarize_agent_alignment input_output.json
       [--labels independent_labels.json]
See docs/agent_alignment.md for definitions, denominators, and label format.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from align_system.utils.agent_actions import action_identity
from align_system.cli.agent_event_summary import load_event_report, summarize_event_reports


def _action_key(action):
    return json.dumps(action_identity(action), sort_keys=True)


def _label_action_key(action):
    required = {'action_id', 'action_type', 'character_id', 'parameters'}
    if not isinstance(action, dict) or not required.issubset(action):
        raise ValueError('Label actions need action_id, action_type, character_id, parameters')
    if (not isinstance(action['action_id'], str) or not action['action_id']
            or not isinstance(action['action_type'], str)
            or (action['character_id'] is not None and not isinstance(action['character_id'], str))
            or (action['parameters'] is not None and not isinstance(action['parameters'], dict))
            or type(action.get('intent_action', False)) is not bool):
        raise ValueError('Label actions need string IDs/types, object parameters, and a boolean intent flag')
    return _action_key(action)


def _index_labels(labels, n_records):
    indexed = {}
    if labels is None:
        return indexed
    if not isinstance(labels, dict) or not isinstance(labels.get('decisions'), list):
        raise ValueError('Labels must be an object with a decisions list')
    if not isinstance(labels.get('source'), str) or not labels['source'].strip():
        raise ValueError('Labels must identify their independent source')
    for label in labels['decisions']:
        if not isinstance(label, dict):
            raise ValueError('Each decision label must be an object')
        idx = label['record_index']
        if type(idx) is not int or not 0 <= idx < n_records or idx in indexed:
            raise ValueError(f'Duplicate or out-of-range record_index: {idx}')
        result = {}
        if 'preferred_actions' in label:
            if not isinstance(label['preferred_actions'], list) or not label['preferred_actions']:
                raise ValueError('preferred_actions must be a nonempty list (ties allowed)')
            result['preferred'] = {_label_action_key(a) for a in label['preferred_actions']}
        if 'action_utilities' in label:
            if not isinstance(label['action_utilities'], list):
                raise ValueError('action_utilities must be a list')
            utilities = {}
            for item in label['action_utilities']:
                key, utility = _label_action_key(item['action']), item['utility']
                if (key in utilities or type(utility) not in (int, float)
                        or not math.isfinite(utility)):
                    raise ValueError('Action utilities must be unique and finite numbers')
                utilities[key] = utility
            if not utilities:
                raise ValueError('action_utilities must be nonempty')
            result['utilities'] = utilities
        if not result:
            raise ValueError('Each label needs preferred_actions or action_utilities')
        indexed[idx] = result
    return indexed


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _mean(values):
    return mean(values) if values else None


def target_key(record):
    """Group unnamed targets by their full entries when available."""
    target = record['input'].get('alignment_target_id')
    if target is None:
        info = record.get('choice_info', {})
        target = json.dumps(info.get('alignment_target_entries',
                                    info.get('alignment_target_values', {})), sort_keys=True)
    return str(target)


def _checked_time(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Recorded times must be finite nonnegative numbers')
    return value


def _summarize_group(rows):
    sources = Counter()
    preferences, regrets, decision_times = [], [], []
    for record, label in rows:
        info = record.get('choice_info', {})
        agent = info.get('langchain_agent', {})
        sources[agent.get('decision_source', 'unknown')] += 1
        if agent.get('decision_time_s') is not None:
            decision_times.append(_checked_time(agent['decision_time_s']))
        chosen = _action_key(record['output']['action'])
        if 'preferred' in label:
            preferences.append(int(chosen in label['preferred']))
        if 'utilities' in label:
            utilities = label['utilities']
            if chosen not in utilities:
                raise ValueError('action_utilities must score the executed action')
            regrets.append(max(utilities.values()) - utilities[chosen])
    known_sources = sum(sources[source] for source in ('agent', 'fallback', 'forced_end_scene'))
    return {
        'executed_actions': len(rows),
        'scenario_target_pairs': len({
            (r['input']['scenario_id'], target_key(r)) for r, _ in rows}),
        'decision_sources': dict(sources),
        'source_labeled_decisions': known_sources,
        'fallback_rate': _ratio(sources['fallback'], known_sources),
        'forced_end_rate': _ratio(sources['forced_end_scene'], known_sources),
        'timed_decisions': len(decision_times),
        'mean_decision_time_s': _mean(decision_times),
        'preference_labeled_decisions': len(preferences),
        'preference_agreement': _mean(preferences),
        'utility_labeled_decisions': len(regrets),
        'mean_annotated_regret': _mean(regrets),
    }


def summarize(records, labels=None):
    """Compute micro averages overall and by target; missing metrics are null.

    Only independently supplied labels contribute to agreement or regret.
    Regret is relative to the annotated alternatives, not an unobserved oracle.
    """
    validate_records(records)
    indexed = _index_labels(labels, len(records))
    rows = [(record, indexed.get(i, {})) for i, record in enumerate(records)]
    groups = defaultdict(list)
    for record, label in rows:
        groups[target_key(record)].append((record, label))
    by_target = {target: _summarize_group(group) for target, group in sorted(groups.items())}
    agreements = [group['preference_agreement'] for group in by_target.values()
                  if group['preference_agreement'] is not None]
    return {
        'annotation_source': labels['source'] if labels is not None else None,
        'overall': _summarize_group(rows),
        'by_target': by_target,
        'worst_labeled_target_agreement': min(agreements) if agreements else None,
    }


def validate_records(records):
    if not isinstance(records, list):
        raise ValueError('input_output.json must be a list of decision records')
    for i, record in enumerate(records):
        if (not isinstance(record, dict)
                or not isinstance(record.get('input'), dict)
                or not isinstance(record['input'].get('scenario_id'), str)
                or not isinstance(record.get('output'), dict)
                or not isinstance(record['output'].get('action'), dict)
                or not isinstance(record.get('choice_info', {}), dict)):
            raise ValueError(f'Invalid decision record at index {i}')


def load_run(input_output, labels_path=None):
    """Read a run and optional labels, enforcing the exact-file association."""
    raw = Path(input_output).read_bytes()
    records = json.loads(raw)
    validate_records(records)
    labels = json.loads(Path(labels_path).read_text()) if labels_path is not None else None
    if labels_path is not None:
        if not isinstance(labels, dict):
            raise ValueError('Labels must be an object')
        if labels.get('input_sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError('Label input_sha256 does not match this input_output.json')
    return records, labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_output', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--events', type=Path, help='Defaults to adjacent agent_events.jsonl')
    args = parser.parse_args(argv)
    try:
        records, labels = load_run(args.input_output, args.labels)
        report = summarize(records, labels)
        report['reliability'] = summarize_event_reports([
            load_event_report(args.input_output, records, args.events)])
        output = json.dumps(report, indent=2, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == '__main__':
    main()
