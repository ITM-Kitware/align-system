#!/usr/bin/env python3
"""Build a fixed corpus, replay captured Qwen requests, and gate matched live runs.

Corpus snapshots are observational screens, not closed-loop trajectories. Only
captured wire requests can be replayed faithfully. Promotion requires live runs
and a separately completed semantic audit; JSON validity is not action quality.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import sys
import time
import urllib.error
import urllib.request


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def episode_key(record):
    source = record['input']
    return source['scenario_id'], source['alignment_target_id']


def group_records(records):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[episode_key(record)].append((index, record))
    return dict(groups)


def multi_agent(record):
    return record.get('choice_info', {}).get('multi_agent', {})


def action(record):
    return record['output']['action']


def build_corpus(run, output, expected_episodes=24):
    run, output = Path(run), Path(output)
    records = read_json(run / 'input_output.json')
    groups = group_records(records)
    if len(groups) != expected_episodes:
        raise ValueError(f'Expected {expected_episodes} episodes, found {len(groups)}.')
    selected = {}
    for key, rows in groups.items():
        for name, offset in [('early', 0), ('middle', len(rows) // 2), ('late', len(rows) - 1)]:
            selected.setdefault(rows[offset][0], set()).add(name)
        for index, record in rows:
            review = multi_agent(record)
            if (review.get('review_status') != 'reviewed'
                    or any(attempt.get('status') != 'reviewed'
                           for attempt in review.get('review_attempts', []))):
                selected.setdefault(index, set()).add('reviewer_failure')
            if (action(record)['action_type'] == 'END_SCENE'
                    or review.get('proposal', {}).get('action_type') == 'END_SCENE'):
                selected.setdefault(index, set()).add('scene_ending')
    index_by_action = {}
    entries = []
    for key, rows in groups.items():
        for offset, (index, record) in enumerate(rows):
            index_by_action[(*key, offset)] = index
            if index in selected:
                entries.append({'record_index': index, 'scenario_id': key[0],
                                'target_id': key[1], 'action_index': offset,
                                'reasons': sorted(selected[index]), 'record': record})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'snapshots.json', entries)
    captured = run / 'model_calls.jsonl'
    requests = []
    missing_requests = 0
    selected_calls = 0
    if captured.exists():
        for line in captured.read_text().splitlines():
            call = json.loads(line)
            key = (call.get('scenario_id'), call.get('target_id'), call.get('action_index'))
            if index_by_action.get(key) not in selected:
                continue
            selected_calls += 1
            if not call.get('request'):
                missing_requests += 1
                continue
            # Keep all stages/repairs of a selected action, in captured order.
            requests.append({key: call.get(key) for key in (
                'scenario_id', 'target_id', 'action_index', 'call_id', 'phase', 'request')})
    with (output / 'requests.jsonl').open('w') as handle:
        for request in requests:
            handle.write(json.dumps(request, allow_nan=False) + '\n')
    manifest = {'schema_version': 1, 'source': str(run.resolve()),
                'input_output_sha256': digest(run / 'input_output.json'),
                'episode_count': len(groups), 'snapshot_count': len(entries),
                'request_count': len(requests),
                'snapshots_sha256': digest(output / 'snapshots.json'),
                'requests_sha256': digest(output / 'requests.jsonl'),
                'selection': 'early/middle/late per episode plus all reviewer failures and proposed/executed scene endings',
                'selected_call_count': selected_calls,
                'missing_request_count': missing_requests,
                'replay_ready': bool(requests) and missing_requests == 0,
                'limitation': 'Snapshot replay does not measure live alignment or counterfactual trajectory quality.'}
    write_json(output / 'manifest.json', manifest)
    return manifest


def prepared_request(call, profile, thinking_budget=None):
    request = deepcopy(call['request'])
    if request['api'] != 'chat_completions':
        raise ValueError('This Qwen screen requires captured Chat Completions requests.')
    body = request['body']
    if body.get('model') != 'Qwen/Qwen3.8-27B-FP8':
        raise ValueError('Corpus must use the fixed Qwen/Qwen3.8-27B-FP8 checkpoint.')
    if body.get('stream'):
        raise ValueError('Streaming captures require a separate streaming replay client.')
    reviewer = call.get('phase') in ('review', 'review_repair')
    effort = 'low' if profile == 'both_low' or (profile == 'reviewer_low' and reviewer) else 'xhigh'
    if profile == 'both_medium':
        effort = 'medium'
    # These are already serialized HTTP bodies, not OpenAI SDK kwargs.
    template = body.setdefault('chat_template_kwargs', {})
    template.update(enable_thinking=profile != 'no_thinking', reasoning_effort=effort,
                    preserve_thinking=True)
    if profile == 'no_thinking':
        template.pop('reasoning_effort', None)
    body.pop('reasoning_effort', None)
    # vLLM accepts either cap name; never send conflicting aliases.
    cap_name = 'max_completion_tokens' if 'max_completion_tokens' in body else 'max_tokens'
    body.pop('max_tokens' if cap_name == 'max_completion_tokens' else 'max_completion_tokens', None)
    body[cap_name] = 8192
    if thinking_budget is not None:
        body['thinking_token_budget'] = thinking_budget
    else:
        body.pop('thinking_token_budget', None)
    body['temperature'] = 0.0
    return body


def response_validation(body, response):
    """Match local parsing behavior; finish/refusal metadata remains observational."""
    if not isinstance(response, dict):
        return ['invalid_response_shape']
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return ['invalid_choices_shape']
    choice = choices[0]
    message = choice.get('message')
    if not isinstance(message, dict):
        return ['invalid_message_shape']
    errors = []
    if not message.get('content') and not message.get('tool_calls'):
        errors.append('empty_response')
    schema = body.get('response_format', {}).get('json_schema', {}).get('schema')
    schema = schema or body.get('structured_outputs', {}).get('json')
    if schema:
        try:
            import jsonschema
            jsonschema.validate(json.loads(message.get('content') or ''), schema)
        except ImportError:
            errors.append('jsonschema_validator_unavailable')
        except (ValueError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
            errors.append(f'structured_output_invalid: {exc}')
    offered = {tool['function']['name']: tool['function'].get('parameters', {})
               for tool in body.get('tools', []) if tool.get('type') == 'function'}
    tool_calls = message.get('tool_calls') or []
    if not isinstance(tool_calls, list):
        return errors + ['invalid_tool_calls_shape']
    for tool in tool_calls:
        if not isinstance(tool, dict) or not isinstance(tool.get('function'), dict):
            errors.append('invalid_tool_call_shape')
            continue
        function = tool.get('function', {})
        if function.get('name') not in offered:
            errors.append('unknown_tool')
            continue
        try:
            import jsonschema
            jsonschema.validate(json.loads(function.get('arguments', '')), offered[function['name']])
        except ImportError:
            errors.append('jsonschema_validator_unavailable')
        except (ValueError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
            errors.append(f'tool_arguments_invalid: {exc}')
    return errors


def post(base_url, body):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('VLLM_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
    request = urllib.request.Request(base_url.rstrip('/') + '/chat/completions',
                                     json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def server_metrics(base_url):
    from urllib.parse import urlsplit, urlunsplit
    url = urlsplit(base_url)
    metrics_url = urlunsplit((url.scheme, url.netloc, '/metrics', '', ''))
    try:
        with urllib.request.urlopen(metrics_url, timeout=10) as response:
            return response.read().decode()
    except (OSError, ValueError) as exc:
        return '# metrics unavailable: ' + str(exc)


def replay(corpus, output, profile, base_url, sender=post, *, thinking_budget=None,
           cache_state='unspecified', metrics_reader=server_metrics):
    corpus, output = Path(corpus), Path(output)
    manifest = read_json(corpus / 'manifest.json')
    if digest(corpus / 'requests.jsonl') != manifest['requests_sha256']:
        raise ValueError('Wire request corpus hash mismatch.')
    calls = [json.loads(line) for line in (corpus / 'requests.jsonl').read_text().splitlines()]
    if not calls:
        raise ValueError('No captured requests. Run the baseline with save_model_requests=true and rebuild the corpus.')
    if manifest.get('missing_request_count', 0):
        raise ValueError('Selected calls have missing request captures; incomplete corpora cannot establish a faithful screen.')
    # Validate every request before issuing the first network request.
    bodies = [prepared_request(call, profile, thinking_budget) for call in calls]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'metrics_before.prom').write_text(metrics_reader(base_url))
    elapsed, failed = [], 0
    with (output / 'responses.jsonl').open('w') as handle:
        for call, body in zip(calls, bodies):
            start = time.monotonic()
            response = None
            try:
                response = sender(base_url, body)
                errors = response_validation(body, response)
            except (OSError, ValueError, urllib.error.HTTPError) as exc:
                errors = [f'{type(exc).__name__}: {exc}']
            seconds = time.monotonic() - start
            elapsed.append(seconds)
            failed += bool(errors)
            visible = deepcopy(response)
            refusal_present = False
            if isinstance(visible, dict):
                for choice in visible.get('choices') or []:
                    message = choice.get('message') if isinstance(choice, dict) else None
                    if isinstance(message, dict):
                        refusal_present |= message.pop('refusal', None) is not None
                        for field in ('reasoning', 'reasoning_content', 'thinking'):
                            message.pop(field, None)
            handle.write(json.dumps({'call_id': call.get('call_id'), 'phase': call.get('phase'),
                                     'elapsed_s': seconds, 'errors': errors,
                                     'refusal_present': refusal_present, 'response': visible}) + '\n')
            handle.flush()
    (output / 'metrics_after.prom').write_text(metrics_reader(base_url))
    report = {'profile': profile, 'thinking_token_budget': thinking_budget,
              'corpus_sha256': digest(corpus / 'requests.jsonl'),
              'request_count': len(calls), 'failed_requests': failed,
              'mean_request_s': statistics.mean(elapsed), 'p90_request_s': percentile(elapsed, .9),
              'passed_transport_screen': failed == 0,
              'quality_verified': False, 'latency_scope': 'non-streaming request wall time; TTFT unavailable',
              'cache_state': cache_state,
              'cache_scope': 'Operator label only: restart server before cold replay; repeat identical ordered corpus for warm replay.',
              'metrics': ['metrics_before.prom', 'metrics_after.prom']}
    write_json(output / 'summary.json', report)
    return report


def read_score(path):
    if not path.exists():
        return None
    match = re.search(r'Session alignment score\s*[:=]\s*([^\s,]+)', path.read_text())
    if not match:
        return None
    try:
        score = float(match.group(1))
        return score if math.isfinite(score) else None
    except ValueError:
        return None


def audit_template(run, output):
    run = Path(run)
    rows = read_json(run / 'input_output.json')
    template = {'input_output_sha256': digest(run / 'input_output.json'), 'reviewer': None,
                'instructions': 'Inspect the executed action, final explanation, and evidence for every record. Set each verdict only after review.',
                'records': [{'record_index': index, 'action_matches_explanation': None,
                             'evidence_valid': None, 'notes': ''} for index in range(len(rows))]}
    if Path(output).exists():
        raise ValueError('Refusing to overwrite an existing audit.')
    write_json(output, template)
    return {'audit_template': str(output), 'records': len(rows)}


def audit_status(run, count):
    path = run / 'consistency_audit.json'
    if not path.exists():
        return {'complete': False, 'errors': ['missing_consistency_audit']}
    audit = read_json(path)
    rows = audit.get('records', [])
    valid_indices = [row.get('record_index') for row in rows]
    complete = (all(type(index) is int for index in valid_indices)
                and audit.get('input_output_sha256') == digest(run / 'input_output.json')
                and isinstance(audit.get('reviewer'), str) and bool(audit['reviewer'].strip())
                and sorted(valid_indices) == list(range(count))
                and all(type(row.get(field)) is bool for row in rows
                        for field in ('action_matches_explanation', 'evidence_valid')))
    return {'complete': complete,
            'contradictions': sum(row.get('action_matches_explanation') is False for row in rows),
            'invalid_evidence': sum(row.get('evidence_valid') is False for row in rows)}


def summarize_run(run, expected_episodes=24):
    run = Path(run)
    records = read_json(run / 'input_output.json')
    groups = group_records(records)
    timing = read_json(run / 'timing.json')
    times = timing.get('raw_times_s', [])
    if len(groups) != expected_episodes or len(times) != len(records):
        raise ValueError(f'{run}: episode or timing counts do not match the required complete run.')
    if not times or any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError(f'{run}: invalid decision timing.')
    episodes = {}
    unavailable = fallback = attempts = truncated = invalid_final = 0
    for key, rows in groups.items():
        patients = {c['id'] for _, record in rows for c in record['input']['full_state'].get('characters', [])}
        treated, tagged = set(), set()
        for _, record in rows:
            chosen, review = action(record), multi_agent(record)
            if chosen['action_type'] == 'TREAT_PATIENT':
                treated.add(chosen.get('character_id'))
            if chosen['action_type'] == 'TAG_CHARACTER':
                tagged.add(chosen.get('character_id'))
            if review.get('review_status') not in ('reviewed', 'unavailable', 'bypassed'):
                raise ValueError(f'{run}: missing or unknown reviewer status.')
            unavailable += review.get('review_status') == 'unavailable'
            fallback += bool(review.get('driver_override'))
            final = review.get('final_review_response_validation') or {}
            invalid_final += final.get('status') == 'invalid'
            for attempt in review.get('review_attempts', []):
                attempts += 1
                generation = attempt.get('generation') or {}
                truncated += (generation.get('finish_reason') == 'length'
                              or generation.get('status') == 'incomplete')
        if not patients or None in treated or None in tagged:
            raise ValueError(f'{run}: cannot compute patient coverage for {key}.')
        filename = f'{key[0]}.{key[1]}.final_state_unstructured.json'
        if not (run / filename).exists():
            raise ValueError(f'{run}: missing completion artifact for {key}.')
        episodes['|'.join(key)] = {'score': read_score(run / filename),
                                 'initial_state': rows[0][1]['input']['full_state'],
                                 'treated': len(treated), 'tagged': len(tagged),
                                 'patients': len(patients), 'actions': len(rows)}
    runtime = read_json(run / 'run_runtime.json') if (run / 'run_runtime.json').exists() else {}
    server_path = run / 'server_runtime.json'
    server = read_json(server_path) if server_path.exists() else {}
    server_verified = (bool(server) and runtime.get('server_manifest_sha256') == digest(server_path)
                       and server.get('revision') == runtime.get('model_revision')
                       and server.get('model') == runtime.get('model') and not server.get('issues'))
    return {'path': str(run), 'episodes': episodes, 'times': times,
            'actions': len(records), 'review_attempts': attempts,
            'review_unavailable': unavailable, 'driver_fallback': fallback,
            'review_truncation': truncated, 'invalid_final': invalid_final,
            'audit': audit_status(run, len(records)), 'runtime': runtime,
            'server_verified': server_verified, 'server': server}


def promotion_report(baselines, candidates, expected_episodes=24, openai_references=None):
    if len(baselines) != 3 or len(candidates) != 3:
        raise ValueError('Promotion requires exactly three baseline and three candidate runs.')
    if len({str(Path(path).resolve()) for path in baselines + candidates}) != 6:
        raise ValueError('Six distinct run directories are required; repeating a run is not replication.')
    base = [summarize_run(path, expected_episodes) for path in baselines]
    cand = [summarize_run(path, expected_episodes) for path in candidates]
    reasons, pair_deltas = [], []
    for run in base + cand:
        if not run['audit']['complete']:
            reasons.append(f'{run["path"]}: consistency audit missing, incomplete, or stale')
        if not run['server_verified']:
            reasons.append(f'{run["path"]}: missing, mismatched, or failed server manifest')
    run_ids = [run['runtime'].get('run_id') for run in base + cand]
    if not all(run_ids) or len(set(run_ids)) != 6:
        reasons.append('Six distinct runtime run_id values are required for independent replication.')
    provenance = ('implementation_fingerprint', 'prompt_fingerprint', 'configuration_fingerprint',
                  'model', 'model_revision')
    reference = base[0]['runtime']
    for run in base + cand:
        for field in provenance:
            if not reference.get(field) or run['runtime'].get(field) != reference[field]:
                reasons.append(f'{run["path"]}: missing or mismatched {field}')
    for name, runs in [('baseline', base), ('candidate', cand)]:
        fingerprints = [run['runtime'].get('experiment_config_fingerprint') for run in runs]
        if not all(fingerprints) or len(set(fingerprints)) != 1:
            reasons.append(f'{name}: experiment configuration differs across repetitions or is unrecorded')
        first_server = runs[0]['server']
        for field in ('packages', 'command', 'environment', 'python', 'torch_cuda'):
            if field not in first_server or any(run['server'].get(field) != first_server[field] for run in runs):
                reasons.append(f'{name}: server {field} differs across repetitions or is unrecorded')
    # The GPU and driver remain fixed while serving packages/settings vary.
    def hardware_identity(run):
        hardware = run['server'].get('hardware', {})
        rows = hardware.get('stdout', '').splitlines()
        return [row.strip() for row in rows] if hardware.get('returncode') == 0 else []
    hardware = hardware_identity(base[0])
    if not hardware or any(hardware_identity(run) != hardware for run in base + cand):
        reasons.append('GPU hardware identity or driver differs across runs or is unrecorded')
    all_keys = set(base[0]['episodes'])
    scores = {'baseline': [], 'candidate': []}
    subgroup = defaultdict(lambda: {'baseline': [], 'candidate': []})
    for repetition, (before, after) in enumerate(zip(base, cand), 1):
        if set(before['episodes']) != all_keys or set(after['episodes']) != all_keys:
            reasons.append(f'repetition {repetition}: episode key mismatch')
            continue
        for key in sorted(all_keys):
            old, new = before['episodes'][key], after['episodes'][key]
            if old['initial_state'] != new['initial_state']:
                reasons.append(f'repetition {repetition}/{key}: initial state mismatch')
            left, right = old['score'], new['score']
            if (left is None) != (right is None):
                reasons.append(f'repetition {repetition}/{key}: numeric score availability mismatch')
            delta = None
            if left is not None and right is not None:
                delta = right - left
                scores['baseline'].append(left)
                scores['candidate'].append(right)
                group = key.rsplit('-', 1)[0]
                subgroup[group]['baseline'].append(left)
                subgroup[group]['candidate'].append(right)
            pair_deltas.append({'repetition': repetition, 'episode': key,
                                'baseline_score': left, 'candidate_score': right, 'delta': delta})
    metrics = {}
    for name, runs in [('baseline', base), ('candidate', cand)]:
        times = [value for run in runs for value in run['times']]
        total = sum(run['actions'] for run in runs)
        attempts = sum(run['review_attempts'] for run in runs)
        episodes = [ep for run in runs for ep in run['episodes'].values()]
        patients = sum(ep['patients'] for ep in episodes)
        metrics[name] = {'mean_decision_s': statistics.mean(times), 'p90_decision_s': percentile(times, .9),
                         'mean_score': statistics.mean(scores[name]) if scores[name] else None,
                         'numeric_scores': len(scores[name]),
                         'treatment_coverage': sum(ep['treated'] for ep in episodes) / patients,
                         'tagging_coverage': sum(ep['tagged'] for ep in episodes) / patients,
                         **{field + '_rate': sum(run[field] for run in runs) / total
                            for field in ('review_unavailable', 'driver_fallback', 'invalid_final')},
                         'review_truncation_rate': sum(run['review_truncation'] for run in runs) / attempts if attempts else None}
        for field in ('contradictions', 'invalid_evidence'):
            metrics[name][field + '_rate'] = sum(run['audit'].get(field, 0) for run in runs) / total
    for field in ('mean_score', 'treatment_coverage', 'tagging_coverage'):
        old, new = metrics['baseline'][field], metrics['candidate'][field]
        if old is None or new is None or new < old:
            reasons.append(f'{field}: unavailable or decreased')
    for field in ('review_unavailable_rate', 'driver_fallback_rate', 'invalid_final_rate', 'review_truncation_rate',
                  'contradictions_rate', 'invalid_evidence_rate', 'mean_decision_s', 'p90_decision_s'):
        old, new = metrics['baseline'][field], metrics['candidate'][field]
        if old is None or new is None or new > old:
            reasons.append(f'{field}: unavailable or increased')
    if metrics['candidate']['mean_decision_s'] >= metrics['baseline']['mean_decision_s']:
        reasons.append('No improvement in mean decision latency.')
    reference_metrics = None
    if openai_references:
        if len(openai_references) != 3 or len({str(Path(p).resolve()) for p in openai_references}) != 3:
            raise ValueError('OpenAI comparison requires three distinct reference runs.')
        if len({str(Path(p).resolve()) for p in baselines + candidates + openai_references}) != 9:
            raise ValueError('OpenAI reference directories must be distinct from the Qwen runs.')
        references = [summarize_run(path, expected_episodes) for path in openai_references]
        reference_ids = [run['runtime'].get('run_id') for run in references]
        if (not all(isinstance(value, str) and value.strip() for value in reference_ids)
                or len(set(reference_ids + run_ids)) != 9):
            raise ValueError('OpenAI references require three distinct, recorded run_id values, separate from Qwen.')
        if any(set(run['episodes']) != all_keys for run in references):
            raise ValueError('OpenAI reference episode keys do not match the Qwen runs.')
        environment_comparison = []
        for repetition, (before, reference_run) in enumerate(zip(base, references), 1):
            fingerprint = before['runtime'].get('implementation_fingerprint')
            if not fingerprint or reference_run['runtime'].get('implementation_fingerprint') != fingerprint:
                raise ValueError(f'OpenAI repetition {repetition}: implementation fingerprint mismatch or missing.')
            if (set(before['episodes']) != all_keys or any(
                    before['episodes'][key]['initial_state'] != reference_run['episodes'][key]['initial_state']
                    for key in all_keys)):
                raise ValueError(f'OpenAI repetition {repetition}: initial full states do not match the baseline.')
            environments = {}
            for name, run in [('qwen_baseline', before), ('openai_reference', reference_run)]:
                environments[name] = {
                    'model': run['runtime'].get('model'),
                    'client_python': run['runtime'].get('python'),
                    'client_versions': run['runtime'].get('versions'),
                    'server_packages': run['server'].get('packages'),
                    'server_hardware': run['server'].get('hardware'),
                }
            differences = [field for field, value in environments['qwen_baseline'].items()
                           if value != environments['openai_reference'][field]]
            environment_comparison.append({'repetition': repetition, **environments,
                                           'different_fields': differences})
        times = [value for run in references for value in run['times']]
        reference_mean = statistics.mean(times)
        if reference_mean <= 0:
            raise ValueError('OpenAI reference mean decision time must be positive.')
        reference_metrics = {'mean_decision_s': reference_mean, 'p90_decision_s': percentile(times, .9),
                             'qwen_candidate_mean_ratio': metrics['candidate']['mean_decision_s'] / reference_mean,
                             'environment_comparison': environment_comparison,
                             'limitations': 'Missing hosted serving versions/hardware remain unknown; client versions are not serving versions.'}
    return {'schema_version': 1, 'promote': not reasons, 'blocking_reasons': sorted(set(reasons)),
            'metrics': metrics, 'paired_episode_deltas': pair_deltas,
            'openai_reference': reference_metrics,
            'subgroup_scores': {key: {name: statistics.mean(values) for name, values in group.items()}
                                for key, group in subgroup.items()},
            'assumptions': ['Run lists pair repetitions in order.', 'Missing scores are excluded, never zero.',
                            'Aggregate score gate; subgroup and per-episode deltas are descriptive.',
                            'Semantic audits are separately supplied evidence, not inferred from provider completion.']}


def build_matrix(output, corpus, revision, *, include_live=False, python=sys.executable):
    """Write an executable, serial experiment matrix without starting any jobs."""
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('An immutable 40-character model revision is required.')
    output, corpus = Path(output).resolve(), Path(corpus).resolve()
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=False)
    trials = [{'name': name, 'profile': name} for name in ('baseline', 'reviewer_low', 'both_low')]
    trials += [{'name': name, 'profile': name} for name in ('both_medium', 'no_thinking')]
    trials += [{'name': f'budget_{budget}', 'profile': 'baseline', 'budget': budget}
               for budget in (512, 1024, 2048)]
    trials += [{'name': f'sequences_{n}', 'profile': 'baseline', 'max_num_seqs': n} for n in (1, 2, 4)]
    trials += [{'name': f'prefill_{n}', 'profile': 'baseline', 'max_num_batched_tokens': n}
               for n in (4096, 8192)]
    trials += [{'name': 'cache_off', 'profile': 'baseline', 'prefix_cache': False}]
    trials += [{'name': f'mtp_{n}', 'profile': 'baseline', 'mtp': n} for n in (1, 3)]
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
             '# Generated only. Executing this file launches inference and, when selected, live TA3 runs.',
             f'cd {shlex.quote(str(root))}', f'export QWEN_MODEL_REVISION={shlex.quote(revision)}',
             'server_pid=', "trap 'if [[ -n ${server_pid:-} ]]; then kill \"$server_pid\" 2>/dev/null || true; wait \"$server_pid\" 2>/dev/null || true; fi' EXIT"]
    corpus_check = ('import hashlib,json,pathlib,sys\n'
                    'p=pathlib.Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text())\n'
                    'assert m.get("replay_ready"), "Build a complete captured request corpus before launching the matrix"\n'
                    'assert hashlib.sha256((p/"requests.jsonl").read_bytes()).hexdigest()==m["requests_sha256"], "Corpus hash mismatch"')
    lines.append(shlex.join([python, '-c', corpus_check, str(corpus)]))
    readiness = ('import os,sys,time,urllib.request\n'
                 'deadline=time.monotonic()+900\n'
                 'while time.monotonic()<deadline:\n'
                 ' try:\n'
                 '  os.kill(int(sys.argv[1]),0)\n'
                 ' except ProcessLookupError:\n'
                 '  sys.exit("Server exited; inspect preflight.log/server.log")\n'
                 ' try:\n'
                 '  urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=3).close()\n'
                 '  sys.exit(0)\n'
                 ' except OSError:\n'
                 '  time.sleep(3)\n'
                 'sys.exit("Server readiness timed out")')
    for trial in trials:
        directory = output / 'runs' / trial['name']
        launch = [python, str(root / 'scripts/qwen_a6000.py'), '--revision', revision,
                  '--output', str(directory / 'server'), '--launch',
                  '--max-num-seqs', str(trial.get('max_num_seqs', 8)),
                  '--max-num-batched-tokens', str(trial.get('max_num_batched_tokens', 2048)),
                  '--mtp', str(trial.get('mtp', 0))]
        if trial.get('prefix_cache') is False:
            launch += ['--no-prefix-cache']
        trial['launch'] = launch
        lines += [f'\n# {trial["name"]}', f'mkdir -p {shlex.quote(str(directory))}',
                  f'{shlex.join(launch)} >{shlex.quote(str(directory / "preflight.log"))} 2>&1 &',
                  'server_pid=$!', f'{shlex.quote(python)} -c {shlex.quote(readiness)} "$server_pid"']
        trial['replay'] = []
        for cache_state in ('cold', 'warm'):
            cmd = [python, str(Path(__file__).resolve()), 'replay', '--corpus', str(corpus),
                   '--output', str(directory / cache_state), '--profile', trial['profile'],
                   '--cache-state', cache_state]
            if trial.get('budget'):
                cmd += ['--thinking-budget', str(trial['budget'])]
            trial['replay'].append(cmd)
            lines.append(shlex.join(cmd))
        trial['live'] = []
        if include_live:
            experiment = trial['profile'] if trial['profile'] in ('baseline', 'reviewer_low', 'both_low') else 'baseline'
            for repetition in range(1, 4):
                cmd = [python, '-m', 'align_system.cli.run_align_system',
                       f'+experiment=phase2_feb_openworld/qwen_a6000_{experiment}',
                       f'model_server_manifest={directory / "server" / "server_manifest.json"}',
                       f'hydra.run.dir={directory / f"live_{repetition}"}']
                for role in ('chat_model', 'reviewer.chat_model'):
                    prefix = f'adm.instance.{role}.extra_body'
                    if trial.get('budget'):
                        cmd += [f'++{prefix}.thinking_token_budget={trial["budget"]}']
                    if trial['profile'] == 'both_medium':
                        cmd += [f'{prefix}.chat_template_kwargs.reasoning_effort=medium']
                    if trial['profile'] == 'no_thinking':
                        cmd += [f'{prefix}.chat_template_kwargs.enable_thinking=false']
                trial['live'].append(cmd)
                lines.append(shlex.join(cmd))
        lines += ['kill "$server_pid"', 'wait "$server_pid" || true', 'server_pid=']
    manifest = {'schema_version': 1, 'revision': revision, 'corpus': str(corpus),
                'include_live': include_live, 'trials': trials,
                'notes': ['One-factor changes relative to baseline; no Cartesian product.',
                          'Port 8000 must be free before execution.',
                          'Cold and warm replay are separated; server restarts between trials.',
                          'Live runs use three repetitions; complete the semantic audits before gate.']}
    write_json(output / 'matrix.json', manifest)
    (output / 'run_matrix.sh').write_text('\n'.join(lines) + '\n')
    return {'matrix': str(output / 'matrix.json'), 'script': str(output / 'run_matrix.sh'),
            'trials': len(trials), 'started_jobs': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    corpus = sub.add_parser('corpus')
    corpus.add_argument('--run', type=Path, required=True)
    corpus.add_argument('--output', type=Path, required=True)
    screen = sub.add_parser('replay')
    screen.add_argument('--corpus', type=Path, required=True)
    screen.add_argument('--output', type=Path, required=True)
    screen.add_argument('--profile', choices=('baseline', 'reviewer_low', 'both_low', 'both_medium', 'no_thinking'), required=True)
    screen.add_argument('--thinking-budget', type=int, choices=(512, 1024, 2048))
    screen.add_argument('--cache-state', choices=('cold', 'warm', 'unspecified'), default='unspecified')
    screen.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    audit = sub.add_parser('audit-template')
    audit.add_argument('--run', type=Path, required=True)
    audit.add_argument('--output', type=Path, required=True)
    gate = sub.add_parser('gate')
    gate.add_argument('--baseline', type=Path, nargs=3, required=True)
    gate.add_argument('--candidate', type=Path, nargs=3, required=True)
    gate.add_argument('--openai-reference', type=Path, nargs=3)
    gate.add_argument('--output', type=Path, required=True)
    matrix = sub.add_parser('matrix', help='Generate runnable serial sweep commands; starts no jobs.')
    matrix.add_argument('--output', type=Path, required=True)
    matrix.add_argument('--corpus', type=Path, required=True)
    matrix.add_argument('--revision', required=True)
    matrix.add_argument('--python', default=sys.executable)
    matrix.add_argument('--include-live', action='store_true', help='Include three live TA3 runs per trial in the generated script.')
    args = parser.parse_args()
    if args.command == 'corpus':
        result = build_corpus(args.run, args.output)
    elif args.command == 'replay':
        result = replay(args.corpus, args.output, args.profile, args.base_url,
                        thinking_budget=args.thinking_budget, cache_state=args.cache_state)
    elif args.command == 'audit-template':
        result = audit_template(args.run, args.output)
    elif args.command == 'matrix':
        result = build_matrix(args.output, args.corpus, args.revision,
                              include_live=args.include_live, python=args.python)
    else:
        result = promotion_report(args.baseline, args.candidate, openai_references=args.openai_reference)
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get('promote') is False or result.get('passed_transport_screen') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
