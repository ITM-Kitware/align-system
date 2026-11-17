#!/usr/bin/env python3
"""Classify Red agents from verified Blue-observation captures.

Run with python -m align_system.cli.blue_observation_classification.
Preparation needs hybrid and Monitor capture directories, each with an audited
summary.json. To finish a prepared sweep's excluded seeds with disjoint prompt
examples, supply --base-run-root and a separate --run-root. Preparation, workers
and reports use this same CLI. Model inference is imported only for workers.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from align_system.cli.red_agent_classification_report import CIA_LABELS, build_report, write_report
from align_system.prompt_engineering.cage_blue_prompts import (
    COLUMNS, EXAMPLES_DIR, SYSTEM_PROMPT_PATH,
    CAGEBlueObservationClassificationPrompt, format_blue_window,
)

LABELS = {label.lower().replace('b-line', 'bline').replace('-', '_'): label for label in CIA_LABELS}
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent
WINDOWS = (5, 7, 9)
START, END = 10, 100
CONDITIONS = ('hybrid', 'monitor')
HISTORICAL_CHECKS = ('fresh_process_repeat_identical', 'historical_initial_network_matches',
                     'historical_metric_scores_match', 'historical_action_trace_matches')
SCHEMA = json.dumps({'type': 'object', 'properties': {
    'action_choice': {'type': 'string', 'enum': list(CIA_LABELS)}},
    'required': ['action_choice'], 'additionalProperties': False})

EXAMPLE_SEEDS = (103, 104, 105)
EXAMPLE_ENDS = (37, 58, 79)
COHORT_NOTE = ('Original 17 episodes/class use the original P4 examples; the remaining '
               '3 episodes/class use replacement examples outside all 20 evaluation seeds. '
               'Each evaluated episode is absent from its own prompt. These are two prompt cohorts.')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def file_digest(path):
    return digest(Path(path).read_bytes())


def metadata(episode):
    return {k: v for k, v in episode.items() if k != 'steps'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_prompt():
    raw = SYSTEM_PROMPT_PATH.read_bytes()
    provenance = json.loads((EXAMPLES_DIR / 'provenance.json').read_text())
    if digest(raw) != provenance['source_system_sha256']:
        raise ValueError('P4 prompt differs from its recorded provenance')
    system = raw.decode()
    if Counter(re.findall(r'Correct training label: (.+)', system)) != Counter(dict.fromkeys(CIA_LABELS, 3)):
        raise ValueError('P4 must contain exactly three examples per CIA class')
    if provenance['example_window_size'] != 7 or len(provenance['example_seeds']) != 3:
        raise ValueError('Unexpected P4 example provenance')
    return system, provenance


def blue_record(step):
    # Explicit allowlist: no Red actions, rewards, raw simulator state or labels.
    rows = [[cell.strip() for cell in line.split('|')[1:-1]]
            for line in step['defender_observation'].splitlines() if line.startswith('|')]
    if not rows or rows[0] != COLUMNS:
        raise ValueError('Missing or unexpected defender observation table')
    action = {key: step['blue_action'][key] for key in ('Action', 'Status', 'Host', 'Subnet')
              if key in step['blue_action']}
    return {'step': step['step'], 'blue_action': action, 'blue_table': rows[1:]}


def load_captures(directory, condition, excluded_seeds):
    directory = directory.resolve()
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'complete' or not summary.get('episodes'):
        raise ValueError(f'Capture audit must be complete: {summary_path}')
    if (condition == 'hybrid') != (summary.get('condition') == 'active_hybrid_ppo'):
        raise ValueError(f'Wrong capture condition for {condition}: {summary_path}')
    selected, excluded, seen = [], [], set()
    for entry in summary['episodes']:
        agent, seed = entry['class_directory'], entry['seed']
        if agent not in LABELS:
            continue
        if type(seed) is not int:
            raise ValueError('Every episode needs an integer seed for example exclusion')
        key = (agent, seed)
        if key in seen:
            raise ValueError(f'Duplicate class/episode seed: {key}')
        seen.add(key)
        if seed in excluded_seeds:
            excluded.append({'class_directory': agent, 'seed': seed})
            continue
        path = (directory / entry['capture_path']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f'Capture path leaves its audit directory: {path}')
        raw = path.read_bytes()
        if digest(raw) != entry['capture_sha256']:
            raise ValueError(f'Capture digest differs from audit: {path}')
        capture = json.loads(raw)
        if capture['seed'] != seed:
            raise ValueError(f'Capture seed differs from audit: {path}')
        expected_agent = ('BlineJumping_' if agent.startswith('bline') else 'MeanderJumping_') + agent[-1].upper()
        if capture['red_agent_name'] != expected_agent:
            raise ValueError(f'Attacker class differs from audit: {path}')
        if condition == 'hybrid':
            checks = ('repeat_capture_identical', 'monitor_control_reproduced', 'initial_network_matches_monitor')
            valid = all(capture.get(check) is True for check in checks)
        else:
            valid = all(entry.get(check) is True for check in HISTORICAL_CHECKS)
        if not valid:
            raise ValueError(f'Capture reproducibility verification failed: {path}')
        if len(capture['steps']) < END:
            raise ValueError(f'Capture has fewer than {END} steps: {path}')
        steps = [blue_record(step) for step in capture['steps'][:END]]
        if [step['step'] for step in steps] != list(range(END)):
            raise ValueError(f'Noncontiguous episode: {path}')
        if condition == 'monitor' and any(step['blue_action']['Action'] != 'Monitor' for step in steps):
            raise ValueError(f'Monitor capture contains another Blue policy: {path}')
        format_blue_window(steps)  # Validate all rows, action results and host inventories now.
        selected.append({'trajectory_id': f'{agent}/seed_{seed}', 'seed': seed,
                         'class_directory': agent, 'true_label': LABELS[agent],
                         'red_agent_name': expected_agent, 'source_path': str(path),
                         'source_sha256': digest(raw), 'steps': steps})
    groups = {agent: {e['seed'] for e in selected if e['class_directory'] == agent} for agent in LABELS}
    if not all(groups.values()) or any(seeds != next(iter(groups.values())) for seeds in groups.values()):
        raise ValueError(f'Need the same nonempty episode seed set for every CIA class: {directory}')
    return sorted(selected, key=lambda e: (e['class_directory'], e['seed'])), {
        'summary_path': str(summary_path), 'summary_sha256': digest(summary_path.read_bytes()),
        'excluded_episodes': excluded,
    }


def model_settings(name):
    backend = os.environ.get('BACKEND', 'local')
    if backend == 'local':
        backend = 'transformers' if name == 'cisco' else 'vllm'
    settings = {
        'backend': backend,
        'model_name': os.environ.get(f'{name.upper()}_MODEL',
            'fdtn-ai/Foundation-Sec-8B-Instruct' if name == 'cisco' else 'Qwen/Qwen3.8-27B'),
        'max_tokens': 128, 'seed': int(os.environ.get('RUN_SEED', '0')),
    }
    if backend == 'hf':
        response_format = os.environ.get('CISCO_RESPONSE_FORMAT', 'json_schema') if name == 'cisco' else 'json_schema'
        if response_format not in ('json_schema', 'json_object', 'null'):
            raise ValueError('Invalid CISCO_RESPONSE_FORMAT')
        settings.update(provider=os.environ.get(f'{name.upper()}_PROVIDER',
                            'featherless-ai' if name == 'cisco' else 'deepinfra'),
                        temperature=0.0, response_format=None if response_format == 'null' else response_format,
                        extra_body={'chat_template_kwargs': {'enable_thinking': False}} if name == 'qwen' else None)
    elif backend == 'vllm':
        settings.update(tensor_parallel_size=int(os.environ.get(f'{name.upper()}_TP', '1' if name == 'cisco' else '2')),
                        dtype='bfloat16', max_model_len=int(os.environ.get('VLLM_MAX_MODEL_LEN', '32768')),
                        gpu_memory_utilization=0.90)
    elif backend == 'transformers':
        settings.update(device='cuda:0', precision='half',
                        model_kwargs={'local_files_only': True},
                        tokenizer_kwargs={'local_files_only': True})
    else:
        raise ValueError('BACKEND must be local, hf or vllm')
    if settings['seed'] < 0 or (backend == 'vllm' and
            (settings['tensor_parallel_size'] < 1 or settings['max_model_len'] <= settings['max_tokens'])):
        raise ValueError('Invalid inference seed, GPU count or context limit')
    return settings


def worker_count(name):
    if name == 'cisco' and os.environ.get('BACKEND', 'local') == 'local':
        per_gpu = int(os.environ.get('CISCO_WORKERS_PER_GPU', '2'))
        if per_gpu not in (1, 2):
            raise ValueError('CISCO_WORKERS_PER_GPU must be 1 or 2')
        return 2 * per_gpu
    return 1


def create_engine(settings):
    settings = dict(settings)
    backend = settings.pop('backend')
    if backend == 'hf':
        from huggingface_hub import get_token
        if not get_token():
            raise ValueError('Set HF_TOKEN or run hf auth login before hosted inference')
        from align_system.algorithms.huggingface_inference_engine import HuggingFaceInferenceEngine
        return HuggingFaceInferenceEngine(**settings)
    if backend == 'transformers':
        import torch
        from align_system.algorithms.cage_local_inference_engine import CAGELocalInferenceEngine
        torch.manual_seed(settings.pop('seed'))
        return CAGELocalInferenceEngine(**settings)
    from align_system.algorithms.cage_vllm_inference_engine import CAGEVLLMInferenceEngine
    return CAGEVLLMInferenceEngine(**settings)


def check_runtime(name):
    """Check cached weights and imports without loading a model or claiming a GPU."""
    import shutil
    from huggingface_hub import hf_hub_download
    import torch

    settings = model_settings(name)
    if torch.version.cuda is None:
        raise ValueError(f'{sys.executable} needs CUDA-enabled torch')
    if settings['backend'] == 'transformers':
        from align_system.algorithms.cage_local_inference_engine import CAGELocalInferenceEngine
    elif settings['backend'] == 'vllm':
        from vllm import LLM
        from vllm.sampling_params import StructuredOutputsParams
        import flashinfer.comm
        from flashinfer.jit.cpp_ext import get_cuda_path
        cuda = Path(get_cuda_path())
        if not shutil.which('ninja') or not (cuda / 'bin/nvcc').is_file() or not (cuda / 'include/cuda_runtime.h').is_file():
            raise ValueError('vLLM requires ninja on PATH and a complete CUDA toolkit')
    else:
        raise ValueError('--check-runtime is for local models')
    model = settings['model_name']
    # Resolve the cached revision without requiring unrelated Hub files such as
    # README/images that the runtime never loads.
    snapshot = Path(model) if Path(model).is_dir() else Path(
        hf_hub_download(model, 'config.json', local_files_only=True)).parent
    index = snapshot / 'model.safetensors.index.json'
    weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not all((snapshot / filename).is_file() for filename in
               (*weights, 'config.json', 'tokenizer_config.json', 'tokenizer.json')):
        raise ValueError(f'Incomplete cached model files: {snapshot}')
    print(f'{name} runtime ready: {sys.executable}; CUDA {torch.version.cuda}; '
          f'cached weights {snapshot}. No model loaded.', flush=True)


def messages(system, episode, start, window):
    user = CAGEBlueObservationClassificationPrompt()(
        format_blue_window(episode['steps'][start:start + window]), CIA_LABELS)
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


def write_run_inputs(root, system, datasets, manifest):
    """Save the same audited artifacts for either prompt cohort."""
    write_json(root / 'sweep_manifest.json', manifest)
    (root / 'system_prompt.txt').write_text(system)
    for condition, episodes in datasets.items():
        for episode in episodes:
            write_json(root / 'replay_inputs' / condition / f"{episode['trajectory_id']}.json", episode)
        for window in WINDOWS:
            write_json(root / 'prompt_previews' / f'{condition}_{window}step.json',
                       messages(system, episodes[0], START, window))


def capture_directories(root, overrides=None):
    """Resolve explicit inputs, environment settings, then a prepared run's audit."""
    overrides = overrides or {}
    manifest_path = root / 'sweep_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    directories = {}
    for condition in CONDITIONS:
        directory = overrides.get(condition) or os.environ.get(f'{condition.upper()}_CAPTURE_DIR')
        if not directory:
            audit = previous.get('audits', {}).get(condition, {})
            if audit.get('summary_path'):
                directory = Path(audit['summary_path']).parent
        if not directory:
            raise ValueError(f'Provide --{condition}-capture-dir or '
                             f'{condition.upper()}_CAPTURE_DIR for an audited capture directory')
        directories[condition] = Path(directory).expanduser().resolve()
    return directories


def source_hashes(models):
    """Record package sources needed by the selected backends, independent of launchers."""
    sources = {Path(__file__).resolve(),
               PACKAGE_ROOT / 'prompt_engineering/cage_blue_prompts.py',
               PACKAGE_ROOT / 'cli/red_agent_classification_report.py'}
    for settings in models.values():
        backend = settings['backend']
        if backend == 'hf':
            sources.add(PACKAGE_ROOT / 'algorithms/huggingface_inference_engine.py')
        elif backend == 'transformers':
            sources.update((PACKAGE_ROOT / 'algorithms/cage_local_inference_engine.py',
                            PACKAGE_ROOT / 'algorithms/outlines_inference_engine.py'))
        elif backend == 'vllm':
            sources.add(PACKAGE_ROOT / 'algorithms/cage_vllm_inference_engine.py')
        else:
            raise ValueError(f'Unknown inference backend: {backend}')
    return {str(path.relative_to(SOURCE_ROOT)): digest(path.read_bytes())
            for path in sorted(sources)}


def prepare(root, dry_run=False, capture_dirs=None):
    root = root.expanduser().resolve()
    system, provenance = load_prompt()
    excluded = set(provenance['example_seeds'])
    datasets, audits = {}, {}
    for condition, directory in capture_directories(root, capture_dirs).items():
        datasets[condition], audits[condition] = load_captures(
            directory, condition, excluded)
    if {e['seed'] for e in datasets['hybrid']} != {e['seed'] for e in datasets['monitor']}:
        raise ValueError('Hybrid and Monitor must use matching episode seeds')
    models = {name: model_settings(name) for name in ('cisco', 'qwen')}
    manifest = {'protocol': 'blue_only_p4_5_7_9', 'prompt_provenance': provenance,
                'example_episode_seeds_excluded_from_all_classes_and_conditions': sorted(excluded),
                'windows': list(WINDOWS), 'start_step': START, 'episode_end': END, 'stride': 1,
                'models': models, 'audits': audits,
                'workers_per_model': {name: worker_count(name) for name in ('cisco', 'qwen')},
                'episodes': {condition: [{k: v for k, v in e.items() if k != 'steps'} for e in episodes]
                             for condition, episodes in datasets.items()},
                'sources_sha256': source_hashes(models)}
    path = root / 'sweep_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != manifest:
            changed = sorted(key for key in previous.keys() | manifest.keys()
                             if previous.get(key) != manifest.get(key))
            if 'sources_sha256' in changed:
                changed.extend(filename for filename in
                               previous['sources_sha256'].keys() | manifest['sources_sha256'].keys()
                               if previous['sources_sha256'].get(filename)
                               != manifest['sources_sha256'].get(filename))
            raise ValueError(f'Inputs, prompt or settings changed ({", ".join(changed)}); '
                             f'use a new RUN_ROOT instead of {root}')
        # Retain the recorded provenance when resuming an identical run.
        manifest = previous
    elif root.exists() and any(p.name != '.sweep.lock' for p in root.iterdir()):
        raise ValueError(f'Outputs exist without a matching sweep manifest: {root}')
    if not dry_run:
        write_run_inputs(root, system, datasets, manifest)
    total = 0
    print(f'P4: 18 examples, 7-step demonstrations; excluded episode seeds {sorted(excluded)}', flush=True)
    for condition, episodes in datasets.items():
        counts = [len(episodes) * (END - START - window + 1) for window in WINDOWS]
        total += sum(counts)
        print(f'{condition}: {len(episodes) // 6} episodes/class; 5/7/9-step calls per model: {counts}', flush=True)
        print(f'  Capture audit: {audits[condition]["summary_path"]}', flush=True)
    for name, settings in manifest['models'].items():
        print(f'{name}: {settings["model_name"]} via {settings["backend"]}'
              + (f'/{settings["provider"]}' if settings['backend'] == 'hf' else
                 f' ({settings["tensor_parallel_size"]} GPUs)' if settings['backend'] == 'vllm' else
                 f' ({manifest["workers_per_model"][name]} FP16 worker shards)'), flush=True)
    print(f'Both models: {2 * total:,} calls before retries. Output: {root}', flush=True)
    return system, datasets, manifest


def read_predictions(path, expected):
    """Reject wrong windows and duplicates; tolerate a crash-truncated final line."""
    if not path.exists():
        return {}
    records = {}
    with path.open('rb+') as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                stream.truncate(position)
                break
            record = json.loads(line)
            key = record['scenario_id']
            if key in records or key not in expected or record['predicted_label'] not in CIA_LABELS:
                raise ValueError(f'Duplicate or unexpected saved prediction: {path}: {key}')
            if any(record.get(k) != v for k, v in expected[key].items()):
                raise ValueError(f'Saved prediction metadata changed: {path}: {key}')
            records[key] = record
    return records


def expected_windows(episodes, window):
    expected, queries = {}, {}
    for episode in episodes:
        for index, start in enumerate(range(START, END - window + 1)):
            key = f"{episode['trajectory_id']}#steps={start}:{start + window}"
            expected[key] = {k: episode[k] for k in ('trajectory_id', 'seed', 'true_label', 'red_agent_name')}
            expected[key].update(scenario_id=key, window_index=index, window_start=start,
                                 window_end=start + window, window_size=window)
            queries[key] = (episode, start)
    return expected, queries


def save_report(output, ordered):
    write_json(output / 'classification_results.json',
               {'complete': True, 'labels': list(CIA_LABELS), 'predictions': ordered})
    report = build_report(ordered)
    write_report(report, ordered, output)
    print(f'{output}: accuracy {report["overall_accuracy"]:.2%}', flush=True)


def merge_shards(root, name, datasets, count):
    for condition, episodes in datasets.items():
        for window in WINDOWS:
            output = root / name / condition / f'{window}step'
            all_expected, _ = expected_windows(episodes, window)
            saved = {}
            for index in range(count):
                shard = output / 'shards' / f'worker_{index:02d}'
                marker = shard / 'classification_results.json'
                if not marker.exists() or not json.loads(marker.read_text()).get('complete'):
                    raise ValueError(f'Worker {index} is incomplete: {output}')
                expected, _ = expected_windows(episodes[index::count], window)
                records = read_predictions(shard / 'predictions.jsonl', expected)
                if records.keys() != expected.keys() or saved.keys() & records.keys():
                    raise ValueError(f'Missing or duplicated worker predictions: {output}')
                saved.update(records)
            if saved.keys() != all_expected.keys():
                raise ValueError(f'Incomplete merged predictions: {output}')
            save_report(output, [saved[key] for key in all_expected])


def run_model(root, name, system, datasets, manifest, shard_index=0, shard_count=1):
    if not 0 <= shard_index < shard_count:
        raise ValueError('Invalid worker index/count')
    engine = None
    for condition, episodes in datasets.items():
        assigned = episodes[shard_index::shard_count]
        if not assigned:
            raise ValueError('Each worker must have at least one episode')
        for window in WINDOWS:
            output = root / name / condition / f'{window}step'
            if shard_count > 1:
                output = output / 'shards' / f'worker_{shard_index:02d}'
            output.mkdir(parents=True, exist_ok=True)
            expected, queries = expected_windows(assigned, window)
            saved = read_predictions(output / 'predictions.jsonl', expected)
            print(f'{name}/{condition}/{window}step: {len(saved)}/{len(expected)} completed', flush=True)
            with (output / 'predictions.jsonl').open('a') as predictions, \
                    (output / 'request_history.jsonl').open('a') as history:
                for key, metadata in expected.items():
                    if key in saved:
                        continue
                    if engine is None:
                        engine = create_engine(manifest['models'][name])
                    episode, start = queries[key]
                    dialog = messages(system, episode, start, window)
                    started = perf_counter()
                    try:
                        result = engine.run_inference(engine.dialog_to_prompt(dialog), SCHEMA)
                    finally:
                        for request in engine.request_history:
                            history.write(json.dumps({'scenario_id': key, 'request': request}) + '\n')
                        history.flush()
                        engine.request_history.clear()
                    if set(result) != {'action_choice'} or result['action_choice'] not in CIA_LABELS:
                        raise ValueError(f'Invalid classification: {result}')
                    record = {**metadata, 'predicted_label': result['action_choice'],
                              'classification_time_s': perf_counter() - started}
                    predictions.write(json.dumps(record) + '\n')
                    predictions.flush()
                    saved[key] = record
                    if len(saved) % 100 == 0:
                        print(f'{name}/{condition}/{window}step: {len(saved)}/{len(expected)}', flush=True)
            ordered = [saved[key] for key in expected]
            save_report(output, ordered)


def load_base(base_root):
    base = read_json(base_root / 'sweep_manifest.json')
    if (base['protocol'] != 'blue_only_p4_5_7_9' or base['windows'] != list(WINDOWS)
            or (base['start_step'], base['episode_end'], base['stride']) != (START, END, 1)):
        raise ValueError('Base sweep has a different window protocol')
    for filename, expected in base['sources_sha256'].items():
        # Hosted transport is recorded by the original manifest even for local
        # runs, but neither local inference engine imports or executes it.
        if (filename == 'align_system/algorithms/huggingface_inference_engine.py'
                and all(s['backend'] != 'hf' for s in base['models'].values())):
            continue
        if file_digest(SOURCE_ROOT / filename) != expected:
            raise ValueError(f'Base sweep source changed: {filename}')
    prompt = (base_root / 'system_prompt.txt').read_text()
    if digest(prompt.encode()) != base['prompt_provenance']['source_system_sha256']:
        raise ValueError('Base system prompt differs from its provenance')
    missing = set(base['example_episode_seeds_excluded_from_all_classes_and_conditions'])
    if len(missing) != 3 or missing != set(base['prompt_provenance']['example_seeds']):
        raise ValueError('Expected exactly three excluded example seeds')
    all_datasets, extra = {}, {}
    for condition in ('hybrid', 'monitor'):
        audit = base['audits'][condition]
        path = Path(audit['summary_path'])
        if file_digest(path) != audit['summary_sha256']:
            raise ValueError(f'Base capture audit changed: {path}')
        episodes, _ = load_captures(path.parent, condition, set())
        if Counter(e['true_label'] for e in episodes) != Counter(dict.fromkeys(CIA_LABELS, 20)):
            raise ValueError(f'Expected 20 audited trajectories per class: {condition}')
        retained = {e['trajectory_id']: metadata(e) for e in episodes if e['seed'] not in missing}
        if retained != {e['trajectory_id']: e for e in base['episodes'][condition]}:
            raise ValueError(f'Base evaluation inputs changed: {condition}')
        extra[condition] = [e for e in episodes if e['seed'] in missing]
        if Counter(e['true_label'] for e in extra[condition]) != Counter(dict.fromkeys(CIA_LABELS, 3)):
            raise ValueError(f'Missing supplemental trajectories: {condition}')
        all_datasets[condition] = episodes
    if {e['seed'] for e in all_datasets['hybrid']} != {e['seed'] for e in all_datasets['monitor']}:
        raise ValueError('Hybrid and Monitor evaluation seeds differ')
    return base, prompt, all_datasets, extra


def simulator_digest(root):
    result = hashlib.sha256()
    files = sorted((root / 'CybORG').rglob('*.py'))
    if not files:
        raise ValueError(f'No simulator sources found: {root}')
    for path in files:
        result.update(str(path.relative_to(root)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def capture_specification(reference):
    scenario = Path(os.environ.get('SCENARIO_PATH') or reference['scenario_path']).expanduser().resolve()
    if os.environ.get('CAGE_ROOT'):
        cage_root = Path(os.environ['CAGE_ROOT']).expanduser().resolve()
    else:
        cage_root = next((parent for parent in scenario.parents if (parent / 'CybORG').is_dir()), None)
        if cage_root is None:
            raise ValueError('Set CAGE_ROOT to the audited simulator checkout containing CybORG')
    if file_digest(scenario) != reference['scenario_sha256']:
        raise ValueError('Example scenario must match the base capture scenario')
    fingerprint = simulator_digest(cage_root)
    if fingerprint != reference['cage_python_source_sha256']:
        raise ValueError('Example simulator sources must match the base capture simulator')
    return {'scenario_path': str(scenario), 'scenario_sha256': file_digest(scenario),
            'cage_root': str(cage_root), 'cage_python_source_sha256': fingerprint,
            'collector_sha256': file_digest(__file__), 'steps': END,
            'python_hash_seed': '0'}


def collect_monitor(agent, seed, spec):
    """CPU-only simulation; store only the Blue evidence needed for examples."""
    sys.path.insert(0, spec['cage_root'])
    import numpy as np
    from CybORG import CybORG
    from CybORG.Agents import GreenAgent
    from CybORG.Agents.SimpleAgents.B_line_resilience import BlineJumping_C, BlineJumping_I, BlineJumping_A
    from CybORG.Agents.SimpleAgents.Meander_Resilience import MeanderJumping_C, MeanderJumping_I, MeanderJumping_A
    from CybORG.Agents.Wrappers import BlueTableWrapper
    from CybORG.Shared.Actions import Monitor

    attackers = dict(zip(('bline_c', 'bline_i', 'bline_a', 'meander_c', 'meander_i', 'meander_a'),
                        (BlineJumping_C, BlineJumping_I, BlineJumping_A,
                         MeanderJumping_C, MeanderJumping_I, MeanderJumping_A)))
    random.seed(seed)
    np.random.seed(seed)
    env = CybORG(spec['scenario_path'], 'sim', agents={'Red': attackers[agent], 'Green': GreenAgent})
    env.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    wrapper = BlueTableWrapper(env, output_mode='table')
    wrapper.reset(agent='Blue')
    records = []
    for index in range(spec['steps']):
        result = wrapper.step(agent='Blue', action=Monitor(agent='Blue', session=0))
        records.append(blue_record({
            'step': index, 'defender_observation': str(result.observation),
            'blue_action': {'Action': 'Monitor', 'Host': 'Monitor',
                            'Status': str(env.get_observation('Blue')['success'])}}))
        if result.done and index + 1 != spec['steps']:
            raise ValueError('Example episode ended before the required horizon')
    format_blue_window(records)
    return {'class_directory': agent, 'seed': seed, 'steps': records}


def collect_repeated(agent, seed, spec):
    """Repeat in fresh processes, with fixed Python/NumPy/hash seeds."""
    captures = []
    with TemporaryDirectory(prefix='cia-blue-example-') as directory:
        spec_path = Path(directory) / 'spec.json'
        write_json(spec_path, spec)
        for repeat in range(2):
            output = Path(directory) / f'repeat_{repeat}.json'
            command = [sys.executable, '-m', 'align_system.cli.blue_observation_classification',
                       '--capture-example', agent,
                       str(seed), '--capture-spec', str(spec_path), '--capture-output', str(output)]
            result = subprocess.run(command, env={**os.environ, 'PYTHONHASHSEED': '0',
                                    'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'},
                                    capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f'Example capture failed for {agent}/{seed}:\n{result.stderr[-4000:]}')
            captures.append(read_json(output))
    if captures[0] != captures[1]:
        raise ValueError(f'Fresh-process repeats differ for {agent}/{seed}')
    return {**captures[0], 'repeat_capture_identical': True, 'capture_specification': spec}


def prepare_examples(root, reference_episode, seeds):
    reference = read_json(reference_episode['source_path'])
    spec = capture_specification(reference)
    specification = {'capture': spec, 'example_seeds': list(seeds),
                     'control_source_sha256': reference_episode['source_sha256']}
    path = root / 'example_captures' / 'manifest.json'
    audit = read_json(path) if path.exists() else {'specification': specification, 'episodes': {}}
    if audit['specification'] != specification:
        raise ValueError('Example settings changed; use a new completion RUN_ROOT')
    if not audit.get('control_matches_base'):
        control = collect_repeated(reference_episode['class_directory'], reference_episode['seed'], spec)
        if control['steps'] != reference_episode['steps']:
            raise ValueError('Example runtime does not reproduce the audited Monitor control')
        audit['control_matches_base'] = True
        write_json(path, audit)
    examples = {}
    for seed in seeds:
        for agent in LABELS:
            name = f'{agent}/seed_{seed}.json'
            target = path.parent / name
            if name in audit['episodes']:
                if file_digest(target) != audit['episodes'][name]:
                    raise ValueError(f'Example capture changed: {target}')
                capture = read_json(target)
            else:
                if target.exists():
                    raise ValueError(f'Untracked example capture: {target}')
                capture = collect_repeated(agent, seed, spec)
                write_json(target, capture)
                audit['episodes'][name] = file_digest(target)
                write_json(path, audit)
                print(f'Example captured and repeated: {agent}/seed_{seed}', flush=True)
            if (capture['seed'] != seed or capture['class_directory'] != agent
                    or capture['capture_specification'] != spec or capture['repeat_capture_identical'] is not True):
                raise ValueError(f'Example metadata mismatch: {target}')
            examples[agent, seed] = capture
    if simulator_digest(Path(spec['cage_root'])) != spec['cage_python_source_sha256']:
        raise ValueError('Simulator changed during example generation')
    return examples, audit


def replacement_prompt(original, examples, seeds):
    header, separator, _ = original.partition('\nTraining example 1:')
    if not separator:
        raise ValueError('Cannot identify the original P4 instruction/example boundary')
    prompt, provenance = header, []
    for seed, end in zip(seeds, EXAMPLE_ENDS):
        for agent, label in LABELS.items():
            window = examples[agent, seed]['steps'][end - 7:end]
            if len(window) != 7:
                raise ValueError('Replacement example must have seven steps')
            text = format_blue_window(window)
            prompt += f'\nTraining example {len(provenance) + 1}:\n{text}\nCorrect training label: {label}\n'
            provenance.append({'class_directory': agent, 'true_label': label, 'seed': seed,
                               'window_start': end - 7, 'window_end': end})
    return prompt, provenance


def prepare_completion(root, base_root, seeds=EXAMPLE_SEEDS, dry_run=False):
    root, base_root = root.resolve(), base_root.resolve()
    if root == base_root or root.is_relative_to(base_root) or base_root.is_relative_to(root):
        raise ValueError('Completion RUN_ROOT must be separate from the original sweep')
    base, original, all_datasets, extra = load_base(base_root)
    evaluation_seeds = {e['seed'] for eps in all_datasets.values() for e in eps}
    if (len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or set(seeds) & evaluation_seeds):
        raise ValueError('Use three distinct example seeds outside ALL 20 evaluation seeds')
    for name, settings in base['models'].items():
        if settings != model_settings(name) or base['workers_per_model'][name] != worker_count(name):
            raise ValueError(f'{name} runtime settings/workers must match the base sweep')
    print(f'Completion: {sorted(set(e["seed"] for e in extra["monitor"]))}; '
          f'replacement Monitor examples: {list(seeds)}', flush=True)
    counts = [len(extra['monitor']) * (END - START - w + 1) for w in WINDOWS]
    print('3 trajectories/class/condition; 18 trajectories/condition; '
          f'5/7/9-step calls: {counts} per model/condition; {4 * sum(counts):,} total.', flush=True)
    manifest_path = root / 'sweep_manifest.json'
    old = read_json(manifest_path) if manifest_path.exists() else None
    if old:
        # Validate before any preparation writes, including resume after code edits.
        if (old['base_manifest_sha256'] != file_digest(base_root / 'sweep_manifest.json')
                or old['sources_sha256'] != source_hashes(base['models'])
                or old['prompt_provenance']['example_seeds'] != list(seeds)):
            raise ValueError('Completion settings or sources changed; use a new RUN_ROOT')
        examples, audit = load_saved_examples(root, seeds)
    elif dry_run:
        reference = read_json(all_datasets['monitor'][0]['source_path'])
        capture_specification(reference)
        print('Preparation will generate and repeat 18 Monitor example episodes on CPU.', flush=True)
        return None, extra, None
    else:
        if root.exists() and any(p.name not in {'.sweep.lock', 'example_captures'} for p in root.iterdir()):
            raise ValueError('Completion outputs exist without a matching manifest')
        examples, audit = prepare_examples(root, all_datasets['monitor'][0], seeds)
    system, example_windows = replacement_prompt(original, examples, seeds)
    manifest = {'protocol': 'blue_only_p4_missing_three_5_7_9',
                'base_run_root': str(base_root), 'base_manifest_sha256': file_digest(base_root / 'sweep_manifest.json'),
                'interpretation': COHORT_NOTE, 'windows': list(WINDOWS),
                'start_step': START, 'episode_end': END, 'stride': 1,
                'models': base['models'], 'workers_per_model': base['workers_per_model'],
                'source_validation': 'Base classifier, prompt, report and selected inference sources must match.',
                'audits': base['audits'], 'sources_sha256': source_hashes(base['models']),
                'prompt_provenance': {'variant': 'P4-replacement-examples', 'example_seeds': list(seeds),
                    'examples_per_class': 3, 'example_window_size': 7, 'examples': example_windows,
                    'source_system_sha256': digest(system.encode()),
                    'original_system_sha256': base['prompt_provenance']['source_system_sha256'],
                    'example_captures_manifest_sha256': digest((root / 'example_captures/manifest.json').read_bytes())},
                'episodes': {condition: [metadata(e) for e in eps] for condition, eps in extra.items()}}
    if old and old != manifest:
        raise ValueError('Completion manifest changed; use a new RUN_ROOT')
    if old and (root / 'system_prompt.txt').read_text() != system:
        raise ValueError('Saved replacement prompt changed')
    if not dry_run:
        write_run_inputs(root, system, extra, manifest)
    return system, extra, manifest


def load_saved_examples(root, seeds):
    audit = read_json(root / 'example_captures/manifest.json')
    if audit['specification']['example_seeds'] != list(seeds) or not audit['control_matches_base']:
        raise ValueError('Invalid replacement example audit')
    examples = {}
    for seed in seeds:
        for agent in LABELS:
            name = f'{agent}/seed_{seed}.json'
            path = root / 'example_captures' / name
            if file_digest(path) != audit['episodes'][name]:
                raise ValueError(f'Example capture changed: {path}')
            examples[agent, seed] = read_json(path)
    return examples, audit


def completed_records(directory, expected):
    """Read original results without invoking the replay reader's truncation repair."""
    data = read_json(directory / 'classification_results.json')
    if data.get('complete') is not True or data.get('labels') != list(CIA_LABELS):
        raise ValueError(f'Incomplete or incompatible results: {directory}')
    records = {}
    for row in data['predictions']:
        key = row['scenario_id']
        if (key in records or key not in expected or row['predicted_label'] not in CIA_LABELS
                or any(row.get(k) != v for k, v in expected[key].items())):
            raise ValueError(f'Duplicate or incompatible prediction: {directory}: {key}')
        records[key] = row
    if records.keys() != expected.keys():
        raise ValueError(f'Missing predictions: {directory}')
    return records


def merge_completed(root, base_root, manifest, require_all=False):
    base, _, all_datasets, extra = load_base(base_root)
    pending = []
    for name in ('qwen', 'cisco'):
        for condition, episodes in all_datasets.items():
            for window in WINDOWS:
                relative = Path(name) / condition / f'{window}step'
                original_dir, extra_dir = base_root / relative, root / relative
                if not all((p / 'classification_results.json').exists() for p in (original_dir, extra_dir)):
                    pending.append(str(relative))
                    continue
                original_expected, _ = expected_windows(base['episodes'][condition], window)
                extra_expected, _ = expected_windows(extra[condition], window)
                rows = {}
                inputs = []
                for cohort, directory, expected, prompt_hash in (
                        ('original_p4', original_dir, original_expected, base['prompt_provenance']['source_system_sha256']),
                        ('replacement_examples', extra_dir, extra_expected, manifest['prompt_provenance']['source_system_sha256'])):
                    part = completed_records(directory, expected)
                    if rows.keys() & part.keys():
                        raise ValueError('Original and supplemental predictions overlap')
                    rows.update({key: {**row, 'prompt_cohort': cohort, 'system_prompt_sha256': prompt_hash,
                                      'source_file': str(directory / 'classification_results.json')}
                                 for key, row in part.items()})
                    inputs.append({'cohort': cohort, 'path': str(directory / 'classification_results.json'),
                                   'sha256': file_digest(directory / 'classification_results.json')})
                expected, _ = expected_windows(episodes, window)
                if rows.keys() != expected.keys():
                    raise ValueError('Combined results do not cover all twenty trajectories/class')
                ordered = [rows[key] for key in expected]
                output = root / 'combined_20_per_class' / relative
                save_report(output, ordered)
                report = read_json(output / 'classification_report.json')
                report.update(interpretation=COHORT_NOTE, prompt_cohorts={
                    'original_p4': {'trajectories_per_class': 17, 'example_seeds': base['prompt_provenance']['example_seeds']},
                    'replacement_examples': {'trajectories_per_class': 3, 'example_seeds': manifest['prompt_provenance']['example_seeds']}})
                write_json(output / 'classification_report.json', report)
                with (output / 'classification_report.txt').open('a') as stream:
                    stream.write('\n' + COHORT_NOTE + '\n')
                write_json(output / 'merge_provenance.json', {'interpretation': COHORT_NOTE, 'inputs': inputs})
    if pending:
        print('Combined reports pending: ' + ', '.join(pending), flush=True)
        if require_all:
            raise ValueError('Both sweeps must finish before all combined reports can be generated')
    else:
        print(f'All 12 combined reports: {root / "combined_20_per_class"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path,
                        help='Output directory (required except for --check-runtime)')
    parser.add_argument('--hybrid-capture-dir', type=Path,
                        help='Audited hybrid captures; defaults to HYBRID_CAPTURE_DIR or the saved run')
    parser.add_argument('--monitor-capture-dir', type=Path,
                        help='Audited Monitor captures; defaults to MONITOR_CAPTURE_DIR or the saved run')
    parser.add_argument('--base-run-root', type=Path,
                        help='Finish this sweep\'s excluded seeds using replacement prompt examples')
    parser.add_argument('--example-seeds', nargs=3, type=int,
                        help='Replacement example seeds with --base-run-root (default: 103 104 105)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--model', choices=('cisco', 'qwen'))
    mode.add_argument('--report-only', choices=('cisco', 'qwen'))
    mode.add_argument('--check-runtime', choices=('cisco', 'qwen'))
    mode.add_argument('--merge-ready', action='store_true',
                      help='Merge available reports with --base-run-root')
    mode.add_argument('--merge-only', action='store_true',
                      help='Merge reports, requiring both sweeps complete; needs --base-run-root')
    mode.add_argument('--capture-example', nargs=2, metavar=('CLASS', 'SEED'), help=argparse.SUPPRESS)
    parser.add_argument('--capture-spec', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--capture-output', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.capture_example:
            if args.capture_spec is None or args.capture_output is None:
                raise ValueError('--capture-example requires --capture-spec and --capture-output')
            agent, seed = args.capture_example
            write_json(args.capture_output, collect_monitor(agent, int(seed), read_json(args.capture_spec)))
            return
        if args.check_runtime:
            check_runtime(args.check_runtime)
            return
        if args.run_root is None:
            raise ValueError('Provide --run-root for the output directory')
        args.run_root = args.run_root.expanduser().resolve()
        if args.base_run_root:
            args.base_run_root = args.base_run_root.expanduser().resolve()
            if args.hybrid_capture_dir or args.monitor_capture_dir:
                raise ValueError('--base-run-root uses its saved capture audit; omit capture-directory flags')
        elif args.merge_ready or args.merge_only or args.example_seeds is not None:
            raise ValueError('--merge-ready, --merge-only and --example-seeds require --base-run-root')
        # The shell prepares once under its sweep lock. Workers only validate;
        # they never rewrite shared inputs or manifests concurrently.
        if not (args.dry_run or args.prepare_only) and not (args.run_root / 'sweep_manifest.json').is_file():
            raise ValueError('Run the shell launcher or --prepare-only before starting workers')
        if args.base_run_root:
            system, datasets, manifest = prepare_completion(
                args.run_root, args.base_run_root, tuple(args.example_seeds or EXAMPLE_SEEDS),
                dry_run=not args.prepare_only)
        else:
            system, datasets, manifest = prepare(args.run_root, not args.prepare_only, {
                'hybrid': args.hybrid_capture_dir, 'monitor': args.monitor_capture_dir})
        if args.model:
            if args.shard_count != manifest['workers_per_model'][args.model]:
                raise ValueError('Worker count differs from sweep manifest')
            run_model(args.run_root, args.model, system, datasets, manifest, args.shard_index, args.shard_count)
        elif args.report_only:
            merge_shards(args.run_root, args.report_only, datasets, manifest['workers_per_model'][args.report_only])
        elif args.merge_ready or args.merge_only:
            merge_completed(args.run_root, args.base_run_root, manifest, require_all=args.merge_only)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
