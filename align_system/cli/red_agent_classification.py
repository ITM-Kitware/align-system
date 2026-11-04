import json
from copy import deepcopy
import atexit
import os
import platform
import shutil
from pathlib import Path
from importlib.metadata import version

from rich.logging import RichHandler
from rich.console import Console
from rich.highlighter import JSONHighlighter
from swagger_client.models import ActionTypeEnum
import hydra
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer

from align_system.utils import logging
from align_system.utils.hydra_utils import initialize_with_custom_references

log = logging.getLogger(__name__)
JSON_HIGHLIGHTER = JSONHighlighter()
from collections import deque


def _write_json(path, data):
    """Replace checkpoints atomically so interruptions leave a readable file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
    temporary.replace(path)


def _completed_replay_records(interface, records):
    """Keep only games with every expected window; restart partial games."""
    if getattr(interface, 'trajectory_dir', None) is None:
        raise ValueError('Resuming classification requires trajectory replay input')
    by_trajectory = {}
    for record in records:
        by_trajectory.setdefault(record['trajectory_id'], []).append(record)
    completed = set()
    expected_trajectories = set()
    for path in interface.trajectory_paths:
        trajectory_id = str(path.relative_to(interface.trajectory_dir))
        expected_trajectories.add(trajectory_id)
        saved = by_trajectory.get(trajectory_id, [])
        if not saved:
            continue
        with path.open() as stream:
            trajectory = json.load(stream)
        end = min(interface.episode_length, *(len(trajectory[field])
                  for field in interface.observation_fields))
        if interface.window_size is None:
            expected = [trajectory_id]
        else:
            expected = [f'{trajectory_id}#steps={start}:{start + interface.window_size}'
                        for start in range(interface.start_step,
                                           end - interface.window_size + 1,
                                           interface.window_stride)]
        saved_ids = [record['scenario_id'] for record in saved]
        if len(saved_ids) != len(set(saved_ids)) or not set(saved_ids) <= set(expected):
            raise ValueError(f'Checkpoint windows do not match replay input: {trajectory_id}')
        if saved_ids == expected:
            completed.add(trajectory_id)
    if set(by_trajectory) - expected_trajectories:
        raise ValueError('Checkpoint contains trajectories outside the configured replay input')
    return [record for record in records if record['trajectory_id'] in completed], completed


def _validate_resume_manifest(previous, current):
    # Output schema and token budget may change when repairing a failed run.
    # Replay inputs, labels, model, and seeds must still describe the same run.
    for key, value in previous['replay'].items():
        if current['replay'].get(key) != value:
            raise ValueError(f'Cannot resume with changed replay setting {key}; use a new output directory')
    for key in ('random_seed', 'numpy_random_seed', 'torch_random_seed', 'python_hash_seed'):
        if previous.get(key) != current.get(key):
            raise ValueError(f'Cannot resume with changed {key}; use a new output directory')
    old_engine, new_engine = previous.get('inference_engine'), current.get('inference_engine')
    if old_engine and new_engine and 'HuggingFaceInferenceEngine(' in old_engine:
        old_settings = json.loads(old_engine.split('(', 1)[1][:-1])
        new_settings = json.loads(new_engine.split('(', 1)[1][:-1])
        for key in ('model_name', 'provider', 'temperature', 'top_p', 'seed'):
            if old_settings.get(key) != new_settings.get(key):
                raise ValueError(f'Cannot resume with changed {key}; use a new output directory')


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="action_based")
def main(cfg: DictConfig) -> None:
    choice_info = None

    # Using the hydra generated output directory for the run
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    classification_results_path = Path(output_dir) / 'classification_results.json'
    previous_results = None
    if classification_results_path.exists():
        if not cfg.get('resume', False):
            raise ValueError('Classification results already exist; enable resume or use a new output directory')
        previous_results = json.loads(classification_results_path.read_text())
        if previous_results.get('complete'):
            log.info(f'Skipping completed classification run: {output_dir}')
            return

    logfile_path = None
    if cfg.save_log:
        logfile_path = os.path.join(output_dir, "align_system.log")

    raw_logfile_path = None
    if cfg.save_raw_log:
        raw_logfile_path = os.path.join(output_dir, "raw_align_system.log")

    save_input_output_to_path = None
    if cfg.save_input_output:
        save_input_output_to_path = os.path.join(output_dir, "input_output.json")

    save_alignment_score_to_path = None
    if cfg.save_scoring_output:
        save_alignment_score_to_path = os.path.join(output_dir, "scores.json")

    save_alignment_targets_to_path = None
    if cfg.save_alignment_targets:
        save_alignment_targets_to_path = os.path.join(output_dir, "targets")
        os.makedirs(save_alignment_targets_to_path, exist_ok=True)

    save_timing_to_path = None
    if cfg.save_timing:
        save_timing_to_path = os.path.join(output_dir, "timing.json")

    # Set log level on root logger (such that child loggers respect
    # the set log level)
    root_logger = logging.getLogger()
    root_logger.setLevel(cfg.loglevel)

    def _log_closer_closure(logfile_filehandler):
        # Capture the logfile filehandler object we want to close
        # in a closure to keep a reference to it when it comes
        # time to close it
        def _close_logfile():
            logfile_filehandler.close()

        return _close_logfile

    if logfile_path is not None:
        logfile = open(logfile_path, 'a' if cfg.get('resume', False) else 'w')
        # Ensure the opened logfile is closed when the program exits
        atexit.register(_log_closer_closure(logfile))

        filehandler = RichHandler(
            console=Console(file=logfile, color_system=None))
        root_logger.addHandler(filehandler)

    if raw_logfile_path is not None:
        # Using Python stdlib logging.FileHandler
        from logging import FileHandler
        filehandler = FileHandler(raw_logfile_path)

        root_logger.addHandler(filehandler)

    if cfg.get('force_determinism', False) or 'torch_random_seed' in cfg:
        import torch
        torch_seed = cfg.get('torch_random_seed', 0)
        log.info(f"Setting `torch.manual_seed` to: {torch_seed}")
        torch.manual_seed(torch_seed)

    if cfg.get('force_determinism', False) or 'torch_use_deterministic_algorithms' in cfg:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        log.info("Setting `torch_use_deterministic_algorithms` to True")
        torch.use_deterministic_algorithms(
            cfg.get('torch_use_deterministic_algorithms', True),
            warn_only=True)

    if cfg.get('force_determinism', False) or 'random_seed' in cfg:
        import random
        random_seed = cfg.get('random_seed', 0)
        log.info(f"Setting `random.seed` to: {random_seed}")
        random.seed(random_seed)

    if cfg.get('force_determinism', False) or 'numpy_random_seed' in cfg:
        import numpy as np
        numpy_random_seed = cfg.get('numpy_random_seed', 0)
        log.info(f"Setting `numpy.random.seed` to: {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if cfg.get('force_determinism', False) or 'sort_available_actions' in cfg:
        log.info("Setting `sort_available_actions` to True")
        sort_available_actions = cfg.get('sort_available_actions', True)
    else:
        sort_available_actions = False

    # Seed before constructors can consume randomness (including local models).
    cfg = initialize_with_custom_references(cfg)
    interface = cfg.interface
    adm = cfg.adm.instance
    inference_engine = cfg.adm.get('structured_inference_engine')
    request_history = getattr(inference_engine, 'request_history', None)
    request_history_path = os.path.join(output_dir, 'request_history.json')

    def save_request_history():
        if request_history is not None:
            _write_json(request_history_path, request_history)

    # Record resolved settings, without persisting credentials. Hydra also saves
    # the experiment configuration and CLI overrides in .hydra/ for each run.
    manifest = {
        'python_version': platform.python_version(),
        'force_determinism': cfg.get('force_determinism', False),
        'random_seed': cfg.get('random_seed'),
        'numpy_random_seed': cfg.get('numpy_random_seed'),
        'torch_random_seed': cfg.get('torch_random_seed'),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'inference_engine': (inference_engine.cache_repr()
                             if hasattr(inference_engine, 'cache_repr') else None),
        'replay': {
            'n_rollouts': getattr(interface, 'n_rollouts', None),
            'episode_length': getattr(interface, 'episode_length', None),
            'start_step': getattr(interface, 'start_step', None),
            'window_size': getattr(interface, 'window_size', None),
            'window_stride': getattr(interface, 'window_stride', None),
            'observation_fields': getattr(interface, 'observation_fields', None),
            'red_agent_labels': getattr(interface, 'red_agent_labels', None),
            'trajectory_paths': [str(path) for path in getattr(interface, 'trajectory_paths', [])],
        },
    }
    if request_history is not None:
        manifest['huggingface_hub_version'] = version('huggingface-hub')
        manifest['reproducibility_note'] = (
            'Fixed seeds and temperature zero request reproducible inference; '
            'hosted providers do not guarantee identical outputs across requests '
            'or backend updates. Retain request_history.json and the input trajectories.')
        log.info(manifest['reproducibility_note'])
    classification_records = []
    completed_trajectories = set()
    manifest_path = Path(output_dir) / 'run_manifest.json'
    if cfg.get('resume', False) and manifest_path.exists():
        _validate_resume_manifest(json.loads(manifest_path.read_text()), manifest)
        if previous_results is not None:
            if set(previous_results['labels']) - set(interface.red_agent_labels):
                raise ValueError('Checkpoint labels do not match the configured classifier')
            classification_records, completed_trajectories = _completed_replay_records(
                interface, previous_results['predictions'])
        # Preserve old request indices, including failed attempts and windows
        # that will be regenerated. New calls append to the existing history.
        if request_history is not None and Path(request_history_path).exists():
            request_history.extend(json.loads(Path(request_history_path).read_text()))
        elif any(record.get('inference_request_indices') for record in classification_records):
            raise ValueError('Cannot resume: request_history.json is missing')
        backup_root = Path(output_dir) / 'resume_backups'
        backup_root.mkdir(exist_ok=True)
        attempt = 1
        while (backup_root / f'attempt_{attempt:03d}').exists():
            attempt += 1
        backup = backup_root / f'attempt_{attempt:03d}'
        backup.mkdir()
        for filename in ('classification_results.json', 'run_manifest.json', 'timing.json'):
            source = Path(output_dir) / filename
            if source.exists():
                shutil.copy2(source, backup / filename)
        manifest['resume'] = {
            'completed_trajectories': sorted(completed_trajectories),
            'retained_predictions': len(classification_records),
            'previous_request_count': len(request_history) if request_history is not None else 0,
            'backup_dir': str(backup),
        }
        log.info(f'Resuming: keeping {len(completed_trajectories)} completed games '
                 f'({len(classification_records)} predictions); restarting unfinished games')
    elif previous_results is not None:
        raise ValueError('Cannot resume: run_manifest.json is missing')
    _write_json(manifest_path, manifest)
    save_request_history()

    if 'filter_tag_character' in cfg:
        filter_tag_character = cfg.get('filter_tag_character', True)
    else:
        filter_tag_character = False

    # HACK: need to invoke 'load_model' for ADMs that require it,
    # maybe it makes more sense to load_model in the init method for
    # those ADMs
    if hasattr(adm, 'load_model'):
        adm.load_model()

    # Capture inputs and outputs in a similar format to what's used by
    # our internal evaluation framework code
    inputs_outputs = []

    session_alignment_scores = []

    # Capture time it takes to choose each action
    action_times = { "scenarios": [] }
    def _compute_time_stats(times_s):
        n_times = len(times_s)
        total_time_s = sum(times_s)
        return {
            "n_actions_taken": n_times,
            "total_time_s": total_time_s,
            "avg_time_s": total_time_s / n_times if n_times else 0.,
            "max_time_s": max(times_s) if n_times else 0.,
            "raw_times_s": times_s
        }
    predictions = dict.fromkeys(previous_results['labels'], 0) if previous_results else {}
    n_labeled, n_correct = 0, 0
    for record in classification_records:
        predictions[record['predicted_label']] += 1
        if record.get('true_label') is not None:
            n_labeled += 1
            n_correct += record['predicted_label'] == record['true_label']
        action_times['scenarios'].append({
            'scenario_id': record['scenario_id'],
            **_compute_time_stats([record['classification_time_s']]),
        })

    def save_classification_results(complete=False):
        results = {
            'complete': complete,
            'evaluation_unit': 'window' if getattr(interface, 'window_size', None) is not None else 'trajectory',
            'n_trajectories': len({record['trajectory_id'] for record in classification_records}),
            'window_size': getattr(interface, 'window_size', None),
            'window_stride': getattr(interface, 'window_stride', None),
            'labels': list(predictions),
            'prediction_counts': predictions,
            'n_labeled': n_labeled,
            'n_correct': n_correct,
            'accuracy': n_correct / n_labeled if n_labeled else None,
            'predictions': classification_records,
        }
        _write_json(classification_results_path, results)

    save_classification_results()
    # Loop through available scenarios
    while scenario := interface.start_scenario():
        if getattr(scenario, 'trajectory_id', None) in completed_trajectories:
            continue
        collected_observations = deque(maxlen = 2)
        if scenario.id() == '':
            log.info("Next scenario ID is blank, assuming we're done, exiting")
            break
        log.info(f'[bold]*Scenario ID*[/bold]: {scenario.id()}')

        # Reset any decision or chat history for a new scenario
        if hasattr(adm, 'reset_history'):
            log.info("[bold]*Resetting choice history*[/bold]")
            adm.reset_history()

        if 'alignment_target' in cfg:
            alignment_target = cfg.alignment_target

            # Alignment targets specified in hydra configs require
            # some nested conversion to dict (from OmegaConf objects)
            # otherwise this can cause some downstream issues with
            # serialization
            alignment_target.kdma_values = [OmegaConf.to_container(c)
                                            if isinstance(c, DictConfig) else c
                                            for c in alignment_target.kdma_values]
        elif cfg.align_to_target:
            alignment_target = scenario.get_alignment_target()
        else:
            alignment_target = None

        log.info('[bold]*ALIGNMENT TARGET*[/bold]')
        if alignment_target is None:
            log.info('Alignment target is `None`')
        else:
            log.info(alignment_target)
            if save_alignment_targets_to_path is not None:
                alignment_target_path = os.path.join(save_alignment_targets_to_path, f"{alignment_target.id}.json")

                with open(alignment_target_path, "w") as f:
                    json.dump(alignment_target.to_dict(), f, indent=2)

        current_state = scenario.get_state()
        scenario_complete = current_state.scenario_complete

        # Tracking these to prevent getting stuck in a loop
        noop_actions = []

        sce_times_s = []

        last_scene_id = None

        while not scenario_complete:
            collected_observations.append(current_state.unstructured)
            current_scene_id = current_state.meta_info.scene_id
            if last_scene_id != current_scene_id:
                log.info(f"[bold]*CHANGED SCENE TO*: {current_scene_id}[/bold]",
                         extra={"markup": True})
                last_scene_id = current_scene_id

            available_actions = scenario.get_available_actions()
            action_to_take = available_actions[0]
            
            if save_input_output_to_path is not None:
                with open(save_input_output_to_path, 'w') as f:
                    json.dump(inputs_outputs, f, indent=2)

            try:
                if hasattr(action_to_take, "intent_action") and action_to_take.intent_action:
                    current_state = scenario.intend_action(action_to_take)
                else:
                    current_state = scenario.take_action(action_to_take)
            except Exception as e:
                log.info(action_to_take)
                raise e

            scenario_complete = current_state.scenario_complete

        
        # Replays start complete; their state contains only the current window
        # (or the full slice when window_size is null). The deque is live-only.
        if collected_observations:
            observation = ''
            for i in range(len(collected_observations)):
                observation += 'Observation {0}:'.format(i) + '\n'
                observation += collected_observations[i] + '\n'
            current_state.unstructured = observation

        log.info(current_state.unstructured)
        available_actions = scenario.get_available_Red_actions()
        for action in available_actions:
            predictions.setdefault(action.name, 0)

        request_start = len(request_history) if request_history is not None else 0
        classification_started = timer()
        try:
            choose_action_result = adm.choose_action(
                current_state,
                [deepcopy(a) for a in available_actions],
                alignment_target if cfg.align_to_target else None,
                scenario_id=scenario.id(),
                **cfg.adm.get('inference_kwargs', {}))
        finally:
            save_request_history()
        classification_time_s = timer() - classification_started
        action_times['scenarios'].append({
            'scenario_id': scenario.id(),
            **_compute_time_stats([classification_time_s]),
        })

        # Handle choose action result (for backwards compatibility if no choice_info)
        if isinstance(choose_action_result, tuple):
            action_to_take, choice_info = choose_action_result
            if 'choice_info' in choice_info:
                # Handle pipeline_adm
                choice_info = choice_info['choice_info']
        else:
            action_to_take = choose_action_result
            choice_info = {}


        log.info("[bold]*ACTION BEING TAKEN*[/bold]",
                    extra={"markup": True})
        
        if isinstance(action_to_take, dict):
                log.info(json.dumps(action_to_take, indent=4),
                         extra={"highlighter": JSON_HIGHLIGHTER})
        else:
            log.info(json.dumps(action_to_take.to_dict(), indent=4),
                        extra={"highlighter": JSON_HIGHLIGHTER})
            
        predictions[action_to_take.name] += 1

        # Only trajectory replays know which red agent was attacking
        true_label = None
        if hasattr(scenario, 'get_red_agent_label'):
            true_label = scenario.get_red_agent_label()
            log.info(f'True red agent: {scenario.red_agent_name} ({true_label}), '
                     f'predicted: {action_to_take.name}')
            if true_label is not None:
                n_labeled += 1
                n_correct += action_to_take.name == true_label

        classification_records.append({
            'scenario_id': scenario.id(),
            'trajectory_id': getattr(scenario, 'trajectory_id', scenario.id()),
            'window_index': getattr(scenario, 'window_index', None),
            'window_start': getattr(scenario, 'window_start', None),
            'window_end': getattr(scenario, 'window_end', None),
            'window_size': getattr(scenario, 'window_size', None),
            'classification_time_s': classification_time_s,
            'seed': getattr(scenario, 'trajectory', {}).get('seed'),
            'red_agent_name': getattr(scenario, 'red_agent_name', None),
            'true_label': true_label,
            'predicted_label': action_to_take.name,
            'inference_request_indices': (list(range(request_start, len(request_history)))
                                          if request_history is not None else []),
        })
        save_classification_results()

    if save_timing_to_path is not None:
        all_times = []
        for sce in action_times["scenarios"]:
            all_times.extend(sce["raw_times_s"])

        action_times.update(_compute_time_stats(all_times))

        with open(save_timing_to_path, 'w') as f:
            json.dump(action_times, f, indent=2)

    if len(session_alignment_scores) > 0:
        if save_alignment_score_to_path is not None:
            with open(save_alignment_score_to_path, 'w') as f:
                json.dump([(s if isinstance(s, dict) else s.to_dict())
                           for s in session_alignment_scores], f, indent=2)
    log.info('Red agents classification results:')
    log.info(predictions)
    if n_labeled > 0:
        log.info(f'Accuracy: {n_correct}/{n_labeled} ({n_correct / n_labeled:.1%})')
    save_classification_results(complete=True)
    log.info(f'Classification results saved to {classification_results_path}')

if __name__ == "__main__":
    main()
