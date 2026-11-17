"""Run/resume live LLM-only CAGE episodes, checkpointing each completed seed."""

from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import signal
from time import perf_counter

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import OmegaConf

from align_system.utils import logging
from align_system.utils.hydra_utils import initialize_with_custom_references


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _manifest(cfg):
    config = OmegaConf.to_container(cfg, resolve=False)
    config.pop('resume', None)
    # Include the scenario and prompt/component source in resume identity.
    root = Path(__file__).resolve().parents[1]
    sources = [root / 'prompt_engineering/outlines_prompts.py',
               root / 'interfaces/cage_llm_service.py',
               root / 'algorithms/cage_llm_adm_component.py',
               root / 'algorithms/cage_action_parameter_completion_adm_component.py',
               root / 'algorithms/outlines_baseline_adm_component.py',
               root / 'algorithms/huggingface_inference_engine.py',
               Path(to_absolute_path(cfg.interface.scenario_path))]
    return {'config': config, 'source_sha256': {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def run(cfg, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Also protect direct CLI launches, not just the shell batch launcher.
    with (output_dir / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(cfg, output_dir)


def _run_locked(cfg, output_dir):
    manifest = _manifest(cfg)
    manifest_path = output_dir / 'run_manifest.json'
    if manifest_path.exists():
        if not cfg.resume:
            raise ValueError('Output already exists; enable resume or choose a new output directory')
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume configuration/source differs; use a new output directory')
    else:
        _write_json(manifest_path, manifest)

    cfg = initialize_with_custom_references(cfg)
    interface, adm = cfg.interface, cfg.adm.instance
    engine = cfg.adm.structured_inference_engine
    logging.getLogger().setLevel(cfg.loglevel)
    results = []
    for index in range(interface.n_rollouts):
        seed = interface.start_seed + index
        result_path = output_dir / f'episode_{seed}.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('complete') and result['seed'] == seed:
                results.append(result)
                continue
        # A failed episode restarts at its original seed. Preserve previous
        # attempts so their paid calls remain available for cost accounting.
        attempts = output_dir / f'episode_{seed}'
        attempts.mkdir(exist_ok=True)
        attempt = 1
        while (attempts / f'attempt_{attempt:03d}').exists():
            attempt += 1
        attempt_dir = attempts / f'attempt_{attempt:03d}'
        attempt_dir.mkdir()
        interface.current_rollout = index
        scenario = interface.start_scenario()
        if hasattr(adm, 'reset_history'):
            adm.reset_history()
        state = scenario.get_state()
        rewards, cia_scores, action_times = [], [], []
        usage = {'prompt_tokens': 0, 'completion_tokens': 0}
        request_count = 0
        started = perf_counter()
        with (attempt_dir / 'steps.jsonl').open('w') as steps, \
                (attempt_dir / 'request_history.jsonl').open('w') as requests:
            while not state.scenario_complete:
                actions = scenario.get_available_actions()
                # LLM libraries/retry logic must not change the simulator RNG.
                python_rng, numpy_rng = random.getstate(), np.random.get_state()
                step_started = perf_counter()
                try:
                    action, _ = adm.choose_action(state, deepcopy(actions), None,
                                                  scenario_id=scenario.id())
                finally:
                    random.setstate(python_rng)
                    np.random.set_state(numpy_rng)
                    for request in engine.request_history:
                        request_count += 1
                        for key in usage:
                            usage[key] += (request.get('usage') or {}).get(key, 0)
                        requests.write(json.dumps({'step': scenario.scenario_count,
                                                   **request}) + '\n')
                    requests.flush()
                    engine.request_history.clear()
                elapsed = perf_counter() - step_started
                previous_state = state.to_dict()
                state = scenario.take_action(action)
                rewards.append(float(scenario.reward))
                cia_scores.append([float(v) for v in scenario.cia_scores])
                action_times.append(elapsed)
                steps.write(json.dumps({
                    'step': scenario.scenario_count - 1, 'state': previous_state,
                    'action': action.to_dict(), 'reward': rewards[-1],
                    'cia_scores': cia_scores[-1], 'decision_time_s': elapsed,
                }) + '\n')
                steps.flush()
        # An early terminal episode should not silently count as 100 steps.
        if len(rewards) != interface.episode_length:
            raise RuntimeError(f'Seed {seed} terminated after {len(rewards)} steps; '
                               f'expected {interface.episode_length}')
        result = {
            'complete': True, 'seed': seed, 'red_agent': interface.red_agent,
            'steps': len(rewards), 'return': sum(rewards),
            'rewards': rewards, 'cia_scores': cia_scores,
            'action_times_s': action_times, 'elapsed_s': perf_counter() - started,
            'requests': request_count, 'usage': usage, 'attempt': attempt,
        }
        _write_json(result_path, result)
        results.append(result)
        print(f'{interface.red_agent}: seed {seed} complete, '
              f'return={result["return"]:.2f}, requests={request_count}', flush=True)

    summary = {
        'complete': True, 'red_agent': interface.red_agent,
        'episodes': len(results), 'steps': sum(r['steps'] for r in results),
        'seeds': [r['seed'] for r in results],
        'mean_return': float(np.mean([r['return'] for r in results])),
        'mean_cia_drop_per_step': np.mean(
            [score for r in results for score in r['cia_scores']], axis=0).tolist(),
        # Summaries describe completed attempts; request JSONL also keeps failures.
        'requests': sum(r['requests'] for r in results),
        'usage': {key: sum(r['usage'][key] for r in results)
                  for key in ('prompt_tokens', 'completion_tokens')},
        'episode_elapsed_s': sum(r['elapsed_s'] for r in results),
    }
    _write_json(output_dir / 'results.json', summary)
    return summary


@hydra.main(version_base=None, config_path='../configs', config_name='action_based')
def main(cfg):
    # Flush outstanding request metadata on launcher termination as on Ctrl-C.
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    run(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == '__main__':
    main()
