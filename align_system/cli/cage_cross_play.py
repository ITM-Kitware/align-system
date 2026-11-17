"""Evaluate fixed CIA PPO policies against the schedules of an existing batch."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from omegaconf import OmegaConf

from align_system.cli import cage_hybrid as shared
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ('bline_c', 'bline_i', 'bline_a', 'meander_c', 'meander_i', 'meander_a',
            'all_cia_hybrid')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(root, policy, seed):
    suffix = policy if policy == 'all_cia_hybrid' else policy[:-1] + policy[-1].upper()
    return Path(root) / f'PPO_base_jumping_{suffix}_seed{seed}.zip'


def load_jobs(reference, checkpoint_dir, policies, episodes=None, deterministic_policy=True):
    jobs = []
    for manifest_path in sorted(reference.glob('*/*/run_manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        cfg = copy.deepcopy(manifest['config'])
        # Reuse the oracle's plan validation and environment construction only.
        # Fixed-policy execution below never constructs a classifier or switches Blue.
        cfg.pop('inference_engine', None)
        cfg['mode'] = 'oracle'
        # Old reference batches sampled PPO actions. New evaluations use the
        # CAGE2 deterministic protocol unless explicitly overridden.
        cfg['deterministic_policy'] = deterministic_policy
        if episodes is not None:
            cfg['n_rollouts'] = episodes
        plan = shared.episode_plan(OmegaConf.create(cfg))
        if episodes is None and [list(p) for p in plan] != manifest['episodes']:
            raise ValueError(f'Reference plan mismatch: {manifest_path}')
        scenario = Path(cfg['scenario_path'])
        if sha256(scenario) != manifest['source_sha256'][str(scenario)]:
            raise ValueError(f'Scenario changed since reference run: {scenario}')
        group, case = manifest_path.parent.relative_to(reference).parts
        for policy in policies:
            paths = {str(s): str(checkpoint(checkpoint_dir, policy, s))
                     for s in sorted({p for _, p in plan})}
            digests = {path: sha256(path) for path in paths.values()}
            for path, digest in digests.items():
                expected = manifest['checkpoint_sha256'].get(path)
                if expected is not None and expected != digest:
                    raise ValueError(f'Checkpoint changed since reference run: {path}')
            jobs.append(dict(policy=policy, group=group, case=case, config=cfg,
                             episodes=[list(p) for p in plan], checkpoints=paths,
                             checkpoint_sha256=digests,
                             reference_manifest=str(manifest_path),
                             reference_manifest_sha256=sha256(manifest_path)))
    if len(jobs) != 12 * len(policies):
        raise ValueError('Expected the complete 12-case reference suite')
    return jobs


def checked_policy(path, env):
    with shared.preserve_rng():
        policy = shared.load_policy(path)
    if (policy.observation_space.shape != env.observation_space.shape
            or policy.action_space.n != env.action_space.n):
        raise ValueError(f'Checkpoint spaces do not match the reference environment: {path}')
    return policy


def run_episode(cfg, seed, policy_seed, policy_name, env, observation, policy, destination):
    metric = ResilienceMetric()
    schedule = shared.attacker_schedule(cfg)
    transitions = {phase['step']: phase['red_agent'] for phase in schedule}
    active = cfg.red_agent
    rewards, scores = [], []
    started = time.monotonic()
    with (destination / 'steps.jsonl').open('w') as stream:
        for step in range(cfg.episode_length):
            switched = step > 0 and step in transitions
            if switched:
                active = transitions[step]
                shared.switch_attacker(env, active)
            action, _ = policy.predict(observation, deterministic=cfg.deterministic_policy)
            observation, reward, terminated, truncated, _ = env.step(int(action))
            cia = [float(v) for v in metric.calculate_scores(
                env, blue_action=env.get_last_action('Blue'))]
            rewards.append(float(reward))
            scores.append(cia)
            record = dict(step=step, policy_seed=policy_seed, active_red_agent=active,
                          true_label=shared.LABELS[active], attacker_switched=switched,
                          selected_policy=policy_name, selection_source='fixed_policy',
                          action_index=int(action), reward=float(reward), cia_scores=cia,
                          blue_action=shared.action_record(env, 'Blue'),
                          red_action=shared.action_record(env, 'Red'))
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if terminated or truncated:
                break
    if len(rewards) != cfg.episode_length:
        raise ValueError(f'Episode ended early: {len(rewards)} steps')
    return dict(complete=True, mode='fixed_policy', fixed_policy=policy_name,
                seed=seed, policy_seed=policy_seed, red_agent=cfg.red_agent,
                red_schedule=schedule, steps=len(rewards), rewards=rewards,
                return_=sum(rewards), cia_scores=scores, elapsed_s=time.monotonic() - started)


def worker(job_file):
    job = json.loads(job_file.read_text())
    output = job_file.parent
    cfg = OmegaConf.create(job['config'])
    stored_config = {**job['config'], 'mode': 'fixed_policy', 'fixed_policy': job['policy']}
    manifest = {**job, 'config': stored_config}
    manifest_path = output / 'run_manifest.json'
    with (output / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f'Resume manifest differs: {output}')
        shared._write_json(manifest_path, manifest)
        results, loaded_seed, policy = [], None, None
        for seed, policy_seed in job['episodes']:
            result_path = output / f'episode_{seed}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if (result.get('complete') and result['seed'] == seed
                        and result['policy_seed'] == policy_seed and result['steps'] == cfg.episode_length):
                    results.append(result)
                    continue
            attempt = 1
            while (output / f'episode_{seed}/attempt_{attempt:03d}').exists():
                attempt += 1
            destination = output / f'episode_{seed}/attempt_{attempt:03d}'
            destination.mkdir(parents=True)
            env, observation = shared.make_environment(cfg, seed)
            try:
                if loaded_seed != policy_seed:
                    policy = checked_policy(job['checkpoints'][str(policy_seed)], env)
                    loaded_seed = policy_seed
                result = run_episode(cfg, seed, policy_seed, job['policy'], env, observation,
                                     policy, destination)
            finally:
                env.close()
            result['return'] = result.pop('return_')
            result['attempt'] = attempt
            shared._write_json(result_path, result)
            results.append(result)
            print(f"{job['policy']}/{job['group']}/{job['case']}: seed={seed}, "
                  f"RL seed={policy_seed}, return={result['return']:.2f}", flush=True)
        summary = dict(complete=True, mode='fixed_policy', fixed_policy=job['policy'],
                       red_agent=cfg.red_agent, red_schedule=shared.attacker_schedule(cfg),
                       episodes=len(results), steps=sum(r['steps'] for r in results),
                       mean_return=float(np.mean([r['return'] for r in results])),
                       mean_cia_drop_per_step=np.mean([r['cia_scores'] for r in results],
                                                     axis=(0, 1)).tolist())
        shared._write_json(output / 'results.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-batch', type=Path,
                        default=ROOT / 'outputs/cia_hybrid_local_full_observability/Foundation-Sec-8B-Instruct/9step/run_4ynkkfcy')
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=Path('/data/shared/aaai_aics2027/rl_checkpoints/CAGE2_CIA'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--episodes', type=int, help='Override for smoke tests; default matches reference')
    parser.add_argument('--deterministic-policy', action=argparse.BooleanOptionalAction,
                        default=True, help='Use greedy PPO actions (default), independent of the reference batch')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not 1 <= args.parallel <= 84 or len(set(args.policies)) != len(args.policies):
        parser.error('Use 1..84 workers and distinct policies')
    jobs = load_jobs(args.reference_batch.resolve(), args.checkpoint_dir.resolve(),
                     args.policies, args.episodes, args.deterministic_policy)
    source_paths = [Path(__file__), Path(shared.__file__),
                    Path(sys.modules[ResilienceMetric.__module__].__file__)]
    for name in ('CybORG.Agents.Castle.scorer_mappings', 'CybORG.Agents.Castle.impact_reporters'):
        source_paths.append(Path(sys.modules[name].__file__))
    sources = {str(p): sha256(p) for p in source_paths}
    for job in jobs:
        scenario = job['config']['scenario_path']
        job['source_sha256'] = {**sources, scenario: sha256(scenario)}
    print(f'{len(jobs)} policy/case jobs, {sum(len(j["episodes"]) for j in jobs)} episodes; '
          f'{args.parallel} concurrent workers; no LLM calls.', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"{job['policy']}/{job['group']}/{job['case']}: "
                  f"{len(job['episodes'])} episodes, schedule={job['config']['red_schedule']}")
        return
    # Validate every checkpoint against the actual wrapper before launching any case.
    cfg = OmegaConf.create(jobs[0]['config'])
    env, _ = shared.make_environment(cfg, jobs[0]['episodes'][0][0])
    try:
        paths = sorted({p for job in jobs for p in job['checkpoints'].values()})
        for path in paths:
            checked_policy(path, env)
    finally:
        env.close()
    print(f'Preflight passed: {len(paths)} checkpoints.', flush=True)
    if args.check:
        return
    root = ROOT / 'outputs/cia_cross_play_local'
    root.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix='run_', dir=root))
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = dict(jobs=jobs, reference_batch=str(args.reference_batch.resolve()))
        marker = output / 'settings.json'
        if marker.exists() and json.loads(marker.read_text()) != settings:
            raise ValueError('Batch settings changed; choose a new OUTPUT_DIR')
        shared._write_json(marker, settings)
        print(f'Cross-play output: {output}', flush=True)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        active, pending = {}, iter(jobs)
        exhausted = False
        try:
            while active or not exhausted:
                while len(active) < args.parallel and not exhausted:
                    job = next(pending, None)
                    if job is None:
                        exhausted = True
                        break
                    folder = output / job['policy'] / job['group'] / job['case']
                    folder.mkdir(parents=True, exist_ok=True)
                    job_file = folder / 'job.json'
                    shared._write_json(job_file, job)
                    stream = (folder / 'run.log').open('a')
                    process = subprocess.Popen([sys.executable, '-m', __spec__.name,
                                                '--worker', str(job_file)],
                                               stdout=stream, stderr=subprocess.STDOUT)
                    active[process] = stream
                for process in list(active):
                    if process.poll() is None:
                        continue
                    active.pop(process).close()
                    if process.returncode:
                        raise RuntimeError(f'Worker failed; inspect run.log under {output}')
                if active:
                    time.sleep(0.2)
        finally:
            for process in active:
                if process.poll() is None:
                    process.terminate()
            for process, stream in active.items():
                process.wait()
                stream.close()
    print(f'Cross-play complete: {output}', flush=True)


if __name__ == '__main__':
    main()
