"""Live CIA classification -> PPO selection, with a no-LLM oracle comparison."""

from collections import deque
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import random
import signal
from time import perf_counter
from zipfile import ZipFile

from gymnasium import spaces
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
import torch

from CybORG import CybORG
from CybORG.Agents.Wrappers.ChallengeWrapper import ChallengeWrapper
from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric
from CybORG.Agents.SimpleAgents.B_line_resilience import (
    BlineJumping_C, BlineJumping_I, BlineJumping_A,
)
from CybORG.Agents.SimpleAgents.Meander_Resilience import (
    MeanderJumping_C, MeanderJumping_I, MeanderJumping_A,
)


# Keep oracle execution independent of the LLM service and its dependencies.
ATTACKERS = {
    'bline_c': BlineJumping_C, 'bline_i': BlineJumping_I, 'bline_a': BlineJumping_A,
    'meander_c': MeanderJumping_C, 'meander_i': MeanderJumping_I,
    'meander_a': MeanderJumping_A,
}


LABELS = {
    'bline_c': 'B-line-C', 'bline_i': 'B-line-I', 'bline_a': 'B-line-A',
    'meander_c': 'Meander-C', 'meander_i': 'Meander-I', 'meander_a': 'Meander-A',
}


@contextmanager
def preserve_rng():
    """Model loading and inference must not perturb simulator/PPO RNGs."""
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)


def episode_plan(cfg):
    if cfg.mode not in ('hybrid', 'oracle'):
        raise ValueError('mode must be hybrid or oracle')
    if cfg.red_agent not in LABELS:
        raise ValueError(f'red_agent must be one of {list(LABELS)}')
    for name, minimum in (('n_rollouts', 1), ('episode_length', 1), ('start_seed', 0),
                          ('episodes_per_policy_seed', 1), ('window_stride', 1)):
        value = cfg[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if type(cfg.window_size) is not int or cfg.window_size not in (5, 7, 9):
        raise ValueError('Few-shot window_size must be 5, 7, or 9')
    if (not cfg.policy_seeds or len(set(cfg.policy_seeds)) != len(cfg.policy_seeds)
            or any(type(s) is not int or s < 0 for s in cfg.policy_seeds)):
        raise ValueError('policy_seeds must be distinct nonnegative integers')
    if cfg.n_rollouts > cfg.episodes_per_policy_seed * len(cfg.policy_seeds):
        raise ValueError('Not enough policy_seeds for n_rollouts / episodes_per_policy_seed')
    if cfg.start_seed + cfg.n_rollouts > 2**32:
        raise ValueError('Episode seeds must fit in uint32')
    attacker_schedule(cfg)
    return [(cfg.start_seed + i, cfg.policy_seeds[i // cfg.episodes_per_policy_seed])
            for i in range(cfg.n_rollouts)]


def attacker_schedule(cfg):
    """Phase starts count completed actions: step=50 applies to action 51."""
    configured = cfg.get('red_schedule')
    if configured is None:
        return [{'step': 0, 'red_agent': cfg.red_agent}]
    schedule = OmegaConf.to_container(configured, resolve=True)
    if not isinstance(schedule, list) or not schedule:
        raise ValueError('red_schedule must be a nonempty list of {step, red_agent}')
    previous = -1
    for phase in schedule:
        if not isinstance(phase, dict) or set(phase) != {'step', 'red_agent'}:
            raise ValueError('Each red_schedule phase must contain step and red_agent')
        step = phase['step']
        if type(step) is not int or not previous < step < cfg.episode_length:
            raise ValueError('red_schedule steps must strictly increase within the episode')
        if phase['red_agent'] not in ATTACKERS:
            raise ValueError(f'Unknown scheduled attacker: {phase["red_agent"]}')
        previous = step
    if schedule[0] != {'step': 0, 'red_agent': cfg.red_agent}:
        raise ValueError('red_schedule must start at step 0 with red_agent')
    return schedule


def switch_attacker(env, red_agent):
    """Replace Red's private policy state, retaining the live game and sessions.

    Bootstrap private memory and action knowledge from the scenario's Red INT
    view of the current world. B-line must not consume the outgoing agent's
    success as its own (which would skip its first scan). Meander likewise
    requires discovery history consistent with its fresh private memory.
    The world, sessions, Blue wrapper, and reward calculators are untouched.
    """
    controller = env.cyborg_env.environment_controller
    interface = controller.agent_interfaces['Red']
    observation = controller._filter_obs(controller.get_agent_state('Red'), 'Red')
    # ActionSpace.reset() in CAGE2 leaves old hostname knowledge behind; use a
    # fresh space, populated with the same unknown/known masks as initial setup.
    space = interface.action_space
    interface.action_space = type(space)(interface.actions, 'Red', space.allowed_subnets)
    interface.update(controller.init_state, known=False)
    interface.update(observation, known=True)
    incoming = ATTACKERS[red_agent]()
    incoming.set_initial_values(action_space=interface.action_space.get_max_action_space(),
                                observation=observation.data)
    interface.agent = incoming
    controller.observation['Red'] = observation


def checkpoint_paths(cfg, policy_seed):
    root = Path(to_absolute_path(cfg.checkpoint_dir))
    paths = {}
    for attacker, label in LABELS.items():
        family, dimension = attacker.split('_')
        path = root / f'PPO_base_jumping_{family}_{dimension.upper()}_seed{policy_seed}.zip'
        if not path.is_file():
            raise FileNotFoundError(f'Missing policy for {label}, RL seed {policy_seed}: {path}')
        paths[label] = path
    return paths


def make_environment(cfg, seed):
    # ChallengeWrapper is the actual PPO training wrapper: retain both its
    # observation encoding and action-index ordering, without rebuilding either.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cyborg = CybORG(to_absolute_path(cfg.scenario_path), 'sim',
                    agents={'Red': ATTACKERS[cfg.red_agent]})
    env = ChallengeWrapper('Blue', cyborg, max_steps=cfg.episode_length)
    # Wrapper construction resets internally. Seed again before the real reset.
    cyborg.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation, _ = env.reset()
    return env, observation


def load_policy(path):
    # These evaluation checkpoints were saved with NumPy 2. Its pickled arrays
    # and space RNGs cannot be read by the NumPy 1 runtime used by CAGE2. Rebuild
    # ONLY the spaces from SB3's plain JSON metadata and discard training buffers;
    # PPO still loads the actual saved policy architecture and tensor weights.
    with ZipFile(path) as archive:
        data = json.loads(archive.read('data'))
    box, discrete = data['observation_space'], data['action_space']
    if (box[':type:'] != "<class 'gymnasium.spaces.box.Box'>"
            or discrete[':type:'] != "<class 'gymnasium.spaces.discrete.Discrete'>"):
        raise ValueError(f'Expected Box/Discrete CAGE2 checkpoint spaces: {path}')
    shape, dtype = tuple(box['_shape']), np.dtype(box['dtype'])
    bounds = [np.fromstring(box[name].strip('[]'), sep=' ', dtype=dtype)
              for name in ('low', 'high')]
    if any(b.size != int(np.prod(shape)) for b in bounds):
        raise ValueError(f'Incomplete observation-space bounds in checkpoint metadata: {path}')
    return PPO.load(str(path), device='cpu', custom_objects={
        '_last_obs': None, '_last_episode_starts': None,
        'observation_space': spaces.Box(bounds[0].reshape(shape), bounds[1].reshape(shape), dtype=dtype),
        'action_space': spaces.Discrete(int(discrete['n']), start=int(discrete.get('start', 0))),
    })


def load_policies(cfg, policy_seed, env):
    with preserve_rng():
        policies = {label: load_policy(path)
                    for label, path in checkpoint_paths(cfg, policy_seed).items()}
    for label, policy in policies.items():
        if (policy.observation_space.shape != env.observation_space.shape
                or policy.action_space.n != env.action_space.n):
            raise ValueError(f'Incompatible checkpoint for {label}, RL seed {policy_seed}: '
                             f'expected {env.observation_space.shape} observations and '
                             f'{env.action_space.n} actions; got {policy.observation_space.shape} '
                             f'and {policy.action_space.n}')
    return policies


def action_record(env, agent):
    """Same target resolution as CAGE2's AgentTrajectory, without its logging side effects."""
    action = env.get_last_action(agent)
    record = {'Action': type(action).__name__,
              'Status': str(env.get_observation(agent).get('success', 'UNKNOWN'))}
    if hasattr(action, 'hostname'):
        record['Host'] = str(action.hostname)
    elif hasattr(action, 'impacted_hostname'):
        record['Host'] = str(action.impacted_hostname)
    elif hasattr(action, 'ip_address'):
        ip_map = {str(ip): host for host, ip in env.get_ip_map().items()}
        record['Host'] = ip_map.get(str(action.ip_address), str(action.ip_address))
    elif hasattr(action, 'subnet'):
        subnet_map = env.cyborg_env.environment_controller.subnet_cidr_map
        cidr_map = {str(cidr): name for name, cidr in subnet_map.items()}
        record['Subnet'] = cidr_map.get(str(action.subnet), str(action.subnet))
    return record


def format_window(window):
    from align_system.interfaces.cia_cage2_service import _format_trajectory_action
    lines = []
    for record in window:
        lines.extend((f'Step {record["step"]}:',
                      f'  Blue: {_format_trajectory_action(record["blue_action"])}',
                      f'  Red: {_format_trajectory_action(record["red_action"])}'))
    return '\n'.join(lines)


def make_classifier(cfg):
    # Oracle execution never imports prompt machinery or constructs an inference engine.
    from align_system.prompt_engineering.cage_prompts import (
        CAGEHybridCIAFewshotSystemPrompt, CAGERedAgentClassificationPrompt,
        CAGERedAgentClassificationSchema,
    )
    engine = instantiate(cfg.inference_engine)
    system = CAGEHybridCIAFewshotSystemPrompt(cfg.window_size)()
    user_template = CAGERedAgentClassificationPrompt()
    schema = CAGERedAgentClassificationSchema()(list(LABELS.values()))

    def classify(window):
        prompt = engine.dialog_to_prompt([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_template(format_window(window), list(LABELS.values()))},
        ])
        result = engine.run_inference(prompt, schema)
        label = result['action_choice']
        if label not in LABELS.values():
            raise ValueError(f'Unknown predicted CIA label: {label}')
        return label

    return engine, classify, system


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _manifest(cfg, plan):
    settings = OmegaConf.to_container(cfg, resolve=True)
    for key in ('resume', 'check_only'):
        settings.pop(key, None)
    if cfg.mode == 'oracle':
        # Neither model settings nor trajectory windows affect the oracle.
        for key in ('inference_engine', 'random_seed', 'window_size', 'window_stride'):
            settings.pop(key, None)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), Path(to_absolute_path(cfg.scenario_path))]
    if cfg.mode == 'hybrid':
        sources += [root / 'prompt_engineering/cage_prompts.py',
                    root / 'interfaces/cia_cage2_service.py']
        # Hash the selected backend; direct vLLM runs need no hosted client,
        # Outlines runtime, LLM-only service, or shell launcher source files.
        target = cfg.inference_engine.get('_target_')
        if target == (
                'align_system.algorithms.huggingface_inference_engine.HuggingFaceInferenceEngine'):
            sources.append(root / 'algorithms/huggingface_inference_engine.py')
        elif target == (
                'align_system.algorithms.cage_local_inference_engine.CAGELocalInferenceEngine'):
            sources += [root / 'algorithms/cage_local_inference_engine.py',
                        root / 'algorithms/outlines_inference_engine.py']
        elif target == (
                'align_system.algorithms.cage_vllm_inference_engine.CAGEVLLMInferenceEngine'):
            sources.append(root / 'algorithms/cage_vllm_inference_engine.py')
    paths = {path for _, policy_seed in plan for path in checkpoint_paths(cfg, policy_seed).values()}
    return {'config': settings, 'episodes': [list(p) for p in plan],
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'checkpoint_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(paths)}}


def run_episode(cfg, seed, policy_seed, env, observation, policies, engine, classify, attempt_dir):
    window = deque(maxlen=cfg.window_size)
    warmup_rng = random.Random(seed)  # Does not consume simulator randomness.
    metric = ResilienceMetric()
    schedule = attacker_schedule(cfg)
    transitions = {phase['step']: phase['red_agent'] for phase in schedule}
    active_red_agent = cfg.red_agent
    predicted = None
    rewards, cia_scores, selections = [], [], []
    requests_count, correct, classifications = 0, 0, 0
    usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    started = perf_counter()
    with (attempt_dir / 'steps.jsonl').open('w') as steps, \
            (attempt_dir / 'request_history.jsonl').open('w') as requests:
        for step in range(cfg.episode_length):
            switched = step > 0 and step in transitions
            if switched:
                active_red_agent = transitions[step]
                switch_attacker(env, active_red_agent)
            truth = LABELS[active_red_agent]
            decision_started = perf_counter()
            bounds = None
            if cfg.mode == 'oracle':
                selected, source = truth, 'oracle'
            elif len(window) < cfg.window_size:
                selected, source = warmup_rng.choice(list(policies)), 'random_warmup'
            elif (step - cfg.window_size) % cfg.window_stride == 0:
                bounds = [window[0]['step'], window[-1]['step'] + 1]
                try:
                    with preserve_rng():
                        predicted = classify(list(window))
                finally:
                    # Keep failed requests too, and flush on interruption.
                    for request in engine.request_history:
                        requests_count += 1
                        for key in usage:
                            usage[key] += (request.get('usage') or {}).get(key, 0) or 0
                        requests.write(json.dumps({'decision_step': step, **request}) + '\n')
                    requests.flush()
                    engine.request_history.clear()
                classifications += 1
                correct += predicted == truth
                selected, source = predicted, 'llm'
            else:
                selected, source = predicted, 'previous_prediction'

            action, _ = policies[selected].predict(observation, deterministic=cfg.deterministic_policy)
            decision_time = perf_counter() - decision_started
            observation, reward, terminated, truncated, _ = env.step(int(action))
            cia = [float(v) for v in metric.calculate_scores(
                env, blue_action=env.get_last_action('Blue'))]
            record = {'step': step, 'blue_action': action_record(env, 'Blue'),
                      'red_action': action_record(env, 'Red')}
            window.append(record)
            rewards.append(float(reward))
            cia_scores.append(cia)
            selections.append(selected)
            steps.write(json.dumps({**record, 'policy_seed': policy_seed,
                                    'active_red_agent': active_red_agent, 'true_label': truth,
                                    'attacker_switched': switched,
                                    'selected_policy': selected, 'selection_source': source,
                                    'prediction': predicted, 'classification_window': bounds,
                                    'action_index': int(action), 'reward': float(reward),
                                    'cia_scores': cia, 'decision_time_s': decision_time}) + '\n')
            steps.flush()
            if terminated or truncated:
                break
    if len(rewards) != cfg.episode_length:
        raise RuntimeError(f'Seed {seed} ended after {len(rewards)} steps; expected {cfg.episode_length}')
    return {'complete': True, 'seed': seed, 'policy_seed': policy_seed,
            'red_agent': cfg.red_agent, 'mode': cfg.mode, 'steps': len(rewards),
            'red_schedule': schedule,
            'return': sum(rewards), 'rewards': rewards, 'cia_scores': cia_scores,
            'selected_policies': selections, 'classifications': classifications,
            'correct_classifications': correct, 'requests': requests_count, 'usage': usage,
            'elapsed_s': perf_counter() - started}


def preflight(cfg):
    """Check every configured checkpoint against the real simulator; no inference."""
    plan = episode_plan(cfg)
    env, _ = make_environment(cfg, plan[0][0])
    try:
        for policy_seed in sorted({p for _, p in plan}):
            load_policies(cfg, policy_seed, env)
    finally:
        env.close()
    if cfg.mode == 'hybrid':
        from align_system.prompt_engineering.cage_prompts import CAGEHybridCIAFewshotSystemPrompt
        CAGEHybridCIAFewshotSystemPrompt(cfg.window_size)()
    print(f'Preflight passed: {len(plan)} episodes, {cfg.episode_length} steps, '
          f'RL seeds {sorted({p for _, p in plan})}; no inference performed.', flush=True)


def run(cfg, output_dir):
    plan = episode_plan(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = _manifest(cfg, plan)
        manifest_path = output / 'run_manifest.json'
        if manifest_path.exists():
            if not cfg.resume or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Existing run configuration/source/checkpoints differ; use a new directory')
        else:
            _write_json(manifest_path, manifest)
        engine = classify = None
        loaded_seed, policies = None, None
        results = []
        for seed, policy_seed in plan:
            result_path = output / f'episode_{seed}.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if (result.get('complete') and result['seed'] == seed
                        and result['policy_seed'] == policy_seed and result['steps'] == cfg.episode_length):
                    results.append(result)
                    continue
            if cfg.mode == 'hybrid' and engine is None:
                with preserve_rng():
                    engine, classify, system = make_classifier(cfg)
                (output / 'system_prompt.txt').write_text(system + '\n')
            attempts = output / f'episode_{seed}'
            attempts.mkdir(exist_ok=True)
            attempt = 1
            while (attempts / f'attempt_{attempt:03d}').exists():
                attempt += 1
            attempt_dir = attempts / f'attempt_{attempt:03d}'
            attempt_dir.mkdir()
            env, observation = make_environment(cfg, seed)
            try:
                if loaded_seed != policy_seed:
                    policies = load_policies(cfg, policy_seed, env)
                    loaded_seed = policy_seed
                result = run_episode(cfg, seed, policy_seed, env, observation,
                                     policies, engine, classify, attempt_dir)
            finally:
                env.close()
            result['attempt'] = attempt
            _write_json(result_path, result)
            results.append(result)
            print(f'{cfg.mode}/{cfg.red_agent}: seed={seed}, RL seed={policy_seed}, '
                  f'return={result["return"]:.2f}, classifications={result["classifications"]}', flush=True)
        count = sum(r['classifications'] for r in results)
        summary = {'complete': True, 'mode': cfg.mode, 'red_agent': cfg.red_agent,
                   'red_schedule': attacker_schedule(cfg),
                   'episodes': len(results), 'seeds': [r['seed'] for r in results],
                   'policy_seeds': [r['policy_seed'] for r in results],
                   'steps': sum(r['steps'] for r in results),
                   'mean_return': float(np.mean([r['return'] for r in results])),
                   'std_return': float(np.std([r['return'] for r in results])),
                   'mean_cia_drop_per_step': np.mean([r['cia_scores'] for r in results], axis=(0, 1)).tolist(),
                   'classifications': count,
                   'classification_accuracy': sum(r['correct_classifications'] for r in results) / count if count else None,
                   'requests': sum(r['requests'] for r in results),
                   'usage': {k: sum(r['usage'][k] for r in results)
                             for k in ('prompt_tokens', 'completion_tokens')}}
        _write_json(output / 'results.json', summary)
        return summary


@hydra.main(version_base=None, config_path='../configs', config_name='cage_hybrid')
def main(cfg):
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    if cfg.check_only:
        preflight(cfg)
    else:
        run(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == '__main__':
    main()
