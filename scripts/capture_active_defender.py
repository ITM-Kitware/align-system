#!/usr/bin/env python3
"""Capture seeded hybrid-PPO defender episodes paired with verified Monitor controls."""

import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys

from capture_defender_observations import (
    action_record, capture, digest, json_safe, simulator_source_digest, verified_monitor_episodes,
)
from run_classifier_replay import write_json


def active_episode(reference, scenario, policy, deterministic):
    import numpy as np
    import torch
    from CybORG import CybORG
    from CybORG.Agents import GreenAgent
    from CybORG.Agents.SimpleAgents.B_line_resilience import BlineJumping_C, BlineJumping_I, BlineJumping_A
    from CybORG.Agents.SimpleAgents.Meander_Resilience import MeanderJumping_C, MeanderJumping_I, MeanderJumping_A
    from CybORG.Agents.Wrappers import ChallengeWrapper
    from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric

    classes = {cls.__name__: cls for cls in (BlineJumping_C, BlineJumping_I, BlineJumping_A,
                                            MeanderJumping_C, MeanderJumping_I, MeanderJumping_A)}
    seed = reference["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = CybORG(str(scenario), "sim", agents={
        "Red": classes[reference["red_agent_name"]], "Green": GreenAgent})
    wrapper = ChallengeWrapper("Blue", env)
    env.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation, _ = wrapper.reset()
    if observation.shape != policy.observation_space.shape:
        raise ValueError("Checkpoint and simulator observation spaces differ")
    if wrapper.action_space.n != policy.action_space.n:
        raise ValueError("Checkpoint and simulator action spaces differ")
    network = {key: json_safe(value) for key, value in env.get_agent_state("Blue").items()
               if key != "success"}
    if network != reference["network_topology"]:
        raise ValueError("Active episode initial network differs from the Monitor control")
    initial = str(wrapper.get_observation_table(is_true_table=False))
    action_mapping = [{"action": type(a).__name__, "parameters": json_safe(a.get_params())}
                      for a in wrapper.get_attr("possible_actions")]
    ip_to_host = {ip: hostname for hostname, ip in env.get_ip_map().items()}
    cidr_to_subnet = {cidr: name for name, cidr in env.environment_controller.subnet_cidr_map.items()}
    metric = ResilienceMetric()
    metric.reset()
    rows = []
    for step in range(len(reference["blue_actions"])):
        action, _ = policy.predict(observation, deterministic=deterministic)
        observation, reward, done, truncated, _ = wrapper.step(action=int(action))
        metric.calculate_scores(env)
        rows.append({
            "step": step, "blue_action_index": int(action), "reward": float(reward),
            "defender_observation": str(wrapper.get_observation_table(is_true_table=False)),
            "defender_raw_observation": json_safe(env.get_observation("Blue")),
            "metric_scores": metric.step_scores(),
            "blue_action": action_record(env.get_last_action("Blue"), env.get_observation("Blue"),
                                         ip_to_host, cidr_to_subnet, blue=True),
            "red_action": action_record(env.get_last_action("Red"), env.get_observation("Red"),
                                        ip_to_host, cidr_to_subnet),
        })
        if (done or truncated) and step + 1 != len(reference["blue_actions"]):
            raise ValueError("Active episode ended before the paired Monitor control")
    return {"initial_defender_observation": initial, "steps": rows,
            "initial_network_matches_monitor": True, "action_mapping": action_mapping,
            "observation_shape": list(observation.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-capture-dir", type=Path, required=True)
    parser.add_argument("--cage-root", type=Path, required=True)
    parser.add_argument("--checkpoint-template", required=True, help="Path containing {seed}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-per-class", action="store_true")
    parser.add_argument("--policy-deterministic", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.monitor_capture_dir / "summary.json").read_text())
    episodes = verified_monitor_episodes(summary)
    source_sha = simulator_source_digest(args.cage_root)
    for episode in episodes:
        path = args.monitor_capture_dir / episode['capture_path']
        if digest(path) != episode['capture_sha256']:
            raise ValueError('Monitor capture hash changed')
        if json.loads(path.read_text())['cage_python_source_sha256'] != source_sha:
            raise ValueError('Simulator source differs from the verified Monitor capture')
    sys.path.insert(0, str(args.cage_root.resolve()))
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    indices = collections.Counter()
    assignments = []
    for row in episodes:
        kind = row["class_directory"]
        seed = indices[kind] % 5
        indices[kind] += 1
        if args.first_per_class and indices[kind] != 1:
            continue
        assignments.append((row, seed))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = args.output_dir / "source"
    sources.mkdir()
    root = Path(__file__).resolve().parents[1]
    for path in (Path(__file__), root / "scripts/capture_defender_observations.py",
                 root / "scripts/run_classifier_replay.py", root / "requirements-active-capture.in",
                 root / "requirements-active-capture.txt", root / "align_system/interfaces/cage_observations.py"):
        shutil.copyfile(path, sources / path.name)
    policies = {}
    results = []
    for episode, training_seed in assignments:
        monitor_path = args.monitor_capture_dir / episode["capture_path"]
        if digest(monitor_path) != episode["capture_sha256"]:
            raise ValueError("Monitor capture hash changed")
        monitor = json.loads(monitor_path.read_text())
        reference_path = Path(monitor["reference_path"])
        if digest(reference_path) != monitor["reference_sha256"]:
            raise ValueError("Historical reference changed")
        reference = json.loads(reference_path.read_text())
        scenario = Path(monitor["scenario_path"])
        if digest(scenario) != monitor["scenario_sha256"]:
            raise ValueError("Scenario changed")
        control = capture(reference, scenario)
        if (control["steps"] != monitor["steps"] or not all(control[field] is True for field in (
            "historical_initial_network_matches", "historical_metric_scores_match", "historical_action_trace_matches",
        ))):
            raise ValueError("CPU PPO runtime changed the Monitor control replay")
        checkpoint = Path(args.checkpoint_template.format(seed=training_seed))
        if training_seed not in policies:
            policies[training_seed] = PPO.load(str(checkpoint), device="cpu")
        first = active_episode(reference, scenario, policies[training_seed], args.policy_deterministic)
        second = active_episode(reference, scenario, policies[training_seed], args.policy_deterministic)
        if first != second:
            raise RuntimeError("Seeded active-defender replay is not deterministic")
        counts = collections.Counter(row["blue_action"]["Action"] for row in first["steps"])
        result = {
            "format_version": 1, "condition": "active_hybrid_ppo", "seed": reference["seed"],
            "red_agent_name": reference["red_agent_name"], "training_seed": training_seed,
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "policy_deterministic": args.policy_deterministic, "policy_device": "cpu",
            "monitor_capture_sha256": episode["capture_sha256"],
            "scenario_sha256": monitor["scenario_sha256"], "cage_revision": monitor["cage_revision"],
            "cage_python_source_sha256": source_sha,
            "collector_sha256": digest(Path(__file__)),
            "observation_protocol_sha256": digest(root / "align_system/interfaces/cage_observations.py"),
            "dependency_lock_sha256": digest(root / "requirements-active-capture.txt"),
            "software": {name: importlib.metadata.version(name) for name in
                         ("torch", "stable-baselines3", "numpy", "gymnasium")},
            "repeat_capture_identical": True, "monitor_control_reproduced": True,
            "observation_timing": "post-action; step 0 follows the first Blue action",
            "blue_action_counts": counts,
            **first,
        }
        name = f"{episode['class_directory']}-{Path(episode['trajectory_id']).stem}.json"
        path = args.output_dir / name
        write_json(path, result)
        results.append({"class_directory": episode["class_directory"],
                        "trajectory_id": episode["trajectory_id"], "capture_path": name,
                        "capture_sha256": digest(path), "training_seed": training_seed,
                        "seed": reference["seed"], "blue_action_counts": counts})
        write_json(args.output_dir / "summary.json", {
            "status": "complete" if len(results) == len(assignments) else "partial",
            "condition": "active_hybrid_ppo", "episodes": results,
            "monitor_audit_sha256": digest(args.monitor_capture_dir / "summary.json"),
        })
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
