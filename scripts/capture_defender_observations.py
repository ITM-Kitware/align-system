#!/usr/bin/env python3
"""Capture paired attacker-action and defender-table views of seeded episodes."""

import argparse
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def action_record(action, observation, ip_to_host, cidr_to_subnet, blue=False):
    result = {"Action": type(action).__name__, "Status": str(observation["success"])}
    if hasattr(action, "hostname"):
        result["Host"] = action.hostname
    elif blue:
        result["Host"] = type(action).__name__
    elif hasattr(action, "impacted_hostname"):
        result["Host"] = str(action.impacted_hostname)
    elif hasattr(action, "ip_address"):
        result["Host"] = ip_to_host.get(action.ip_address, str(action.ip_address))
    elif hasattr(action, "subnet"):
        result["Subnet"] = cidr_to_subnet.get(action.subnet, str(action.subnet))
    return result


def capture(reference, scenario):
    import numpy as np
    from CybORG import CybORG
    from CybORG.Agents import GreenAgent
    from CybORG.Agents.SimpleAgents.B_line_resilience import BlineJumping_C, BlineJumping_I, BlineJumping_A
    from CybORG.Agents.SimpleAgents.Meander_Resilience import MeanderJumping_C, MeanderJumping_I, MeanderJumping_A
    from CybORG.Agents.Wrappers import BlueTableWrapper
    from CybORG.Shared.Actions import Monitor
    from CybORG.AlignmentMetric.resilience_measure import ResilienceMetric

    classes = {cls.__name__: cls for cls in (BlineJumping_C, BlineJumping_I, BlineJumping_A,
                                            MeanderJumping_C, MeanderJumping_I, MeanderJumping_A)}
    if any(a["Action"] != "Monitor" for a in reference["blue_actions"]):
        raise ValueError("This capture protocol supports Monitor-only episodes")
    seed = reference["seed"]
    random.seed(seed)
    np.random.seed(seed)
    env = CybORG(str(scenario), "sim", agents={
        "Red": classes[reference["red_agent_name"]], "Green": GreenAgent})
    env.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    wrapper = BlueTableWrapper(env, output_mode="table")
    initial = str(wrapper.reset(agent="Blue").observation)
    network = {key: json_safe(value) for key, value in env.get_agent_state("Blue").items()
               if key != "success"}
    metric = ResilienceMetric()
    metric.reset()
    ip_to_host = {ip: hostname for hostname, ip in env.get_ip_map().items()}
    cidr_to_subnet = {cidr: name for name, cidr in env.environment_controller.subnet_cidr_map.items()}
    rows = []
    for step in range(len(reference["blue_actions"])):
        result = wrapper.step(agent="Blue", action=Monitor(agent="Blue", session=0))
        metric.calculate_scores(env)
        rows.append({
            "step": step,
            "defender_observation": str(result.observation),
            "defender_raw_observation": json_safe(env.get_observation("Blue")),
            "metric_scores": metric.step_scores(),
            "blue_action": action_record(env.get_last_action("Blue"), env.get_observation("Blue"),
                                         ip_to_host, cidr_to_subnet, blue=True),
            "red_action": action_record(env.get_last_action("Red"), env.get_observation("Red"),
                                        ip_to_host, cidr_to_subnet),
        })
        if result.done and step + 1 != len(reference["blue_actions"]):
            raise ValueError("Episode ended before the recorded trajectory")
    mismatches = [
        {"step": row["step"], "side": side, "recorded": reference[f"{side}_actions"][row["step"]],
         "replayed": row[f"{side}_action"]}
        for row in rows for side in ("blue", "red")
        if row[f"{side}_action"] != reference[f"{side}_actions"][row["step"]]
    ]
    metric_mismatches = [row["step"] for row in rows
                         if row["metric_scores"] != reference["metric_scores"][row["step"]]]
    return {"initial_defender_observation": initial, "steps": rows,
            "historical_initial_network_matches": network == reference["network_topology"],
            "historical_metric_scores_match": not metric_mismatches,
            "historical_metric_mismatch_count": len(metric_mismatches),
            "historical_action_trace_matches": not mismatches,
            "historical_action_mismatch_count": len(mismatches),
            "first_historical_action_mismatch": mismatches[0] if mismatches else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cage-root", type=Path, required=True,
                        help="Directory containing the CybORG package")
    parser.add_argument("--cage-revision", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new file")
    sys.path.insert(0, str(args.cage_root.resolve()))
    reference = json.loads(args.reference.read_text())
    first = capture(reference, args.scenario)
    second = capture(reference, args.scenario)
    if first != second:
        raise RuntimeError("Repeated seeded capture changed observations or actions")
    source_files = sorted((args.cage_root / "CybORG").rglob("*.py"))
    source_digest = hashlib.sha256()
    for path in source_files:
        source_digest.update(str(path.relative_to(args.cage_root)).encode())
        source_digest.update(path.read_bytes())
    result = {
        "format_version": 1, "reference_sha256": digest(args.reference),
        "reference_path": str(args.reference), "seed": reference["seed"],
        "red_agent_name": reference["red_agent_name"], "blue_policy": "Monitor",
        "scenario_sha256": digest(args.scenario), "scenario_path": str(args.scenario),
        "cage_revision": args.cage_revision, "cage_python_source_sha256": source_digest.hexdigest(),
        "collector_sha256": digest(Path(__file__)),
        "dependency_lock_sha256": digest(Path(__file__).resolve().parents[1] / "requirements-observation-capture.txt"),
        "software": {name: importlib.metadata.version(name) for name in
                     ("numpy", "gymnasium", "prettytable", "PyYAML", "paramiko")},
        "python_version": sys.version, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "repeat_capture_identical": True,
        "observation_timing": "post-action; step 0 follows the first Monitor action",
        "observation_source": "BlueTableWrapper.step(agent='Blue').observation; no true table",
        "limitation": "Matching actions does not prove unrecorded historical observations were identical.",
        **first,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("steps", "initial_defender_observation")}), flush=True)


if __name__ == "__main__":
    main()
