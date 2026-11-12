#!/usr/bin/env python3
"""Pair historical classifier requests with verified defender-only observations."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from run_classifier_replay import digest, read_requests, write_json


def format_action(action):
    text = action["Action"]
    if "Subnet" in action:
        text += f" on subnet {action['Subnet']}"
    elif action.get("Host") not in (None, action["Action"]):
        text += f" on {action['Host']}"
    return f"{text} (success: {action['Status']})"


def paired_request(item, captured, capture_sha):
    provenance = item["provenance"]
    start, end = provenance["window_start"], provenance["window_end"]
    window = captured["steps"][start:end]
    if [row["step"] for row in window] != list(range(start, end)):
        raise ValueError("Captured observations do not cover the requested window")
    privileged = []
    defender = []
    for row in window:
        blue = f"  Blue: {format_action(row['blue_action'])}"
        privileged.extend((f"Step {row['step']}:", blue,
                           f"  Red: {format_action(row['red_action'])}"))
        defender.extend((f"Step {row['step']}:", blue, "  Blue observation:",
                         row["defender_observation"]))
    original_user = item["messages"][-1]
    if original_user["role"] != "user":
        raise ValueError("Expected the final message to be the classification request")
    prefix = "# OBSERVATIONS\n"
    boundary = "\n\n# AVAILABLE LABELS\n"
    if not original_user["content"].startswith(prefix):
        raise ValueError("Unknown historical request format")
    before, after = original_user["content"].split(boundary)
    if before != prefix + "\n".join(privileged):
        raise ValueError("Recovered action window differs from the historical model input")
    result = copy.deepcopy(item)
    result["messages"][-1]["content"] = prefix + "\n".join(defender) + boundary + after
    result["original_request_id"] = item["request_id"]
    result["input_view"] = "defender_table_and_blue_action"
    message_sha = hashlib.sha256(json.dumps(result["messages"], sort_keys=True).encode()).hexdigest()
    result["request_id"] = hashlib.sha256(json.dumps({
        "original_request_id": item["request_id"], "messages_sha256": message_sha,
        "capture_sha256": capture_sha, "view": result["input_view"],
    }, sort_keys=True).encode()).hexdigest()
    result["provenance"].update({
        "historical_messages_sha256": provenance["messages_sha256"],
        "messages_sha256": message_sha, "capture_sha256": capture_sha,
        "observation_timing": captured["observation_timing"],
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requests = read_requests(args.requests)
    summary = json.loads((args.capture_dir / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("Episode audit must finish before preparing a classifier comparison")
    captures = {}
    for episode in summary["episodes"]:
        required = ("fresh_process_repeat_identical", "historical_initial_network_matches",
                    "historical_metric_scores_match", "historical_action_trace_matches")
        if not all(episode[field] for field in required):
            raise ValueError(f"Historical episode verification failed: {episode['trajectory_id']}")
        path = args.capture_dir / episode["capture_path"]
        if digest(path) != episode["capture_sha256"]:
            raise ValueError(f"Captured evidence changed: {path}")
        captures[(episode["class_directory"], episode["trajectory_id"])] = (
            json.loads(path.read_text()), episode["capture_sha256"])
    paired = [paired_request(item, *captures[(item["provenance"]["class_directory"],
                                              item["trajectory_id"])]) for item in requests]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    privileged_path = args.output_dir / "privileged-requests.jsonl"
    defender_path = args.output_dir / "defender-requests.jsonl"
    shutil.copyfile(args.requests, privileged_path)
    with defender_path.open("x") as stream:
        for row in paired:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    read_requests(defender_path)
    manifest = {
        "n_requests_per_view": len(requests), "capture_audit_sha256": digest(args.capture_dir / "summary.json"),
        "privileged_requests_sha256": digest(privileged_path), "defender_requests_sha256": digest(defender_path),
        "preparation_script_sha256": digest(Path(__file__)),
        "system_prompts_unchanged": True, "generation_settings_unchanged": True,
        "classification_labels_unchanged": True, "windows_unchanged": True,
        "comparison": "Change only the current observation section; retain historical prompts and examples.",
        "limitations": [
            "Defender observations are reconstructed, not independently saved historical observations.",
            "Historical prompts and few-shot examples describe attacker actions; this tests information restriction before prompt adaptation.",
            "Blue always uses Monitor; active RL defender policies may produce different observations.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
