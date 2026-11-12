#!/usr/bin/env python3
"""Pair historical classifier requests with verified defender-only observations."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from align_system.interfaces.cage_observations import (
    capture_paths, format_window,
)

from run_classifier_replay import digest, read_requests, write_json


def request_sections(item):
    user = item["messages"][-1]
    if user["role"] != "user" or not user["content"].startswith("# OBSERVATIONS\n"):
        raise ValueError("Unknown historical request format")
    before, after = user["content"].split("\n\n# AVAILABLE LABELS\n")
    return before, after


def validate_historical_window(item, captured):
    provenance = item["provenance"]
    start, end = provenance["window_start"], provenance["window_end"]
    before, _ = request_sections(item)
    expected = format_window(captured, start, end, ('blue_actions', 'red_actions'))
    if before != "# OBSERVATIONS\n" + expected:
        raise ValueError("Recovered action window differs from the historical model input")


def defender_request(item, captured, capture_sha):
    provenance = item["provenance"]
    start, end = provenance["window_start"], provenance["window_end"]
    _, after = request_sections(item)
    observations = format_window(captured, start, end)
    result = copy.deepcopy(item)
    result["messages"][-1]["content"] = "# OBSERVATIONS\n" + observations + "\n\n# AVAILABLE LABELS\n" + after
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
    if summary.get('condition') == 'active_hybrid_ppo':
        raise ValueError('Use prepare_active_replay.py for active-defender captures')
    captures = {}
    for episode, path in zip(summary["episodes"], capture_paths(args.capture_dir)):
        captures[(episode["class_directory"], episode["trajectory_id"])] = (
            json.loads(path.read_text()), episode["capture_sha256"])
    paired = []
    for item in requests:
        captured, capture_sha = captures[(item["provenance"]["class_directory"], item["trajectory_id"])]
        validate_historical_window(item, captured)
        paired.append(defender_request(item, captured, capture_sha))
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
        "observation_formatter_sha256": digest(Path(__file__).resolve().parents[1] / "align_system/interfaces/cage_observations.py"),
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
