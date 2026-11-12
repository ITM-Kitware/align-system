#!/usr/bin/env python3
"""Prepare active-defender classifier inputs with the historical prompt held fixed."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from prepare_defender_replay import format_action, paired_request
from run_classifier_replay import digest, read_requests, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requests = read_requests(args.requests)
    summary = json.loads((args.capture_dir / "summary.json").read_text())
    if summary["status"] != "complete" or summary["condition"] != "active_hybrid_ppo":
        raise ValueError("Active-defender capture is incomplete")
    captures = {}
    for episode in summary["episodes"]:
        path = args.capture_dir / episode["capture_path"]
        if digest(path) != episode["capture_sha256"]:
            raise ValueError("Active capture changed")
        record = json.loads(path.read_text())
        if not all(record[key] for key in ("repeat_capture_identical", "monitor_control_reproduced",
                                           "initial_network_matches_monitor")):
            raise ValueError("Active replay or Monitor control verification failed")
        captures[(episode["class_directory"], episode["trajectory_id"])] = (record, episode["capture_sha256"])
    paired = []
    for original in requests:
        item = copy.deepcopy(original)
        captured, capture_sha = captures[(item["provenance"]["class_directory"], item["trajectory_id"])]
        start, end = item["provenance"]["window_start"], item["provenance"]["window_end"]
        action_lines = []
        for row in captured["steps"][start:end]:
            action_lines.extend((f"Step {row['step']}:",
                                 f"  Blue: {format_action(row['blue_action'])}",
                                 f"  Red: {format_action(row['red_action'])}"))
        boundary = "\n\n# AVAILABLE LABELS\n"
        suffix = item["messages"][-1]["content"].split(boundary)[1]
        item["messages"][-1]["content"] = "# OBSERVATIONS\n" + "\n".join(action_lines) + boundary + suffix
        item["historical_prediction"] = None
        result = paired_request(item, captured, capture_sha)
        result["scenario_id"] = f"active-hybrid/seed{captured['training_seed']}/{original['scenario_id']}"
        result["condition"] = "active_hybrid_ppo"
        result["provenance"].update({"checkpoint_sha256": captured["checkpoint_sha256"],
                                    "training_seed": captured["training_seed"],
                                    "policy_deterministic": captured["policy_deterministic"],
                                    "monitor_request_id": original["request_id"]})
        paired.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "defender-requests.jsonl"
    with output.open("x") as stream:
        for row in paired:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    read_requests(output)
    shutil.copyfile(Path(__file__), args.output_dir / Path(__file__).name)
    manifest = {
        "condition": "active_hybrid_ppo", "n_requests": len(paired),
        "monitor_reference_requests_sha256": digest(args.requests), "requests_sha256": digest(output),
        "capture_summary_sha256": digest(args.capture_dir / "summary.json"),
        "preparation_script_sha256": digest(Path(__file__)),
        "shared_preparation_script_sha256": digest(Path(__file__).with_name("prepare_defender_replay.py")),
        "system_prompts_unchanged": True, "generation_settings_unchanged": True,
        "window_schedule_unchanged": True,
        "interpretation": "New active-defender episodes paired to Monitor by environment seed, initial network, attacker class, and window schedule. Current attacker actions are excluded from the LLM input.",
        "prompt_limitation": "Historical instructions and few-shot examples still describe attacker actions; this precedes defender-specific prompt adaptation.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
