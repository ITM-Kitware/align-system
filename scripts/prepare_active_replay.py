#!/usr/bin/env python3
"""Prepare active-defender classifier inputs with the historical prompt held fixed."""

import argparse
import json
from pathlib import Path
import shutil

from prepare_defender_replay import capture_paths, defender_request
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
    for episode, path in zip(summary["episodes"], capture_paths(args.capture_dir)):
        record = json.loads(path.read_text())
        captures[(episode["class_directory"], episode["trajectory_id"])] = (record, episode["capture_sha256"])
    paired = []
    for original in requests:
        captured, capture_sha = captures[(original["provenance"]["class_directory"], original["trajectory_id"])]
        result = defender_request(original, captured, capture_sha)
        result["historical_prediction"] = None
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
        "observation_formatter_sha256": digest(Path(__file__).resolve().parents[1] / "align_system/interfaces/cage_observations.py"),
        "system_prompts_unchanged": True, "generation_settings_unchanged": True,
        "window_schedule_unchanged": True,
        "interpretation": "New active-defender episodes paired to Monitor by environment seed, initial network, attacker class, and window schedule. Current attacker actions are excluded from the LLM input.",
        "prompt_limitation": "Historical instructions and few-shot examples still describe attacker actions; this precedes defender-specific prompt adaptation.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
