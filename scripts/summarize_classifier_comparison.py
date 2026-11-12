#!/usr/bin/env python3
"""Validate classifier outputs and write paired scores for shared experiment records."""

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

from run_classifier_replay import LABELS, write_json


def read_experiment(root, spec):
    requests_path = root / spec["requests"]
    predictions_path = root / spec["output"] / "predictions.jsonl"
    if not requests_path.exists():
        return {**spec, "status": "inputs_pending", "n_samples": 0}, {}
    inputs = {}
    requests_hash = hashlib.sha256()
    for line in requests_path.open("rb"):
        requests_hash.update(line)
        row = json.loads(line)
        if row["request_id"] in inputs:
            raise ValueError("Duplicate input identity")
        inputs[row["request_id"]] = {
            "true_label": row["true_label"],
            "paired_id": row.get("original_request_id", row["request_id"]),
            "trajectory_id": row["trajectory_id"],
        }
    manifest_path = predictions_path.with_name("run_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["requests_sha256"] != requests_hash.hexdigest():
            raise ValueError("Run manifest does not match the configured input file")
        if spec["model_directory"] not in Path(manifest["model_snapshot"]).parts:
            raise ValueError("Run manifest does not match the configured model")
    elif predictions_path.exists():
        raise ValueError("Predictions are missing their run manifest")
    predictions = {}
    if predictions_path.exists():
        with predictions_path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                row = json.loads(line)
                identity = row["request_id"]
                if identity not in inputs or identity in predictions:
                    raise ValueError(f"Unknown or duplicate output identity: {identity}")
                if row["true_label"] != inputs[identity]["true_label"] or row["predicted_label"] not in LABELS:
                    raise ValueError("Saved output label is invalid")
                predictions[identity] = row
    matrix = [[0] * len(LABELS) for _ in LABELS]
    paired = {}
    for identity, row in predictions.items():
        matrix[LABELS.index(row["true_label"])][LABELS.index(row["predicted_label"])] += 1
        paired_id = inputs[identity]["paired_id"]
        if paired_id in paired:
            raise ValueError("Duplicate paired window identity")
        paired[paired_id] = {"true_label": row["true_label"], "predicted_label": row["predicted_label"],
                             "correct": row["true_label"] == row["predicted_label"]}
    n = len(predictions)
    correct = sum(matrix[i][i] for i in range(len(LABELS)))
    result = {
        **spec, "status": "complete" if n == len(inputs) else "partial" if n else "pending",
        "expected_samples": len(inputs), "n_samples": n, "n_correct": correct,
        "accuracy": correct / n if n else None,
        "family_accuracy": sum(r["true_label"].rsplit("-", 1)[0] == r["predicted_label"].rsplit("-", 1)[0]
                               for r in predictions.values()) / n if n else None,
        "cia_accuracy": sum(r["true_label"][-1] == r["predicted_label"][-1]
                            for r in predictions.values()) / n if n else None,
        "labels": LABELS, "confusion_matrix": matrix,
        "matrix_orientation": "rows=true, columns=predicted",
        "mean_inference_seconds": sum(r["inference_seconds"] for r in predictions.values()) / n if n else None,
        "per_class": {label: {"correct": matrix[i][i], "samples": sum(matrix[i])}
                      for i, label in enumerate(LABELS)},
    }
    return result, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = {}
        paired = {}
        for spec in cfg["experiments"]:
            result, pairs = read_experiment(Path(cfg["shared_root"]), spec)
            runs[spec["name"]] = result
            paired[spec["name"]] = pairs
        comparisons = []
        for left, right in cfg["comparisons"]:
            common = paired[left].keys() & paired[right].keys()
            for identity in common:
                if paired[left][identity]["true_label"] != paired[right][identity]["true_label"]:
                    raise ValueError("Paired true labels disagree")
            a = sum(paired[left][identity]["correct"] for identity in common)
            b = sum(paired[right][identity]["correct"] for identity in common)
            comparisons.append({"left": left, "right": right, "paired_windows": len(common),
                                "left_correct": a, "right_correct": b,
                                "accuracy_difference_right_minus_left": (b-a)/len(common) if common else None})
        result = {
            "status": "complete" if all(r["status"] == "complete" for r in runs.values()) else "partial",
            "runs": runs, "paired_comparisons": comparisons,
            "interpretation": cfg["interpretation"],
            "statistical_unit": "Reported accuracy counts overlapping windows; windows from one episode are not independent trials.",
        }
        write_json(args.output_dir / "comparison.json", result)
        temporary = args.output_dir / "scores.csv.tmp"
        columns = ("name", "model", "input_view", "blue_policy", "status", "n_samples", "n_correct",
                   "accuracy", "family_accuracy", "cia_accuracy", "mean_inference_seconds")
        with temporary.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(runs.values())
        temporary.replace(args.output_dir / "scores.csv")
        print(json.dumps({"status": result["status"], "runs": {
            name: {k:r.get(k) for k in ("status", "n_samples", "accuracy")} for name,r in runs.items()}}))


if __name__ == "__main__":
    main()
