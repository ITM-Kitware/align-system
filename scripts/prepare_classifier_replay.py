#!/usr/bin/env python3
"""Prepare exact recorded classifier dialogues for local model evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys


CLASSES = {
    "bline_c": "B-line-C",
    "bline_i": "B-line-I",
    "bline_a": "B-line-A",
    "meander_c": "Meander-C",
    "meander_i": "Meander-I",
    "meander_a": "Meander-A",
}
SCENARIO_PREFIX = "[bold]*Scenario ID*[/bold]: "
WINDOW_PATTERN = re.compile(r"^(.*)#steps=(\d+):(\d+)$")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_dialogues(path):
    dialogues = {}
    scenario_id = None
    repeated = 0
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.startswith(SCENARIO_PREFIX):
                scenario_id = line[len(SCENARIO_PREFIX):].strip()
            elif line.startswith('[{"role": "system"'):
                require(scenario_id is not None, f"{path}:{number}: dialogue has no scenario ID")
                dialogue = json.loads(line)
                require(
                    isinstance(dialogue, list)
                    and len(dialogue) == 2
                    and [item.get("role") for item in dialogue] == ["system", "user"]
                    and all(isinstance(item.get("content"), str) for item in dialogue),
                    f"{path}:{number}: expected a system/user dialogue",
                )
                messages = [{"role": item["role"], "content": item["content"]} for item in dialogue]
                if scenario_id in dialogues:
                    require(dialogues[scenario_id] == messages,
                            f"{path}:{number}: conflicting dialogues for {scenario_id}")
                    repeated += 1
                else:
                    dialogues[scenario_id] = messages
    require(dialogues, f"{path}: no recorded dialogues")
    return dialogues, repeated


def prepare_class(run_dir, directory, label):
    root = run_dir / directory
    sources = [root / filename for filename in (
        "classification_results.json", "request_history.json", "raw_align_system.log",
        "run_manifest.json", ".hydra/config.yaml",
    )]
    for path in sources:
        require(path.is_file(), f"Missing evidence: {path}")
    results = read_json(sources[0])
    history = read_json(sources[1])
    dialogues, repetitions = read_dialogues(sources[2])
    require(results.get("complete") is True, f"{directory}: classification is incomplete")
    require(results.get("evaluation_unit") == "window", f"{directory}: expected window evaluation")
    require(set(results["labels"]) == set(CLASSES.values()), f"{directory}: unexpected labels")
    records = results["predictions"]
    require(records and len(records) == results["n_labeled"], f"{directory}: inconsistent record count")
    require(len(records) == len({p["scenario_id"] for p in records}),
            f"{directory}: duplicate prediction IDs")
    require(set(dialogues) == {p["scenario_id"] for p in records},
            f"{directory}: missing or extra dialogues relative to predictions")
    correct = sum(p["predicted_label"] == p["true_label"] for p in records)
    require(correct == results["n_correct"], f"{directory}: inconsistent correct count")
    require(abs(correct / len(records) - results["accuracy"]) < 1e-12,
            f"{directory}: inconsistent accuracy")
    output = []
    for index, prediction in enumerate(records):
        sid = prediction["scenario_id"]
        require(prediction["true_label"] == label, f"{directory}: wrong true label for {sid}")
        require(prediction["predicted_label"] in CLASSES.values(), f"{directory}: unknown prediction")
        match = WINDOW_PATTERN.fullmatch(sid)
        require(match is not None, f"{directory}: malformed window ID {sid}")
        start, end = prediction["window_start"], prediction["window_end"]
        require(match[1] == prediction["trajectory_id"] and int(match[2]) == start
                and int(match[3]) == end and end - start == prediction["window_size"],
                f"{directory}: window metadata does not match {sid}")
        messages = dialogues[sid]
        steps = [int(value) for value in re.findall(r"^Step (\d+):$", messages[1]["content"], re.MULTILINE)]
        require(steps == list(range(start, end)), f"{directory}: dialogue steps do not match {sid}")
        indices = prediction["inference_request_indices"]
        require(indices and len(indices) == len(set(indices))
                and all(type(i) is int and 0 <= i < len(history) for i in indices),
                f"{directory}: invalid request-history indices for {sid}")
        request = history[indices[-1]]
        require(request.get("status") == "completed", f"{directory}: final request failed for {sid}")
        response = json.loads(request["response_text"])
        require(response.get("action_choice") == prediction["predicted_label"],
                f"{directory}: final response disagrees with prediction for {sid}")
        response_format = request["response_format"]
        require(response_format.get("type") == "json_schema", f"{directory}: missing JSON schema")
        schema = response_format["json_schema"]["schema"]
        require(set(schema["properties"]["action_choice"]["enum"]) == set(CLASSES.values()),
                f"{directory}: response schema labels do not match")
        generation = {key: request[key] for key in ("temperature", "top_p", "seed", "max_tokens")}
        model = {"requested_model": request["requested_model"],
                 "request_model": request["request_model"], "provider": request["provider"],
                 "served_model": request["model"]}
        identity = {"scenario_id": sid, "messages": messages, "response_format": response_format,
                    "generation": generation, "model": model}
        output.append({
            "request_id": digest(identity), "scenario_id": sid,
            "trajectory_id": prediction["trajectory_id"],
            "true_label": label, "historical_prediction": prediction["predicted_label"],
            "messages": messages, "response_format": response_format,
            "generation": generation, "model": model,
            "provenance": {
                "class_directory": directory, "source_prediction_index": index,
                "window_start": start, "window_end": end, "window_size": prediction["window_size"],
                "inference_request_indices": indices,
                "recorded_request_sha256": request.get("request_sha256"),
                "system_prompt_sha256": hashlib.sha256(messages[0]["content"].encode()).hexdigest(),
                "messages_sha256": digest(messages),
            },
        })
    return output, sources, repetitions


def prepare(run_dir, output_dir, per_class=None):
    require(per_class is None or per_class > 0, "--per-class must be positive")
    require(not output_dir.exists(), f"Output directory already exists: {output_dir}")
    all_records, selected, sources = [], [], []
    repetitions = 0
    for directory, label in CLASSES.items():
        records, files, duplicates = prepare_class(run_dir, directory, label)
        require(per_class is None or per_class <= len(records),
                f"--per-class exceeds available records for {label}")
        all_records.extend(records)
        sources.extend(files)
        repetitions += duplicates
        positions = (range(len(records)) if per_class is None
                     else sorted(random.Random(0).sample(range(len(records)), per_class)))
        selected.extend(records[i] for i in positions)
    require(len({r["request_id"] for r in all_records}) == len(all_records), "Duplicate request identities")
    counts = Counter(r["true_label"] for r in all_records)
    require(len(set(counts.values())) == 1, "Source dataset is not balanced across classes")
    report_path = run_dir / "classification_report.json"
    require(report_path.is_file(), f"Missing evidence: {report_path}")
    report = read_json(report_path)
    correct = sum(r["true_label"] == r["historical_prediction"] for r in all_records)
    require(report["n_samples"] == len(all_records) and report["n_correct"] == correct,
            "Aggregate report does not match per-class predictions")
    require(abs(report["overall_accuracy"] - correct / len(all_records)) < 1e-12,
            "Aggregate accuracy does not match per-class predictions")
    sources.append(report_path)
    manifest = {
        "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir.resolve()),
        "source_files": {str(p.relative_to(run_dir)): file_digest(p) for p in sources},
        "source_count": len(all_records), "request_count": len(selected),
        "source_class_counts": dict(counts),
        "selected_class_counts": dict(Counter(r["true_label"] for r in selected)),
        "source_accuracy": correct / len(all_records),
        "selected_historical_accuracy": sum(r["true_label"] == r["historical_prediction"] for r in selected) / len(selected),
        "selection": {"per_class": per_class, "method": "all" if per_class is None else "seeded_sample_per_class", "seed": 0},
        "identical_resumed_dialogues_collapsed": repetitions,
        "source_final_request_token_caps": dict(Counter(str(r["generation"]["max_tokens"]) for r in all_records)),
        "source_models": [json.loads(s) for s in sorted({canonical_json(r["model"]) for r in all_records})],
        "system_prompt_sha256": sorted({r["provenance"]["system_prompt_sha256"] for r in all_records}),
        "notes": [
            "Messages are copied from recorded dialogues, not rendered from the current prompt template.",
            "The historical seven-step few-shot prompt includes timing rules removed at 5faddefc1c802df6513b3d32367f78d79ecaa49b.",
            "Per-record generation settings come from each prediction's final successful request; resumed runs may mix token limits.",
            "Local model inference is a new backend; identical messages do not guarantee identical hosted-provider predictions.",
            "true_label and historical_prediction are scoring metadata and must never be included in model input.",
        ],
    }
    output_dir.mkdir(parents=True)
    requests_path = output_dir / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(canonical_json(record) + "\n")
    manifest["requests_sha256"] = file_digest(requests_path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-class", type=int, help="Deterministic balanced smoke subset; default: all windows")
    args = parser.parse_args()
    try:
        manifest = prepare(args.run_dir, args.output_dir, args.per_class)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"request_count": manifest["request_count"], "source_count": manifest["source_count"],
                      "source_accuracy": manifest["source_accuracy"], "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
