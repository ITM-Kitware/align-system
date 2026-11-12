#!/usr/bin/env python3
"""Evaluate recorded classifier dialogues with a local structured Llama model."""

import argparse
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time


LABELS = ("B-line-C", "B-line-I", "B-line-A", "Meander-C", "Meander-I", "Meander-A")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def configure_chat_encoding(model):
    """The rendered chat template already contains its special tokens."""
    original_encode = model.tokenizer.encode

    def encode(prompt, **kwargs):
        kwargs["add_special_tokens"] = False
        return original_encode(prompt, **kwargs)

    model.tokenizer.encode = encode


def render_chat_input(model, messages):
    tokenizer = model.tokenizer.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    expected_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    input_ids, _ = model.tokenizer.encode(prompt)
    actual_ids = input_ids[0].tolist()
    if actual_ids != expected_ids:
        raise ValueError("Outlines input tokens differ from the tokenizer's chat template")
    return prompt, len(actual_ids)


def read_requests(path):
    requests = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        item = json.loads(line)
        identity = item["request_id"]
        if identity in seen:
            raise ValueError(f"Duplicate request_id on line {line_number}: {identity}")
        seen.add(identity)
        if item["true_label"] not in LABELS:
            raise ValueError(f"Unknown true label on line {line_number}")
        messages = item["messages"]
        if not messages or any(
            m.get("role") not in ("system", "user") or not isinstance(m.get("content"), str)
            for m in messages
        ):
            raise ValueError(f"Expected system/user input messages on line {line_number}")
        schema = item["response_format"]["json_schema"]["schema"]
        choices = schema["properties"]["action_choice"]["enum"]
        if set(choices) != set(LABELS):
            raise ValueError(f"Unexpected output labels on line {line_number}")
        settings = item["generation"]
        if settings["temperature"] != 0 or settings.get("top_p", 1) != 1:
            raise ValueError("This greedy baseline requires temperature=0 and top_p=1")
        if not isinstance(settings["max_tokens"], int) or settings["max_tokens"] <= 0:
            raise ValueError("max_tokens must be a positive integer")
        requests.append(item)
    if not requests:
        raise ValueError("No replay requests")
    return requests


def read_results(path, requests):
    if not path.exists():
        return {}
    expected = {r["request_id"]: r for r in requests}
    completed = {}
    offset = 0
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        # A killed process can leave an incomplete final append. Preserve it
        # separately before replaying that request; complete invalid lines fail.
        if index == len(lines) - 1 and not line.endswith(b"\n"):
            tail = path.with_name(f"incomplete-tail-{time.time_ns()}.bin")
            tail.write_bytes(line)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        row = json.loads(line)
        identity = row["request_id"]
        if identity not in expected or identity in completed:
            raise ValueError(f"Unknown or duplicate saved request: {identity}")
        if row["true_label"] != expected[identity]["true_label"]:
            raise ValueError(f"Saved true label differs: {identity}")
        if row["predicted_label"] not in LABELS:
            raise ValueError(f"Invalid saved prediction: {identity}")
        completed[identity] = row
        offset += len(line)
    return completed


def report(output, requests, completed, gpu_memory_bytes=None):
    report_path = output / "classification_report.json"
    if report_path.exists():
        saved_peak = json.loads(report_path.read_text()).get("peak_gpu_allocated_bytes")
        if saved_peak is not None:
            gpu_memory_bytes = max(saved_peak, gpu_memory_bytes or 0)
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for row in completed.values():
        matrix[LABELS.index(row["true_label"])][LABELS.index(row["predicted_label"])] += 1
    correct = sum(matrix[i][i] for i in range(len(LABELS)))
    seconds = sum(r["inference_seconds"] for r in completed.values())
    result = {
        "evaluation_unit": "window",
        "status": "complete" if len(completed) == len(requests) else "partial",
        "expected_samples": len(requests),
        "n_samples": len(completed),
        "n_correct": correct,
        "overall_accuracy": correct / len(completed) if completed else None,
        "labels": LABELS,
        "matrix_orientation": "rows=true_label, columns=predicted_label",
        "confusion_matrix": matrix,
        "per_class": {
            label: {
                "support": sum(matrix[i]),
                "correct": matrix[i][i],
                "accuracy": matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None,
            }
            for i, label in enumerate(LABELS)
        },
        "total_inference_seconds": seconds,
        "mean_inference_seconds": seconds / len(completed) if completed else None,
        "peak_gpu_allocated_bytes": gpu_memory_bytes,
        "interpretation": "Local greedy replay of historical inputs; not a reproduction of the hosted Turbo backend.",
    }
    write_json(report_path, result)
    return result


def run(args, requests):
    import outlines
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run inside a Slurm GPU allocation")
    model_path = args.model.resolve()
    if not (model_path / "config.json").is_file():
        raise ValueError("--model must be a complete local model snapshot")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {
        "format_version": 1,
        "requests_sha256": digest(args.requests),
        "runner_sha256": digest(__file__),
        "dependency_lock_sha256": digest(
            Path(__file__).resolve().parents[1] / "requirements-classifier-replay.txt"
        ),
        "model_snapshot": str(model_path),
        "model_config_sha256": digest(model_path / "config.json"),
        "tokenizer_config_sha256": digest(model_path / "tokenizer_config.json"),
        "precision": args.precision,
        "sampler": "greedy",
        "chat_template_add_generation_prompt": True,
        "tokenization_add_special_tokens": False,
        "whitespace_pattern": "[ ]?",
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "outlines", "outlines-core", "huggingface-hub")},
        "n_requests": len(requests),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run identity changed; use a new output directory")
    elif (output / "predictions.jsonl").exists():
        raise ValueError("Predictions exist without a run manifest")
    else:
        write_json(manifest_path, manifest)
    predictions_path = output / "predictions.jsonl"
    completed = read_results(predictions_path, requests)
    remaining = [r for r in requests if r["request_id"] not in completed]
    if not remaining:
        print(json.dumps(report(output, requests, completed)), flush=True)
        return
    print(json.dumps({"stage": "loading_model", "remaining": len(remaining),
                      "gpu": torch.cuda.get_device_name(),
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID")}), flush=True)
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    model = outlines.models.transformers(
        str(model_path), device="cuda",
        model_kwargs={"torch_dtype": dtype, "local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )
    configure_chat_encoding(model)
    generators = {}
    torch.cuda.reset_peak_memory_stats()
    try:
        with predictions_path.open("a") as stream:
            for item in remaining:
                settings = item["generation"]
                seed = settings.get("seed", 0)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                schema = json.dumps(item["response_format"]["json_schema"]["schema"], sort_keys=True)
                if schema not in generators:
                    generators[schema] = outlines.generate.json(
                        model, schema, sampler=outlines.samplers.GreedySampler(),
                        whitespace_pattern=r"[ ]?",
                    )
                # Ground-truth labels and historical answers stay outside the
                # model input. Only the recorded system/user messages are rendered.
                prompt, input_tokens = render_chat_input(model, item["messages"])
                started = time.monotonic()
                prediction = generators[schema](prompt, max_tokens=settings["max_tokens"])
                elapsed = time.monotonic() - started
                label = prediction["action_choice"]
                if label not in LABELS:
                    raise ValueError(f"Unexpected model response: {prediction!r}")
                row = {
                    "request_id": item["request_id"], "scenario_id": item["scenario_id"],
                    "true_label": item["true_label"], "predicted_label": label,
                    "historical_prediction": item["historical_prediction"],
                    "inference_seconds": elapsed,
                    "input_tokens": input_tokens,
                    "max_tokens": settings["max_tokens"],
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                completed[item["request_id"]] = row
                print(json.dumps({"completed": len(completed), "total": len(requests),
                                  "true_label": item["true_label"], "predicted_label": label,
                                  "seconds": round(elapsed, 3)}), flush=True)
                if len(completed) % 10 == 0:
                    report(output, requests, completed, torch.cuda.max_memory_allocated())
    finally:
        result = report(output, requests, completed, torch.cuda.max_memory_allocated())
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    requests = read_requests(args.requests)
    if args.validate_only:
        print(json.dumps({"n_requests": len(requests),
                          "per_class": dict(collections.Counter(r["true_label"] for r in requests)),
                          "requests_sha256": digest(args.requests)}))
        return
    if args.model is None or args.output_dir is None:
        parser.error("--model and --output-dir are required for inference")
    run(args, requests)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, RuntimeError, BlockingIOError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
