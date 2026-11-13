#!/usr/bin/env python3
"""Replay recorded classification inputs using a local Qwen vLLM runtime."""

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

from run_classifier_replay import LABELS, digest, input_protocol, read_requests, read_results, report, write_json


INTERPRETATION = (
    "Qwen FP8 local classification with thinking disabled and historical instructions and examples held fixed. "
    "Observation views are recorded in input_protocol. Not an exact reproduction of a hosted backend."
)


def summarize(output, requests, completed):
    result = report(output, requests, completed)
    result["interpretation"] = interpretation(requests)
    result["memory_measurement"] = "vLLM worker allocations are recorded in the Slurm logs"
    write_json(output / "classification_report.json", result)
    return result


def interpretation(requests):
    protocol = input_protocol(requests)['prompt_protocol']
    if protocol == 'historical_instructions_and_examples_held_fixed':
        return INTERPRETATION
    if protocol == 'fresh_seed_validation':
        return ('Frozen prompt comparison on newly seeded Monitor episodes, each captured twice. '
                'Fresh scores are excluded from candidate selection; shared-seed groups are the resampling unit.')
    if protocol == 'legal_event_history':
        return ('Blue table and raw-observation representations with short or cumulative histories. '
                'No true Red actions or hidden outcomes are supplied. Historical reserved episodes are exploratory.')
    return ('Blue-only prompt adaptation with recorded instructions, examples, and history variants; '
            'Qwen FP8 with thinking disabled. Reserved episodes are separate from prompt development, '
            'but contributed to previously inspected aggregate baselines.')


def run(args, requests):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    model = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prompts = {}
    for item in requests:
        tokens = tokenizer.apply_chat_template(
            item["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False,
        )
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError("Chat template must return a nonempty flat token list")
        if len(tokens) + item["generation"]["max_tokens"] > args.max_model_len:
            raise ValueError(f"Request exceeds context capacity: {item['request_id']}")
        prompts[item["request_id"]] = tokens

    settings = {
        "dtype": "bfloat16", "quantization": "fp8", "tensor_parallel_size": 1,
        "max_model_len": args.max_model_len, "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_model_len,
        "gpu_memory_utilization": 0.88, "enforce_eager": True,
        "language_model_only": True, "enable_prefix_caching": True, "seed": 0,
    }
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "scripts/run_classifier_replay.py",
               root / "scripts/qwen_classifier_replay.slurm",
               root / "requirements-qwen-replay.in", root / "requirements-qwen-replay.txt"]
    manifest = {
        "format_version": 1, "requests_sha256": digest(args.requests),
        "input_protocol": input_protocol(requests),
        "source_sha256": {p.name: digest(p) for p in sources},
        "model_snapshot": str(model),
        "model_files_sha256": {name: digest(model / name) for name in
                               ("config.json", "tokenizer_config.json", "tokenizer.json",
                                "chat_template.jinja", "model.safetensors.index.json")},
        "engine_settings": settings, "enable_thinking": False,
        "structured_output_backend": "xgrammar", "temperature": 0, "top_p": 1,
        "generation_limits": "recorded per request", "n_requests": len(requests),
        "max_input_tokens": max(map(len, prompts.values())),
        "software": {name: importlib.metadata.version(name) for name in
                     ("vllm", "torch", "transformers", "xgrammar", "huggingface-hub")},
        "interpretation": interpretation(requests),
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Run identity changed; use a new output directory")
        elif (output / "predictions.jsonl").exists():
            raise ValueError("Predictions exist without a run manifest")
        else:
            write_json(manifest_path, manifest)
            evidence = output / "source"
            evidence.mkdir(exist_ok=True)
            for path in sources:
                shutil.copyfile(path, evidence / path.name)
        completed = read_results(output / "predictions.jsonl", requests)
        remaining = [item for item in requests if item["request_id"] not in completed]
        if not remaining:
            print(json.dumps(summarize(output, requests, completed)), flush=True)
            return
        print(json.dumps({"stage": "loading_model", "remaining": len(remaining),
                          "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                          "max_input_tokens": manifest["max_input_tokens"]}), flush=True)
        llm = LLM(model=str(model), tokenizer=str(model),
                  structured_outputs_config={"backend": "xgrammar"}, **settings)
        try:
            with (output / "predictions.jsonl").open("a") as stream:
                for item in remaining:
                    generation = item["generation"]
                    sampling = SamplingParams(
                        temperature=0, top_p=1, seed=generation.get("seed", 0),
                        max_tokens=generation["max_tokens"],
                        structured_outputs=StructuredOutputsParams(
                            json=item["response_format"]["json_schema"]["schema"]),
                    )
                    started = time.monotonic()
                    result = llm.generate(
                        [{"prompt_token_ids": prompts[item["request_id"]]}],
                        sampling_params=sampling, use_tqdm=False,
                    )[0].outputs[0]
                    elapsed = time.monotonic() - started
                    if result.finish_reason != "stop":
                        raise ValueError(f"Incomplete generation: {result.finish_reason}: {result.text!r}")
                    label = json.loads(result.text)["action_choice"]
                    if label not in LABELS:
                        raise ValueError(f"Unknown predicted label: {label}")
                    row = {
                        "request_id": item["request_id"], "scenario_id": item["scenario_id"],
                        "true_label": item["true_label"], "predicted_label": label,
                        "historical_prediction": item["historical_prediction"],
                        "inference_seconds": elapsed,
                        "input_tokens": len(prompts[item["request_id"]]),
                        "output_tokens": len(result.token_ids), "raw_response": result.text,
                        "max_tokens": generation["max_tokens"],
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed[item["request_id"]] = row
                    print(json.dumps({"completed": len(completed), "total": len(requests),
                                      "seconds": round(elapsed, 3)}), flush=True)
                    if len(completed) % 10 == 0:
                        summarize(output, requests, completed)
        finally:
            print(json.dumps(summarize(output, requests, completed)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    requests = read_requests(args.requests)
    if args.validate_only:
        print(json.dumps({"n_requests": len(requests), "requests_sha256": digest(args.requests)}))
        return
    if args.model is None or args.output_dir is None:
        parser.error("--model and --output-dir are required for inference")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Run inside a Slurm GPU allocation")
    run(args, requests)


if __name__ == "__main__":
    main()
