#!/usr/bin/env python3
"""Check every recorded prompt against a local model's context capacity on CPU."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    count = maximum = 0
    source_hash = hashlib.sha256()
    for line in args.requests.open("rb"):
        source_hash.update(line)
        item = json.loads(line)
        tokens = tokenizer.apply_chat_template(item["messages"], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False,
                                               return_dict=False)
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
            raise ValueError("Expected a flat token list")
        length = len(tokens) + item["generation"]["max_tokens"]
        if length > args.limit:
            raise ValueError(f"Context capacity exceeded: {item['request_id']}: {length}")
        maximum = max(maximum, length)
        count += 1
    result = {"n_requests": count, "requests_sha256": source_hash.hexdigest(),
              "model_snapshot": str(args.model.resolve()), "enable_thinking": False,
              "max_prompt_plus_generation_tokens": maximum, "context_limit": args.limit}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
