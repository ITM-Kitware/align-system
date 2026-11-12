#!/usr/bin/env python3
"""Count conflicting attacker labels attached to exactly identical model inputs."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = collections.defaultdict(collections.Counter)
    for line in args.requests.open():
        row = json.loads(line)
        key = hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()
        groups[key][row["true_label"]] += 1
    total = sum(sum(counts.values()) for counts in groups.values())
    conflicts = [counts for counts in groups.values() if len(counts) > 1]
    ceiling_correct = sum(max(counts.values()) for counts in groups.values())
    result = {
        "n_samples": total, "unique_message_sequences": len(groups),
        "conflicting_groups": len(conflicts),
        "samples_in_conflicting_groups": sum(sum(counts.values()) for counts in conflicts),
        "optimistic_in_sample_ceiling_correct": ceiling_correct,
        "optimistic_in_sample_ceiling_accuracy": ceiling_correct / total,
        "conflict_label_sets": dict(collections.Counter(
            ", ".join(sorted(counts)) for counts in conflicts)),
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "interpretation": "A deterministic classifier using only these exact messages must give identical predictions for identical inputs. Choosing each observed group's majority label gives an optimistic in-sample ceiling, not a generalization estimate. No labels or time metadata are removed from the input messages for this grouping.",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
