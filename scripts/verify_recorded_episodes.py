#!/usr/bin/env python3
"""Verify all unique recorded episodes referenced by classifier replay inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--cage-root", type=Path, required=True)
    parser.add_argument("--cage-revision", required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-per-class", action="store_true")
    args = parser.parse_args()
    references = sorted({(row["provenance"]["class_directory"], row["trajectory_id"])
                         for row in map(json.loads, args.requests.read_text().splitlines())})
    if args.first_per_class:
        references = list({kind: next(item for item in references if item[0] == kind)
                           for kind in sorted({item[0] for item in references})}.values())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("capture_defender_observations.py")
    source = args.output_dir / "source"
    source.mkdir()
    root = script.resolve().parents[1]
    for path in (script, Path(__file__), root / "requirements-observation-capture.in",
                 root / "requirements-observation-capture.txt"):
        shutil.copyfile(path, source / path.name)
    results = []
    for kind, trajectory in references:
        reference = args.trajectory_root / kind / "trajectories" / trajectory
        name = f"{kind}-{Path(trajectory).stem}"
        paths = []
        for repeat in (1, 2):
            output = args.output_dir / f"{name}-repeat{repeat}.json"
            command = [sys.executable, str(script), "--reference", str(reference),
                       "--cage-root", str(args.cage_root), "--cage-revision", args.cage_revision,
                       "--scenario", str(args.scenario), "--output", str(output)]
            with (args.output_dir / f"{name}-repeat{repeat}.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            paths.append(output)
        repeated_identically = paths[0].read_bytes() == paths[1].read_bytes()
        if not repeated_identically:
            raise RuntimeError(f"Fresh-process replay differs: {name}")
        captured = json.loads(paths[0].read_text())
        row = {
            "class_directory": kind, "trajectory_id": trajectory, "seed": captured["seed"],
            "capture_path": paths[0].name, "steps": len(captured["steps"]),
            "capture_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            "fresh_process_repeat_identical": repeated_identically,
            **{key: value for key, value in captured.items() if key.startswith("historical_")},
        }
        results.append(row)
        report = {"status": "complete" if len(results) == len(references) else "partial",
                  "expected_episodes": len(references), "checked_episodes": len(results),
                  "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                  "episodes": results}
        temporary = args.output_dir / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output_dir / "summary.json")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
