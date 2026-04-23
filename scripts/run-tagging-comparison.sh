#!/bin/bash
# Runs the tagging baseline and RAG baseline experiments back-to-back for comparison.
#
# Usage:
#   bash scripts/run-tagging-comparison.sh [output_dir]
#
# Output dir defaults to: tagging_comparison_results
# Each run gets its own subdirectory: <output_dir>/<experiment>/<timestamp>/

set -e

OUTPUT_BASEDIR=${1:-tagging_comparison_results}
DATE_NOW=$(date +"%Y-%m-%d__%H-%M-%S")

EXPERIMENTS=(
    "tagging/tagging_baseline"
    "tagging/tagging_rag_baseline"
)

for exp in "${EXPERIMENTS[@]}"; do
    exp_name=$(basename "$exp")
    out_dir="${OUTPUT_BASEDIR}/${exp_name}/${DATE_NOW}"
    echo "========================================"
    echo "Running: ${exp}"
    echo "Output:  ${out_dir}"
    echo "========================================"

    uv run run_align_system \
        +experiment="${exp}" \
        hydra.run.dir="${out_dir}"

    echo "Done: ${exp_name}"
    echo
done

echo "Both runs complete. Results in: ${OUTPUT_BASEDIR}/"
