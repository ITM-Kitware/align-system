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

# BASELINE Experiments

# EXPERIMENTS=(
#     "tagging/tagging_baseline"
#     "tagging/tagging_rag_baseline"
# )

# for exp in "${EXPERIMENTS[@]}"; do
#     exp_name=$(basename "$exp")
#     out_dir="${OUTPUT_BASEDIR}/${exp_name}/${DATE_NOW}"
#     echo "========================================"
#     echo "Running: ${exp}"
#     echo "Output:  ${out_dir}"
#     echo "========================================"

#     uv run run_align_system \
#         +experiment="${exp}" \
#         hydra.run.dir="${out_dir}"

#     echo "Done: ${exp_name}"
#     echo
# done

# ALIGNEMENT Experiments
EXPERIMENTS=(
    # "tagging/tagging_fewshot_aligned"
    "tagging/tagging_fewshot_aligned_rag"
)

ALIGNMENT_TARGET=(
    # "tagging/bcd"
    "tagging/start"
    # "tagging/salt"
)

INTERFACE=(
    "/data/users/yonatan.gefen/align-system/start-protocol-tagging-exp/itm_eval_align_tag_example_start_color_only_treated_interventions.json"
)

for exp in "${EXPERIMENTS[@]}"; do
    for at in "${ALIGNMENT_TARGET[@]}"; do
        exp_name=$(basename "$exp")
        at_name=$(basename "$at")
        out_dir="${OUTPUT_BASEDIR}/${exp_name}/${at_name}/${DATE_NOW}"
        echo "========================================"
        echo "Running: ${exp} with alignment target ${at}"
        echo "Output:  ${out_dir}"
        echo "========================================"

        uv run run_align_system \
            +experiment="${exp}" \
            +alignment_target="${at}" \
            interface.input_output_filepath="${INTERFACE}" \
            hydra.run.dir="${out_dir}"

        echo "Done: ${exp_name} with alignment taregt: ${at_name}"
        echo
    done
done

echo "All runs complete. Results in: ${OUTPUT_BASEDIR}/"
