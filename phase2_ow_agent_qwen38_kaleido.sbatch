#!/bin/bash
#SBATCH --account=itm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=ow_agent_qwen38_kaleido
#SBATCH --output=slurm-%x-%j.out

# Submit from the repo root, or set PROJECT_DIR to the checkout.
# TOOLKIT=baseline|kaleido selects the toolkit on generic launchers.
# AUTO_ASSESS=true|false controls automatic Kaleido assessments.
# All usernames start with testrun-, including live runs and custom names.
# For a remote model, set VLLM_URL; use --gres=none if this job needs no GPU
# (Kaleido still needs a GPU unless KALEIDO_DEVICE=cpu).
# For bf16 on 48GB cards, request --gres=gpu:2 and set TP=2 and MODEL.
# See README.md for examples. DRY_RUN=1 prints the command without running it.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
export TOOLKIT=kaleido
export EXPERIMENT=phase2_langchain_agent_ow_kaleido_live
exec bash ./phase2_ow_agent_qwen38.sh "$@"
