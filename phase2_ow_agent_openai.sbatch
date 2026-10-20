#!/bin/bash
#SBATCH --account=itm
#SBATCH --nodes=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=ow_agent_openai
#SBATCH --output=slurm-%x-%j.out

# Activate the align environment and export OPENAI_API_KEY before sbatch.
# MODEL (or OPENAI_MODEL) selects the hosted model; default gpt-4.1.
# Defaults: compreg, evidence, on-demand, live, APRIL_OPENWORLD3.
# Shared runner settings and trailing Hydra overrides apply.
# TOOLKIT=kaleido still loads Kaleido locally; default it to CPU here.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
export MODEL_BACKEND=openai
export TOOLKIT="${TOOLKIT:-compreg}"
case "${EXPERIMENT:-${TOOLKIT}}" in
  compreg|comparative_regression|phase2_langchain_agent_ow_comparative_regression_live)
    export ASSESSMENT_MODE="${ASSESSMENT_MODE:-evidence}" ;;
esac
export KALEIDO_DEVICE="${KALEIDO_DEVICE:-cpu}"
export PYTHONUNBUFFERED=1
exec bash ./phase2_ow_agent_qwen38.sh "$@"
