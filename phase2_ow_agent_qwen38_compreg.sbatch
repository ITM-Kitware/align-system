#!/bin/bash
#SBATCH --account=itm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=ow_agent_qwen38_compreg
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%j.out

# Submit from the repo root, or set PROJECT_DIR to the checkout.
# TOOLKIT=baseline|kaleido|compreg selects the toolkit on generic launchers.
# ASSESSMENT_MODE=ranked|evidence applies to CR; AUTO_ASSESS=true|false applies to both toolkits.
# All usernames start with testrun-, including live runs and custom names.
# For a remote model, set VLLM_URL; use --gres=none if this job needs no GPU
# (Kaleido still needs a GPU unless KALEIDO_DEVICE=cpu).
# For bf16 on 48GB cards, request --gres=gpu:2 and set TP=2 and MODEL.
# See README.md for examples. DRY_RUN=1 prints the command without running it.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
export TOOLKIT=compreg
export EXPERIMENT=phase2_langchain_agent_ow_comparative_regression_live
# Alignment scores exist only for live eval sessions against a TA3 server
# that can reach TA1: the server scores each scenario at its end and the
# driver records the result in scoring_status.json. Test sessions never
# score, and the OpenWorld profiles have no training scenarios, so
# TRAINING_SESSION=full does not apply. The job fails if a scenario ends
# without a score (REQUIRE_SCORES=false to only warn).
# Job array: task 0 runs April OpenWorld, task 1 runs February OpenWorld, as
# separate jobs with their own output directories. Submit with --array=0 or
# --array=1 for one of them, or set ADM_PROFILE to run a single profile.
export RUN_MODE="${RUN_MODE:-live}"
profiles=(APRIL_OPENWORLD3 FEB_OPENWORLD3)
export ADM_PROFILE="${ADM_PROFILE:-${profiles[${SLURM_ARRAY_TASK_ID:-0}]}}"
exec bash ./phase2_ow_agent_qwen38.sh "$@"
