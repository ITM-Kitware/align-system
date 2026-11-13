#!/usr/bin/env bash
#SBATCH --job-name=blue-prompt-screen
#SBATCH --partition=community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:30:00

set -euo pipefail
umask 0002
: "${REPLAY_CHECKOUT:?Set immutable source directory}"
: "${REPLAY_PYTHON:?Set existing inference environment Python}"
: "${REPLAY_CACHE_DIR:?Set data-drive cache}"
: "${REPLAY_TMP_ROOT:?Set data-drive temporary root}"
: "${REPLAY_STOP_AT_EPOCH:?Set absolute cutoff}"
if [[ $# -ne 3 ]]; then
    echo 'Usage: blue_prompt_screen.slurm INPUTS OUTPUT MODEL' >&2
    exit 2
fi
remaining=$((REPLAY_STOP_AT_EPOCH - $(date +%s) - 120))
if ((remaining <= 0)); then
    exit 75
fi
export TMPDIR="${REPLAY_TMP_ROOT}/blue-screen-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$REPLAY_CACHE_DIR"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE=/data/shared/models/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="${REPLAY_CACHE_DIR}/vllm"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$2"
cp "$0" "$2/submitted_job.sh"
cd "$REPLAY_CHECKOUT"
exec timeout --signal=TERM --kill-after=45s "${remaining}s" \
    "$REPLAY_PYTHON" -u scripts/run_blue_prompt_screen.py \
    --inputs "$1" --output "$2" --model "$3" --stop-at "$REPLAY_STOP_AT_EPOCH"
