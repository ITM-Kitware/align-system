#!/usr/bin/env bash
#SBATCH --job-name=classifier-followup
#SBATCH --partition=community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
umask 0002
: "${REPLAY_CHECKOUT:?Set immutable source}"
: "${REPLAY_PYTHON:?Set inference Python}"
: "${REPLAY_CACHE_DIR:?Set data-drive cache}"
: "${REPLAY_TMP_ROOT:?Set data-drive temporary root}"
: "${REPLAY_STOP_AT_EPOCH:?Set absolute cutoff}"
: "${REPLAY_MAX_JOB_SECONDS:?Set allocation time budget}"
: "${FOLLOWUP_ROOT:?Set owned follow-up study directory}"
: "${FRESH_CAPTURE_ROOT:?Set completed fresh capture directory}"
: "${PRIMARY_SCREEN_ROOT:?Set frozen primary screen attempt}"
: "${LEGACY_BLUE_REQUESTS:?Set completed matched Blue input source}"
: "${REPLAY_MODEL:?Set existing cached model snapshot}"
if [[ $# -ne 1 || ( "$1" != raw && "$1" != fresh ) ]]; then
    echo 'Usage: overnight_followups.slurm raw|fresh' >&2
    exit 2
fi
remaining=$((REPLAY_STOP_AT_EPOCH - $(date +%s) - 120))
if ((remaining <= 0)); then exit 75; fi
export TMPDIR="${REPLAY_TMP_ROOT}/followup-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$REPLAY_CACHE_DIR"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE=/data/shared/models/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="${REPLAY_CACHE_DIR}/vllm"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$FOLLOWUP_ROOT/$1/output"
cp "$0" "$FOLLOWUP_ROOT/$1/output/submitted_job.sh"
cd "$REPLAY_CHECKOUT"
context_args=(--context-limit 32768)
if [[ "$1" == raw ]]; then
    context_args=(--context-limit 65536 --prefill-batch-tokens 8192)
fi
exec timeout --signal=TERM --kill-after=45s "${remaining}s" \
    "$REPLAY_PYTHON" -u scripts/run_overnight_followups.py "$1" \
    --inputs "$FOLLOWUP_ROOT/$1/inputs" --output "$FOLLOWUP_ROOT/$1/output" \
    --fresh-captures "$FRESH_CAPTURE_ROOT" --screen-inputs "$PRIMARY_SCREEN_ROOT/inputs" \
    --primary-output "$PRIMARY_SCREEN_ROOT/output" --legacy-requests "$LEGACY_BLUE_REQUESTS" \
    --model "$REPLAY_MODEL" --stop-at "$REPLAY_STOP_AT_EPOCH" "${context_args[@]}"
