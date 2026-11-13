#!/usr/bin/env bash
#SBATCH --job-name=active-transfer
#SBATCH --partition=community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00

set -euo pipefail
umask 0002
: "${REPLAY_CHECKOUT:?Set immutable source}"
: "${REPLAY_PYTHON:?Set existing inference Python}"
: "${REPLAY_CACHE_DIR:?Set data-drive cache}"
: "${REPLAY_TMP_ROOT:?Set data-drive temporary root}"
: "${TRANSFER_STOP_AT_EPOCH:?Set this run's absolute deadline}"
: "${TRANSFER_MAX_JOB_SECONDS:?Set this run's allocation limit}"
if [[ $# -ne 3 ]]; then echo 'Usage: active_transfer.slurm INPUTS OUTPUT MODEL' >&2; exit 2; fi
remaining=$((TRANSFER_STOP_AT_EPOCH - $(date +%s) - 120))
if ((remaining <= 0)); then exit 75; fi
export TMPDIR="${REPLAY_TMP_ROOT}/active-transfer-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$REPLAY_CACHE_DIR" VLLM_CACHE_ROOT="${REPLAY_CACHE_DIR}/vllm"
export HF_HOME="${HOME}/.cache/huggingface" HF_HUB_CACHE=/data/shared/models/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$2"
cp "$0" "$2/submitted_job.sh"
cd "$REPLAY_CHECKOUT"
exec timeout --signal=TERM --kill-after=45s "${remaining}s" \
    "$REPLAY_PYTHON" -u scripts/run_active_transfer.py --inputs "$1" --output "$2" --model "$3" --stop-at "$TRANSFER_STOP_AT_EPOCH"
