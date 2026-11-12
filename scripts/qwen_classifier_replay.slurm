#!/usr/bin/env bash
#SBATCH --job-name=qwen-classifier
#SBATCH --partition=community
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append

set -euo pipefail
umask 0002

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch qwen_classifier_replay.slurm REQUESTS OUTPUT_DIR MODEL_SNAPSHOT" >&2
    exit 2
fi
: "${REPLAY_CHECKOUT:?Set REPLAY_CHECKOUT to the data-drive source directory}"
: "${REPLAY_CACHE_DIR:?Set REPLAY_CACHE_DIR to a personal data-drive cache}"
: "${REPLAY_TMP_ROOT:?Set REPLAY_TMP_ROOT to a personal data-drive temporary directory}"
export TMPDIR="${REPLAY_TMP_ROOT}/qwen-job-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$REPLAY_CACHE_DIR"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE=/data/shared/models/huggingface
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE=/data/shared/datasets/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="${REPLAY_CACHE_DIR}/vllm"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"
(umask 0077; mkdir -p "$HF_HOME")
test -f "$3/config.json"
test -d "$2" || mkdir "$2"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
cd "$REPLAY_CHECKOUT"
exec .venv-qwen/bin/python -u scripts/run_qwen_classifier_replay.py \
    --requests "$1" --output-dir "$2" --model "$3"
