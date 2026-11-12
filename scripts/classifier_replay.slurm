#!/usr/bin/env bash
#SBATCH --job-name=llama-classifier
#SBATCH --partition=community
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append

set -euo pipefail
umask 0002

if [[ $# -ne 3 ]]; then
    echo "Usage: sbatch classifier_replay.slurm REQUESTS OUTPUT_DIR MODEL_SNAPSHOT" >&2
    exit 2
fi

: "${REPLAY_CHECKOUT:?Set REPLAY_CHECKOUT to the checkout on the data drive}"
: "${REPLAY_CACHE_DIR:?Set REPLAY_CACHE_DIR to a personal data-drive cache}"
: "${REPLAY_TMP_ROOT:?Set REPLAY_TMP_ROOT to a personal data-drive temporary directory}"

export TMPDIR="${REPLAY_TMP_ROOT}/job-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${REPLAY_CACHE_DIR}"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="/data/shared/models/huggingface"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="/data/shared/datasets/huggingface"
export OUTLINES_CACHE_DIR="${REPLAY_CACHE_DIR}/outlines"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONHASHSEED=0
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$OUTLINES_CACHE_DIR" "$2"
(umask 0077; mkdir -p "$HF_HOME")

cd "$REPLAY_CHECKOUT"
exec .venv/bin/python -u scripts/run_classifier_replay.py \
    --requests "$1" --output-dir "$2" --model "$3"
