#!/usr/bin/env bash
#SBATCH --job-name=active-defender-capture
#SBATCH --partition=community
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -euo pipefail
umask 0002
if [[ $# -ne 4 ]]; then
    echo "Usage: sbatch capture_active_defender.slurm MONITOR_CAPTURE CAGE_ROOT CHECKPOINT_TEMPLATE OUTPUT" >&2
    exit 2
fi
: "${REPLAY_CHECKOUT:?Set the data-drive source directory}"
: "${REPLAY_TMP_ROOT:?Set a personal data-drive temporary directory}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export TMPDIR="${REPLAY_TMP_ROOT}/active-capture-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$REPLAY_CHECKOUT"
exec .venv-active/bin/python -u scripts/capture_active_defender.py \
    --monitor-capture-dir "$1" --cage-root "$2" --checkpoint-template "$3" --output-dir "$4"
