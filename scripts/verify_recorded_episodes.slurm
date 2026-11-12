#!/usr/bin/env bash
#SBATCH --job-name=seed-replay-check
#SBATCH --partition=community
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -euo pipefail
umask 0002
if [[ $# -ne 6 ]]; then
    echo "Usage: sbatch verify_recorded_episodes.slurm REQUESTS TRAJECTORY_ROOT CAGE_ROOT REVISION SCENARIO OUTPUT" >&2
    exit 2
fi
: "${REPLAY_CHECKOUT:?Set the data-drive source directory}"
: "${REPLAY_TMP_ROOT:?Set a personal data-drive temporary directory}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export TMPDIR="${REPLAY_TMP_ROOT}/capture-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$REPLAY_CHECKOUT"
exec .venv-observations/bin/python -u scripts/verify_recorded_episodes.py \
    --requests "$1" --trajectory-root "$2" --cage-root "$3" \
    --cage-revision "$4" --scenario "$5" --output-dir "$6"
