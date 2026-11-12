#!/usr/bin/env bash
#SBATCH --job-name=classifier-report
#SBATCH --partition=community
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --open-mode=append
set -euo pipefail
umask 0002
if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch classifier_comparison_report.slurm CONFIG OUTPUT" >&2
    exit 2
fi
: "${REPLAY_CHECKOUT:?Set the data-drive source directory}"
cd "$REPLAY_CHECKOUT"
exec .venv-observations/bin/python scripts/summarize_classifier_comparison.py \
    --config "$1" --output-dir "$2"
