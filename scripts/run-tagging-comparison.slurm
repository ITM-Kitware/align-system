#!/bin/bash
#SBATCH --job-name=align-integ-tests
#SBATCH --output=output/slurm-%j.out
#SBATCH --error=output/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=24G
#SBATCH --account=itm

cd /data/users/yonatan.gefen/align-system
mkdir -p tagging-output
bash ./scripts/run-tagging-comparison.sh "$@"
