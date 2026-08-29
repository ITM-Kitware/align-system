#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00
#SBATCH --mem=24G
#SBATCH --error=feb2026_sweep_sbatch.%J.err
#SBATCH --account=itm

# Set cache directories for huggingface
export HF_HOME="/data/shared/models/huggingface"
export TRANSFORMERS_CACHE="/data/shared/models/huggingface"
export HUGGINGFACE_HUB_CACHE="/data/shared/models/huggingface"
export HF_DATASETS_CACHE="/data/shared/datasets/huggingface"

uv run python -m align_system.cli.run_align_system +experiment=statefulness_poc/baseline_open_world_dialog