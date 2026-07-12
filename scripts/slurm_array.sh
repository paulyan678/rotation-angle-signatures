#!/usr/bin/env bash
#SBATCH --job-name=rotation-patterns
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch --array=0-N scripts/slurm_array.sh CONFIG MANIFEST" >&2
  exit 2
fi

CONFIG=$1
MANIFEST=$2
INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

rotation-patterns run-task --config "$CONFIG" --manifest "$MANIFEST" --index "$INDEX"

