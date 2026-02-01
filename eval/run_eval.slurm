#!/usr/bin/env bash
#SBATCH --job-name=rlm-eval
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=256
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

mkdir -p logs

EVAL_SCRIPT="eval/codeqa_example.py"
if [ "$#" -gt 0 ]; then
  EVAL_SCRIPT="$1"
  shift
fi

# Optional: activate your virtual environment before running.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

uv run python "$EVAL_SCRIPT" "$@"
