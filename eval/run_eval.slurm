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

ARGS=("$@")
BACKEND="openai"
VLLM_MODEL="qwen3-8b"
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  arg="${ARGS[$i]}"
  case "$arg" in
    --backend)
      BACKEND="${ARGS[$((i + 1))]}"
      i=$((i + 1))
      ;;
    --backend=*)
      BACKEND="${arg#*=}"
      ;;
    --vllm-model)
      VLLM_MODEL="${ARGS[$((i + 1))]}"
      i=$((i + 1))
      ;;
    --vllm-model=*)
      VLLM_MODEL="${arg#*=}"
      ;;
  esac
done

# Optional: activate your virtual environment before running.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

wait_for_vllm() {
  local port="$1"
  VLLM_PORT="$port" python - <<'PY'
import os
import time
import urllib.request

port = os.environ["VLLM_PORT"]
url = f"http://localhost:{port}/v1/models"
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if resp.status == 200:
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit(f"Timed out waiting for vLLM to start on port {port}")
PY
}

start_vllm() {
  local model="$1"
  local log_path="logs/vllm-${SLURM_JOB_ID:-local}-${model}.log"

  case "$model" in
    qwen3-8b)
      VLLM_PORT=8001
      vllm serve Qwen/Qwen3-8B --port 8001 --data-parallel-size 8 >"$log_path" 2>&1 &
      ;;
    qwen3-coder-480b-a35b-fp8)
      VLLM_PORT=8000
      vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
        --port 8000 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.9 \
        --enable-expert-parallel \
        --data-parallel-size 8 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder >"$log_path" 2>&1 &
      ;;
    qwen3-coder-30b-a3b)
      VLLM_PORT=8002
      vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --port 8002 \
        --data-parallel-size 8 \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.90 \
        --kv-cache-dtype fp8_e4m3 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 16k >"$log_path" 2>&1 &
      ;;
    qwen3-coder-next)
      VLLM_PORT=8003
      vllm serve Qwen/Qwen3-Coder-Next \
      --port 8003 \
      --tensor-parallel-size 2 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --gpu-memory-utilization 0.9 \
      --max-model-len 65536 \
      --max-num-seqs 8 \
      --max-num-batched-tokens 16k >"$log_path" 2>&1 &
      ;;
    *)
      echo "Unknown vLLM model preset: $model" >&2
      exit 1
      ;;
  esac

  VLLM_PID=$!
  trap 'if [ -n "${VLLM_PID:-}" ]; then kill "$VLLM_PID" 2>/dev/null || true; fi' EXIT
  wait_for_vllm "$VLLM_PORT"
}

if [ "$BACKEND" = "vllm" ]; then
  start_vllm "$VLLM_MODEL"
fi

uv run python "$EVAL_SCRIPT" "${ARGS[@]}"
