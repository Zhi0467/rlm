"""
Example: Oolong Benchmark sample run from the RLM paper: https://arxiv.org/abs/2512.24601v1
Runs multiple RLM configurations and appends summary accuracy to a CSV after each run.

Usage:
  uv run python eval/oolong_example.py --num-samples 10 --start-index 0
  uv run python eval/oolong_example.py --all
"""

import argparse
import sys
from itertools import islice
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rlm.logger import RLMLogger

from utils import (
    VLLM_MODEL_CONFIGS,
    append_summary,
    build_backend_selection,
    build_run_configs,
    configure_eval,
    print_summary,
    run_config_over_rows,
)

load_dotenv()

try:
    from datasets import load_dataset
except ImportError:
    print(
        "Please install the 'datasets' library to run this example. Run `uv pip install datasets`"
    )
    sys.exit(1)

RESULTS_CSV = Path("logs") / "oolong_results.csv"
DATASET_NAME = "oolongbench/oolong-real"
DATASET_CONFIG = "toy_dnd"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Oolong eval run."""
    parser = argparse.ArgumentParser(description="Run Oolong example with multiple RLM configs.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to run")
    parser.add_argument("--start-index", type=int, default=1, help="Start index in the dataset")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all rows (ignores --num-samples and --start-index)",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "vllm"],
        default="openai",
        help="Backend to use for the eval (openai or vllm)",
    )
    parser.add_argument(
        "--vllm-model",
        choices=sorted(VLLM_MODEL_CONFIGS.keys()),
        default="qwen3-8b",
        help="vLLM model preset (used when --backend vllm)",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help="Override vLLM base URL (defaults to preset value)",
    )
    parser.add_argument(
        "--vllm-model-name",
        default=None,
        help="Override vLLM model name (defaults to preset value)",
    )
    return parser.parse_args()


def load_oolong_rows(start_index: int, count: int) -> list[dict]:
    """Load a slice of rows from the Oolong benchmark."""
    if count <= 0:
        raise ValueError("num_samples must be > 0")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    streaming_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test", streaming=True)
    return list(islice(streaming_ds, start_index, start_index + count))


def load_all_oolong_rows() -> list[dict]:
    """Load all rows from the Oolong benchmark."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    return list(dataset)


def evaluate_response(expected_answer: str, response: str) -> bool:
    """Check whether the model response matches the expected answer."""
    expected = expected_answer.lower().strip()
    actual = response.lower().strip()
    return expected in actual or actual in expected


def build_context(row: dict[str, Any]) -> str:
    """Extract the context field from a dataset row."""
    return row["context_window_text"]


def build_root_prompt_from_row(row: dict[str, Any]) -> str:
    """Build the root prompt for a dataset row."""
    return row["question"]


def build_expected_answer(row: dict[str, Any]) -> str:
    """Extract the expected answer from a dataset row."""
    return row["answer"]


def main() -> None:
    """Entry point for the Oolong eval runner."""
    args = parse_args()

    if args.all:
        rows = load_all_oolong_rows()
    else:
        rows = load_oolong_rows(args.start_index, args.num_samples)
    if not rows:
        raise ValueError("No rows loaded from dataset")

    configure_eval(
        context_builder=build_context,
        expected_answer_builder=build_expected_answer,
        root_prompt_builder=build_root_prompt_from_row,
        evaluate_response=evaluate_response,
    )

    logger = RLMLogger(log_dir="./logs")
    backend, backend_kwargs, other_backend_kwargs, max_iterations = build_backend_selection(
        args
    )
    run_configs = build_run_configs(
        backend, backend_kwargs, other_backend_kwargs, max_iterations
    )

    for run_config in run_configs:
        metrics = run_config_over_rows(logger, run_config, rows)
        append_summary(RESULTS_CSV, run_config, metrics)
        print_summary(run_config, metrics)


if __name__ == "__main__":
    main()
