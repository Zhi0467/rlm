"""
Example: CodeQA evaluation on LongBench-v2, mirroring eval/oolong_example.py.
Runs multiple RLM configurations and appends summary accuracy to a CSV after each run.

Usage:
  sbatch eval/run_eval.slurm eval/codeqa_example.py \
    --backend vllm \
    --vllm-model qwen3-8b \
    --all

  sbatch eval/run_eval.slurm eval/codeqa_example.py \
    --backend vllm \
    --vllm-model qwen3-coder-480b-a35b-fp8 \
    --all

  sbatch eval/run_eval.slurm eval/codeqa_example.py \
    --backend vllm \
    --vllm-model qwen3-coder-30b-a3b \
    --all

vLLM serve commands for each preset are executed inside eval/run_eval.slurm.

to debug, run without --all.
"""

import argparse
import re
import sys
from typing import Any

from dotenv import load_dotenv

from rlm.logger import RLMLogger

from utils import (
    VLLM_MODEL_CONFIGS,
    append_summary,
    build_backend_selection,
    build_results_csv_path,
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

RESULTS_CSV_PREFIX = "codeqa_results"
DATASET_NAME = "zai-org/LongBench-v2"
DATASET_SPLIT = "train"
CODEQA_PREFIX = "code"
LETTER_BY_NUMBER = {"1": "A", "2": "B", "3": "C", "4": "D"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the CodeQA eval run."""
    parser = argparse.ArgumentParser(description="Run CodeQA example with multiple RLM configs.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to run")
    parser.add_argument("--start-index", type=int, default=1, help="Start index in the dataset")
    parser.add_argument(
        "--filter-field",
        choices=["domain", "sub_domain"],
        default="domain",
        help="Field used to filter CodeQA rows",
    )
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


def load_codeqa_rows(start_index: int, count: int, filter_field: str) -> list[dict]:
    """Load a slice of CodeQA rows from LongBench-v2."""
    if count <= 0:
        raise ValueError("num_samples must be > 0")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if filter_field == "domain":
        filtered = dataset.filter(
            lambda ex: str(ex["domain"]).lower().startswith(CODEQA_PREFIX)
        )
    else:
        filtered = dataset.filter(
            lambda ex: str(ex["sub_domain"]).lower().startswith(CODEQA_PREFIX)
        )

    if len(filtered) == 0:
        raise ValueError("No CodeQA rows matched the dataset filter")
    if start_index >= len(filtered):
        return []

    end_index = min(start_index + count, len(filtered))
    return list(filtered.select(range(start_index, end_index)))


def load_all_codeqa_rows(filter_field: str) -> list[dict]:
    """Load all CodeQA rows from LongBench-v2."""
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if filter_field == "domain":
        filtered = dataset.filter(
            lambda ex: str(ex["domain"]).lower().startswith(CODEQA_PREFIX)
        )
    else:
        filtered = dataset.filter(
            lambda ex: str(ex["sub_domain"]).lower().startswith(CODEQA_PREFIX)
        )
    expected_count = 50
    if len(filtered) != expected_count:
        raise ValueError(
            f"Expected {expected_count} CodeQA rows after filter, found {len(filtered)}"
        )
    return list(filtered)


def build_choices_text(row: dict[str, Any]) -> str:
    """Format multiple-choice options from a dataset row."""
    return "\n".join(
        [
            f"A. {row['choice_A']}",
            f"B. {row['choice_B']}",
            f"C. {row['choice_C']}",
            f"D. {row['choice_D']}",
        ]
    )


def build_root_prompt(question: str, choices_text: str) -> str:
    """Build the root prompt containing the question and choices."""
    return (
        f"{question}\n\nChoices:\n{choices_text}\n\n"
        "Answer with a single letter: A, B, C, or D."
    )


def normalize_expected_answer(expected_answer: str) -> str:
    """Normalize the expected answer to A/B/C/D."""
    expected = expected_answer.strip().upper()
    if expected in LETTER_BY_NUMBER:
        expected = LETTER_BY_NUMBER[expected]
    if expected not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unexpected answer format: {expected_answer}")
    return expected


def extract_choice(response: str) -> str | None:
    """Extract a final choice token (A-D or 1-4) from the response."""
    if not response:
        return None
    matches = re.findall(r"\b([A-Da-d]|[1-4])\b", response)
    if not matches:
        return None
    token = matches[-1].upper()
    return LETTER_BY_NUMBER.get(token, token)


def evaluate_response(expected_answer: str, response: str) -> bool:
    """Check whether the model response matches the expected answer."""
    expected = normalize_expected_answer(expected_answer)
    actual = extract_choice(response)
    if actual is None:
        return False
    return actual == expected


def build_context(row: dict[str, Any]) -> str:
    """Extract the context field from a dataset row."""
    return row["context"]


def build_root_prompt_from_row(row: dict[str, Any]) -> str:
    """Build the root prompt for a dataset row."""
    choices_text = build_choices_text(row)
    question = row["question"]
    return build_root_prompt(question, choices_text)


def build_expected_answer(row: dict[str, Any]) -> str:
    """Extract the expected answer from a dataset row."""
    return row["answer"]


def main() -> None:
    """Entry point for the CodeQA eval runner."""
    args = parse_args()

    if args.all:
        rows = load_all_codeqa_rows(args.filter_field)
    else:
        rows = load_codeqa_rows(args.start_index, args.num_samples, args.filter_field)
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
    results_csv_path = build_results_csv_path(
        RESULTS_CSV_PREFIX, backend_kwargs.get("model_name", "unknown")
    )

    for run_config in run_configs:
        metrics = run_config_over_rows(logger, run_config, rows)
        append_summary(results_csv_path, run_config, metrics)
        print_summary(run_config, metrics)


if __name__ == "__main__":
    main()
