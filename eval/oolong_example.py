"""
Example: Oolong Benchmark sample run from the RLM paper: https://arxiv.org/abs/2512.24601v1
Runs multiple RLM configurations and appends summary accuracy to a CSV after each run.

Usage:
  uv run python eval/oolong_example.py --num-samples 10 --start-index 0
  uv run python eval/oolong_example.py --all
"""

import argparse
import csv
import os
import socket
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

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


@dataclass(frozen=True)
class RLMRunConfig:
    name: str
    recursive_max_depth: int
    backend_kwargs: dict[str, Any]
    other_backends: list[str] | None = None
    other_backend_kwargs: list[dict[str, Any]] | None = None


@dataclass
class RunMetrics:
    correct: int = 0
    total: int = 0
    input_tokens_sum: int = 0
    output_tokens_sum: int = 0

    def record(self, is_correct: bool, input_tokens: int, output_tokens: int) -> None:
        self.total += 1
        if is_correct:
            self.correct += 1
        self.input_tokens_sum += input_tokens
        self.output_tokens_sum += output_tokens


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


def get_api_key() -> str:
    """Fetch the OpenAI API key from the environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def build_run_configs(api_key: str) -> list[RLMRunConfig]:
    """Build RLM run configurations for the eval."""
    return [
        RLMRunConfig(
            name="depth-1",
            recursive_max_depth=1,
            backend_kwargs={"model_name": "gpt-5-mini", "api_key": api_key},
            other_backends=["openai"],
            other_backend_kwargs=[{"model_name": "gpt-5-nano", "api_key": api_key}],
        ),
        RLMRunConfig(
            name="depth-2",
            recursive_max_depth=2,
            backend_kwargs={"model_name": "gpt-5-mini", "api_key": api_key},
            other_backends=["openai", "openai"],
            other_backend_kwargs=[
                {"model_name": "gpt-5-nano", "api_key": api_key},
                {"model_name": "gpt-5-nano", "api_key": api_key},
            ],
        ),
    ]


def build_rlm(logger: RLMLogger, run_config: RLMRunConfig) -> RLM:
    """Construct an RLM instance for a run configuration."""
    return RLM(
        backend="openai",
        backend_kwargs=run_config.backend_kwargs,
        environment="local",
        max_iterations=20,
        recursive_max_depth=run_config.recursive_max_depth,
        other_backends=run_config.other_backends,
        other_backend_kwargs=run_config.other_backend_kwargs,
        logger=logger,
        verbose=True,
    )


def evaluate_response(expected_answer: str, response: str) -> bool:
    """Check whether the model response matches the expected answer."""
    expected = expected_answer.lower().strip()
    actual = response.lower().strip()
    return expected in actual or actual in expected


def is_timeout_response(response: str) -> bool:
    """Detect timeout-style errors in a response string."""
    if not response:
        return False
    normalized = response.lower()
    if "error" not in normalized:
        return False
    return "timeout" in normalized or "timed out" in normalized


def ensure_results_csv(path: Path) -> None:
    """Create or reset the results CSV with the expected header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_name",
        "recursive_max_depth",
        "root_model",
        "other_models",
        "correct",
        "total",
        "accuracy",
        "avg_input_tokens",
        "avg_output_tokens",
    ]

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as file:
            reader = csv.reader(file)
            existing_header = next(reader, [])
        if existing_header == header:
            return

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


def append_summary(path: Path, run_config: RLMRunConfig, metrics: RunMetrics) -> None:
    """Append a summary row for one run configuration."""
    ensure_results_csv(path)
    if metrics.total <= 0:
        raise ValueError("No samples were evaluated")

    other_models = []
    if run_config.other_backend_kwargs:
        other_models = [
            kwargs.get("model_name", "unknown") for kwargs in run_config.other_backend_kwargs
        ]

    accuracy = metrics.correct / metrics.total
    avg_input = metrics.input_tokens_sum / metrics.total
    avg_output = metrics.output_tokens_sum / metrics.total

    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                run_config.name,
                run_config.recursive_max_depth,
                run_config.backend_kwargs.get("model_name", "unknown"),
                "|".join(other_models),
                metrics.correct,
                metrics.total,
                f"{accuracy:.4f}",
                int(avg_input),
                int(avg_output),
            ]
        )
        file.flush()


def get_tokens_from_usage_summary(usage_summary) -> tuple[int, int]:
    """Extract total input/output tokens from an aggregated usage summary."""
    total_input = 0
    total_output = 0
    for usage in usage_summary.model_usage_summaries.values():
        total_input += usage.total_input_tokens
        total_output += usage.total_output_tokens
    return total_input, total_output


def run_single_example(
    logger: RLMLogger,
    run_config: RLMRunConfig,
    context: str,
    question: str,
    expected_answer: str,
) -> tuple[bool, int, int] | None:
    """Run a single row and return correctness and token counts (or None on timeout)."""
    try:
        rlm = build_rlm(logger, run_config)
        result = rlm.completion(prompt=context, root_prompt=question)
    except (TimeoutError, socket.timeout) as exc:
        print(f"Skipping row due to timeout: {exc}")
        return None

    response_text = result.response
    if is_timeout_response(response_text):
        print("Skipping row due to timeout response")
        return None

    input_tokens, output_tokens = get_tokens_from_usage_summary(result.usage_summary)

    is_correct = evaluate_response(expected_answer, response_text)
    return is_correct, input_tokens, output_tokens


def run_config_over_rows(
    logger: RLMLogger, run_config: RLMRunConfig, rows: list[dict]
) -> RunMetrics:
    """Evaluate all rows for one configuration and accumulate metrics."""
    metrics = RunMetrics()
    for row in rows:
        context = row["context_window_text"]
        question = row["question"]
        expected_answer = row["answer"]

        result = run_single_example(
            logger, run_config, context, question, expected_answer
        )
        if result is None:
            continue
        is_correct, input_tokens, output_tokens = result
        metrics.record(is_correct, input_tokens, output_tokens)

    return metrics


def print_summary(run_config: RLMRunConfig, metrics: RunMetrics) -> None:
    """Print a brief summary for a run configuration."""
    accuracy = metrics.correct / metrics.total if metrics.total else 0.0
    avg_input = metrics.input_tokens_sum / metrics.total if metrics.total else 0.0
    avg_output = metrics.output_tokens_sum / metrics.total if metrics.total else 0.0

    print("-" * 50)
    print(f"Run: {run_config.name}")
    print(f"Accuracy: {metrics.correct} / {metrics.total} ({accuracy:.2%})")
    print(f"Avg input tokens: {avg_input:.0f}")
    print(f"Avg output tokens: {avg_output:.0f}")


def main() -> None:
    """Entry point for the Oolong eval runner."""
    args = parse_args()
    api_key = get_api_key()

    if args.all:
        rows = load_all_oolong_rows()
    else:
        rows = load_oolong_rows(args.start_index, args.num_samples)
    if not rows:
        raise ValueError("No rows loaded from dataset")

    logger = RLMLogger(log_dir="./logs")
    run_configs = build_run_configs(api_key)

    for run_config in run_configs:
        metrics = run_config_over_rows(logger, run_config, rows)
        append_summary(RESULTS_CSV, run_config, metrics)
        print_summary(run_config, metrics)


if __name__ == "__main__":
    main()
