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
import csv
import os
import re
import socket
import sys
from dataclasses import dataclass
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

RESULTS_CSV_PREFIX = "codeqa_results"
DATASET_NAME = "zai-org/LongBench-v2"
DATASET_SPLIT = "train"
CODEQA_PREFIX = "code"
LETTER_BY_NUMBER = {"1": "A", "2": "B", "3": "C", "4": "D"}
VLLM_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3-coder-480b-a35b-fp8": {
        "model_name": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "base_url": "http://localhost:8000/v1",
        "max_iterations": 20,
    },
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B-Instruct",
        "base_url": "http://localhost:8001/v1",
        "max_iterations": 20,
    },
    "qwen3-coder-30b-a3b": {
        "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "base_url": "http://localhost:8002/v1",
        "max_iterations": 20,
    },
}


def sanitize_model_name(model_name: str) -> str:
    """Normalize a model name to a filesystem-safe suffix."""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", model_name.strip()).strip("-").lower()
    return sanitized or "unknown-model"


def build_results_csv_path(model_name: str) -> Path:
    """Build the results CSV path with a model name suffix."""
    suffix = sanitize_model_name(model_name)
    return Path("logs") / f"{RESULTS_CSV_PREFIX}_{suffix}.csv"


@dataclass(frozen=True)
class RLMRunConfig:
    name: str
    recursive_max_depth: int
    backend: str
    backend_kwargs: dict[str, Any]
    max_iterations: int
    environment: str = "local"
    environment_kwargs: dict[str, Any] | None = None
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


def get_api_key(backend: str) -> str:
    """Fetch the API key for the requested backend."""
    if backend == "vllm":
        return os.environ.get("VLLM_API_KEY", "EMPTY")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return api_key


def build_backend_selection(
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], int]:
    """Resolve backend settings from CLI args."""
    if args.backend == "vllm":
        model_config = VLLM_MODEL_CONFIGS[args.vllm_model]
        base_url = args.vllm_base_url or model_config["base_url"]
        model_name = args.vllm_model_name or model_config["model_name"]
        api_key = get_api_key("vllm")
        backend_kwargs = {
            "base_url": base_url,
            "model_name": model_name,
            "api_key": api_key,
        }
        other_backend_kwargs = [backend_kwargs.copy(), backend_kwargs.copy()]
        return "vllm", backend_kwargs, other_backend_kwargs, model_config["max_iterations"]

    api_key = get_api_key("openai")
    backend_kwargs = {"model_name": "gpt-5-mini", "api_key": api_key}
    other_backend_kwargs = [
        {"model_name": "gpt-5-nano", "api_key": api_key},
        {"model_name": "gpt-5-nano", "api_key": api_key},
    ]
    return "openai", backend_kwargs, other_backend_kwargs, 20


def build_run_configs(
    backend: str,
    backend_kwargs: dict[str, Any],
    other_backend_kwargs: list[dict[str, Any]],
    max_iterations: int,
) -> list[RLMRunConfig]:
    """Build RLM run configurations for the eval."""
    return [
        RLMRunConfig(
            name="depth-1",
            recursive_max_depth=1,
            backend=backend,
            backend_kwargs=backend_kwargs,
            max_iterations=max_iterations,
            other_backends=[backend],
            other_backend_kwargs=other_backend_kwargs[:1],
        ),
        RLMRunConfig(
            name="depth-2",
            recursive_max_depth=2,
            backend=backend,
            backend_kwargs=backend_kwargs,
            max_iterations=max_iterations,
            other_backends=[backend, backend],
            other_backend_kwargs=other_backend_kwargs[:2],
        ),
    ]


def build_rlm(logger: RLMLogger, run_config: RLMRunConfig) -> RLM:
    """Construct an RLM instance for a run configuration."""
    return RLM(
        backend=run_config.backend,
        backend_kwargs=run_config.backend_kwargs,
        environment=run_config.environment,
        environment_kwargs=run_config.environment_kwargs,
        max_iterations=run_config.max_iterations,
        recursive_max_depth=run_config.recursive_max_depth,
        other_backends=run_config.other_backends,
        other_backend_kwargs=run_config.other_backend_kwargs,
        logger=logger,
        verbose=True,
    )


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
    choices_text: str,
    expected_answer: str,
) -> tuple[bool, int, int] | None:
    """Run a single row and return correctness and token counts (or None on timeout)."""
    root_prompt = build_root_prompt(question, choices_text)
    try:
        rlm = build_rlm(logger, run_config)
        result = rlm.completion(prompt=context, root_prompt=root_prompt)
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
        context = row["context"]
        question = row["question"]
        expected_answer = row["answer"]
        choices_text = build_choices_text(row)

        result = run_single_example(
            logger, run_config, context, question, choices_text, expected_answer
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
    """Entry point for the CodeQA eval runner."""
    args = parse_args()

    if args.all:
        rows = load_all_codeqa_rows(args.filter_field)
    else:
        rows = load_codeqa_rows(args.start_index, args.num_samples, args.filter_field)
    if not rows:
        raise ValueError("No rows loaded from dataset")

    logger = RLMLogger(log_dir="./logs")
    backend, backend_kwargs, other_backend_kwargs, max_iterations = build_backend_selection(
        args
    )
    run_configs = build_run_configs(
        backend, backend_kwargs, other_backend_kwargs, max_iterations
    )
    results_csv_path = build_results_csv_path(backend_kwargs.get("model_name", "unknown"))

    for run_config in run_configs:
        metrics = run_config_over_rows(logger, run_config, rows)
        append_summary(results_csv_path, run_config, metrics)
        print_summary(run_config, metrics)


if __name__ == "__main__":
    main()
