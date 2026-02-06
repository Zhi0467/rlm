"""Shared helpers for eval scripts."""

import argparse
import csv
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rlm import RLM
from rlm.logger import RLMLogger

ContextBuilder = Callable[[dict[str, Any]], str]
ExpectedAnswerBuilder = Callable[[dict[str, Any]], str]
RootPromptBuilder = Callable[[dict[str, Any]], str]
EvaluateResponse = Callable[[str, str], bool]

CONTEXT_BUILDER: ContextBuilder | None = None
EXPECTED_ANSWER_BUILDER: ExpectedAnswerBuilder | None = None
ROOT_PROMPT_BUILDER: RootPromptBuilder | None = None
EVALUATE_RESPONSE: EvaluateResponse | None = None
CURRENT_ROOT_PROMPT: str | None = None
MAX_ITERATIONS: int = 10

VLLM_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3-coder-480b-a35b-fp8": {
        "model_name": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "base_url": "http://localhost:8000/v1",
    },
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B",
        "base_url": "http://localhost:8001/v1",
    },
    "qwen3-coder-30b-a3b": {
        "model_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "base_url": "http://localhost:8002/v1",
    },
    "qwen3-coder-next": {
        "model_name": "Qwen/Qwen3-Coder-Next",
        "base_url": "http://localhost:8003/v1",
    },
}


@dataclass(frozen=True)
class RLMRunConfig:
    name: str
    recursive_max_depth: int
    backend_kwargs: dict[str, Any]
    backend: str = "openai"
    max_iterations: int = MAX_ITERATIONS
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


def sanitize_model_name(model_name: str) -> str:
    """Normalize a model name to a filesystem-safe suffix."""
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", model_name.strip()).strip("-").lower()
    return sanitized or "unknown-model"


def configure_eval(
    *,
    context_builder: ContextBuilder,
    expected_answer_builder: ExpectedAnswerBuilder,
    root_prompt_builder: RootPromptBuilder,
    evaluate_response: EvaluateResponse,
) -> None:
    """Configure row-to-prompt mapping and evaluation for shared runners."""
    global CONTEXT_BUILDER
    global EXPECTED_ANSWER_BUILDER
    global ROOT_PROMPT_BUILDER
    global EVALUATE_RESPONSE
    global CURRENT_ROOT_PROMPT

    CONTEXT_BUILDER = context_builder
    EXPECTED_ANSWER_BUILDER = expected_answer_builder
    ROOT_PROMPT_BUILDER = root_prompt_builder
    EVALUATE_RESPONSE = evaluate_response
    CURRENT_ROOT_PROMPT = None


def build_results_csv_path(results_prefix: str, model_name: str) -> Path:
    """Build the results CSV path with a model name suffix."""
    suffix = sanitize_model_name(model_name)
    return Path("logs") / f"{results_prefix}_{suffix}.csv"


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
        return "vllm", backend_kwargs, other_backend_kwargs, MAX_ITERATIONS

    api_key = get_api_key("openai")
    backend_kwargs = {"model_name": "gpt-5-mini", "api_key": api_key}
    other_backend_kwargs = [
        {"model_name": "gpt-5-nano", "api_key": api_key},
        {"model_name": "gpt-5-nano", "api_key": api_key},
    ]
    return "openai", backend_kwargs, other_backend_kwargs, MAX_ITERATIONS


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
    env_kwargs = (
        run_config.environment_kwargs.copy() if run_config.environment_kwargs else {}
    )
    if "code_execution_timeout" not in env_kwargs:
        env_kwargs["code_execution_timeout"] = 300
    return RLM(
        backend=run_config.backend,
        backend_kwargs=run_config.backend_kwargs,
        environment=run_config.environment,
        environment_kwargs=env_kwargs,
        max_iterations=run_config.max_iterations,
        recursive_max_depth=run_config.recursive_max_depth,
        other_backends=run_config.other_backends,
        other_backend_kwargs=run_config.other_backend_kwargs,
        logger=logger,
        verbose=True,
    )


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
    expected_answer: str,
) -> tuple[bool, int, int] | None:
    """Run a single row and return correctness and token counts (or None on timeout)."""
    if EVALUATE_RESPONSE is None:
        raise ValueError("Evaluation is not configured. Call configure_eval() first.")
    if CURRENT_ROOT_PROMPT is None:
        raise ValueError("Root prompt is not configured. Call run_config_over_rows().")

    try:
        rlm = build_rlm(logger, run_config)
        result = rlm.completion(prompt=context, root_prompt=CURRENT_ROOT_PROMPT)
    except (TimeoutError, socket.timeout) as exc:
        print(f"Skipping row due to timeout: {exc}")
        return None

    response_text = result.response
    if is_timeout_response(response_text):
        print("Skipping row due to timeout response")
        return None

    input_tokens, output_tokens = get_tokens_from_usage_summary(result.usage_summary)

    is_correct = EVALUATE_RESPONSE(expected_answer, response_text)
    return is_correct, input_tokens, output_tokens


def run_config_over_rows(
    logger: RLMLogger, run_config: RLMRunConfig, rows: list[dict]
) -> RunMetrics:
    """Evaluate all rows for one configuration and accumulate metrics."""
    global CURRENT_ROOT_PROMPT
    if CONTEXT_BUILDER is None or EXPECTED_ANSWER_BUILDER is None:
        raise ValueError("Row builders are not configured. Call configure_eval() first.")
    if ROOT_PROMPT_BUILDER is None:
        raise ValueError("Root prompt builder is not configured. Call configure_eval() first.")

    metrics = RunMetrics()
    for row in rows:
        context = CONTEXT_BUILDER(row)
        expected_answer = EXPECTED_ANSWER_BUILDER(row)
        root_prompt = ROOT_PROMPT_BUILDER(row)
        CURRENT_ROOT_PROMPT = root_prompt

        result = run_single_example(logger, run_config, context, expected_answer)
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
