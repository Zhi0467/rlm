"""
Example: AIME25 evaluation.
Runs multiple RLM configurations and appends summary accuracy to a CSV after each run.

Usage:
  sbatch eval/run_eval.slurm eval/aime_example.py \
    --backend vllm \
    --vllm-model qwen3-8b \
    --all

  sbatch eval/run_eval.slurm eval/aime_example.py \
    --backend vllm \
    --vllm-model qwen3-coder-480b-a35b-fp8 \
    --all

  sbatch eval/run_eval.slurm eval/aime_example.py \
    --backend vllm \
    --vllm-model qwen3-coder-30b-a3b \
    --all

vLLM serve commands for each preset are executed inside eval/run_eval.slurm.

to debug, run without --all.
"""

import argparse
import sys
from typing import Any, List, Optional, Tuple

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

RESULTS_CSV_PREFIX = "aime_results"
DATASET_NAME = "math-ai/aime25"
DATASET_SPLIT = "test"
_MATH_VERIFY_CONFIGS: Optional[Tuple[List[Any], List[Any]]] = None


def _build_math_verify_configs(LatexExtractionConfig, ExprExtractionConfig) -> Tuple[
    List[Any], List[Any]
]:
    try:
        pred_config = [
            LatexExtractionConfig(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                equations=False,
                boxed="all",
                boxed_match_priority=0,
            ),
            ExprExtractionConfig(),
        ]
    except TypeError:
        try:
            from latex2sympy2_extended.math_normalization import NormalizationConfig
        except Exception:
            pred_config = [LatexExtractionConfig(), ExprExtractionConfig()]
        else:
            pred_config = [
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    normalization_config=NormalizationConfig(
                        basic_latex=True,
                        units=True,
                        malformed_operators=False,
                        nits=False,
                        boxed="all",
                        equations=False,
                    ),
                ),
                ExprExtractionConfig(),
            ]

    gold_config = [LatexExtractionConfig(), ExprExtractionConfig()]
    return pred_config, gold_config


def _math_verify(pred: str, gold: str) -> Optional[bool]:
    try:
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
    except Exception:
        return None

    global _MATH_VERIFY_CONFIGS
    if _MATH_VERIFY_CONFIGS is None:
        _MATH_VERIFY_CONFIGS = _build_math_verify_configs(
            LatexExtractionConfig, ExprExtractionConfig
        )

    pred_config, gold_config = _MATH_VERIFY_CONFIGS
    try:
        parsed_pred = parse(pred, extraction_config=pred_config)
        parsed_gold = parse(gold, extraction_config=gold_config)
        return bool(verify(parsed_gold, parsed_pred))
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the AIME eval run."""
    parser = argparse.ArgumentParser(description="Run AIME example with multiple RLM configs.")
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


def load_aime_rows(start_index: int, count: int) -> list[dict]:
    """Load a slice of rows from AIME25."""
    if count <= 0:
        raise ValueError("num_samples must be > 0")
    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    if start_index >= len(dataset):
        return []

    end_index = min(start_index + count, len(dataset))
    return list(dataset.select(range(start_index, end_index)))


def load_all_aime_rows() -> list[dict]:
    """Load all rows from AIME25."""
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return list(dataset)


def build_root_prompt() -> str:
    """Build the root prompt for AIME problems."""
    return (
        "Solve the problem in the provided context. "
        "You must wrap the final integer answer in /boxed{...} in your final response."
    )


def evaluate_response(expected_answer: str, response: str) -> bool:
    """Check whether the model response matches the expected answer."""
    verified = _math_verify(response, expected_answer)
    if verified is None:
        raise ValueError(
            "math_verify is not available. Install it with `uv pip install math-verify`."
        )
    return verified


def build_context(row: dict[str, Any]) -> str:
    """Extract the context field from a dataset row."""
    return row["problem"]


def build_root_prompt_from_row(_: dict[str, Any]) -> str:
    """Build the root prompt for a dataset row."""
    return build_root_prompt()


def build_expected_answer(row: dict[str, Any]) -> str:
    """Extract the expected answer from a dataset row."""
    return row["answer"]


def main() -> None:
    """Entry point for the AIME eval runner."""
    args = parse_args()

    if args.all:
        rows = load_all_aime_rows()
    else:
        rows = load_aime_rows(args.start_index, args.num_samples)
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
