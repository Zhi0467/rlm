"""
Parsing utilities for RLM trjaectories.
"""

import re
from typing import TYPE_CHECKING

from rlm.core.types import REPLResult, RLMIteration

if TYPE_CHECKING:
    from rlm.environments.base_env import BaseEnv


def find_code_blocks(text: str) -> list[str]:
    """
    Find REPL code blocks in text wrapped in triple backticks and return List of content(s).
    Returns None if no code blocks are found.
    """
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []

    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results


def strip_fenced_code(text: str) -> str:
    """Remove triple-backtick fenced code blocks from text."""
    return re.sub(r"```.*?```", "\n", text, flags=re.DOTALL)


def resolve_final_var(variable_name: str, environment: "BaseEnv | None") -> str | None:
    if environment is None:
        return None
    result = environment.execute_code(f"print(FINAL_VAR({variable_name!r}))")
    final_answer = result.stdout.strip()
    if final_answer == "":
        final_answer = result.stderr.strip() or ""
    return final_answer


def resolve_fstring(content: str, environment: "BaseEnv | None") -> str | None:
    """
    Resolve a simple f-string with {identifier} placeholders using environment variables.
    Returns the resolved string, or None if it cannot be resolved.
    """
    if environment is None:
        return None

    match = re.fullmatch(r'''f(["'])(.*)\1''', content, re.DOTALL)
    if not match:
        return None

    raw = match.group(2)
    placeholders = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", raw)
    if not placeholders:
        return raw

    resolved = raw
    for name in dict.fromkeys(placeholders):
        value = resolve_final_var(name, environment)
        if value is None:
            return None
        resolved = resolved.replace(f"{{{name}}}", value)
    return resolved


def extract_standalone_boxed_answer(cleaned_text: str) -> str | None:
    """Return a boxed answer only when it is the sole non-empty line and non-empty."""
    lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None

    line = lines[0]
    for prefix in ("\\boxed{", "/boxed{"):
        if line.startswith(prefix) and line.endswith("}"):
            content = line[len(prefix) : -1]
            if content.strip():
                return line
            return None
    return None


def find_final_answer(text: str, environment: "BaseEnv | None" = None) -> str | None:
    """
    Find FINAL(...) or FINAL_VAR(...) statement in response and return the final answer string.

    If FINAL_VAR is found and an environment is provided, executes code to retrieve the variable value.
    Returns None if neither pattern is found.

    Args:
        text: The response text to parse
        environment: Optional environment to execute code for FINAL_VAR retrieval

    Returns:
        The final answer string, or None if no final answer pattern is found
    """
    cleaned_text = strip_fenced_code(text)

    # Check for FINAL_VAR pattern first - must be at start of line (code fences allowed)
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        return resolve_final_var(variable_name, environment)

    # Check for FINAL pattern - must be at start of line (outside code fences)
    # Use greedy matching to capture content with nested parentheses
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, cleaned_text, re.MULTILINE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        resolved = resolve_fstring(content, environment)
        return resolved if resolved is not None else content

    # Check for a strict standalone \boxed{...} (or /boxed{...}) fallback
    boxed_answer = extract_standalone_boxed_answer(cleaned_text)
    if boxed_answer is not None:
        return boxed_answer

    return None


def format_iteration(
    iteration: RLMIteration, max_character_length: int = 20000
) -> list[dict[str, str]]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.

    Args:
        iteration: The iteration to format
        max_character_length: The maximum character length of the result

    Returns:
        A list of messages to add to the next prompt
    """
    messages = [{"role": "assistant", "content": iteration.response}]

    for code_block in iteration.code_blocks:
        code = code_block.code
        result = code_block.result
        result = format_execution_result(result)
        if len(result) > max_character_length:
            result = (
                result[:max_character_length]
                + f"... + [{len(result) - max_character_length} chars...]"
            )

        execution_message = {
            "role": "user",
            "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}",
        }
        messages.append(execution_message)
    return messages


################
# TODO: Remove and refactor these soon
################


def format_execution_result(result: REPLResult) -> str:
    """
    Format the execution result as a string for display.

    Args:
        result: The REPLResult object to format.
    """
    result_parts = []

    if result.stdout:
        result_parts.append(f"\n{result.stdout}")

    if result.stderr:
        result_parts.append(f"\n{result.stderr}")

    # Show some key variables (excluding internal ones)
    important_vars = {}
    for key, value in result.locals.items():
        if not key.startswith("_") and key not in [
            "__builtins__",
            "__name__",
            "__doc__",
        ]:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                important_vars[key] = ""

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def check_for_final_answer(response: str, repl_env, logger) -> str | None:
    """Check if response contains a final answer."""
    # Use the new find_final_answer function which handles both FINAL and FINAL_VAR
    return find_final_answer(response, environment=repl_env)


def convert_context_for_repl(context):
    """
    Convert REPL context to either some
    """
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None

    return context_data, context_str
