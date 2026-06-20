from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .common import (
    BenchmarkResult,
    BenchmarkSpec,
    coerce_steps,
    extract_json_payload,
    load_records,
    load_spec,
    score_toolchain,
)

DEFAULT_SPEC = BenchmarkSpec(
    name="btl1",
    kind="messages",
    data_path_env="BTL_BTL1_PATH",
    default_path="data/final/eval.jsonl",
)


def _prompt_from_messages(messages: list[dict[str, Any]], tokenizer: Any | None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chunks = []
    for message in messages:
        role = str(message.get("role", "")).upper()
        content = str(message.get("content", ""))
        chunks.append(f"{role}: {content}")
    chunks.append("ASSISTANT:")
    return "\n".join(chunks)


def evaluate_btl1(
    predict_fn: Callable[[str], str],
    dataset_path: Path,
    tokenizer: Any | None = None,
    limit: int | None = None,
) -> BenchmarkResult:
    spec = load_spec("btl1", DEFAULT_SPEC)
    rows = 0
    valid_json = 0
    exact_sequence = 0
    tool_accuracy = 0.0
    tool_f1 = 0.0
    param_accuracy = 0.0
    dependency_accuracy = 0.0
    step_exact_rate = 0.0
    refusal_hits = 0
    orchestration_score = 0.0
    error_totals = {
        "parse_fail": 0,
        "overcall": 0,
        "undercall": 0,
        "refusal_miss": 0,
        "tool_mismatch": 0,
        "param_mismatch": 0,
        "dependency_mismatch": 0,
        "sequence_mismatch": 0,
    }

    for row in load_records(dataset_path, limit=limit):
        rows += 1
        messages = row.get("messages", [])
        assistant = next((msg for msg in messages if msg.get("role") == "assistant"), None)
        if not assistant:
            continue

        expected = coerce_steps(extract_json_payload(str(assistant.get("content", ""))))
        prompt = _prompt_from_messages(messages[:-1], tokenizer)
        predicted_raw = predict_fn(prompt)
        predicted = coerce_steps(extract_json_payload(str(predicted_raw)))

        valid_json += int(bool(predicted or predicted_raw.strip() == "[]"))
        refusal_hits += int(len(expected) == 1 and expected[0].get("tool") == "reasoning" and len(predicted) == 1 and predicted[0].get("tool") == "reasoning")

        scores = score_toolchain(expected, predicted, allow_refusal=spec.allow_refusal)
        exact_sequence += scores["exact_sequence"]
        tool_accuracy += scores["tool_accuracy"]
        tool_f1 += scores["tool_f1"]
        param_accuracy += scores["param_accuracy"]
        dependency_accuracy += scores["dependency_accuracy"]
        step_exact_rate += scores["step_exact_rate"]
        orchestration_score += scores["tool_orchestration_score"]
        for key, value in scores["errors"].items():
            error_totals[key] += int(value)

    denom = max(rows, 1)
    metrics = {
        "json_valid_rate": valid_json / denom,
        "exact_sequence_rate": exact_sequence / denom,
        "tool_accuracy": tool_accuracy / denom,
        "tool_f1": tool_f1 / denom,
        "param_accuracy": param_accuracy / denom,
        "dependency_accuracy": dependency_accuracy / denom,
        "step_exact_rate": step_exact_rate / denom,
        "refusal_accuracy": refusal_hits / denom,
        "tool_orchestration_score": orchestration_score / denom,
        "errors": {key: value / denom for key, value in error_totals.items()},
    }
    return BenchmarkResult(name=spec.name, rows=rows, metrics=metrics, note=str(dataset_path))
