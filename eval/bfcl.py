from __future__ import annotations

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
    select_first,
)

DEFAULT_SPEC = BenchmarkSpec(
    name="bfcl",
    kind="steps",
    data_path_env="BTL_BFCL_PATH",
    default_path="eval/data/bfcl.jsonl",
    prompt_selectors=("prompt", "question", "query", "input", "instruction", "user_query", "messages[?role=user].content"),
    expected_selectors=("expected", "answer", "label", "ground_truth", "tool_calls", "function_call", "reference", "messages[-1].content"),
)


def evaluate_bfcl(
    predict_fn: Callable[[str], str],
    dataset_path: Path,
    limit: int | None = None,
) -> BenchmarkResult:
    spec = load_spec("bfcl", DEFAULT_SPEC)
    rows = 0
    json_valid = 0
    tool_exact = 0
    param_exact = 0
    abstain_exact = 0
    tool_f1 = 0.0
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
        prompt = str(select_first(row, spec.prompt_selectors) or "")
        expected = coerce_steps(select_first(row, spec.expected_selectors))
        predicted_raw = predict_fn(prompt)
        predicted = coerce_steps(extract_json_payload(str(predicted_raw)))

        json_valid += int(bool(predicted))
        if not expected:
            abstain_exact += int(not predicted or (len(predicted) == 1 and predicted[0].get("tool") == "reasoning"))
            continue

        scores = score_toolchain(expected, predicted, allow_refusal=spec.allow_refusal)
        tool_exact += int(scores["tool_accuracy"] == 1.0 and scores["predicted_steps"] == scores["expected_steps"])
        param_exact += int(scores["param_accuracy"] == 1.0 and scores["predicted_steps"] == scores["expected_steps"])
        tool_f1 += scores["tool_f1"]
        orchestration_score += scores["tool_orchestration_score"]
        for key, value in scores["errors"].items():
            error_totals[key] += int(value)

    denom = max(rows, 1)
    metrics = {
        "json_valid_rate": json_valid / denom,
        "tool_exact_rate": tool_exact / denom,
        "param_exact_rate": param_exact / denom,
        "abstain_exact_rate": abstain_exact / denom,
        "tool_f1": tool_f1 / denom,
        "tool_orchestration_score": orchestration_score / denom,
        "errors": {key: value / denom for key, value in error_totals.items()},
    }
    return BenchmarkResult(name=spec.name, rows=rows, metrics=metrics, note=str(dataset_path))
