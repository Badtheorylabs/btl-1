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
    name="toolbench",
    kind="steps",
    data_path_env="BTL_TOOLBENCH_PATH",
    default_path="eval/data/toolbench.jsonl",
    prompt_selectors=("prompt", "question", "query", "input", "instruction", "messages[?role=user].content"),
    expected_selectors=("tool_calls", "expected", "answer", "label", "ground_truth", "messages[-1].content"),
)


def evaluate_toolbench(
    predict_fn: Callable[[str], str],
    dataset_path: Path,
    limit: int | None = None,
) -> BenchmarkResult:
    spec = load_spec("toolbench", DEFAULT_SPEC)
    rows = 0
    exact_sequence = 0
    step_exact = 0.0
    dependency_accuracy = 0.0
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
        predicted = coerce_steps(extract_json_payload(str(predict_fn(prompt))))
        expected = coerce_steps(select_first(row, spec.expected_selectors))
        scores = score_toolchain(expected, predicted, allow_refusal=spec.allow_refusal)
        exact_sequence += scores["exact_sequence"]
        step_exact += scores["step_exact_rate"]
        dependency_accuracy += scores["dependency_accuracy"]
        tool_f1 += scores["tool_f1"]
        orchestration_score += scores["tool_orchestration_score"]
        for key, value in scores["errors"].items():
            error_totals[key] += int(value)

    denom = max(rows, 1)
    metrics = {
        "exact_sequence_rate": exact_sequence / denom,
        "step_exact_rate": step_exact / denom,
        "dependency_accuracy": dependency_accuracy / denom,
        "tool_f1": tool_f1 / denom,
        "tool_orchestration_score": orchestration_score / denom,
        "errors": {key: value / denom for key, value in error_totals.items()},
    }
    return BenchmarkResult(name=spec.name, rows=rows, metrics=metrics, note=str(dataset_path))
