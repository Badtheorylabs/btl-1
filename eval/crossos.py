from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .common import (
    BenchmarkResult,
    BenchmarkSpec,
    extract_json_payload,
    infer_shell_intent,
    load_records,
    load_spec,
    normalize_text,
    score_toolchain,
    select_first,
)

DEFAULT_SPEC = BenchmarkSpec(
    name="crossos",
    kind="portable_steps",
    data_path_env="BTL_CROSSOS_PATH",
    default_path="eval/data/crossos.jsonl",
    prompt_selectors=("prompt", "question", "query", "input", "instruction"),
    expected_selectors=("steps", "expected", "tool_calls", "answer"),
)


def _portable_steps(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        payload = payload["steps"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    steps: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        params = dict(item.get("params") if isinstance(item.get("params"), dict) else {})
        tool = str(item.get("tool") or "")
        if tool == "shell_command":
            intent = item.get("intent") or params.get("intent") or params.get("command") or ""
            params["command"] = normalize_text(intent) if intent else infer_shell_intent(params.get("command"))
        steps.append(
            {
                "id": str(item.get("id") or f"step_{index + 1}"),
                "tool": tool,
                "params": params,
                "depends_on": [str(dep) for dep in item.get("depends_on", [])] if isinstance(item.get("depends_on"), list) else [],
            }
        )
    return steps


def evaluate_crossos(
    predict_fn: Callable[[str], str],
    dataset_path: Path,
    limit: int | None = None,
) -> BenchmarkResult:
    spec = load_spec("crossos", DEFAULT_SPEC)
    rows = 0
    json_valid = 0
    shell_steps = 0
    shell_hits = 0
    orchestration_score = 0.0
    portable_step_score = 0.0
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
        expected = _portable_steps(select_first(row, spec.expected_selectors))
        predicted = _portable_steps(extract_json_payload(str(predict_fn(prompt))))
        scores = score_toolchain(expected, predicted, allow_refusal=spec.allow_refusal)
        orchestration_score += scores["tool_orchestration_score"]
        portable_step_score += scores["step_exact_rate"]
        json_valid += int(bool(predicted))
        for exp, got in zip(expected, predicted):
            if exp.get("tool") == "shell_command":
                shell_steps += 1
                shell_hits += int(exp.get("params", {}).get("command") == got.get("params", {}).get("command"))
        for key, value in scores["errors"].items():
            error_totals[key] += int(value)

    denom = max(rows, 1)
    metrics = {
        "json_valid_rate": json_valid / denom,
        "portable_shell_accuracy": shell_hits / max(shell_steps, 1),
        "portable_step_exact_rate": portable_step_score / denom,
        "tool_orchestration_score": orchestration_score / denom,
        "errors": {key: value / denom for key, value in error_totals.items()},
    }
    return BenchmarkResult(name=spec.name, rows=rows, metrics=metrics, note=str(dataset_path))
