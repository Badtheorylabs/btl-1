from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .common import BenchmarkResult, BenchmarkSpec, load_records, load_spec, score_text_answers, select_first

DEFAULT_SPEC = BenchmarkSpec(
    name="gaia",
    kind="text",
    data_path_env="BTL_GAIA_PATH",
    default_path="eval/data/gaia.jsonl",
    prompt_selectors=("prompt", "question", "query", "input", "instruction", "messages[?role=user].content"),
    expected_selectors=("answer", "reference", "expected", "label", "ground_truth", "solution", "messages[-1].content"),
)


def evaluate_gaia(
    predict_fn: Callable[[str], str],
    dataset_path: Path,
    limit: int | None = None,
) -> BenchmarkResult:
    spec = load_spec("gaia", DEFAULT_SPEC)
    rows = 0
    em = 0.0
    contains = 0.0
    f1_total = 0.0

    for row in load_records(dataset_path, limit=limit):
        rows += 1
        prompt = str(select_first(row, spec.prompt_selectors) or "")
        expected = str(select_first(row, spec.expected_selectors) or "")
        predicted = predict_fn(prompt)
        scores = score_text_answers(expected, predicted)
        em += scores["exact_match"]
        contains += scores["contains"]
        f1_total += scores["token_f1"]

    denom = max(rows, 1)
    metrics = {
        "exact_match": em / denom,
        "contains_rate": contains / denom,
        "token_f1": f1_total / denom,
    }
    return BenchmarkResult(name=spec.name, rows=rows, metrics=metrics, note=str(dataset_path))
