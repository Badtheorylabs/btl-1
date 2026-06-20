from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .bfcl import evaluate_bfcl
from .btl1 import evaluate_btl1
from .common import BenchmarkResult, BenchmarkSpec, contamination_report, load_records, load_spec
from .crossos import evaluate_crossos
from .gaia import evaluate_gaia
from .toolbench import evaluate_toolbench


@dataclass
class SuiteResult:
    results: list[BenchmarkResult]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "results": [result.to_dict() for result in self.results],
        }


def _resolve_path(value: str | None, default: Path) -> Path | None:
    if not value:
        return default if default.exists() else None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _dataset_path(project_root: Path, spec: BenchmarkSpec) -> Path | None:
    default_path = project_root / spec.default_path
    return _resolve_path(os.environ.get(spec.data_path_env), default_path)


def _looks_placeholder(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    try:
        for row in load_records(path, limit=5):
            assistant = None
            if isinstance(row, dict):
                assistant_content = row.get("assistant_content")
                if isinstance(assistant_content, str):
                    assistant = assistant_content
                else:
                    messages = row.get("messages")
                    if isinstance(messages, list) and messages:
                        last = messages[-1]
                        if isinstance(last, dict):
                            content = last.get("content")
                            if isinstance(content, str):
                                assistant = content
            if assistant and "dry run" in assistant.lower():
                return True
        return False
    except Exception:
        return False


def _latest_teacher_completed(project_root: Path) -> Path | None:
    teacher_root = project_root / "data" / "teacher" / "runs"
    if not teacher_root.exists():
        return None
    candidates = sorted(
        teacher_root.rglob("completed.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if not _looks_placeholder(candidate):
            return candidate
    return None


def _default_paths(project_root: Path) -> dict[str, Path | None]:
    btl1_spec = load_spec("btl1", BenchmarkSpec("btl1", "messages", "BTL_BTL1_PATH", "data/final/eval.jsonl"))
    bfcl_spec = load_spec("bfcl", BenchmarkSpec("bfcl", "steps", "BTL_BFCL_PATH", "eval/data/bfcl.jsonl"))
    toolbench_spec = load_spec("toolbench", BenchmarkSpec("toolbench", "steps", "BTL_TOOLBENCH_PATH", "eval/data/toolbench.jsonl"))
    gaia_spec = load_spec("gaia", BenchmarkSpec("gaia", "text", "BTL_GAIA_PATH", "eval/data/gaia.jsonl"))
    crossos_spec = load_spec("crossos", BenchmarkSpec("crossos", "portable_steps", "BTL_CROSSOS_PATH", "eval/data/crossos.jsonl"))
    return {
        "btl1": _dataset_path(project_root, btl1_spec),
        "bfcl": _dataset_path(project_root, bfcl_spec),
        "toolbench": _dataset_path(project_root, toolbench_spec),
        "gaia": _dataset_path(project_root, gaia_spec),
        "crossos": _dataset_path(project_root, crossos_spec),
    }


def _skip(name: str, path: Path | None) -> BenchmarkResult:
    note = "missing benchmark data"
    if path is not None:
        note = str(path)
    return BenchmarkResult(name=name, rows=0, metrics={}, skipped=True, note=note)


def _north_star_score(results: list[BenchmarkResult]) -> float:
    scores: list[float] = []
    for result in results:
        metric = result.metrics.get("tool_orchestration_score")
        if isinstance(metric, (int, float)) and metric > 0:
            scores.append(float(metric))
    if not scores:
        return 0.0
    inv_sum = sum(1.0 / max(score, 1e-6) for score in scores)
    return len(scores) / inv_sum


def run_suite(
    predict_fn: Callable[[str], str],
    project_root: Path,
    tokenizer: Any | None = None,
    limit: int | None = None,
) -> SuiteResult:
    paths = _default_paths(project_root)
    results: list[BenchmarkResult] = []

    btl1_path = paths["btl1"]
    if _looks_placeholder(btl1_path):
        btl1_path = None
    results.append(evaluate_btl1(predict_fn, btl1_path, tokenizer=tokenizer, limit=limit) if btl1_path else _skip("btl1", btl1_path))
    results.append(evaluate_bfcl(predict_fn, paths["bfcl"], limit=limit) if paths["bfcl"] else _skip("bfcl", paths["bfcl"]))
    results.append(evaluate_toolbench(predict_fn, paths["toolbench"], limit=limit) if paths["toolbench"] else _skip("toolbench", paths["toolbench"]))
    results.append(evaluate_gaia(predict_fn, paths["gaia"], limit=limit) if paths["gaia"] else _skip("gaia", paths["gaia"]))
    results.append(evaluate_crossos(predict_fn, paths["crossos"], limit=limit) if paths["crossos"] else _skip("crossos", paths["crossos"]))

    train_btl1_path = _resolve_path(os.environ.get("BTL_BTL1_TRAIN_PATH"), None)
    if train_btl1_path is None:
        train_btl1_path = _latest_teacher_completed(project_root)
    if train_btl1_path is None:
        train_btl1_path = _resolve_path(None, project_root / "data" / "final" / "train.jsonl")
    if _looks_placeholder(train_btl1_path):
        train_btl1_path = None

    transfer_btl1_result = next((result for result in results if result.name == "btl1"), None)
    train_btl1_result = None
    crr = None
    if train_btl1_path is not None:
        train_btl1_result = evaluate_btl1(predict_fn, train_btl1_path, tokenizer=tokenizer, limit=limit)
        if transfer_btl1_result is not None:
            train_score = float(train_btl1_result.metrics.get("tool_orchestration_score", 0.0))
            transfer_score = float(transfer_btl1_result.metrics.get("tool_orchestration_score", 0.0))
            crr = transfer_score / max(train_score, 1e-6)

    train_path = project_root / "data" / "final" / "train.jsonl"
    eval_path = project_root / "data" / "final" / "eval.jsonl"
    scan_limit = limit if limit is not None else 2000
    contamination = {
        "sample_limit": scan_limit,
        "train_vs_eval": contamination_report(
            load_records(train_path, limit=scan_limit),
            load_records(eval_path, limit=scan_limit),
            sample_limit=scan_limit,
        ),
        "external": {},
    }
    for name in ("bfcl", "toolbench", "gaia", "crossos"):
        path = paths.get(name)
        if path is None:
            contamination["external"][name] = {"skipped": True, "note": "missing benchmark data"}
            continue
        contamination["external"][name] = contamination_report(
            load_records(train_path, limit=scan_limit),
            load_records(path, limit=scan_limit),
            sample_limit=scan_limit,
        )

    summary = {
        "north_star_tool_orchestration_score": _north_star_score(results),
        "btl1": {
            "train_path": str(train_btl1_path) if train_btl1_path is not None else None,
            "transfer_path": str(paths["btl1"]) if paths["btl1"] is not None else None,
            "train_tool_orchestration_score": None if train_btl1_result is None else train_btl1_result.metrics.get("tool_orchestration_score"),
            "transfer_tool_orchestration_score": None if transfer_btl1_result is None else transfer_btl1_result.metrics.get("tool_orchestration_score"),
            "crr": crr,
        },
        "contamination": contamination,
        "benchmarks_scored": sum(1 for result in results if not result.skipped),
    }
    return SuiteResult(results=results, summary=summary)


def render_suite(result: SuiteResult) -> str:
    return json.dumps(result.to_dict(), indent=2, sort_keys=True)
