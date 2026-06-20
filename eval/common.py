from __future__ import annotations

import gzip
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Any, Iterator


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    kind: str
    data_path_env: str
    default_path: str
    prompt_selectors: tuple[str, ...] = ()
    expected_selectors: tuple[str, ...] = ()
    allow_refusal: bool = True

def load_jsonl(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
def _open_record_stream(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _record_items(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("data", "items", "examples", "rows", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


def _load_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(path)
        return [row for row in table.to_pylist() if isinstance(row, dict)]
    except Exception:
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_parquet(path)
            return [row for row in frame.to_dict(orient="records") if isinstance(row, dict)]
        except Exception as exc:
            raise RuntimeError(f"Unable to read parquet file: {path}") from exc


def load_records(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    if path.is_dir():
        seen = 0
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            suffixes = [suffix.lower() for suffix in child.suffixes]
            if not (
                child.suffix.lower() in {".jsonl", ".json", ".parquet"}
                or suffixes[-2:] in ([".jsonl", ".gz"], [".json", ".gz"])
            ):
                continue
            for row in load_records(child, None if limit is None else limit - seen):
                yield row
                seen += 1
                if limit is not None and seen >= limit:
                    return
        return

    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-2:] == [".jsonl", ".gz"] or path.suffix.lower() == ".jsonl":
        with _open_record_stream(path) as handle:
            for index, line in enumerate(handle):
                if limit is not None and index >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return

    if suffixes[-2:] == [".json", ".gz"] or path.suffix.lower() == ".json":
        with _open_record_stream(path) as handle:
            payload = json.load(handle)
        count = 0
        for item in _record_items(payload):
            if limit is not None and count >= limit:
                return
            count += 1
            yield item
        return

    if path.suffix.lower() == ".parquet" or suffixes[-2:] == [".parquet", ".gz"]:
        count = 0
        for item in _load_parquet_records(path):
            if limit is not None and count >= limit:
                return
            count += 1
            yield item
        return

    raise ValueError(f"Unsupported record file: {path}")


def spec_dir() -> Path:
    return Path(__file__).resolve().parent / "specs"


def load_spec(name: str, fallback: BenchmarkSpec) -> BenchmarkSpec:
    path = spec_dir() / f"{name}.json"
    if not path.exists():
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = asdict(fallback)
    for key in ("name", "kind", "data_path_env", "default_path", "allow_refusal"):
        if key in payload:
            data[key] = payload[key]
    for key in ("prompt_selectors", "expected_selectors"):
        if key in payload and isinstance(payload[key], list):
            data[key] = tuple(str(item) for item in payload[key])
    return BenchmarkSpec(**data)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text.strip()).lower()


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<reasoning>.*?</reasoning>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL)
    return text.strip()


def split_selector(selector: str) -> list[str]:
    parts: list[str] = []
    buf = []
    depth = 0
    for char in selector:
        if char == "." and depth == 0:
            if buf:
                parts.append("".join(buf))
                buf = []
            continue
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        buf.append(char)
    if buf:
        parts.append("".join(buf))
    return parts


def _walk_selector(node: Any, selector: str) -> list[Any]:
    current = [node]
    for part in split_selector(selector):
        next_nodes: list[Any] = []
        match = re.fullmatch(r"([^\[]+)(?:\[(.+)\])?", part)
        if not match:
            return []
        key, suffix = match.group(1), match.group(2)
        for item in current:
            value = item
            if key:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    continue
            if suffix is None:
                next_nodes.append(value)
                continue
            if not isinstance(value, list):
                continue
            if suffix.startswith("?"):
                cond = suffix[1:]
                cond_match = re.fullmatch(r"([A-Za-z0-9_]+)=([^=]+)", cond)
                if not cond_match:
                    continue
                cond_key, cond_value = cond_match.group(1), cond_match.group(2)
                for element in value:
                    if isinstance(element, dict) and normalize_text(element.get(cond_key)) == normalize_text(cond_value):
                        next_nodes.append(element)
            else:
                try:
                    index = int(suffix)
                except ValueError:
                    continue
                try:
                    next_nodes.append(value[index])
                except IndexError:
                    continue
        current = next_nodes
    return current


def select_first(node: Any, selectors: tuple[str, ...]) -> Any:
    for selector in selectors:
        for value in _walk_selector(node, selector):
            if value not in (None, "", [], {}):
                return value
    return None


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_payload(text: str) -> Any | None:
    cleaned = strip_reasoning(text)
    candidates = [cleaned]

    first = cleaned.find("[")
    last = cleaned.rfind("]")
    if 0 <= first < last:
        candidates.append(cleaned[first : last + 1])

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if 0 <= first_obj < last_obj:
        candidates.append(cleaned[first_obj : last_obj + 1])

    for candidate in candidates:
        parsed = _try_json(candidate)
        if parsed is not None:
            return parsed
    return None


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(value[key]) for key in sorted(value)}
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(normalize_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def coerce_steps(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    steps: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        step = {
            "id": str(item.get("id") or f"step_{index + 1}"),
            "tool": str(item.get("tool") or ""),
            "params": item.get("params") if isinstance(item.get("params"), dict) else {},
            "depends_on": [str(dep) for dep in item.get("depends_on", [])] if isinstance(item.get("depends_on"), list) else [],
        }
        steps.append(step)
    return steps


def coerce_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("answer", "text", "content", "response", "result"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(payload, ensure_ascii=False)


def is_refusal_like(steps: list[dict[str, Any]]) -> bool:
    if not steps:
        return True
    if len(steps) == 1 and steps[0].get("tool") == "reasoning":
        return True
    return False


def score_step_lists(expected: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    expected = coerce_steps(expected)
    predicted = coerce_steps(predicted)

    total = max(len(expected), len(predicted), 1)
    aligned = min(len(expected), len(predicted))

    tool_hits = 0
    param_hits = 0
    dep_hits = 0
    exact_hits = 0

    for index in range(aligned):
        exp = expected[index]
        got = predicted[index]
        tool_ok = normalize_text(exp.get("tool")) == normalize_text(got.get("tool"))
        param_ok = canonical_json(exp.get("params", {})) == canonical_json(got.get("params", {}))
        dep_ok = canonical_json(sorted(exp.get("depends_on", []))) == canonical_json(sorted(got.get("depends_on", [])))
        tool_hits += int(tool_ok)
        param_hits += int(param_ok)
        dep_hits += int(dep_ok)
        exact_hits += int(tool_ok and param_ok and dep_ok)

    tool_precision = tool_hits / max(len(predicted), 1)
    tool_recall = tool_hits / max(len(expected), 1)
    tool_f1 = 0.0 if tool_precision + tool_recall == 0 else (2 * tool_precision * tool_recall) / (tool_precision + tool_recall)

    return {
        "json_valid": 1,
        "expected_steps": len(expected),
        "predicted_steps": len(predicted),
        "tool_accuracy": tool_hits / total,
        "tool_f1": tool_f1,
        "param_accuracy": param_hits / total,
        "dependency_accuracy": dep_hits / total,
        "step_exact_rate": exact_hits / total,
        "exact_sequence": int(canonical_json(expected) == canonical_json(predicted)),
    }


def score_toolchain(expected: list[dict[str, Any]], predicted: list[dict[str, Any]], allow_refusal: bool = True) -> dict[str, Any]:
    expected = coerce_steps(expected)
    predicted = coerce_steps(predicted)
    base = score_step_lists(expected, predicted)

    refused = allow_refusal and is_refusal_like(predicted)
    expected_refusal = is_refusal_like(expected)
    refusal_ok = int(refused == expected_refusal)
    overcall = int(not expected_refusal and refused)
    undercall = int(expected_refusal and not refused)

    coverage = 0.5 * base["tool_accuracy"] + 0.3 * base["param_accuracy"] + 0.2 * base["dependency_accuracy"]
    strictness = 0.4 * base["json_valid"] + 0.35 * base["exact_sequence"] + 0.25 * base["step_exact_rate"]
    tool_orchestration_score = 100.0 * (0.55 * coverage + 0.25 * strictness + 0.20 * refusal_ok)

    errors = {
        "parse_fail": int(len(predicted) == 0 and len(expected) > 0),
        "overcall": overcall,
        "undercall": undercall,
        "refusal_miss": int(not refusal_ok),
        "tool_mismatch": int(base["tool_accuracy"] < 1.0),
        "param_mismatch": int(base["param_accuracy"] < 1.0),
        "dependency_mismatch": int(base["dependency_accuracy"] < 1.0),
        "sequence_mismatch": int(base["exact_sequence"] < 1.0),
    }

    return {
        **base,
        "refusal_ok": refusal_ok,
        "tool_orchestration_score": tool_orchestration_score,
        "errors": errors,
    }


def contamination_report(train_rows: Iterator[dict[str, Any]], eval_rows: Iterator[dict[str, Any]], sample_limit: int | None = None) -> dict[str, Any]:
    train_hashes: set[str] = set()
    eval_hashes: set[str] = set()
    train_count = 0
    eval_count = 0

    for row in train_rows:
        train_count += 1
        train_hashes.add(stable_hash(row))
        if sample_limit is not None and train_count >= sample_limit:
            break

    for row in eval_rows:
        eval_count += 1
        eval_hashes.add(stable_hash(row))
        if sample_limit is not None and eval_count >= sample_limit:
            break

    overlap = len(train_hashes & eval_hashes)
    return {
        "train_seen": train_count,
        "eval_seen": eval_count,
        "hash_overlap": overlap,
        "overlap_rate": overlap / max(len(eval_hashes), 1),
    }


def token_f1(expected: str, predicted: str) -> float:
    exp_tokens = normalize_text(expected).split()
    got_tokens = normalize_text(predicted).split()
    if not exp_tokens or not got_tokens:
        return 0.0
    exp_counts: dict[str, int] = {}
    got_counts: dict[str, int] = {}
    for token in exp_tokens:
        exp_counts[token] = exp_counts.get(token, 0) + 1
    for token in got_tokens:
        got_counts[token] = got_counts.get(token, 0) + 1
    overlap = 0
    for token, count in exp_counts.items():
        overlap += min(count, got_counts.get(token, 0))
    precision = overlap / len(got_tokens)
    recall = overlap / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_text_answers(expected: str, predicted: str) -> dict[str, Any]:
    exp_norm = normalize_text(expected)
    got_norm = normalize_text(predicted)
    return {
        "exact_match": int(exp_norm == got_norm),
        "contains": int(bool(exp_norm) and exp_norm in got_norm),
        "token_f1": token_f1(expected, predicted),
    }


def normalize_shell_command(command: Any) -> str:
    text = normalize_text(command)
    text = text.replace("\\", "/")
    text = re.sub(r"\s+", " ", text)
    return text


def infer_shell_intent(command: Any) -> str:
    text = normalize_shell_command(command)
    if not text:
        return ""

    intent_map = (
        ("git_status", r"\bgit\s+status\b"),
        (
            "latest_markdown",
            r"((\.md\b|markdown).*(sort-object|sort\s*-|head\s*-n\s*1|select-object\s*-first|tail\s*-n\s*1|latest|newest|recent|lastwritetime)"
            r"|((sort-object|sort\s*-).*(\.md\b|markdown))|latest.*(\.md\b|markdown))",
        ),
        ("todo_search", r"(todo|fixme).*(rg\b|grep\b|select-string\b|findstr\b)|\b(rg|grep|select-string|findstr)\b.*(todo|fixme)"),
        (
            "latest_png",
            r"((\.png\b|png).*(sort-object|sort\s*-|head\s*-n\s*1|select-object\s*-first|tail\s*-n\s*1|latest|newest|recent|lastwritetime)"
            r"|((sort-object|sort\s*-).*(\.png\b|png))|latest.*(\.png\b|png))",
        ),
        ("read_file", r"(get-content|type\b|cat\b|sed\b.*-n)"),
        ("write_file", r"(set-content|out-file|new-item|add-content|tee\b|printf\b.*>|echo\b.*>)"),
        ("list_files", r"(get-childitem|dir\b|ls\b|find\b)"),
        ("search_text", r"(select-string|grep\b|rg\b|findstr\b)"),
    )

    for intent, pattern in intent_map:
        if re.search(pattern, text):
            return intent
    return "shell_command"


@dataclass
class BenchmarkResult:
    name: str
    rows: int
    metrics: dict[str, Any]
    skipped: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
