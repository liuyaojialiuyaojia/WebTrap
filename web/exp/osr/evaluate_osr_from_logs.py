#!/usr/bin/env python3
"""Evaluate OSR tasks from GPT-web-tools trace logs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

_NAV_STATUS_RE = re.compile(r"Information navigation → page\s+(\d+)")
_HASH_URL_RE = re.compile(r"url:\s*[^\s]+#p(\d+)", re.IGNORECASE)
_INDEX_ROOT_RE = re.compile(r"url:\s*[^\s]*index\.html(?!#p)", re.IGNORECASE)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_page_index(content: str) -> Optional[int]:
    if not content:
        return None
    status_matches = _NAV_STATUS_RE.findall(content)
    if status_matches:
        try:
            return int(status_matches[-1])
        except ValueError:
            pass
    hash_matches = _HASH_URL_RE.findall(content)
    if hash_matches:
        try:
            return int(hash_matches[-1])
        except ValueError:
            pass
    if _INDEX_ROOT_RE.search(content):
        return 0
    return None


def _iter_trace_messages_last_snapshot(path: Path) -> Iterable[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return
    for candidate in (lines[-1], *reversed(lines[:-1])):
        try:
            messages = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if isinstance(msg, dict):
                yield msg
        return


def _visited_page_indices_from_trace(path: Path) -> set[int]:
    visited: set[int] = set()
    for msg in _iter_trace_messages_last_snapshot(path):
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        page_index = _detect_page_index(content)
        if page_index is not None:
            visited.add(page_index)
    return visited


def _final_page_index_from_trace(path: Path) -> Optional[int]:
    current_page_index: Optional[int] = 0
    for msg in _iter_trace_messages_last_snapshot(path):
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        detected = _detect_page_index(content)
        if detected is not None:
            current_page_index = detected
    return current_page_index


def _discover_tasks(task_folder: Path) -> List[Path]:
    tasks: List[Path] = []
    for path in task_folder.glob("*.json"):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        if path.stem.isdigit():
            tasks.append(path)
    return sorted(tasks, key=lambda p: int(p.stem))


def _resolve_log_paths(log_folder: Path, task_id: int) -> List[Path]:
    candidates: List[Path] = []
    direct_trace = log_folder / f"trace_{task_id}.jsonl"
    if direct_trace.is_file():
        candidates.append(direct_trace)

    trial_traces = sorted(log_folder.glob(f"trace_{task_id}_r*.jsonl"))
    candidates.extend([path for path in trial_traces if path.is_file()])

    numeric = log_folder / f"{task_id}.jsonl"
    if numeric.is_file():
        candidates.append(numeric)

    seen: set[Path] = set()
    unique: List[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


@dataclass(frozen=True)
class TrialResult:
    log: str
    final_page_index: Optional[int]
    visited_expected_page: Optional[bool]
    visited_pages: List[int]
    passed: bool


@dataclass(frozen=True)
class TaskResult:
    task_id: int
    passed: bool
    expected_page_index: int
    expected_leaf_marker: Optional[str]
    matched_log: Optional[str]
    trials: List[TrialResult]


def _evaluate_one(task_path: Path, log_folder: Path) -> TaskResult:
    task = _load_json(task_path)
    metadata = task.get("metadata") if isinstance(task, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError(f"Task missing metadata dict: {task_path}")
    try:
        task_id = int(task.get("task_id"))
        expected_page_index = int(metadata.get("page_index"))
    except Exception as exc:
        raise ValueError(f"Task missing numeric ids: {task_path}") from exc

    expected_leaf_marker = str(metadata.get("leaf_marker") or "").strip() or None
    log_paths = _resolve_log_paths(log_folder, task_id)
    trials: List[TrialResult] = []
    passed = False
    matched_log: Optional[str] = None

    for log_path in log_paths:
        visited_pages = sorted(_visited_page_indices_from_trace(log_path))
        visited_expected_page = expected_page_index in visited_pages
        trial_passed = bool(visited_expected_page)
        trials.append(
            TrialResult(
                log=str(log_path),
                final_page_index=_final_page_index_from_trace(log_path),
                visited_expected_page=visited_expected_page,
                visited_pages=visited_pages,
                passed=trial_passed,
            )
        )
        if trial_passed and not passed:
            passed = True
            matched_log = str(log_path)

    if not log_paths:
        trials.append(
            TrialResult(
                log="(missing)",
                final_page_index=None,
                visited_expected_page=None,
                visited_pages=[],
                passed=False,
            )
        )

    return TaskResult(
        task_id=task_id,
        passed=passed,
        expected_page_index=expected_page_index,
        expected_leaf_marker=expected_leaf_marker,
        matched_log=matched_log,
        trials=trials,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-folder", type=Path, required=True)
    parser.add_argument("--task-folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.log_folder.is_dir():
        raise FileNotFoundError(f"log folder not found: {args.log_folder}")
    if not args.task_folder.is_dir():
        raise FileNotFoundError(f"task folder not found: {args.task_folder}")

    results = [
        _evaluate_one(task_path, args.log_folder)
        for task_path in _discover_tasks(args.task_folder)
    ]
    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]

    payload = {
        "evaluator": "web/exp/osr/evaluate_osr_from_logs.py",
        "rule": "best_of_trials_visited_expected_page_index",
        "total_scores": float(len(passed)),
        "cnt_tasks": float(len(results)),
        "passed_task_ids": [result.task_id for result in passed],
        "failed_task_ids": [result.task_id for result in failed],
        "details": [
            {
                "task_id": result.task_id,
                "passed": result.passed,
                "expected_page_index": result.expected_page_index,
                "expected_leaf_marker": result.expected_leaf_marker,
                "matched_log": result.matched_log,
                "trials": [
                    {
                        "log": trial.log,
                        "final_page_index": trial.final_page_index,
                        "visited_expected_page": trial.visited_expected_page,
                        "visited_pages": trial.visited_pages,
                        "passed": trial.passed,
                    }
                    for trial in result.trials
                ],
            }
            for result in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
