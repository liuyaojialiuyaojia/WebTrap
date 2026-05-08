#!/usr/bin/env python3
"""Shared helpers for file/exp staged pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def case_id_from_index(index: int) -> str:
    return f"case_{index + 1:04d}"


def to_posix(path: Path) -> str:
    return path.as_posix()


def safe_relative(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    try:
        return path.relative_to(root)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Path {path} is outside root {root}") from exc


def iter_case_ids(attack_cases_path: Path) -> Iterator[str]:
    for row in read_jsonl(attack_cases_path):
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            yield case_id


def find_directory_readme(directory: Path) -> Optional[Path]:
    candidates = sorted(
        [p for p in directory.iterdir() if p.is_file() and p.name.lower().startswith("readme.")],
        key=lambda p: p.name,
    )
    if candidates:
        return candidates[0]
    return None


def ensure_directory_readme(directory: Path, suffix: str = ".md") -> Path:
    readme = find_directory_readme(directory)
    if readme is not None:
        return readme
    if not suffix.startswith("."):
        suffix = "." + suffix
    readme = directory / f"readme{suffix}"
    if not readme.exists():
        readme.write_text("", encoding="utf-8")
    return readme
