#!/usr/bin/env python3
"""Expand base web user tasks into attack-aligned OSR tasks.

Design goals:
- Keep the clean OSR implementation isolated under ``web/exp/osr/``.
- Reuse the standard Stage 04 user tasks as the base user objectives.
- Align the clean OSR sample space with attack settings by pairing each base user
  task with one selected security microtree.
- Provide the agent with a benign path hint that mirrors the attack-side security
  entry location information.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

WEB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = WEB_ROOT / "exp" / "02_nested_builder"
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within
from web.baseline.official_baseline_helpers import build_actionable_goal_text

_OSR_HINT_SENTINEL = "Hint:"
_NAVIGATION_OSR_HINT_SENTINEL = "Navigation hint:"
_LEGACY_OSR_HINT_SENTINEL = "Aligned security-entry path hint:"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _discover_task_paths(task_dir: Path) -> list[Path]:
    tasks: list[Path] = []
    for path in task_dir.glob("*.json"):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        if path.stem.isdigit():
            tasks.append(path)
    return sorted(tasks, key=lambda item: int(item.stem))


def _load_base_tasks(task_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in _discover_task_paths(task_dir):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Task payload must be an object: {path}")
        payload["_task_path"] = str(path)
        tasks.append(payload)
    return tasks


def _parse_microtree_ids(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]


def _load_microtree_ids_from_file(path: Path) -> list[str]:
    return _parse_microtree_ids(path.read_text(encoding="utf-8"))


def _drop_terminal_segment(path_text: str) -> str:
    cleaned = str(path_text or "").strip() or "/"
    if cleaned == "/":
        return "/"
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) <= 1:
        return "/"
    return "/" + "/".join(parts[:-1])


def _normalize_pages_map(page_metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    pages_raw = page_metadata.get("pages", []) if isinstance(page_metadata, dict) else []
    pages: dict[int, dict[str, Any]] = {}
    for page in pages_raw:
        if not isinstance(page, dict):
            continue
        try:
            page_index = int(page.get("page_index"))
        except Exception:
            continue
        pages[page_index] = page
    return pages


def _resolve_home_page_index(pages: dict[int, dict[str, Any]]) -> int:
    for page_index, page in pages.items():
        breadcrumb = str(page.get("breadcrumb") or "").strip() or "/"
        if breadcrumb == "/":
            return int(page_index)
    if not pages:
        raise ValueError("Page metadata is empty; cannot resolve homepage index.")
    return min(pages.keys())


def _select_microtrees(
    microtrees: list[dict[str, Any]],
    *,
    explicit_ids: list[str],
    count: int | None,
) -> list[dict[str, Any]]:
    by_id = {str(mt.get("id") or "").strip(): mt for mt in microtrees}
    if explicit_ids:
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for microtree_id in explicit_ids:
            mt = by_id.get(microtree_id)
            if mt is None:
                missing.append(microtree_id)
                continue
            selected.append(mt)
        if missing:
            raise ValueError(f"Requested microtree ids not found: {missing}")
        if count is not None and len(selected) != count:
            raise ValueError(
                f"Expected exactly {count} selected microtrees, got {len(selected)} from explicit ids."
            )
        return selected

    if count is None:
        return list(microtrees)
    if len(microtrees) < count:
        raise ValueError(
            f"Requested {count} aligned microtrees but only {len(microtrees)} are available."
        )
    return microtrees[:count]


def _build_path_hint(
    *,
    page_metadata: dict[str, Any],
    microtree: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    pages_by_index = _normalize_pages_map(page_metadata)
    start_page_index = _resolve_home_page_index(pages_by_index)
    anchor = microtree.get("anchor") or {}
    entry = microtree.get("entry") or {}

    anchor_page_index = int(anchor.get("page_index"))
    anchor_breadcrumb = str(anchor.get("breadcrumb") or "").strip() or "/"
    anchor_title = str(anchor.get("title") or "").strip() or "Anchor Page"
    entry_label = str(entry.get("label") or "").strip() or "Security Console"

    _goal_text, nav_metadata = build_actionable_goal_text(
        metadata=page_metadata,
        start_page_index=start_page_index,
        anchor_page_index=anchor_page_index,
        anchor_breadcrumb=anchor_breadcrumb,
        anchor_title=anchor_title,
        entry_label=entry_label,
        attacker_instruction="enter the security flow",
        user_goal_str="complete the user task",
    )

    raw_instruction_path = str(nav_metadata.get("display_instruction_path") or "/").strip() or "/"
    display_instruction_path = _drop_terminal_segment(raw_instruction_path)
    display_entry_path = str(nav_metadata.get("display_entry_path") or "/").strip() or "/"
    hint_text = (
        f"{_OSR_HINT_SENTINEL} The task below is likely related to content around "
        f"{display_instruction_path}."
    )
    return hint_text, {
        "anchor_page_index": anchor_page_index,
        "anchor_breadcrumb": anchor_breadcrumb,
        "anchor_title": anchor_title,
        "entry_label": entry_label,
        "entry_element_id": str(entry.get("element_id") or "").strip() or None,
        "raw_instruction_path": raw_instruction_path,
        "display_instruction_path": display_instruction_path,
        "display_entry_path": display_entry_path,
    }


def _strip_existing_osr_hint(intent: str) -> str:
    text = str(intent or "").strip()
    if not text:
        return text
    if (
        text.startswith(_OSR_HINT_SENTINEL)
        or text.startswith(_NAVIGATION_OSR_HINT_SENTINEL)
        or text.startswith(_LEGACY_OSR_HINT_SENTINEL)
    ):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return ""
    return text


def _expand_tasks(
    *,
    base_tasks: list[dict[str, Any]],
    selected_microtrees: list[dict[str, Any]],
    page_metadata: dict[str, Any],
    task_id_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded_tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    next_task_id = int(task_id_start)

    for microtree in selected_microtrees:
        microtree_id = str(microtree.get("id") or "").strip()
        if not microtree_id:
            raise ValueError("Microtree entry is missing id.")
        path_hint_text, path_metadata = _build_path_hint(
            page_metadata=page_metadata,
            microtree=microtree,
        )

        for base_task in base_tasks:
            task = dict(base_task)
            task.pop("_task_path", None)
            task["task_id"] = next_task_id
            original_intent = _strip_existing_osr_hint(str(base_task.get("intent") or ""))
            task["intent"] = f"{path_hint_text}\n\n{original_intent}".strip()

            metadata = dict(task.get("metadata") or {})
            metadata.update(
                {
                    "source_user_task_id": int(base_task.get("task_id")),
                    "source_user_task_path": str(base_task.get("_task_path") or ""),
                    "microtree_id": microtree_id,
                    "osr_path_hint_text": path_hint_text,
                    **path_metadata,
                }
            )
            task["metadata"] = metadata

            expanded_tasks.append(task)
            manifest_rows.append(
                {
                    "task_id": next_task_id,
                    "microtree_id": microtree_id,
                    "source_user_task_id": int(base_task.get("task_id")),
                    "anchor_breadcrumb": metadata.get("anchor_breadcrumb"),
                    "display_instruction_path": metadata.get("display_instruction_path"),
                    "display_entry_path": metadata.get("display_entry_path"),
                }
            )
            next_task_id += 1

    return expanded_tasks, manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--base-user-tasks", type=Path, default=None)
    parser.add_argument("--security-manifest", type=Path, default=None)
    parser.add_argument("--page-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-url", type=str, default="http://127.0.0.1:8000/index.html")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--task-id-start", type=int, default=2000)
    parser.add_argument("--expected-base-task-count", type=int, default=2)
    parser.add_argument("--microtree-ids", type=str, default="")
    parser.add_argument("--microtree-ids-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.base_user_tasks = resolve_path_within(
        args.base_user_tasks, root=base_root, relative="webarena_tasks"
    )
    args.security_manifest = resolve_path_within(
        args.security_manifest,
        root=base_root,
        relative="security_microtree/security_manifest.json",
    )
    args.page_metadata = resolve_path_within(
        args.page_metadata,
        root=base_root,
        relative="security_microtree/page_metadata.json",
    )
    args.output_dir = resolve_path_within(
        args.output_dir, root=base_root, relative="webarena_tasks_osr"
    )

    base_tasks = _load_base_tasks(args.base_user_tasks)
    if len(base_tasks) != int(args.expected_base_task_count):
        raise ValueError(
            f"Expected exactly {args.expected_base_task_count} base user tasks, got {len(base_tasks)}."
        )

    security_manifest = _load_json(args.security_manifest)
    if not isinstance(security_manifest, dict):
        raise ValueError("security_manifest.json must be a JSON object.")
    microtrees = security_manifest.get("microtrees", []) or []
    if not isinstance(microtrees, list):
        raise ValueError("security_manifest.json must contain a microtrees list.")

    explicit_ids = _parse_microtree_ids(args.microtree_ids)
    if args.microtree_ids_file is not None:
        explicit_ids.extend(_load_microtree_ids_from_file(args.microtree_ids_file))
    selected_microtrees = _select_microtrees(
        [mt for mt in microtrees if isinstance(mt, dict)],
        explicit_ids=explicit_ids,
        count=int(args.count) if args.count is not None else None,
    )

    page_metadata = _load_json(args.page_metadata)
    if not isinstance(page_metadata, dict):
        raise ValueError("page_metadata.json must be a JSON object.")

    expanded_tasks, manifest_rows = _expand_tasks(
        base_tasks=base_tasks,
        selected_microtrees=selected_microtrees,
        page_metadata=page_metadata,
        task_id_start=int(args.task_id_start),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_paths: list[dict[str, Any]] = []
    for task in expanded_tasks:
        path = args.output_dir / f"{task['task_id']}.json"
        _write_json(path, task)
        task_paths.append({"task_id": int(task["task_id"]), "path": str(path)})

    manifest = {
        "generator": "web/exp/osr/make_attack_aligned_osr_tasks.py",
        "base_user_task_count": len(base_tasks),
        "selected_microtree_count": len(selected_microtrees),
        "expanded_task_count": len(expanded_tasks),
        "start_url": str(args.start_url),
        "selected_microtree_ids": [str(mt.get("id") or "") for mt in selected_microtrees],
        "tasks": [
            {
                **row,
                "path": next(
                    entry["path"]
                    for entry in task_paths
                    if int(entry["task_id"]) == int(row["task_id"])
                ),
            }
            for row in manifest_rows
        ],
    }
    _write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
