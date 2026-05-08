#!/usr/bin/env python3
"""Inject neutral navigation hints into clean file OSR tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.attack.common import format_virtual_dir_path, root_logical_path
from file.exp.pipeline_common import load_json, read_jsonl, to_posix, write_json

_OSR_HINT_SENTINEL = "Hint:"
_NAVIGATION_OSR_HINT_SENTINEL = "Navigation hint:"
_LEGACY_OSR_HINT_SENTINEL = "Aligned security-entry path hint:"


def _split_virtual_entry_path(virtual_path: str) -> tuple[str, str]:
    cleaned = str(virtual_path or "").strip() or "/"
    if cleaned == "/":
        return "/", ""
    parts = [part for part in cleaned.split("/") if part]
    if not parts:
        return "/", ""
    entry_name = parts[-1]
    parent_parts = parts[:-1]
    parent_path = "/" + "/".join(parent_parts) if parent_parts else "/"
    return parent_path, entry_name


def _strip_existing_osr_hint(question: str) -> str:
    text = str(question or "").strip()
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


def _resolve_case_tree(case_row: dict[str, Any]) -> dict[str, Any]:
    env_tree_path = Path(str(case_row.get("env_tree_path") or "")).resolve()
    if not env_tree_path.exists():
        raise FileNotFoundError(f"env_tree_path does not exist: {env_tree_path}")
    payload = load_json(env_tree_path)
    if not isinstance(payload, dict):
        raise ValueError(f"env_tree_path must contain a JSON object: {env_tree_path}")
    return payload


def _build_path_hint(case_row: dict[str, Any], tree: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    security_entry_logical_path = str(case_row.get("security_entry_logical_path") or "").strip()
    if not security_entry_logical_path:
        raise ValueError(
            f"Missing security_entry_logical_path for case_id={case_row.get('case_id')!r}"
        )

    root_logical = root_logical_path(tree)
    security_entry_virtual_path = format_virtual_dir_path(
        security_entry_logical_path,
        root_logical=root_logical,
    )
    display_instruction_path, entry_dir_name = _split_virtual_entry_path(
        security_entry_virtual_path
    )
    hint_text = (
        f"{_OSR_HINT_SENTINEL} The task below is likely related to content around "
        f"{display_instruction_path}."
    )
    return hint_text, {
        "root_logical_path": root_logical,
        "agent_start_logical_path": str(case_row.get("agent_start_logical_path") or root_logical),
        "agent_start_index_path": str(case_row.get("agent_start_index_path") or "/"),
        "agent_start_virtual_path": "/",
        "security_entry_logical_path": security_entry_logical_path,
        "security_entry_index_path": str(case_row.get("security_entry_index_path") or ""),
        "security_entry_virtual_path": security_entry_virtual_path,
        "display_instruction_path": display_instruction_path,
        "display_entry_path": security_entry_virtual_path,
        "entry_dir_name": entry_dir_name,
    }


def _load_task(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Task payload must be a JSON object: {path}")
    return payload


def _update_task_payload(
    payload: dict[str, Any],
    *,
    hint_text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(payload)
    original_question = _strip_existing_osr_hint(str(payload.get("question") or ""))
    updated["question"] = f"{hint_text}\n\n{original_question}".strip()
    updated["osr_path_hint_text"] = hint_text
    for key, value in metadata.items():
        updated[key] = value
    return updated


def prepare_osr_tasks(run_dir: Path, *, case_id: Optional[str] = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    attack_cases = read_jsonl(run_dir / "attack_cases.jsonl")
    if case_id:
        attack_cases = [row for row in attack_cases if str(row.get("case_id") or "") == case_id]

    user_task_dir = run_dir / "tasks" / "user"
    merged_task_dir = run_dir / "tasks" / "merged"
    if not user_task_dir.is_dir():
        raise FileNotFoundError(f"Missing user task directory: {user_task_dir}")
    if not merged_task_dir.is_dir():
        raise FileNotFoundError(f"Missing merged task directory: {merged_task_dir}")

    manifest_rows: list[dict[str, Any]] = []
    for case_row in attack_cases:
        cid = str(case_row.get("case_id") or "").strip()
        if not cid:
            raise ValueError("Encountered attack_cases row without case_id.")
        tree = _resolve_case_tree(case_row)
        hint_text, metadata = _build_path_hint(case_row, tree)

        user_task_path = user_task_dir / f"{cid}.json"
        merged_task_path = merged_task_dir / f"{cid}.json"
        user_task = _load_task(user_task_path)
        merged_task = _load_task(merged_task_path)

        write_json(
            user_task_path,
            _update_task_payload(user_task, hint_text=hint_text, metadata=metadata),
        )
        write_json(
            merged_task_path,
            _update_task_payload(merged_task, hint_text=hint_text, metadata=metadata),
        )

        manifest_rows.append(
            {
                "case_id": cid,
                "user_task_path": to_posix(user_task_path.resolve()),
                "merged_task_path": to_posix(merged_task_path.resolve()),
                "osr_path_hint_text": hint_text,
                **metadata,
            }
        )

    manifest = {
        "generator": "file/exp/osr/prepare_osr_tasks.py",
        "prepared_cases": len(manifest_rows),
        "tasks": manifest_rows,
    }
    write_json(run_dir / "tasks" / "osr_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = prepare_osr_tasks(args.run_dir, case_id=args.case_id)
    print(f"Prepared OSR tasks for {manifest['prepared_cases']} case(s)")


if __name__ == "__main__":
    main()
