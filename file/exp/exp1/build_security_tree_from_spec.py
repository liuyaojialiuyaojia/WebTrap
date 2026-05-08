#!/usr/bin/env python3
"""Build exp1 security subtree JSON from spec (no filesystem materialization)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.pipeline_common import write_json


def _normalize_suffix(raw: Any, default: str = ".md") -> str:
    value = str(raw or default).strip()
    if not value:
        return default
    if not value.startswith("."):
        value = "." + value
    value = re.sub(r"[^a-zA-Z0-9._-]", "", value)
    if value in {"", "."}:
        return default
    return value


def _readme_name(suffix: str) -> str:
    return f"readme{_normalize_suffix(suffix)}"


def _safe_rel_parts(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text or text == ".":
        return []
    p = Path(text)
    if p.is_absolute():
        raise ValueError(f"Path must be relative: {text}")
    if ".." in p.parts:
        raise ValueError(f"Path cannot contain '..': {text}")
    parts: list[str] = []
    for part in p.parts:
        part = part.strip()
        if part in {"", "."}:
            continue
        if "/" in part or "\\" in part:
            raise ValueError(f"Invalid path component in {text}: {part}")
        parts.append(part)
    return parts


def _new_node(name: str, content: Any = "") -> dict[str, Any]:
    if not str(name).strip():
        raise ValueError("Node name cannot be empty.")
    return {
        "name": str(name),
        "content": str(content),
        "children": [],
    }


def _is_directory(node: dict[str, Any]) -> bool:
    return bool(node.get("children"))


def _find_child(parent: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
    for child in parent.get("children", []):
        if str(child.get("name") or "") == name:
            return child
    return None


def _ensure_directory(parent: dict[str, Any], name: str) -> dict[str, Any]:
    existing = _find_child(parent, name)
    if existing is None:
        node = _new_node(name)
        parent["children"].append(node)
        return node
    if not _is_directory(existing):
        raise ValueError(f"Path conflict: '{name}' exists as a file.")
    return existing


def _upsert_file(parent: dict[str, Any], name: str, content: str) -> dict[str, Any]:
    existing = _find_child(parent, name)
    if existing is not None and _is_directory(existing):
        raise ValueError(f"Path conflict: '{name}' exists as a directory.")
    if existing is None:
        node = _new_node(name, content)
        parent["children"].append(node)
        return node
    existing["content"] = content
    return existing


def _upsert_readme(directory_node: dict[str, Any], *, suffix: str, content: str) -> str:
    filename = _readme_name(suffix)
    _upsert_file(directory_node, filename, content)
    return filename


def _ensure_readme_exists(directory_node: dict[str, Any], default_content: str) -> str:
    for child in directory_node.get("children", []):
        name = str(child.get("name") or "").lower()
        if not child.get("children") and name.startswith("readme."):
            return str(child.get("name"))
    return _upsert_readme(directory_node, suffix=".md", content=default_content)


def _assign_depths(node: dict[str, Any], depth: int) -> None:
    node["depth"] = int(depth)
    for child in node.get("children", []):
        _assign_depths(child, depth + 1)


def _collect_directories(
    *,
    node: dict[str, Any],
    logical_path: str,
    rows: list[dict[str, Any]],
) -> None:
    if not node.get("children"):
        return
    readmes = [
        str(child.get("name") or "")
        for child in node.get("children", [])
        if not child.get("children") and str(child.get("name") or "").lower().startswith("readme.")
    ]
    rows.append(
        {
            "logical_path": logical_path,
            "readmes": sorted(readmes),
        }
    )
    for child in node.get("children", []):
        if child.get("children"):
            child_path = f"{logical_path}/{child['name']}".replace("//", "/")
            _collect_directories(node=child, logical_path=child_path, rows=rows)


def build_security_tree_from_spec_json(
    *,
    spec_path: Path,
) -> dict[str, Any]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Security spec must be a JSON object.")

    entry_cfg = payload.get("entry") if isinstance(payload.get("entry"), dict) else {}
    entry_dir_name = str(entry_cfg.get("dir_name") or "security_entry").strip() or "security_entry"
    if "/" in entry_dir_name or "\\" in entry_dir_name:
        raise ValueError(f"entry.dir_name must be a single path component: {entry_dir_name}")

    entry_suffix = _normalize_suffix(entry_cfg.get("readme_suffix"), default=".md")
    entry_content = str(entry_cfg.get("readme_content") or "")

    entry_node = _new_node(entry_dir_name)
    _upsert_readme(entry_node, suffix=entry_suffix, content=entry_content)

    folders_raw = payload.get("folders") or []
    if not isinstance(folders_raw, list):
        raise ValueError("folders must be a list when provided.")

    for folder in folders_raw:
        if not isinstance(folder, dict):
            raise ValueError("Each folders entry must be an object.")
        parts = _safe_rel_parts(folder.get("path"))
        current = entry_node
        for part in parts:
            current = _ensure_directory(current, part)
        suffix = _normalize_suffix(folder.get("readme_suffix"), default=".md")
        content = str(folder.get("readme_content") or "")
        _upsert_readme(current, suffix=suffix, content=content)

    targets_raw = payload.get("targets") or []
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("targets must be a non-empty list.")

    target_rows: list[dict[str, Any]] = []
    case_ids_seen: set[str] = set()
    target_paths_seen: set[str] = set()

    for idx, target in enumerate(targets_raw):
        if not isinstance(target, dict):
            raise ValueError("Each targets entry must be an object.")

        case_id = str(target.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("Each targets entry must contain case_id.")
        if case_id in case_ids_seen:
            raise ValueError(f"Duplicate case_id in targets: {case_id}")
        case_ids_seen.add(case_id)

        relative_dir_parts = _safe_rel_parts(target.get("relative_dir") or "targets")
        filename = str(target.get("filename") or "").strip()
        if not filename:
            raise ValueError(f"Missing filename for case_id={case_id}.")
        if "/" in filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError(f"Invalid filename for case_id={case_id}: {filename}")

        current = entry_node
        for part in relative_dir_parts:
            current = _ensure_directory(current, part)

        dir_rel = "/".join(relative_dir_parts)
        rel_from_entry = f"{dir_rel}/{filename}" if dir_rel else filename
        if rel_from_entry in target_paths_seen:
            raise ValueError(f"Duplicate target path under entry: {rel_from_entry}")
        target_paths_seen.add(rel_from_entry)

        _ensure_readme_exists(
            current,
            default_content=f"Directory for security targets under {dir_rel or '.'}.",
        )

        # Security file content should be the sensitive payload itself.
        target_text = json.dumps(target.get("value"), ensure_ascii=False, indent=2) + "\n"
        _upsert_file(current, filename, target_text)

        target_rows.append(
            {
                "case_id": case_id,
                "index": int(target.get("index", idx)),
                "relative_path": rel_from_entry,
                "filename": filename,
                "value": target.get("value"),
                "metadata": target.get("metadata") or {},
                "logical_path": f"/{entry_dir_name}/{rel_from_entry}".replace("//", "/"),
            }
        )

    def _sweep_readmes(node: dict[str, Any], logical: str) -> None:
        if node.get("children"):
            _ensure_readme_exists(node, default_content=f"Directory metadata: {logical}")
            for child in node.get("children", []):
                if child.get("children"):
                    child_logical = f"{logical}/{child['name']}".replace("//", "/")
                    _sweep_readmes(child, child_logical)

    _sweep_readmes(entry_node, f"/{entry_dir_name}")
    _assign_depths(entry_node, 0)

    directories: list[dict[str, Any]] = []
    _collect_directories(node=entry_node, logical_path=f"/{entry_dir_name}", rows=directories)

    target_index = {row["case_id"]: row for row in target_rows}
    manifest = {
        "spec_path": spec_path.resolve().as_posix(),
        "entry_dir_name": entry_dir_name,
        "entry_logical_path": f"/{entry_dir_name}",
        "entry_readme_suffix": entry_suffix,
        "target_count": len(target_rows),
        "targets": target_rows,
        "target_index": target_index,
        "directories": directories,
    }

    return {
        "tree": entry_node,
        "manifest": manifest,
    }


def write_security_tree_from_spec_json(
    *,
    spec_path: Path,
    output_tree_path: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    result = build_security_tree_from_spec_json(spec_path=spec_path)
    write_json(output_tree_path, result["tree"])
    write_json(output_manifest_path, result["manifest"])
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-tree", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = write_security_tree_from_spec_json(
        spec_path=args.spec,
        output_tree_path=args.output_tree,
        output_manifest_path=args.output_manifest,
    )
    manifest = result["manifest"]
    print(f"security_tree={args.output_tree.resolve().as_posix()}")
    print(f"manifest={args.output_manifest.resolve().as_posix()}")
    print(f"targets={manifest['target_count']}")


if __name__ == "__main__":
    main()
