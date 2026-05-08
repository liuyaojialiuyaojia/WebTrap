#!/usr/bin/env python3
"""Merge security subtree JSON into user tree JSON at an anchor index path."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.pipeline_common import write_json


def _is_directory(node: dict[str, Any]) -> bool:
    return bool(node.get("children"))


def _find_child(parent: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
    for child in parent.get("children", []):
        if str(child.get("name") or "") == name:
            return child
    return None


def _parse_anchor_index_path(anchor: str) -> list[int]:
    raw = str(anchor or "").strip()
    if raw == "/":
        return []
    if not raw.startswith("/"):
        raise ValueError(
            "ANCHOR_LOGICAL_PATH must use index format like /0/1 (root is /)."
        )
    if raw.endswith("/"):
        raise ValueError(
            "ANCHOR_LOGICAL_PATH must use index format like /0/1 (no trailing /)."
        )

    tokens = raw.split("/")[1:]
    if not tokens:
        return []

    indexes: list[int] = []
    for token in tokens:
        if token == "":
            raise ValueError("ANCHOR_LOGICAL_PATH cannot contain empty segment.")
        if not token.isdigit():
            raise ValueError("ANCHOR_LOGICAL_PATH segments must be digits, e.g. /0/1.")
        indexes.append(int(token))
    return indexes


def _resolve_anchor_by_index_path(
    root: dict[str, Any],
    anchor_index_path: str,
) -> tuple[dict[str, Any], str, str]:
    root_name = str(root.get("name") or "").strip()
    if not root_name:
        raise ValueError("User tree root node must have a non-empty name.")

    indexes = _parse_anchor_index_path(anchor_index_path)
    node = root
    logical_parts = [root_name]

    for index in indexes:
        children = node.get("children") or []
        if index < 0 or index >= len(children):
            raise ValueError(
                f"ANCHOR_LOGICAL_PATH index out of range: {anchor_index_path} "
                f"(index={index}, children={len(children)})."
            )
        node = children[index]
        name = str(node.get("name") or "").strip()
        if not name:
            raise ValueError(f"Unnamed node encountered while resolving {anchor_index_path}.")
        logical_parts.append(name)

    if not _is_directory(node):
        logical_path = "/" + "/".join(logical_parts)
        raise ValueError(
            f"ANCHOR_LOGICAL_PATH resolved to a non-directory node: {anchor_index_path} -> {logical_path}"
        )

    normalized_anchor = "/" if not indexes else "/" + "/".join(str(i) for i in indexes)
    resolved_logical = "/" + "/".join(logical_parts)
    return node, normalized_anchor, resolved_logical


def _assign_depths(node: dict[str, Any], depth: int) -> None:
    node["depth"] = int(depth)
    for child in node.get("children", []):
        _assign_depths(child, depth + 1)


def merge_security_tree_json(
    *,
    user_tree: dict[str, Any],
    security_tree: dict[str, Any],
    security_manifest: dict[str, Any],
    anchor_index_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    merged_root = copy.deepcopy(user_tree)
    entry_node = copy.deepcopy(security_tree)

    if not isinstance(merged_root, dict) or not isinstance(entry_node, dict):
        raise ValueError("user_tree and security_tree must be JSON objects.")

    anchor_node, anchor_index_norm, anchor_logical_path = _resolve_anchor_by_index_path(
        merged_root, anchor_index_path
    )

    entry_name = str(entry_node.get("name") or "").strip()
    if not entry_name:
        raise ValueError("security_tree root node must have a non-empty name.")

    existing = _find_child(anchor_node, entry_name)
    if existing is not None:
        if not overwrite:
            raise FileExistsError(
                f"Security entry already exists under anchor {anchor_logical_path}: {entry_name}"
            )
        anchor_node["children"] = [
            child for child in anchor_node.get("children", []) if str(child.get("name") or "") != entry_name
        ]

    anchor_node.setdefault("children", []).append(entry_node)
    _assign_depths(merged_root, 0)

    entry_logical_path = f"{anchor_logical_path}/{entry_name}".replace("//", "/")

    targets_raw = security_manifest.get("targets") or []
    merged_targets: list[dict[str, Any]] = []
    target_index: dict[str, dict[str, Any]] = {}
    for target in targets_raw:
        if not isinstance(target, dict):
            continue
        row = dict(target)
        case_id = str(row.get("case_id") or "").strip()
        relative_path = str(row.get("relative_path") or "").strip()
        if not relative_path:
            continue
        logical_path = f"{entry_logical_path}/{relative_path}".replace("//", "/")
        row["logical_path"] = logical_path
        merged_targets.append(row)
        if case_id:
            target_index[case_id] = row

    merged_manifest = {
        "anchor_index_path": anchor_index_norm,
        "anchor_logical_path": anchor_logical_path,
        "entry_dir_name": entry_name,
        "entry_logical_path": entry_logical_path,
        "target_count": len(merged_targets),
        "targets": merged_targets,
        "target_index": target_index,
    }

    return {
        "tree": merged_root,
        "manifest": merged_manifest,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-tree", type=Path, required=True)
    parser.add_argument("--security-tree", type=Path, required=True)
    parser.add_argument("--security-manifest", type=Path, required=True)
    parser.add_argument("--anchor-index-path", type=str, default="\\")
    parser.add_argument("--output-tree", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    user_tree = json.loads(args.user_tree.read_text(encoding="utf-8"))
    security_tree = json.loads(args.security_tree.read_text(encoding="utf-8"))
    security_manifest = json.loads(args.security_manifest.read_text(encoding="utf-8"))

    result = merge_security_tree_json(
        user_tree=user_tree,
        security_tree=security_tree,
        security_manifest=security_manifest,
        anchor_index_path=args.anchor_index_path,
        overwrite=bool(args.overwrite),
    )

    write_json(args.output_tree, result["tree"])
    write_json(args.output_manifest, result["manifest"])

    manifest = result["manifest"]
    print(f"merged_tree={args.output_tree.resolve().as_posix()}")
    print(f"manifest={args.output_manifest.resolve().as_posix()}")
    print(f"anchor_index={manifest['anchor_index_path']}")
    print(f"entry={manifest['entry_logical_path']}")
    print(f"targets={manifest['target_count']}")


if __name__ == "__main__":
    main()
