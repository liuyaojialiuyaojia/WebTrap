#!/usr/bin/env python3
"""Build exp1 user subtree as a single tree JSON payload (not filesystem dirs)."""

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

from file.exp.exp1.tree_exploration import extract_subtree
from file.exp.pipeline_common import to_posix, write_json
from file.utils.file_tree_dataset import FileTree


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def _normalize_suffix(raw_name: str, default: str = ".md") -> str:
    name = str(raw_name or "").strip()
    if "." not in name:
        return default
    ext = "." + name.rsplit(".", 1)[1].strip()
    ext = re.sub(r"[^a-zA-Z0-9._-]", "", ext)
    if ext in {"", "."}:
        return default
    return ext


def _readme_name_for(node_name: str) -> str:
    return f"readme{_normalize_suffix(node_name)}"


def _directory_name_without_suffix(node_name: str) -> str:
    raw = str(node_name or "").strip()
    if "." not in raw:
        return raw
    stem = raw.rsplit(".", 1)[0].strip()
    return stem or raw


def _new_node(name: str, content: Any = "") -> dict[str, Any]:
    return {
        "name": str(name),
        "content": _to_text(content),
        "children": [],
    }


def _assign_depths(node: dict[str, Any], depth: int) -> None:
    node["depth"] = int(depth)
    for child in node.get("children", []):
        _assign_depths(child, depth + 1)


def _append_user_children(
    *,
    src_node: Any,
    dst_parent: dict[str, Any],
    logical_parent: str,
    user_leaf_files: list[str],
    readme_files: set[str],
) -> None:
    if src_node.children:
        dir_name = _directory_name_without_suffix(src_node.name)
        logical_path = f"{logical_parent}/{dir_name}".replace("//", "/")
        dir_node = _new_node(dir_name)
        dst_parent["children"].append(dir_node)

        readme_name = _readme_name_for(src_node.name)
        readme_node = _new_node(readme_name, src_node.content)
        dir_node["children"].append(readme_node)
        readme_files.add(f"{logical_path}/{readme_name}")

        for child in src_node.children:
            _append_user_children(
                src_node=child,
                dst_parent=dir_node,
                logical_parent=logical_path,
                user_leaf_files=user_leaf_files,
                readme_files=readme_files,
            )
        return

    logical_path = f"{logical_parent}/{src_node.name}".replace("//", "/")
    leaf_node = _new_node(src_node.name, src_node.content)
    dst_parent["children"].append(leaf_node)
    user_leaf_files.append(logical_path)


def _walk_manifest(
    *,
    node: dict[str, Any],
    logical_path: str,
    readme_files: set[str],
    rows: list[dict[str, Any]],
) -> None:
    children = node.get("children", [])
    kind = "directory" if children else "file"
    rows.append(
        {
            "logical_path": logical_path,
            "logical_name": str(node.get("name") or ""),
            "kind": kind,
            "generated_readme": logical_path in readme_files,
        }
    )
    for child in children:
        child_name = str(child.get("name") or "")
        child_path = f"{logical_path}/{child_name}".replace("//", "/")
        _walk_manifest(node=child, logical_path=child_path, readme_files=readme_files, rows=rows)


def build_user_tree_json(
    *,
    source_tree_path: Path,
    depth: int,
    width: int,
    seed: int,
    root_coord: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    """Extract user subtree and return a JSON tree payload + manifest.

    Output tree is one JSON object using the same shape as FileTreeNode.to_dict,
    with an added rule: every directory contains one generated readme.* child file.
    """

    full_tree = FileTree.load(str(source_tree_path))
    subtree = extract_subtree(
        tree=full_tree,
        depth=depth,
        width=width,
        seed=seed,
        root_coord=root_coord,
    )

    root_name = _directory_name_without_suffix(subtree.root.name).strip() or "root"
    root = _new_node(root_name)
    user_leaf_files: list[str] = []
    readme_files: set[str] = set()

    root_readme_name = _readme_name_for(subtree.root.name)
    root_readme = _new_node(root_readme_name, subtree.root.content)
    root["children"].append(root_readme)
    readme_files.add(f"/{root_name}/{root_readme_name}".replace("//", "/"))

    for child in subtree.root.children:
        _append_user_children(
            src_node=child,
            dst_parent=root,
            logical_parent=f"/{root_name}".replace("//", "/"),
            user_leaf_files=user_leaf_files,
            readme_files=readme_files,
        )

    _assign_depths(root, 0)

    manifest_rows: list[dict[str, Any]] = []
    _walk_manifest(
        node=root,
        logical_path=f"/{root_name}".replace("//", "/"),
        readme_files=readme_files,
        rows=manifest_rows,
    )

    directories = [row["logical_path"] for row in manifest_rows if row["kind"] == "directory"]
    files = [row["logical_path"] for row in manifest_rows if row["kind"] == "file"]

    manifest = {
        "source_tree_path": to_posix(source_tree_path.resolve()),
        "tree_params": {
            "depth": depth,
            "width": width,
            "seed": seed,
            "root_coord": list(root_coord) if root_coord else None,
        },
        "nodes": manifest_rows,
        "directory_count": len(directories),
        "file_count": len(files),
        "user_leaf_files": user_leaf_files,
        "generated_readme_files": sorted(readme_files),
        "directories": directories,
        "all_leaf_files": files,
    }

    return {
        "tree": root,
        "manifest": manifest,
    }


def write_user_tree_json(
    *,
    source_tree_path: Path,
    output_tree_path: Path,
    output_manifest_path: Path,
    depth: int,
    width: int,
    seed: int,
    root_coord: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    result = build_user_tree_json(
        source_tree_path=source_tree_path,
        depth=depth,
        width=width,
        seed=seed,
        root_coord=root_coord,
    )
    write_json(output_tree_path, result["tree"])
    write_json(output_manifest_path, result["manifest"])
    return result


def parse_root_coord(raw: Optional[str]) -> Optional[tuple[int, int]]:
    if raw is None:
        return None
    if "," not in raw:
        raise ValueError("--root-coord must be in the form depth,index")
    left, right = raw.split(",", 1)
    return int(left.strip()), int(right.strip())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-tree",
        type=Path,
        default=Path("file/create/file_tree/custom_d11_w2_r_1.json"),
    )
    parser.add_argument("--output-tree", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-coord", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root_coord = parse_root_coord(args.root_coord)
    result = write_user_tree_json(
        source_tree_path=args.source_tree,
        output_tree_path=args.output_tree,
        output_manifest_path=args.output_manifest,
        depth=args.depth,
        width=args.width,
        seed=args.seed,
        root_coord=root_coord,
    )
    manifest = result["manifest"]
    print(f"user_tree={args.output_tree.resolve().as_posix()}")
    print(f"manifest={args.output_manifest.resolve().as_posix()}")
    print(f"directories={manifest['directory_count']}")
    print(f"files={manifest['file_count']}")


if __name__ == "__main__":
    main()
