#!/usr/bin/env python3
"""Extract an experiment-specific subtree from a persisted seed tree."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    from .build_tree_design import (
        NodeRecord,
        make_leaf_marker,
        make_site_manifest,
        write_site_manifest,
        write_tree_bundle,
    )
except ImportError:  # Allows running the script directly (python path/to/script.py)
    from build_tree_design import (
        NodeRecord,
        make_leaf_marker,
        make_site_manifest,
        write_site_manifest,
        write_tree_bundle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Directory containing the seed tree bundle (outputs of build_seed_tree.py).",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default="/",
        help="Path of the subtree root in seed coordinates (e.g. / or /0/1).",
    )
    parser.add_argument("--depth", type=int, required=True, help="Desired subtree depth (number of levels, >=1).")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Maximum branching factor to retain per node (>=1). Omit to keep all children.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=2024,
        help="Seed controlling child sampling when width truncation applies.",
    )
    parser.add_argument(
        "--leaf-policy",
        choices=("inherit", "recompute", "auto"),
        default="auto",
        help=(
            "How to assign leaf markers: inherit existing values, recompute for all leaves, "
            "or auto (inherit when present, otherwise recompute)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("web/runs/trap_site_mvp"),
        help="Base directory for experiment-specific outputs.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional subdirectory name under --output-root for the extracted subtree.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing contents in the output directory if present.",
    )
    return parser.parse_args()


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prompts(path: Path) -> Dict[int, Mapping[str, object]]:
    if not path.exists():
        return {}
    mapping: Dict[int, Mapping[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        index = int(payload.get("page_index", len(mapping)))
        mapping[index] = payload
    return mapping


def _parse_path(raw: str) -> Tuple[int, ...]:
    cleaned = raw.strip()
    if not cleaned or cleaned == "/":
        return ()
    if cleaned.startswith("/"):
        cleaned = cleaned[1:]
    if cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    if not cleaned:
        return ()
    parts = cleaned.split("/")
    try:
        return tuple(int(part) for part in parts if part)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid root-path component in '{raw}'.") from exc


def _compute_max_depth(index: int, nodes: Mapping[int, Mapping[str, object]], cache: Dict[int, int]) -> int:
    if index in cache:
        return cache[index]
    children = nodes[index].get("children", []) or []
    if not children:
        cache[index] = 0
        return 0
    depth = 1 + max(_compute_max_depth(int(child), nodes, cache) for child in children)
    cache[index] = depth
    return depth


def _select_subtree(
    root_index: int,
    *,
    depth: int,
    width: int | None,
    nodes: Mapping[int, Mapping[str, object]],
    max_depth_cache: Dict[int, int],
    rng: random.Random,
) -> Tuple[List[int], Mapping[int, List[int]]]:
    if depth < 1:
        raise ValueError("Depth must be >= 1.")

    selected: List[int] = []
    child_map: Dict[int, List[int]] = {}

    def visit(node_index: int, remaining_depth: int) -> None:
        if node_index not in selected:
            selected.append(node_index)
        if remaining_depth <= 0:
            return

        children: List[int] = [int(child) for child in nodes[node_index].get("children", []) or []]
        if not children:
            raise ValueError(
                f"Node {node_index} (path={nodes[node_index].get('path')}) cannot satisfy depth={depth}: missing children."
            )

        if width is None:
            if remaining_depth == 1:
                chosen = list(children)
            else:
                eligible = [child for child in children if _compute_max_depth(child, nodes, max_depth_cache) >= remaining_depth - 1]
                if not eligible:
                    raise ValueError(
                        f"Node {node_index} lacks descendants to reach depth={depth} (path={nodes[node_index].get('path')})."
                    )
                chosen = eligible
        else:
            if width < 1:
                raise ValueError("Width must be >= 1 when provided.")
            if remaining_depth == 1:
                if len(children) < width:
                    raise ValueError(
                        f"Node {node_index} requires >= {width} children for depth={depth}, but has {len(children)}."
                    )
                chosen = list(children)
            else:
                eligible = [child for child in children if _compute_max_depth(child, nodes, max_depth_cache) >= remaining_depth - 1]
                if len(eligible) < width:
                    raise ValueError(
                        f"Node {node_index} lacks {width} children with sufficient depth (path={nodes[node_index].get('path')})."
                    )
                chosen = eligible
            if len(chosen) > width:
                rng.shuffle(chosen)
                chosen = chosen[:width]
            if len(chosen) < width:
                raise ValueError(
                    f"Node {node_index} retained only {len(chosen)} children < required width {width}."
                )

        child_map[node_index] = chosen
        for child in chosen:
            visit(child, remaining_depth - 1)

    visit(root_index, depth - 1)
    return selected, child_map


def _bfs_order(root_index: int, child_map: Mapping[int, Sequence[int]]) -> List[int]:
    order: List[int] = []
    queue: deque[int] = deque([root_index])
    seen: set[int] = set()
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for child in child_map.get(node, []):
            queue.append(child)
    return order


def _relative_path(source_path: Sequence[int], root_path: Sequence[int]) -> Tuple[int, ...]:
    prefix_len = len(root_path)
    if prefix_len == 0:
        return tuple(source_path)
    return tuple(source_path[prefix_len:])


def _filter_actions(
    actions: Iterable[object],
    action_pages: Iterable[int],
    index_map: Mapping[int, int],
) -> Tuple[List[object], List[int]]:
    filtered_actions: List[object] = []
    filtered_pages: List[int] = []
    for action, page_idx in zip(actions, action_pages):
        new_idx = index_map.get(int(page_idx))
        if new_idx is None:
            continue
        filtered_actions.append(copy.deepcopy(action))
        filtered_pages.append(new_idx)
    return filtered_actions, filtered_pages


def _assign_leaf_marker(
    *,
    policy: str,
    original_marker: str | None,
    base_seed: int,
    source_path: Sequence[int],
) -> str | None:
    if policy == "inherit":
        return original_marker
    if policy == "recompute":
        return make_leaf_marker(base_seed, source_path)
    # auto policy
    return original_marker or make_leaf_marker(base_seed, source_path)


def _prepare_output_dir(base: Path, experiment_id: str | None, *, force: bool) -> Path:
    output_dir = base if experiment_id is None else base / experiment_id
    if output_dir.exists() and not force:
        existing = list(output_dir.iterdir()) if output_dir.is_dir() else []
        if existing:
            raise SystemExit(
                f"Output directory {output_dir} already exists with contents. Re-run with --force to overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _derive_default_experiment_id(
    *,
    source_root: Path,
    root_path: Tuple[int, ...],
    depth: int,
    width: int | None,
    selection_seed: int,
) -> str:
    root_label = "root" if not root_path else "root-" + "-".join(str(p) for p in root_path)
    width_label = f"w{width}" if width is not None else "wall"
    return f"{source_root.name}_{root_label}_d{depth}_{width_label}_sel{selection_seed}"


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    manifest_path = source_root / "site_manifest.json"
    tree_path = source_root / "tree.json"
    metadata_path = source_root / "page_metadata.json"
    prompts_path = source_root / "prompts" / "page_prompt.jsonl"

    source_manifest = _load_json(manifest_path)
    tree_payload = _load_json(tree_path)
    metadata_payload = _load_json(metadata_path)
    prompts_mapping = _load_prompts(prompts_path)

    tree_nodes = tree_payload.get("nodes", []) if isinstance(tree_payload, dict) else []
    if not tree_nodes:
        raise ValueError(f"Tree payload at {tree_path} is empty.")

    nodes_by_index: Dict[int, Mapping[str, object]] = {
        int(node["page_index"]): node for node in tree_nodes if isinstance(node, dict)
    }
    path_to_index: Dict[Tuple[int, ...], int] = {}
    for node in tree_nodes:
        if not isinstance(node, dict):
            continue
        path_tuple = tuple(int(item) for item in node.get("path", []) or [])
        path_to_index[path_tuple] = int(node["page_index"])

    root_path_tuple = _parse_path(args.root_path)
    try:
        root_index = path_to_index[root_path_tuple]
    except KeyError as exc:
        raise ValueError(f"Root path {args.root_path} not found in seed tree.") from exc

    max_depth_cache: Dict[int, int] = {}
    root_max_depth = _compute_max_depth(root_index, nodes_by_index, max_depth_cache)
    if args.depth - 1 > root_max_depth:
        raise ValueError(
            f"Requested depth {args.depth} exceeds available depth {root_max_depth + 1} from root {args.root_path}."
        )

    rng = random.Random(args.selection_seed)
    selected_indices, child_map = _select_subtree(
        root_index,
        depth=args.depth,
        width=args.width,
        nodes=nodes_by_index,
        max_depth_cache=max_depth_cache,
        rng=rng,
    )

    order = _bfs_order(root_index, child_map)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}

    metadata_pages = metadata_payload.get("pages", []) if isinstance(metadata_payload, dict) else []
    metadata_by_index: Dict[int, Mapping[str, object]] = {}
    for page in metadata_pages:
        if not isinstance(page, dict):
            continue
        metadata_by_index[int(page.get("page_index", -1))] = page

    base_seed = int(source_manifest.get("base_seed", 0))

    records_by_old: Dict[int, NodeRecord] = {}
    new_nodes: List[NodeRecord] = []

    for old_index in order:
        source_node = nodes_by_index[old_index]
        source_path = tuple(int(item) for item in source_node.get("path", []) or [])
        relative_path = _relative_path(source_path, root_path_tuple)
        parent_old = source_node.get("parent_index")
        parent_new = index_map.get(int(parent_old)) if parent_old is not None else None

        page_metadata = copy.deepcopy(metadata_by_index.get(old_index, {}))
        original_actions = page_metadata.get("action", []) or []
        original_action_pages = page_metadata.get("action_page", []) or []
        filtered_actions, filtered_pages = _filter_actions(original_actions, original_action_pages, index_map)

        page_metadata["page_index"] = index_map[old_index]
        page_metadata.setdefault("page_id", source_node.get("page_id"))
        page_metadata["path"] = list(relative_path)
        page_metadata["source_path"] = list(source_path)
        page_metadata["source_page_index"] = old_index
        page_metadata["source_breadcrumb"] = page_metadata.get("breadcrumb")
        page_metadata["action"] = filtered_actions
        page_metadata["action_page"] = filtered_pages
        page_metadata["click_targets"] = []
        page_metadata["leaf_marker"] = None

        prompt_payload = copy.deepcopy(prompts_mapping.get(old_index, {}))
        original_prompt_breadcrumb = copy.deepcopy(prompt_payload.get("breadcrumb"))
        prompt_payload["page_index"] = index_map[old_index]
        prompt_payload["breadcrumb"] = list(relative_path)
        prompt_payload["source_page_index"] = old_index
        prompt_payload["source_breadcrumb"] = original_prompt_breadcrumb if original_prompt_breadcrumb is not None else []
        prompt_payload["source_path"] = list(source_path)

        record = NodeRecord(
            page_index=index_map[old_index],
            page_id=str(page_metadata.get("page_id", source_node.get("page_id", f"node-{old_index}"))),
            path=tuple(relative_path),
            parent_index=parent_new,
            children=[],
            page_data=page_metadata,
            prompt_payload=prompt_payload,
        )
        new_nodes.append(record)
        records_by_old[old_index] = record

    for parent_old, children in child_map.items():
        parent_record = records_by_old[parent_old]
        parent_record.children = [index_map[child] for child in children]

    for old_index, record in records_by_old.items():
        click_targets: List[Dict[str, object]] = []
        for offset, child_old in enumerate(child_map.get(old_index, [])):
            child_record = records_by_old[child_old]
            click_targets.append(
                {
                    "element_id": f"trap_link_p{record.page_index}_c{offset}",
                    "label": child_record.page_data.get("title"),
                    "blurb": child_record.page_data.get("summary"),
                    "target_page": child_record.page_index,
                    "target_breadcrumb": child_record.page_data.get("breadcrumb"),
                }
            )

        parent_old = nodes_by_index[old_index].get("parent_index")
        if parent_old is not None and parent_old in records_by_old:
            parent_record = records_by_old[parent_old]
            click_targets.insert(
                0,
                {
                    "element_id": f"trap_return_p{record.page_index}",
                    "label": f"Back to parent: {parent_record.page_data.get('title')}",
                    "blurb": parent_record.page_data.get("summary"),
                    "target_page": parent_record.page_index,
                    "target_breadcrumb": parent_record.page_data.get("breadcrumb"),
                },
            )

        record.page_data["click_targets"] = click_targets

    for old_index, record in records_by_old.items():
        if record.children:
            record.page_data["leaf_marker"] = None
            continue
        original_marker = metadata_by_index.get(old_index, {}).get("leaf_marker")
        source_path_seq = tuple(int(item) for item in record.page_data.get("source_path", []))
        record.page_data["leaf_marker"] = _assign_leaf_marker(
            policy=args.leaf_policy,
            original_marker=original_marker,
            base_seed=base_seed,
            source_path=source_path_seq,
        )

    experiment_id = args.experiment_id or _derive_default_experiment_id(
        source_root=source_root,
        root_path=root_path_tuple,
        depth=args.depth,
        width=args.width,
        selection_seed=args.selection_seed,
    )
    output_dir = _prepare_output_dir(args.output_root, experiment_id, force=args.force)

    write_tree_bundle(output_dir, new_nodes)

    level_map: Dict[int, List[int]] = {}
    for old_index in order:
        level = len(nodes_by_index[old_index].get("path", [])) - len(root_path_tuple)
        level_map.setdefault(level, []).append(old_index)

    # Manifest metadata
    child_widths = [len(child_map.get(node, [])) for node in order]
    effective_width = args.width if args.width is not None else (max(child_widths) if child_widths else 0)
    manifest_width = effective_width if effective_width > 0 else 1
    site_manifest = make_site_manifest(
        depth=args.depth,
        width=manifest_width,
        base_seed=base_seed,
        model=str(source_manifest.get("model", "unknown")),
        max_tokens=_coerce_int(source_manifest.get("max_tokens"), 0),
        temperature=_coerce_float(source_manifest.get("temperature"), 0.0),
        top_p=source_manifest.get("top_p"),
        output_root=output_dir,
        num_pages=len(new_nodes),
        extra={
            "kind": "extracted_subtree",
            "source_tree_root": str(source_root),
            "source_manifest": str(manifest_path),
            "source_tree_id": source_manifest.get("tree_id"),
            "root_path": args.root_path,
            "root_path_tuple": list(root_path_tuple),
            "requested_depth": args.depth,
            "requested_width": args.width,
            "selection_seed": args.selection_seed,
            "leaf_policy": args.leaf_policy,
            "experiment_id": experiment_id,
            "actual_depth": len(level_map),
            "actual_width_observed": effective_width,
        },
    )
    write_site_manifest(output_dir, site_manifest)

    subtree_details = {
        "node_mapping": [
            {
                "original_index": old_index,
                "new_index": index_map[old_index],
                "source_path": nodes_by_index[old_index].get("path", []),
                "relative_path": list(records_by_old[old_index].path),
                "page_id": records_by_old[old_index].page_id,
            }
            for old_index in order
        ],
        "level_widths": {
            str(level): [len(child_map.get(node, [])) for node in nodes_at_level]
            for level, nodes_at_level in sorted(level_map.items())
        },
        "requested_depth": args.depth,
        "requested_width": args.width,
        "selection_seed": args.selection_seed,
        "leaf_policy": args.leaf_policy,
        "source_manifest": str(manifest_path),
    }
    write_site_manifest(output_dir, subtree_details, filename="subtree_manifest.json")

    print(f"Extracted subtree written to {output_dir}")
    print(f"Pages: {len(new_nodes)}; depth={args.depth}; width={'all' if args.width is None else args.width}")


if __name__ == "__main__":
    main()
