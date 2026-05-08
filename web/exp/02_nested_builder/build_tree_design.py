#!/usr/bin/env python3
"""Build a branching website design by reusing stage-01 node constructors."""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.util
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from datetime import datetime

from experiment_paths import resolve_experiment_root


class _ProgressPrinter:
    """Lightweight stderr progress display for tree construction."""

    def __init__(self, total: int, *, enabled: bool) -> None:
        self._total = max(total, 0)
        self._enabled = enabled and self._total > 0
        self._last_len = 0

    def update(self, current: int) -> None:
        if not self._enabled:
            return
        pct = (current / self._total) * 100 if self._total else 100.0
        message = f"Building tree nodes: {current}/{self._total} ({pct:5.1f}%)"
        padding = " " * max(0, self._last_len - len(message))
        sys.stderr.write("\r" + message + padding)
        sys.stderr.flush()
        self._last_len = len(message)

    def finish(self) -> None:
        if not self._enabled:
            return
        sys.stderr.write("\n")
        sys.stderr.flush()


def _expected_node_count(depth: int, width: int) -> int:
    if depth <= 0 or width <= 0:
        return 0
    if width == 1:
        return depth
    total = 0
    for level in range(depth):
        total += width**level
    return total


def _report_duplicate_titles(nodes: Sequence[NodeRecord]) -> None:
    titles = [str(node.page_data.get("title", "")).strip() for node in nodes]
    counts = Counter(titles)
    duplicates = [(title, count) for title, count in counts.items() if title and count > 1]
    if not duplicates:
        # Handle empty-title duplicates separately to avoid blank line spam
        empty_count = counts.get("", 0)
        if empty_count > 1:
            sys.stderr.write(
                f"[Stage2] Detected {empty_count} nodes with empty titles (treated as duplicates).\n"
            )
        return

    duplicates.sort(key=lambda item: item[1], reverse=True)
    total_duplicates = sum(count for _, count in duplicates)
    sys.stderr.write(
        f"[Stage2] Detected {len(duplicates)} duplicate titles covering {total_duplicates} nodes.\n"
    )
    preview = duplicates[:10]
    for title, count in preview:
        sys.stderr.write(f"  - {count} × {title}\n")
    remaining = len(duplicates) - len(preview)
    if remaining > 0:
        sys.stderr.write(f"  ... {remaining} more duplicate titles not shown.\n")
    empty_count = counts.get("", 0)
    if empty_count > 1:
        sys.stderr.write(
            f"  - {empty_count} × (empty title)\n"
        )


WEB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _load_stage1_module():
    stage_dir = Path(__file__).resolve().parents[1] / "01_node_constructor"
    spec = importlib.util.spec_from_file_location(
        "trap_site_stage01", stage_dir / "__init__.py", submodule_search_locations=[str(stage_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate stage 01 node constructor module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_NODE_CONSTRUCTOR = _load_stage1_module()
PageRequest = _NODE_CONSTRUCTOR.PageRequest  # type: ignore[attr-defined]
SiblingInfo = _NODE_CONSTRUCTOR.SiblingInfo  # type: ignore[attr-defined]
PageMetadata = _NODE_CONSTRUCTOR.PageMetadata  # type: ignore[attr-defined]
build_page = _NODE_CONSTRUCTOR.build_page  # type: ignore[attr-defined]
format_breadcrumb = _NODE_CONSTRUCTOR.format_breadcrumb  # type: ignore[attr-defined]


@dataclass
class NodeRecord:
    page_index: int
    page_id: str
    path: Tuple[int, ...]
    parent_index: int | None
    children: List[int] = field(default_factory=list)
    page_data: Dict[str, object] = field(default_factory=dict)
    prompt_payload: Dict[str, object] = field(default_factory=dict)


def _make_page_id(path: Sequence[int]) -> str:
    return "root" if not path else "node-" + "-".join(str(item) for item in path)


def _derive_seed(base_seed: int, path: Sequence[int]) -> int:
    token = "|".join(str(item) for item in path)
    digest = hashlib.sha256(f"{base_seed}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _make_leaf_marker(base_seed: int, path: Sequence[int]) -> str:
    token = "/".join(str(item) for item in path)
    digest = hashlib.sha256(f"leaf:{base_seed}:{token}".encode("utf-8")).hexdigest()
    return f"LEAF_MARKER:{digest[:32]}"


def make_leaf_marker(base_seed: int, path: Sequence[int]) -> str:
    """Public wrapper for computing stable leaf markers."""

    return _make_leaf_marker(base_seed, path)


def _stage1_metadata_to_page_data(metadata, path: Sequence[int]) -> Dict[str, object]:
    filtered_actions: List[dict] = []
    filtered_pages: List[int] = []
    for concept, page_idx in zip(metadata.action, metadata.action_page):
        concept_str = str(getattr(concept, "concept", concept)).strip()
        if concept_str.lower() in {"", "#none#", "none"}:
            continue
        if hasattr(concept, "to_dict"):
            filtered_actions.append(concept.to_dict())
        elif isinstance(concept, dict):
            payload = dict(concept)
            payload.setdefault("concept", concept_str)
            filtered_actions.append(payload)
        else:
            filtered_actions.append({"concept": concept_str})
        filtered_pages.append(int(page_idx))

    return {
        "page_index": metadata.page_index,
        "page_id": metadata.page_id,
        "path": list(path),
        "breadcrumb": metadata.breadcrumb,
        "title": metadata.title,
        "summary": metadata.summary,
        "body": metadata.body,
        "action": filtered_actions,
        "action_page": filtered_pages,
        "click_targets": [],
        "leaf_marker": None,
        "injections": [],
    }


def _fallback_build_page(request: PageRequest, error: Exception) -> PageMetadata:
    seed = request.seed if request.seed is not None else _derive_seed(0, request.breadcrumb)
    rng = random.Random(seed)
    breadcrumb_str = format_breadcrumb(request.breadcrumb)
    mode = request.gen_mode or "direct"
    scope = "root" if not request.breadcrumb else breadcrumb_str
    sibling_titles = ", ".join(sib.title for sib in request.siblings) or "none"
    title = f"{mode.title()} page {scope}"
    summary = f"{mode} summary for {scope} (siblings: {sibling_titles})."
    body = (
        f"Offline fallback content generated for {scope}. "
        f"Parent summary: {request.parent_summary or 'N/A'}. "
        f"Original error: {error.__class__.__name__}."
    )
    primitive_pool = ["header", "carousel", "deck", "dealmedia", "footer1", "cart", "submit"]
    actions = ["header"]
    actions.extend(rng.sample(primitive_pool[1:], k=2))
    action_tuple = tuple(actions)
    action_page_tuple = tuple(request.page_index for _ in action_tuple)
    return PageMetadata(
        page_id=request.page_id,
        page_index=request.page_index,
        breadcrumb=breadcrumb_str,
        title=title,
        summary=summary,
        body=body,
        action=action_tuple,
        action_page=action_page_tuple,
    )


def _serialize_prompt(request: PageRequest) -> Dict[str, object]:
    siblings = [
        {"title": sibling.title, "summary": sibling.summary}
        for sibling in request.siblings
    ]
    return {
        "page_id": request.page_id,
        "page_index": request.page_index,
        "breadcrumb": list(request.breadcrumb),
        "gen_mode": request.gen_mode,
        "title_hint": request.title_hint,
        "title_path": list(request.title_path),
        "parent_summary": request.parent_summary,
        "siblings": siblings,
        "batch_size": request.batch_size,
        "seed": request.seed,
    }


def _register_click_target(parent: NodeRecord, child: NodeRecord, child_offset: int) -> None:
    parent.page_data.setdefault("click_targets", [])
    element_id = f"trap_link_p{parent.page_index}_c{child_offset}"
    target_entry = {
        "element_id": element_id,
        "label": child.page_data["title"],
        "blurb": child.page_data["summary"],
        "target_page": child.page_index,
        "target_breadcrumb": child.page_data["breadcrumb"],
    }
    parent.page_data["click_targets"].append(target_entry)


def _register_parent_return(child: NodeRecord, parent: NodeRecord) -> None:
    child.page_data.setdefault("click_targets", [])
    element_id = f"trap_return_p{child.page_index}"
    if any(target.get("element_id") == element_id for target in child.page_data["click_targets"]):
        return
    child.page_data["click_targets"].insert(
        0,
        {
            "element_id": element_id,
            "label": f"Back to parent: {parent.page_data['title']}",
            "blurb": parent.page_data["summary"],
            "target_page": parent.page_index,
            "target_breadcrumb": parent.page_data["breadcrumb"],
        },
    )


def _build_tree(
    depth: int,
    width: int,
    base_seed: int,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    progress: bool,
) -> List[NodeRecord]:
    if depth < 1:
        raise ValueError("depth must be >= 1.")
    if width < 1:
        raise ValueError("width must be >= 1.")

    nodes: List[NodeRecord] = []
    expected_nodes = _expected_node_count(depth, width)
    progress_printer = _ProgressPrinter(expected_nodes, enabled=progress)

    def _append_node(
        parent_index: int | None,
        path: Tuple[int, ...],
        gen_mode: str,
        siblings: Sequence[NodeRecord],
    ) -> NodeRecord:
        page_index = len(nodes)
        page_id = _make_page_id(path)
        title_path: Tuple[str, ...]
        parent_summary: str | None
        if parent_index is None:
            title_path = ()
            parent_summary = None
        else:
            titles: List[str] = []
            current = parent_index
            while current is not None:
                ancestor = nodes[current]
                titles.append(str(ancestor.page_data["title"]))
                current = ancestor.parent_index
            title_path = tuple(reversed(titles))
            parent_summary = str(nodes[parent_index].page_data["summary"])

        siblings_info = tuple(
            SiblingInfo(title=str(sib.page_data["title"]), summary=str(sib.page_data["summary"]))
            for sib in siblings
        )
        request = PageRequest(
            page_id=page_id,
            page_index=page_index,
            breadcrumb=path,
            gen_mode=gen_mode,
            title_hint=None,
            title_path=title_path,
            parent_summary=parent_summary,
            siblings=siblings_info,
            batch_size=1,
            seed=_derive_seed(base_seed, path),
        )
        try:
            stage_metadata = build_page(
                request,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=request.seed,
            )
        except Exception as exc:  # noqa: BLE001
            stage_metadata = _fallback_build_page(request, exc)

        page_data = _stage1_metadata_to_page_data(stage_metadata, path)
        node = NodeRecord(
            page_index=page_index,
            page_id=page_id,
            path=path,
            parent_index=parent_index,
            page_data=page_data,
            prompt_payload=_serialize_prompt(request),
        )
        nodes.append(node)
        progress_printer.update(len(nodes))
        if parent_index is not None:
            parent_node = nodes[parent_index]
            parent_node.children.append(page_index)
            _register_click_target(parent_node, node, child_offset=len(parent_node.children) - 1)
            _register_parent_return(node, parent_node)
        return node

    root = _append_node(parent_index=None, path=(), gen_mode="direct", siblings=())
    queue: List[int] = [root.page_index]

    while queue:
        current_index = queue.pop(0)
        current_node = nodes[current_index]
        current_depth = len(current_node.path)
        if current_depth >= depth - 1:
            continue

        if not current_node.children:
            child_path = current_node.path + (0,)
            child_node = _append_node(
                parent_index=current_index,
                path=child_path,
                gen_mode="extend_depth",
                siblings=(),
            )
            queue.append(child_node.page_index)

        while len(current_node.children) < width:
            child_position = len(current_node.children)
            child_path = current_node.path + (child_position,)
            existing_siblings = [nodes[idx] for idx in current_node.children]
            child_node = _append_node(
                parent_index=current_index,
                path=child_path,
                gen_mode="extend_breadth",
                siblings=existing_siblings,
            )
            queue.append(child_node.page_index)

    for node in nodes:
        if not node.children:
            node.page_data["leaf_marker"] = _make_leaf_marker(base_seed, node.path)

    progress_printer.finish()
    return nodes


def _ensure_output_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "prompts": root / "prompts",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_prompts(path: Path, nodes: Iterable[NodeRecord]) -> None:
    lines = [json.dumps(node.prompt_payload, ensure_ascii=False) for node in nodes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _assemble_design(nodes: Sequence[NodeRecord]) -> dict:
    action: List[dict] = []
    action_page: List[int] = []
    for node in nodes:
        page_actions = node.page_data.get("action", [])
        page_indices = node.page_data.get("action_page", [])
        for concept, page_idx in zip(page_actions, page_indices):
            if hasattr(concept, "to_dict"):
                payload = concept.to_dict()
            elif isinstance(concept, dict):
                payload = dict(concept)
            else:
                payload = {"concept": str(concept)}
            action.append(payload)
            action_page.append(int(page_idx))
    return {
        "number_of_pages": len(nodes),
        "action": action,
        "action_page": action_page,
    }


def _assemble_transitions(nodes: Sequence[NodeRecord]) -> dict:
    actions: List[dict] = []
    for node in nodes:
        click_targets = node.page_data.get("click_targets") or []
        forward_targets = {child_index for child_index in node.children}
        matched_forward = set()
        for click_entry in click_targets:
            target_page = click_entry.get("target_page")
            element_id = click_entry.get("element_id")
            if target_page is None or element_id is None:
                continue
            if target_page in forward_targets:
                matched_forward.add(target_page)
            actions.append(
                {
                    "type": "addOpenPageTransition",
                    "is_transition": True,
                    "source_page": node.page_index,
                    "target_page": target_page,
                    "source_group": element_id,
                    "target_group": f"page{target_page}",
                    "controls": {
                        "eventType": "click",
                        "preconditionVisited": [],
                        "shouldSubmitOnFinalPage": False,
                        "conceptual": False,
                        "taskSuccessScale": 1.0,
                    },
                }
            )
        missing_forward = forward_targets - matched_forward
        if missing_forward:
            missing_str = ", ".join(str(idx) for idx in sorted(missing_forward))
            raise RuntimeError(
                f"Page {node.page_index} missing click targets for children: {missing_str}"
            )
    return {"actions": actions}


def _collect_metadata(nodes: Sequence[NodeRecord]) -> dict:
    return {
        "pages": [node.page_data for node in nodes],
    }


def _collect_leaf_index(nodes: Sequence[NodeRecord]) -> List[dict]:
    leaves: List[dict] = []
    for node in nodes:
        marker = node.page_data.get("leaf_marker")
        if node.children or not marker:
            continue
        leaves.append(
            {
                "page_index": node.page_index,
                "page_id": node.page_id,
                "breadcrumb": format_breadcrumb(node.path),
                "leaf_marker": marker,
            }
        )
    return leaves


def _collect_tree(nodes: Sequence[NodeRecord]) -> dict:
    return {
        "nodes": [
            {
                "page_index": node.page_index,
                "page_id": node.page_id,
                "path": list(node.path),
                "parent_index": node.parent_index,
                "children": node.children,
                "title": node.page_data["title"],
            }
            for node in nodes
        ]
    }


def build_tree_nodes(
    depth: int,
    width: int,
    base_seed: int,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    progress: bool = False,
) -> List[NodeRecord]:
    """Public API to materialise website nodes using stage-01 builders."""

    return _build_tree(
        depth,
        width,
        base_seed,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        progress=progress,
    )


def build_tree_bundle(nodes: Sequence[NodeRecord]) -> Dict[str, object]:
    """Assemble design, transitions, metadata, leaf index, and tree dumps."""

    design = _assemble_design(nodes)
    transitions = _assemble_transitions(nodes)
    metadata_bundle = _collect_metadata(nodes)
    leaf_index = _collect_leaf_index(nodes)
    tree_dump = _collect_tree(nodes)
    return {
        "designs": [design],
        "transitions": transitions,
        "metadata": metadata_bundle,
        "leaf_index": leaf_index,
        "tree": tree_dump,
    }


def write_tree_bundle(output_root: Path, nodes: Sequence[NodeRecord]) -> Dict[str, Path]:
    """Persist prompts and bundle artefacts to ``output_root``.

    Returns a mapping of key artefacts to their on-disk locations.
    """

    paths = _ensure_output_dirs(output_root)
    prompts_path = paths["prompts"] / "page_prompt.jsonl"
    _write_prompts(prompts_path, nodes)

    _report_duplicate_titles(nodes)

    bundle = build_tree_bundle(nodes)
    design_path = output_root / "website_designs.json"
    transitions_path = output_root / "transitions.json"
    metadata_path = output_root / "page_metadata.json"
    leaf_index_path = output_root / "leaf_index.json"
    tree_path = output_root / "tree.json"

    _write_json(design_path, bundle["designs"])
    _write_json(transitions_path, bundle["transitions"])
    _write_json(metadata_path, bundle["metadata"])
    _write_json(leaf_index_path, bundle["leaf_index"])
    _write_json(tree_path, bundle["tree"])

    return {
        "prompts": prompts_path,
        "design": design_path,
        "transitions": transitions_path,
        "metadata": metadata_path,
        "leaf_index": leaf_index_path,
        "tree": tree_path,
    }


def make_site_manifest(
    *,
    depth: int,
    width: int,
    base_seed: int,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float | None,
    output_root: Path,
    num_pages: int,
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Build a manifest describing how a tree bundle was generated."""

    manifest: Dict[str, object] = {
        "kind": "full_tree",
        "depth": depth,
        "width": width,
        "base_seed": base_seed,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "num_pages": num_pages,
        "output_root": str(output_root),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": Path(__file__).name,
    }
    if extra:
        manifest.update(dict(extra))
    return manifest


def write_site_manifest(output_root: Path, manifest: Mapping[str, object], *, filename: str = "site_manifest.json") -> Path:
    """Persist a manifest into ``output_root`` and return its path."""

    path = output_root / filename
    _write_json(path, manifest)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3, help="Tree depth (>=1).")
    parser.add_argument("--width", type=int, default=2, help="Branching factor for non-leaf nodes (>=1).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed used to derive deterministic outputs.")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name forwarded to stage-01 builders.",
    )
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum tokens for stage-01 generation.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for stage-01 generation.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus sampling parameter.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for generated artefacts (default: web/runs/trap_site_mvp).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional experiment identifier appended under --output-root.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display even when stderr is a TTY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_experiment_root(args.output_root, args.experiment_id)
    show_progress = not args.no_progress and sys.stderr.isatty()
    nodes = build_tree_nodes(
        args.depth,
        args.width,
        args.seed,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        progress=show_progress,
    )

    write_tree_bundle(output_root, nodes)

    manifest_extra: Dict[str, object] | None = None
    if args.experiment_id:
        manifest_extra = {"experiment_id": args.experiment_id}

    manifest = make_site_manifest(
        depth=args.depth,
        width=args.width,
        base_seed=args.seed,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_root=output_root,
        num_pages=len(nodes),
        extra=manifest_extra,
    )
    write_site_manifest(output_root, manifest)

    print(f"Tree bundle written to {output_root}")
    if manifest_extra:
        print(f"Experiment ID: {args.experiment_id}")


if __name__ == "__main__":
    main()
