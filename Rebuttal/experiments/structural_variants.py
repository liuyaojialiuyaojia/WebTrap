#!/usr/bin/env python3
"""Plan reviewer-requested path variants from structural metadata only.

The planner deliberately whitelists graph and stage-location fields. Page
bodies, prompts, injected text, observations, and trace contents are never
read into the returned structures or written to the result files.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

Node = Hashable

PLACEMENT_VARIANTS: dict[str, tuple[str, ...]] = {
    "shift_s2": ("inertia",),
    "shift_s3": ("payload",),
    "shift_s2s3": ("inertia", "payload"),
}


@dataclass(frozen=True)
class PlacementCandidate:
    """One structurally valid placement condition."""

    system: str
    variant: str
    moved_stages: tuple[str, ...]
    inertia_node: Node
    payload_node: Node
    inertia_depth: int
    payload_depth: int
    inertia_displacement_hops: int
    payload_displacement_hops: int
    route: tuple[Node, ...]
    inertia_position: int
    payload_position: int
    total_hops: int
    added_hops: int
    repeated_hops: int


class Graph:
    """Small directed graph helper with deterministic shortest paths."""

    def __init__(
        self,
        adjacency: Mapping[Node, Iterable[Node]],
        depths: Mapping[Node, int],
    ) -> None:
        self.adjacency = {
            node: tuple(sorted(set(neighbors), key=str))
            for node, neighbors in adjacency.items()
        }
        self.depths = dict(depths)

    def shortest_path(self, start: Node, goal: Node) -> tuple[Node, ...]:
        if start == goal:
            return (start,)
        queue: deque[Node] = deque([start])
        parent: dict[Node, Node | None] = {start: None}
        while queue:
            current = queue.popleft()
            for neighbor in self.adjacency.get(current, ()):
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == goal:
                    queue.clear()
                    break
                queue.append(neighbor)
        if goal not in parent:
            raise ValueError(f"No structural route from {start!r} to {goal!r}")
        result: list[Node] = []
        current: Node | None = goal
        while current is not None:
            result.append(current)
            current = parent[current]
        return tuple(reversed(result))


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _join_paths(*paths: Sequence[Node]) -> tuple[Node, ...]:
    route: list[Node] = []
    for path in paths:
        if not path:
            continue
        if route and route[-1] == path[0]:
            route.extend(path[1:])
        else:
            route.extend(path)
    return tuple(route)


def _rank_candidates(
    *,
    system: str,
    graph: Graph,
    start: Node,
    anchor: Node,
    original_inertia: Node,
    original_payload: Node,
    candidates: Iterable[Node],
    moved_stages: Sequence[str] = ("inertia", "payload"),
    variant: str = "shift_s2s3",
    max_displacement_hops: int = 2,
) -> list[PlacementCandidate]:
    moved = tuple(str(stage) for stage in moved_stages)
    if not moved or any(stage not in {"inertia", "payload"} for stage in moved):
        raise ValueError(f"Invalid moved stages: {moved!r}")
    if max_displacement_hops < 1:
        raise ValueError("max_displacement_hops must be positive")

    original_route = graph.shortest_path(start, anchor)
    original_nodes = set(original_route)
    inertia_depth = graph.depths[original_inertia]
    payload_depth = graph.depths[original_payload]
    candidate_nodes = set(candidates)

    def stage_candidates(
        stage: str,
        original_node: Node,
        original_depth: int,
    ) -> list[tuple[Node, int]]:
        if stage not in moved:
            return [(original_node, 0)]
        rows: list[tuple[Node, int]] = []
        for node in candidate_nodes:
            if graph.depths.get(node) != original_depth or node in original_nodes:
                continue
            try:
                displacement = len(graph.shortest_path(original_node, node)) - 1
            except ValueError:
                continue
            if 1 <= displacement <= max_displacement_hops:
                rows.append((node, displacement))
        return sorted(rows, key=lambda item: (item[1], str(item[0])))

    inertia_candidates = stage_candidates(
        "inertia", original_inertia, inertia_depth
    )
    payload_candidates = stage_candidates(
        "payload", original_payload, payload_depth
    )

    rows: list[PlacementCandidate] = []
    for inertia_node, inertia_displacement in inertia_candidates:
        try:
            first = graph.shortest_path(start, inertia_node)
        except ValueError:
            continue
        for payload_node, payload_displacement in payload_candidates:
            try:
                second = graph.shortest_path(inertia_node, payload_node)
                third = graph.shortest_path(payload_node, anchor)
            except ValueError:
                continue
            route = _join_paths(first, second, third)
            inertia_position = len(first) - 1
            payload_position = inertia_position + len(second) - 1
            total_hops = len(route) - 1
            rows.append(
                PlacementCandidate(
                    system=system,
                    variant=variant,
                    moved_stages=moved,
                    inertia_node=inertia_node,
                    payload_node=payload_node,
                    inertia_depth=inertia_depth,
                    payload_depth=payload_depth,
                    inertia_displacement_hops=inertia_displacement,
                    payload_displacement_hops=payload_displacement,
                    route=route,
                    inertia_position=inertia_position,
                    payload_position=payload_position,
                    total_hops=total_hops,
                    added_hops=total_hops - (len(original_route) - 1),
                    repeated_hops=len(route) - len(set(route)),
                )
            )

    return sorted(
        rows,
        key=lambda row: (
            row.added_hops,
            row.repeated_hops,
            row.total_hops,
            row.inertia_displacement_hops + row.payload_displacement_hops,
            str(row.inertia_node),
            str(row.payload_node),
        ),
    )


def load_browser_graph(metadata_path: Path) -> Graph:
    """Load only page indices, depths, and target-page edges."""

    payload = _read_json(metadata_path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("pages"), list):
        raise ValueError(f"Invalid Browser metadata: {metadata_path}")
    adjacency: dict[int, set[int]] = {}
    depths: dict[int, int] = {}
    for raw_page in payload["pages"]:
        if not isinstance(raw_page, Mapping) or not isinstance(
            raw_page.get("page_index"), int
        ):
            continue
        page_index = int(raw_page["page_index"])
        raw_path = raw_page.get("path")
        depths[page_index] = len(raw_path) if isinstance(raw_path, list) else 0
        adjacency.setdefault(page_index, set())
        raw_targets = raw_page.get("click_targets")
        if not isinstance(raw_targets, list):
            continue
        for target in raw_targets:
            if not isinstance(target, Mapping):
                continue
            target_page = target.get("target_page")
            if isinstance(target_page, int):
                adjacency[page_index].add(int(target_page))
    return Graph(adjacency, depths)


def load_browser_stage_locations(
    injected_metadata_path: Path,
) -> tuple[dict[str, int], int]:
    """Read stage names and page indices while treating injection text as opaque."""

    payload = _read_json(injected_metadata_path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("pages"), list):
        raise ValueError(f"Invalid injected Browser metadata: {injected_metadata_path}")
    stages: dict[str, int] = {}
    anchor: int | None = None
    for raw_page in payload["pages"]:
        if not isinstance(raw_page, Mapping) or not isinstance(
            raw_page.get("page_index"), int
        ):
            continue
        injections = raw_page.get("injections")
        if not isinstance(injections, list):
            continue
        for injection in injections:
            if not isinstance(injection, Mapping):
                continue
            psaa = injection.get("psaa")
            source = injection.get("source")
            if not isinstance(psaa, Mapping):
                continue
            stage = psaa.get("stage")
            if isinstance(stage, str) and stage in {"lure", "inertia", "payload"}:
                stages[stage] = int(raw_page["page_index"])
            if isinstance(source, Mapping) and isinstance(
                source.get("anchor_page_index"), int
            ):
                anchor = int(source["anchor_page_index"])
    missing = {"lure", "inertia", "payload"} - stages.keys()
    if missing or anchor is None:
        raise ValueError(
            f"Missing Browser structural fields: stages={sorted(missing)}, anchor={anchor}"
        )
    return stages, anchor


def _directory_children(node: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw_children = node.get("children")
    if not isinstance(raw_children, list):
        return []
    children: list[Mapping[str, object]] = []
    for child in raw_children:
        if not isinstance(child, Mapping):
            continue
        child_type = str(child.get("type") or "").lower()
        child_children = child.get("children")
        if child_type not in {"file"} and isinstance(child_children, list) and child_children:
            children.append(child)
    return children


def load_file_graph(tree_path: Path) -> Graph:
    """Load only directory names and parent-child relationships."""

    payload = _read_json(tree_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid File tree: {tree_path}")
    adjacency: dict[str, set[str]] = {}
    depths: dict[str, int] = {}

    def visit(node: Mapping[str, object], parent: str | None, depth: int) -> None:
        name = str(node.get("name") or "").strip()
        if parent is None:
            current = f"/{name}" if name else "/root"
        else:
            current = f"{parent.rstrip('/')}/{name}"
        adjacency.setdefault(current, set())
        depths[current] = depth
        if parent is not None:
            adjacency.setdefault(parent, set()).add(current)
            adjacency[current].add(parent)
        for child in _directory_children(node):
            visit(child, current, depth + 1)

    visit(payload, None, 0)
    return Graph(adjacency, depths)


def load_file_stage_locations(
    manifest_path: Path,
) -> tuple[dict[str, str], str, str]:
    """Read only structural columns from the JSONL injection manifest."""

    stages: dict[str, str] = {}
    start: str | None = None
    security_entry: str | None = None
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            continue
        stage = row.get("stage")
        directory = row.get("directory_logical_path")
        if isinstance(stage, str) and isinstance(directory, str):
            if stage in {"lure", "inertia", "payload"}:
                stages[stage] = directory
        if isinstance(row.get("start_directory_logical_path"), str):
            start = str(row["start_directory_logical_path"])
        if isinstance(row.get("security_entry_directory_logical_path"), str):
            security_entry = str(row["security_entry_directory_logical_path"])
    missing = {"lure", "inertia", "payload"} - stages.keys()
    if missing or start is None or security_entry is None:
        raise ValueError(
            "Missing File structural fields: "
            f"stages={sorted(missing)}, start={start}, entry={security_entry}"
        )
    return stages, start, security_entry


def plan_browser(
    metadata_path: Path,
    injected_metadata_path: Path,
) -> tuple[list[PlacementCandidate], dict[str, object]]:
    variants, context = plan_browser_variants(metadata_path, injected_metadata_path)
    return variants["shift_s2s3"], context


def plan_browser_variants(
    metadata_path: Path,
    injected_metadata_path: Path,
) -> tuple[dict[str, list[PlacementCandidate]], dict[str, object]]:
    graph = load_browser_graph(metadata_path)
    stages, anchor = load_browser_stage_locations(injected_metadata_path)
    candidates = {
        node
        for node, depth in graph.depths.items()
        if depth <= graph.depths[stages["payload"]]
    }
    variants = {
        variant: _rank_candidates(
            system="Browser",
            graph=graph,
            start=stages["lure"],
            anchor=anchor,
            original_inertia=stages["inertia"],
            original_payload=stages["payload"],
            candidates=candidates,
            moved_stages=moved_stages,
            variant=variant,
        )
        for variant, moved_stages in PLACEMENT_VARIANTS.items()
    }
    context = {
        "start_node": stages["lure"],
        "anchor_node": anchor,
        "original_stage_nodes": stages,
        "original_route": graph.shortest_path(stages["lure"], anchor),
    }
    return variants, context


def plan_file(
    tree_path: Path,
    manifest_path: Path,
) -> tuple[list[PlacementCandidate], dict[str, object]]:
    variants, context = plan_file_variants(tree_path, manifest_path)
    return variants["shift_s2s3"], context


def plan_file_variants(
    tree_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, list[PlacementCandidate]], dict[str, object]]:
    graph = load_file_graph(tree_path)
    stages, start, security_entry = load_file_stage_locations(manifest_path)
    candidates = {
        node
        for node, depth in graph.depths.items()
        if depth <= graph.depths[stages["payload"]] and "/security_entry" not in node
    }
    variants = {
        variant: _rank_candidates(
            system="File",
            graph=graph,
            start=start,
            anchor=security_entry,
            original_inertia=stages["inertia"],
            original_payload=stages["payload"],
            candidates=candidates,
            moved_stages=moved_stages,
            variant=variant,
        )
        for variant, moved_stages in PLACEMENT_VARIANTS.items()
    }
    context = {
        "start_node": start,
        "anchor_node": security_entry,
        "original_stage_nodes": stages,
        "original_route": graph.shortest_path(start, security_entry),
    }
    return variants, context


def _serialize_candidate(row: PlacementCandidate) -> dict[str, object]:
    payload = asdict(row)
    payload["route"] = list(row.route)
    payload["moved_stages"] = list(row.moved_stages)
    payload["stage_positions"] = {
        "lure": 0,
        "inertia": row.inertia_position,
        "payload": row.payload_position,
    }
    payload["stage_nodes"] = {
        "lure": row.route[0],
        "inertia": row.inertia_node,
        "payload": row.payload_node,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser-metadata",
        type=Path,
        default=Path("web/runs/exp_d10w2_psaa/security_microtree/page_metadata.json"),
    )
    parser.add_argument(
        "--browser-injected-metadata",
        type=Path,
        default=Path("web/runs/exp_d10w2_psaa/static/page_metadata_injected.json"),
    )
    parser.add_argument(
        "--file-tree",
        type=Path,
        default=Path("file/runs/exp_d10w2_psaa/env/env_post_injection_tree.json"),
    )
    parser.add_argument(
        "--file-manifest",
        type=Path,
        default=Path("file/runs/exp_d10w2_psaa/injection/injection_manifest.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Rebuttal/results/placement"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    browser_variants, browser_context = plan_browser_variants(
        args.browser_metadata,
        args.browser_injected_metadata,
    )
    file_variants, file_context = plan_file_variants(
        args.file_tree,
        args.file_manifest,
    )
    if any(not rows for rows in browser_variants.values()) or any(
        not rows for rows in file_variants.values()
    ):
        raise ValueError("No valid suboptimal placement candidates were found")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def serialize_variants(
        variants: Mapping[str, Sequence[PlacementCandidate]],
    ) -> dict[str, object]:
        return {
            variant: {
                "moved_stages": list(PLACEMENT_VARIANTS[variant]),
                "selected": _serialize_candidate(rows[0]),
            }
            for variant, rows in variants.items()
        }

    selected = {
        "schema_version": 2,
        "selection_rule": [
            "only the stages named by each condition are moved",
            "every moved node is outside the original shortest path",
            "every moved node is one or two graph hops from its original node",
            "preserve the original stage depths",
            "minimize added route hops",
            "then minimize repeated nodes, total route hops, total displacement, and node ids",
            "do not inspect task text, page/file content, injection text, or outcomes",
        ],
        "systems": {
            "Browser": {
                **browser_context,
                "variants": serialize_variants(browser_variants),
            },
            "File": {
                **file_context,
                "variants": serialize_variants(file_variants),
            },
        },
    }
    json_path = args.out_dir / "selected_placements.json"
    json_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for variants in (browser_variants, file_variants):
        for variant in PLACEMENT_VARIANTS:
            row = variants[variant][0]
            print(
                f"{row.system}/{variant}: inertia={row.inertia_node}, "
                f"payload={row.payload_node}, hops={row.total_hops}, "
                f"added={row.added_hops}"
            )
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
