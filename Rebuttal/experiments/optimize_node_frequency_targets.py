#!/usr/bin/env python3
"""Compute target-allocation upper bounds for the node-frequency table."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

NEGATIVE_INFINITY = -10**9


@dataclass(eq=False)
class TreeNode:
    """One user-tree node used by the allocation optimizer."""

    key: int | str
    children: list["TreeNode"] = field(default_factory=list)
    targetable: bool = False
    is_directory: bool = False
    display_path: str = ""


CandidatePolicy = Callable[[TreeNode], bool]


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _postorder(root: TreeNode) -> list[TreeNode]:
    rows: list[TreeNode] = []

    def visit(node: TreeNode) -> None:
        for child in node.children:
            visit(child)
        rows.append(node)

    visit(root)
    return rows


def maximize_threshold(
    root: TreeNode,
    *,
    trajectories: int,
    minimum_visits: int,
    is_candidate: CandidatePolicy,
) -> tuple[int, dict[int | str, int]]:
    """Maximize qualifying nodes and return one optimal target allocation."""

    tables: dict[TreeNode, list[int]] = {}
    for node in _postorder(root):
        current = [NEGATIVE_INFINITY] * (trajectories + 1)
        current[0] = 0
        if node.targetable:
            for count in range(1, trajectories + 1):
                current[count] = 0

        for child in node.children:
            child_table = tables[child]
            combined = [NEGATIVE_INFINITY] * (trajectories + 1)
            for already_used, score in enumerate(current):
                if score == NEGATIVE_INFINITY:
                    continue
                for child_count in range(trajectories - already_used + 1):
                    child_score = child_table[child_count]
                    if child_score == NEGATIVE_INFINITY:
                        continue
                    total = already_used + child_count
                    combined[total] = max(
                        combined[total], score + child_score
                    )
            current = combined

        if is_candidate(node):
            for count in range(minimum_visits, trajectories + 1):
                if current[count] != NEGATIVE_INFINITY:
                    current[count] += 1
        tables[node] = current

    allocation: dict[int | str, int] = {}

    def reconstruct(node: TreeNode, assigned: int) -> None:
        if not node.children:
            if node.targetable and assigned:
                allocation[node.key] = assigned
            return

        target_score = tables[node][assigned]
        if is_candidate(node) and assigned >= minimum_visits:
            target_score -= 1

        child_count = len(node.children)
        prefix = [
            [NEGATIVE_INFINITY] * (assigned + 1)
            for _ in range(child_count + 1)
        ]
        prefix[0][0] = 0
        for child_index, child in enumerate(node.children, start=1):
            child_table = tables[child]
            for total in range(assigned + 1):
                prefix[child_index][total] = max(
                    (
                        prefix[child_index - 1][total - child_assigned]
                        + child_table[child_assigned]
                        for child_assigned in range(total + 1)
                        if prefix[child_index - 1][total - child_assigned]
                        != NEGATIVE_INFINITY
                        and child_table[child_assigned]
                        != NEGATIVE_INFINITY
                    ),
                    default=NEGATIVE_INFINITY,
                )
        if prefix[child_count][assigned] != target_score:
            raise RuntimeError(f"Could not reconstruct allocation at {node.key}")

        remaining = assigned
        splits: list[tuple[TreeNode, int]] = []
        for child_index in range(child_count, 0, -1):
            child = node.children[child_index - 1]
            child_table = tables[child]
            selected: int | None = None
            for child_assigned in range(remaining + 1):
                previous = prefix[child_index - 1][
                    remaining - child_assigned
                ]
                if (
                    previous != NEGATIVE_INFINITY
                    and child_table[child_assigned] != NEGATIVE_INFINITY
                    and previous + child_table[child_assigned]
                    == prefix[child_index][remaining]
                ):
                    selected = child_assigned
                    break
            if selected is None:
                raise RuntimeError(
                    f"Could not reconstruct child allocation at {node.key}"
                )
            splits.append((child, selected))
            remaining -= selected
        if remaining:
            raise RuntimeError(f"Unassigned trajectories at {node.key}")
        for child, child_assigned in reversed(splits):
            reconstruct(child, child_assigned)

    reconstruct(root, trajectories)
    return tables[root][trajectories], allocation


def maximize_ranked_er_upper_bounds(
    root: TreeNode,
    *,
    trajectories: int,
    ranks: Sequence[int],
    is_candidate: CandidatePolicy,
) -> dict[int, tuple[int, dict[int | str, int]]]:
    """Maximize each requested ranked node visit count independently.

    Rank ``k`` means the kth-highest individual candidate-node visit count,
    not the encounter rate of the union of the top-k nodes. The returned
    allocation is a witness for that rank's independently optimized maximum.
    """

    if trajectories <= 0:
        raise ValueError("trajectories must be positive")
    unique_ranks = sorted(set(ranks))
    if not unique_ranks or unique_ranks[0] <= 0:
        raise ValueError("ranks must contain positive integers")
    candidate_count = sum(
        is_candidate(node) for node in _postorder(root)
    )
    if unique_ranks[-1] > candidate_count:
        raise ValueError(
            f"rank {unique_ranks[-1]} exceeds {candidate_count} candidates"
        )

    cache: dict[int, tuple[int, dict[int | str, int]]] = {}

    def solve(minimum_visits: int) -> tuple[int, dict[int | str, int]]:
        if minimum_visits not in cache:
            cache[minimum_visits] = maximize_threshold(
                root,
                trajectories=trajectories,
                minimum_visits=minimum_visits,
                is_candidate=is_candidate,
            )
        return cache[minimum_visits]

    maxima: dict[int, tuple[int, dict[int | str, int]]] = {}
    full_er_candidates, full_er_allocation = solve(trajectories)
    for rank in unique_ranks:
        if full_er_candidates >= rank:
            maxima[rank] = trajectories, full_er_allocation
            continue
        lower = 0
        upper = trajectories - 1
        best_visits = 0
        best_allocation: dict[int | str, int] = {}
        while lower <= upper:
            minimum_visits = (lower + upper) // 2
            qualifying_nodes, allocation = solve(minimum_visits)
            if qualifying_nodes >= rank:
                best_visits = minimum_visits
                best_allocation = allocation
                lower = minimum_visits + 1
            else:
                upper = minimum_visits - 1
        maxima[rank] = best_visits, best_allocation
    return maxima


def _pareto(points: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    best_second: dict[int, int] = {}
    for first, second in points:
        best_second[first] = max(best_second.get(first, -1), second)
    frontier: list[tuple[int, int]] = []
    greatest_second = -1
    for first in sorted(best_second, reverse=True):
        second = best_second[first]
        if second > greatest_second:
            frontier.append((first, second))
            greatest_second = second
    return tuple(frontier)


def threshold_pareto_frontier(
    root: TreeNode,
    *,
    trajectories: int,
    minimum_visits_10: int,
    minimum_visits_30: int,
    is_candidate: CandidatePolicy,
) -> tuple[tuple[int, int], ...]:
    """Return nondominated (nodes>=10%, nodes>=30%) count pairs."""

    def solve(node: TreeNode) -> list[tuple[tuple[int, int], ...]]:
        if node.targetable:
            current = []
            for count in range(trajectories + 1):
                candidate = int(is_candidate(node))
                current.append(
                    (
                        (
                            candidate
                            if count >= minimum_visits_10
                            else 0,
                            candidate
                            if count >= minimum_visits_30
                            else 0,
                        ),
                    )
                )
        else:
            current = [tuple() for _ in range(trajectories + 1)]
            current[0] = ((0, 0),)

        for child in node.children:
            child_table = solve(child)
            combined: list[tuple[tuple[int, int], ...]] = [
                tuple() for _ in range(trajectories + 1)
            ]
            for total in range(trajectories + 1):
                points: list[tuple[int, int]] = []
                for child_assigned in range(total + 1):
                    parent_points = current[total - child_assigned]
                    child_points = child_table[child_assigned]
                    points.extend(
                        (
                            parent_10 + child_10,
                            parent_30 + child_30,
                        )
                        for parent_10, parent_30 in parent_points
                        for child_10, child_30 in child_points
                    )
                combined[total] = _pareto(points)
            current = combined

        if node.children and is_candidate(node):
            for count, points in enumerate(current):
                bonus_10 = int(count >= minimum_visits_10)
                bonus_30 = int(count >= minimum_visits_30)
                current[count] = tuple(
                    (first + bonus_10, second + bonus_30)
                    for first, second in points
                )
        return current

    return _pareto(solve(root)[trajectories])


def _subtree_counts(
    root: TreeNode,
    allocation: Mapping[int | str, int],
) -> dict[TreeNode, int]:
    counts: dict[TreeNode, int] = {}

    def visit(node: TreeNode) -> int:
        total = int(allocation.get(node.key, 0))
        total += sum(visit(child) for child in node.children)
        counts[node] = total
        return total

    visit(root)
    return counts


def allocation_metrics(
    root: TreeNode,
    *,
    allocation: Mapping[int | str, int],
    trajectories: int,
    minimum_visits_10: int,
    minimum_visits_30: int,
    is_candidate: CandidatePolicy,
) -> dict[str, float | int]:
    """Evaluate one concrete target allocation."""

    counts = _subtree_counts(root, allocation)
    candidates = [node for node in counts if is_candidate(node)]
    candidate_count = len(candidates)
    visit_counts = sorted(
        (counts[node] for node in candidates),
        reverse=True,
    )

    def ranked_er(rank: int) -> float:
        if rank > candidate_count:
            return 0.0
        return visit_counts[rank - 1] / trajectories

    return {
        "candidate_count": candidate_count,
        "top1_er": ranked_er(1),
        "top2_er": ranked_er(2),
        "top3_er": ranked_er(3),
        "random1_expected_encounters": sum(visit_counts) / candidate_count,
        "nodes_er_ge_10_count": sum(
            count >= minimum_visits_10 for count in visit_counts
        ),
        "nodes_er_ge_30_count": sum(
            count >= minimum_visits_30 for count in visit_counts
        ),
    }


def _maximum_random1_expected_encounters(
    root: TreeNode,
    *,
    trajectories: int,
    is_candidate: CandidatePolicy,
) -> float:
    greatest_path_candidates = 0

    def visit(node: TreeNode, path_candidates: int) -> None:
        nonlocal greatest_path_candidates
        updated = path_candidates + int(is_candidate(node))
        if node.targetable:
            greatest_path_candidates = max(
                greatest_path_candidates, updated
            )
        for child in node.children:
            visit(child, updated)

    visit(root, 0)
    candidate_count = sum(
        is_candidate(node) for node in _postorder(root)
    )
    return trajectories * greatest_path_candidates / candidate_count


def _build_browser_tree(run_root: Path) -> TreeNode:
    metadata = _read_json(run_root / "page_metadata.json")
    pages = metadata.get("pages") if isinstance(metadata, Mapping) else None
    if not isinstance(pages, list):
        raise ValueError("Browser metadata must contain pages")
    page_by_path = {
        tuple(page["path"]): page
        for page in pages
        if isinstance(page, Mapping)
    }
    nodes = {
        path: TreeNode(
            key=int(page["page_index"]),
            is_directory=True,
            display_path=str(page.get("breadcrumb") or "/"),
        )
        for path, page in page_by_path.items()
    }
    for path, node in nodes.items():
        if path:
            nodes[path[:-1]].children.append(node)

    leaves = _read_json(run_root / "leaf_index.json")
    if not isinstance(leaves, list):
        raise ValueError("Browser leaf index must be a list")
    targets = {
        int(row["page_index"])
        for row in leaves
        if isinstance(row, Mapping)
    }
    for node in nodes.values():
        node.targetable = node.key in targets
    return nodes[()]


def _build_file_tree(run_root: Path) -> TreeNode:
    tree = _read_json(run_root / "env" / "user_tree.json")
    manifest = _read_json(run_root / "env" / "user_tree_manifest.json")
    if not isinstance(tree, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("File tree and manifest must be objects")
    target_paths = set(map(str, manifest["user_leaf_files"]))

    def build(row: Mapping[str, object], prefix: str = "") -> TreeNode:
        name = str(row.get("name") or "")
        path = f"{prefix}/{name}".replace("//", "/")
        raw_children = row.get("children")
        child_rows = raw_children if isinstance(raw_children, list) else []
        children = [
            build(child, path)
            for child in child_rows
            if isinstance(child, Mapping)
        ]
        return TreeNode(
            key=path,
            children=children,
            targetable=path in target_paths,
            is_directory=bool(children),
            display_path=path,
        )

    return build(tree)


def _format_allocation(
    root: TreeNode,
    allocation: Mapping[int | str, int],
) -> list[dict[str, object]]:
    by_key = {node.key: node for node in _postorder(root)}
    return [
        {
            "target_node": key,
            "target_path": by_key[key].display_path,
            "trajectories": count,
        }
        for key, count in sorted(allocation.items(), key=lambda item: str(item[0]))
    ]


def _write_table(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "| System | Candidate nodes | Top-1 ER upper bound ↑ | "
        "Top-2 ER upper bound ↑ | Top-3 ER upper bound ↑ | "
        "Random-1 expected encounters (batch) | "
        "max nodes ER ≥ 10% ↑ | max nodes ER ≥ 30% ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['candidate_nodes']} | "
            f"{float(row['top1_er_upper_bound']):.2%} | "
            f"{float(row['top2_er_upper_bound']):.2%} | "
            f"{float(row['top3_er_upper_bound']):.2%} | "
            f"{float(row['random1_expected_encounters']):.3f} | "
            f"{float(row['nodes_er_ge_10_ratio_upper_bound']):.2%} | "
            f"{float(row['nodes_er_ge_30_ratio_upper_bound']):.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser-root",
        type=Path,
        default=Path("web/runs/exp_d10w2_osr_gitlab"),
    )
    parser.add_argument(
        "--file-tree-root",
        type=Path,
        default=Path("Rebuttal/runs/node_frequency_rerun/file"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Rebuttal/results/node_frequency_theoretical_max"),
    )
    parser.add_argument(
        "--coverage-table",
        type=Path,
        default=Path("Rebuttal/results/coverage/table.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    browser_root = _build_browser_tree(args.browser_root)
    file_root = _build_file_tree(args.file_tree_root)
    configurations = [
        (
            "Browser",
            "All non-trivial nodes",
            browser_root,
            72,
            lambda node: node is not browser_root,
        ),
        (
            "Browser",
            "Attacker-writable public nodes",
            browser_root,
            72,
            lambda node: node is not browser_root,
        ),
        (
            "File",
            "All non-trivial nodes",
            file_root,
            60,
            lambda node: node is not file_root,
        ),
        (
            "File",
            "Attacker-writable public nodes",
            file_root,
            60,
            lambda node: node is not file_root and node.is_directory,
        ),
    ]

    rows: list[dict[str, object]] = []
    scenarios: list[dict[str, object]] = []
    frontiers: list[dict[str, object]] = []
    solved: dict[tuple[str, str, int], tuple[int, dict[int | str, int]]] = {}
    for system, label, root, trajectories, policy in configurations:
        minimum_10 = math.ceil(0.10 * trajectories)
        minimum_30 = math.ceil(0.30 * trajectories)
        for minimum in (minimum_10, minimum_30):
            cache_key = (system, label, minimum)
            if cache_key not in solved:
                solved[cache_key] = maximize_threshold(
                    root,
                    trajectories=trajectories,
                    minimum_visits=minimum,
                    is_candidate=policy,
                )

        max_10, allocation_10 = solved[(system, label, minimum_10)]
        max_30, allocation_30 = solved[(system, label, minimum_30)]
        ranked_maxima = maximize_ranked_er_upper_bounds(
            root,
            trajectories=trajectories,
            ranks=(1, 2, 3),
            is_candidate=policy,
        )
        metrics_10 = allocation_metrics(
            root,
            allocation=allocation_10,
            trajectories=trajectories,
            minimum_visits_10=minimum_10,
            minimum_visits_30=minimum_30,
            is_candidate=policy,
        )
        metrics_30 = allocation_metrics(
            root,
            allocation=allocation_30,
            trajectories=trajectories,
            minimum_visits_10=minimum_10,
            minimum_visits_30=minimum_30,
            is_candidate=policy,
        )
        candidate_count = int(metrics_10["candidate_count"])
        rows.append(
            {
                "system": system,
                "candidate_nodes": label,
                "trajectories": trajectories,
                "candidate_count": candidate_count,
                "top1_er_upper_bound": ranked_maxima[1][0] / trajectories,
                "top2_er_upper_bound": ranked_maxima[2][0] / trajectories,
                "top3_er_upper_bound": ranked_maxima[3][0] / trajectories,
                "random1_expected_encounters": (
                    _maximum_random1_expected_encounters(
                        root,
                        trajectories=trajectories,
                        is_candidate=policy,
                    )
                ),
                "nodes_er_ge_10_count_upper_bound": max_10,
                "nodes_er_ge_10_ratio_upper_bound": (
                    max_10 / candidate_count
                ),
                "nodes_er_ge_30_count_upper_bound": max_30,
                "nodes_er_ge_30_ratio_upper_bound": (
                    max_30 / candidate_count
                ),
            }
        )
        scenarios.extend(
            [
                *[
                    {
                        "system": system,
                        "candidate_nodes": label,
                        "optimized_metric": f"top{rank}_er",
                        "rank": rank,
                        "maximum_visits": ranked_maxima[rank][0],
                        "upper_bound": (
                            ranked_maxima[rank][0] / trajectories
                        ),
                        "metrics": allocation_metrics(
                            root,
                            allocation=ranked_maxima[rank][1],
                            trajectories=trajectories,
                            minimum_visits_10=minimum_10,
                            minimum_visits_30=minimum_30,
                            is_candidate=policy,
                        ),
                        "allocation": _format_allocation(
                            root,
                            ranked_maxima[rank][1],
                        ),
                    }
                    for rank in (1, 2, 3)
                ],
                {
                    "system": system,
                    "candidate_nodes": label,
                    "optimized_metric": "nodes_er_ge_10",
                    "minimum_visits": minimum_10,
                    "metrics": metrics_10,
                    "allocation": _format_allocation(root, allocation_10),
                },
                {
                    "system": system,
                    "candidate_nodes": label,
                    "optimized_metric": "nodes_er_ge_30",
                    "minimum_visits": minimum_30,
                    "metrics": metrics_30,
                    "allocation": _format_allocation(root, allocation_30),
                },
            ]
        )
        frontier = threshold_pareto_frontier(
            root,
            trajectories=trajectories,
            minimum_visits_10=minimum_10,
            minimum_visits_30=minimum_30,
            is_candidate=policy,
        )
        frontiers.append(
            {
                "system": system,
                "candidate_nodes": label,
                "count_pairs": [
                    {
                        "nodes_er_ge_10_count": first,
                        "nodes_er_ge_30_count": second,
                    }
                    for first, second in frontier
                ],
                "independent_maxima_jointly_attainable": (
                    (max_10, max_30) in frontier
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "column-wise theoretical target-allocation upper bounds",
        "empirical_result": False,
        "assumptions": {
            "trajectory_model": (
                "each trajectory follows the unique user-tree path to its "
                "assigned target and visits each path node once"
            ),
            "browser_trajectories": 72,
            "file_trajectories": 60,
            "random1_expected_encounters": (
                "expected total encounter count across the full batch after "
                "uniformly sampling one candidate node"
            ),
            "ranked_er_columns": (
                "Top-k is the kth-highest individual candidate-node ER, not "
                "a top-k union; ranks are maximized independently"
            ),
            "threshold_columns": (
                "maximized independently; they need not be attainable by one "
                "common target allocation"
            ),
        },
        "rows": rows,
    }
    (args.out_dir / "theoretical_max.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "target_allocations.json").write_text(
        json.dumps({"scenarios": scenarios}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "pareto_frontiers.json").write_text(
        json.dumps({"frontiers": frontiers}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_table(args.out_dir / "table.md", rows)
    args.coverage_table.parent.mkdir(parents=True, exist_ok=True)
    _write_table(args.coverage_table, rows)
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
