"""Sampling helpers for classify runners.

These utilities replicate the subtree/target selection logic from the standard
exploration runners so the classify workflows operate on identical samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from file.utils.file_tree_dataset import FileTree, FileTreeNode
from file.exp.tree_exploration import (
    extract_subtree,
    get_leaf_nodes,
    get_nodes_at_depth,
    get_tree_max_depth,
    check_target_duplicate_in_tree,
)


@dataclass
class SampleEvent:
    """Represents either a valid sample or a skipped case."""

    kind: str  # "sample" or "skip"
    variation: str  # "depth"
    metadata: Dict[str, object]
    subtree: Optional[FileTree] = None
    target_node: Optional[FileTreeNode] = None


def depth_variation_events(
    full_tree: FileTree,
    tree_path: str,
    fixed_width: int,
    tests_per_depth: int,
) -> Iterator[SampleEvent]:
    """Yield sample/skip events following the depth-variation runner rules."""

    max_depth = get_tree_max_depth(full_tree)
    test_id = 0

    for depth in range(2, max_depth + 1):
        valid_root_nodes: list[Tuple[int, int]] = []
        for start_depth in range(max_depth - depth + 2):
            nodes_at_depth = get_nodes_at_depth(full_tree, start_depth)
            for idx, node in enumerate(nodes_at_depth):
                node_tree = FileTree(node)
                if get_tree_max_depth(node_tree) >= depth - 1:
                    valid_root_nodes.append((start_depth, idx))

        if not valid_root_nodes:
            yield SampleEvent(
                kind="skip",
                variation="depth",
                metadata={
                    "tree_path": tree_path,
                    "depth": depth,
                    "width": fixed_width,
                    "reason": "no_valid_root_nodes",
                },
            )
            continue

        if tests_per_depth <= 0:
            yield SampleEvent(
                kind="skip",
                variation="depth",
                metadata={
                    "tree_path": tree_path,
                    "depth": depth,
                    "width": fixed_width,
                    "reason": "tests_per_depth_zero",
                },
            )
            continue

        if len(valid_root_nodes) >= tests_per_depth:
            nodes_to_test = valid_root_nodes[:tests_per_depth]
            node_test_count = {node: 1 for node in nodes_to_test}
        else:
            nodes_to_test = valid_root_nodes
            base = tests_per_depth // len(valid_root_nodes)
            extra = tests_per_depth % len(valid_root_nodes)
            node_test_count = {}
            for i, node in enumerate(nodes_to_test):
                num_tests = base + (1 if i < extra else 0)
                node_test_count[node] = num_tests

        for (root_depth, root_idx), num_tests in node_test_count.items():
            if num_tests <= 0:
                continue
            for test_number in range(num_tests):
                test_id += 1
                seed = test_id * 1000
                subtree = extract_subtree(
                    full_tree,
                    depth,
                    fixed_width,
                    seed,
                    root_coord=(root_depth, root_idx),
                )
                leaves = get_leaf_nodes(subtree)
                target_node: Optional[FileTreeNode] = None
                target_coord: Optional[int] = None
                for leaf_idx, leaf in enumerate(leaves):
                    if not check_target_duplicate_in_tree(subtree, leaf.name):
                        target_node = leaf
                        target_coord = leaf_idx
                        break

                metadata = {
                    "tree_path": tree_path,
                    "depth": depth,
                    "width": fixed_width,
                    "root_coord": (root_depth, root_idx),
                    "test_id": test_id,
                    "test_number": test_number + 1,
                    "seed": seed,
                    "total_leaves": len(leaves),
                    "target_coord": target_coord,
                }

                if target_node is None:
                    yield SampleEvent(
                        kind="skip",
                        variation="depth",
                        metadata={**metadata, "reason": "no_valid_target"},
                    )
                    continue

                yield SampleEvent(
                    kind="sample",
                    variation="depth",
                    metadata=metadata,
                    subtree=subtree,
                    target_node=target_node,
                )
