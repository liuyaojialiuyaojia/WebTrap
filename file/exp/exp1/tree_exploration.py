#!/usr/bin/env python3
"""Tree exploration utilities for file-tree experiment construction."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add repository root to the import path to avoid shadowing top-level `utils`.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.utils.file_tree_dataset import FileTree, FileTreeNode


def extract_subtree(
    tree: FileTree,
    depth: int,
    width: int,
    seed: int,
    root_coord: Optional[Tuple[int, int]] = None,
    strip_non_leaf_extensions: bool = False,
) -> FileTree:
    """Extract a subtree from a full tree using depth, width, and seed."""
    random.seed(seed)

    if root_coord is None:
        max_start_depth = _get_max_depth(tree.root) - depth + 1
        if max_start_depth <= 0:
            max_start_depth = 1

        start_depth = random.randint(0, max_start_depth - 1)
        nodes_at_depth: List[FileTreeNode] = []
        _collect_nodes_at_depth(tree.root, 0, start_depth, nodes_at_depth)

        if not nodes_at_depth:
            return FileTree(
                FileTreeNode(name=tree.root.name, content=tree.root.content, depth=0)
            )

        subtree_root_original = random.choice(nodes_at_depth)
    else:
        target_depth, target_index = root_coord
        nodes_at_depth = []
        _collect_nodes_at_depth(tree.root, 0, target_depth, nodes_at_depth)

        if target_index >= len(nodes_at_depth):
            raise ValueError(
                f"Invalid root_coord: no node at depth {target_depth}, index {target_index}"
            )

        subtree_root_original = nodes_at_depth[target_index]

    subtree_root = FileTreeNode(
        name=subtree_root_original.name, content=subtree_root_original.content, depth=0
    )
    _build_subtree(subtree_root_original, subtree_root, depth - 1, width, 1)

    if strip_non_leaf_extensions:
        _process_subtree_node_names(subtree_root)

    return FileTree(subtree_root)


def extract_subtree_for_width(
    tree: FileTree,
    depth: int,
    width: int,
    seed: int,
    root_coord: Optional[Tuple[int, int]] = None,
    strip_non_leaf_extensions: bool = False,
) -> FileTree:
    """Extract a subtree while preferring roots that exercise the target width."""
    random.seed(seed)

    if root_coord is None:
        max_start_depth = _get_max_depth(tree.root) - depth + 1
        if max_start_depth <= 0:
            max_start_depth = 1

        start_depth = random.randint(0, max_start_depth - 1)
        nodes_at_depth: List[FileTreeNode] = []
        _collect_nodes_at_depth(tree.root, 0, start_depth, nodes_at_depth)

        if not nodes_at_depth:
            return FileTree(
                FileTreeNode(name=tree.root.name, content=tree.root.content, depth=0)
            )

        valid_nodes = [
            node
            for node in nodes_at_depth
            if _can_node_support_width(node, width, depth - 1)
        ]
        if not valid_nodes:
            valid_nodes = nodes_at_depth

        subtree_root_original = random.choice(valid_nodes)
    else:
        target_depth, target_index = root_coord
        nodes_at_depth = []
        _collect_nodes_at_depth(tree.root, 0, target_depth, nodes_at_depth)

        if target_index >= len(nodes_at_depth):
            raise ValueError(
                f"Invalid root_coord: no node at depth {target_depth}, index {target_index}"
            )

        subtree_root_original = nodes_at_depth[target_index]

    subtree_root = FileTreeNode(
        name=subtree_root_original.name, content=subtree_root_original.content, depth=0
    )
    _build_subtree_for_width(subtree_root_original, subtree_root, depth - 1, width, 1)

    if strip_non_leaf_extensions:
        _process_subtree_node_names(subtree_root)

    return FileTree(subtree_root)


def _can_node_support_width(
    node: FileTreeNode, required_width: int, remaining_depth: int
) -> bool:
    """Return whether any remaining path can expose the requested width."""
    if remaining_depth == 0:
        return True

    if len(node.children) >= required_width:
        return True

    for child in node.children:
        if _can_node_support_width(child, required_width, remaining_depth - 1):
            return True

    return False


def _build_subtree_for_width(
    original_node: FileTreeNode,
    new_node: FileTreeNode,
    remaining_depth: int,
    target_width: int,
    current_depth: int,
):
    """Build a subtree while preferring branches that can satisfy target width."""
    if remaining_depth <= 0 or not original_node.children:
        return

    available_children = list(original_node.children)
    preferred_children = []
    backup_children = []

    for child in available_children:
        if _can_node_support_width(child, target_width, remaining_depth - 1):
            preferred_children.append(child)
        else:
            backup_children.append(child)

    if len(preferred_children) >= target_width:
        selected_children = random.sample(
            preferred_children, min(target_width, len(preferred_children))
        )
    else:
        selected_children = preferred_children[:]
        remaining_slots = target_width - len(preferred_children)
        if remaining_slots > 0 and backup_children:
            additional = random.sample(
                backup_children, min(remaining_slots, len(backup_children))
            )
            selected_children.extend(additional)

    if len(selected_children) < target_width and len(available_children) > len(
        selected_children
    ):
        remaining_children = [
            child for child in available_children if child not in selected_children
        ]
        additional_needed = min(
            target_width - len(selected_children), len(remaining_children)
        )
        if additional_needed > 0:
            additional = random.sample(remaining_children, additional_needed)
            selected_children.extend(additional)

    for original_child in selected_children:
        new_child = FileTreeNode(
            name=original_child.name,
            content=original_child.content,
            depth=current_depth,
            parent=new_node,
        )
        new_node.children.append(new_child)
        _build_subtree_for_width(
            original_child,
            new_child,
            remaining_depth - 1,
            target_width,
            current_depth + 1,
        )


def _process_subtree_node_names(node: FileTreeNode):
    """Strip extensions from non-leaf nodes."""
    if node.children and "." in node.name and node.name != "root":
        node.name = node.name.rsplit(".", 1)[0]

    for child in node.children:
        _process_subtree_node_names(child)


def get_leaf_nodes(tree: FileTree) -> List[FileTreeNode]:
    """Return all leaf nodes in right-to-left depth-first order."""
    leaves = []

    def collect_leaves(node: FileTreeNode):
        if not node.children:
            leaves.append(node)
        else:
            for child in reversed(node.children):
                collect_leaves(child)

    collect_leaves(tree.root)
    return leaves


def check_duplicate_leaf_names(
    leaves: List[FileTreeNode],
) -> Dict[str, List[FileTreeNode]]:
    """Return duplicate leaf names mapped to the nodes using each name."""
    name_to_nodes = {}
    for leaf in leaves:
        if leaf.name not in name_to_nodes:
            name_to_nodes[leaf.name] = []
        name_to_nodes[leaf.name].append(leaf)

    return {
        name: nodes for name, nodes in name_to_nodes.items() if len(nodes) > 1
    }


def get_random_target_file(tree: FileTree, seed: int) -> str:
    """Select one random leaf filename as the target."""
    random.seed(seed)
    leaves = get_leaf_nodes(tree)

    if not leaves:
        raise ValueError("Tree has no leaf file nodes.")

    duplicates = check_duplicate_leaf_names(leaves)
    if duplicates:
        error_msg = "Cannot select a target file because duplicate leaf names exist:\n"
        for name, nodes in duplicates.items():
            error_msg += f"  - '{name}' appears {len(nodes)} times\n"
        raise ValueError(error_msg)

    target = random.choice(leaves)
    return target.name


def check_target_duplicate_in_tree(tree: FileTree, target_name: str) -> bool:
    """Return whether a target filename appears more than once in the tree."""
    count = 0

    def count_occurrences(node: FileTreeNode):
        nonlocal count
        if node.name == target_name:
            count += 1
        for child in node.children:
            count_occurrences(child)

    count_occurrences(tree.root)
    return count > 1


def get_target_file_by_coord(
    tree: FileTree, target_coord: int, check_only_target: bool = True
) -> str:
    """Select a target leaf by right-to-left depth-first leaf index."""
    leaves = get_leaf_nodes(tree)

    if not leaves:
        raise ValueError("Tree has no leaf file nodes.")

    if target_coord < 0 or target_coord >= len(leaves):
        raise ValueError(
            f"Invalid target coordinate {target_coord}; leaf count is {len(leaves)}."
        )

    target = leaves[target_coord]

    if check_only_target:
        if check_target_duplicate_in_tree(tree, target.name):
            raise ValueError(f"Target file '{target.name}' is duplicated in the tree.")
    else:
        duplicates = check_duplicate_leaf_names(leaves)
        if duplicates:
            error_msg = "Cannot select a target file because duplicate leaf names exist:\n"
            for name, nodes in duplicates.items():
                error_msg += f"  - '{name}' appears {len(nodes)} times\n"
            raise ValueError(error_msg)

    return target.name


def get_tree_max_depth(tree: FileTree) -> int:
    """Return the maximum depth of a tree."""
    return _get_max_depth(tree.root)


def get_nodes_at_depth(tree: FileTree, depth: int) -> List[FileTreeNode]:
    """Return all nodes at a target depth in breadth-first order."""
    nodes = []
    _collect_nodes_at_depth(tree.root, 0, depth, nodes)
    return nodes


def count_possible_subtrees(tree: FileTree, subtree_depth: int) -> int:
    """Count nodes that can form a subtree of the requested depth."""
    if subtree_depth <= 0:
        return 0

    max_depth = get_tree_max_depth(tree)
    count = 0

    for start_depth in range(max_depth - subtree_depth + 2):
        nodes_at_depth = get_nodes_at_depth(tree, start_depth)

        for node in nodes_at_depth:
            node_max_depth = _get_max_depth(node)
            if node_max_depth >= subtree_depth - 1:
                count += 1

    return count


def _get_max_depth(node: FileTreeNode, current_depth: int = 0) -> int:
    """Return the maximum descendant depth from a node."""
    if not node.children:
        return current_depth

    max_child_depth = current_depth
    for child in node.children:
        child_depth = _get_max_depth(child, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)

    return max_child_depth


def _collect_nodes_at_depth(
    node: FileTreeNode,
    current_depth: int,
    target_depth: int,
    result: List[FileTreeNode],
) -> None:
    """Collect nodes at a target depth."""
    if current_depth == target_depth:
        result.append(node)
        return

    for child in node.children:
        _collect_nodes_at_depth(child, current_depth + 1, target_depth, result)


def _build_subtree(
    original_node: FileTreeNode,
    new_node: FileTreeNode,
    remaining_depth: int,
    width: int,
    current_depth: int,
) -> None:
    """Recursively build a subtree with depth and width limits."""
    if remaining_depth <= 0 or not original_node.children:
        return

    children_to_keep = original_node.children.copy()
    if len(children_to_keep) > width:
        random.shuffle(children_to_keep)
        children_to_keep = children_to_keep[:width]

    for child in children_to_keep:
        new_child = FileTreeNode(
            name=child.name, content=child.content, depth=current_depth, parent=new_node
        )
        new_node.children.append(new_child)
        _build_subtree(child, new_child, remaining_depth - 1, width, current_depth + 1)
