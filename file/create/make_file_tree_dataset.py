#!/usr/bin/env python3
"""Create file trees using API-based generation and extraction.

This module provides functions to create file trees through API calls,
then extract subtrees based on depth/width/seed parameters for testing.
"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from file.utils.file_tree_dataset import FileTree, FileTreeNode
from file.utils.generate import generate
from file.create.tree_generation_utils import generate_with_retry, log_generation_error
from file.create.tree_validation import validate_tree_structure


def create_fan_tree(depth: int, width: int, breadth_mode: str = "mixed", related_ratio: float = 0.7, model: str = "gpt-4o-mini", max_retries: int = 3) -> FileTree:
    """Create a fan-shaped tree by progressively extending from minimal tree.

    Each level first adds one depth child to every frontier node, then extends
    breadth until each frontier node reaches the requested width.

    Args:
        depth: Tree depth.
        width: Number of children for each non-leaf node.
        breadth_mode: "related", "unrelated", or "mixed".
        related_ratio: Probability of related expansion in mixed mode.
        model: Model used for generation.
    
    Returns:
        Complete fan-shaped tree.
    """
    tree = create_minimal_tree(model, max_retries)
    
    if depth <= 1:
        return tree
    
    total_api_calls = 1
    for d in range(1, depth + 1):
        nodes_at_depth = width ** (d - 1)
        total_api_calls += nodes_at_depth
        total_api_calls += nodes_at_depth
    
    with tqdm(total=total_api_calls, desc=f"Generating fan tree (depth={depth}, width={width})") as pbar:
        pbar.update(1)
        
        for current_depth in range(1, depth + 1):
            nodes_to_extend = []
            _collect_nodes_at_depth(tree.root, 0, current_depth - 1, nodes_to_extend)
            
            for node in nodes_to_extend:
                path = tree.get_path_to_node(node)
                tree = extend_depth(tree, path, model, max_retries)
                pbar.update(1)
            
            for node in nodes_to_extend:
                if len(node.children) < width:
                    nodes_needed = width - len(node.children)
                    
                    if breadth_mode == "related":
                        tree = extend_breadth_related(tree, current_depth, nodes_needed, node, model, max_retries)
                    elif breadth_mode == "unrelated":
                        tree = extend_breadth_unrelated(tree, current_depth, nodes_needed, node, model, max_retries)
                    else:  # mixed mode
                        if random.random() < related_ratio:
                            tree = extend_breadth_related(tree, current_depth, nodes_needed, node, model, max_retries)
                        else:
                            tree = extend_breadth_unrelated(tree, current_depth, nodes_needed, node, model, max_retries)
                    
                pbar.update(1)
    
    # Final validation
    is_valid, error_msg = validate_tree_structure(tree, depth, width)
    if not is_valid:
        print(f"Warning: Generated tree failed validation: {error_msg}")
    
    return tree


def _collect_nodes_at_depth(node: FileTreeNode, current_depth: int, target_depth: int, result: List[FileTreeNode]) -> None:
    """Collect all nodes at a target depth."""
    if current_depth == target_depth:
        result.append(node)
        return
    
    for child in node.children:
        _collect_nodes_at_depth(child, current_depth + 1, target_depth, result)


def create_minimal_tree(model: str = "gpt-4o-mini", max_retries: int = 3) -> FileTree:
    """Create a minimal tree with just root and one child."""
    prompt = """Create a minimal file tree with:
1. A root folder named "root" (content: "Root directory of the project")
2. ONE child file or folder

Requirements:
- The child should be a general topic that can be expanded
- Include both name (with file extension if it's a file) and content
- Return in JSON format with this structure:

{
    "name": "root",
    "content": "Root directory of the project",
    "children": [
        {"name": "example.txt", "content": "Example content", "children": []}
    ]
}

Filename format requirements:
- Only filenames with extensions (e.g., "report.pdf", "config.json")
- No slashes or special characters"""
    
    try:
        tree_data = generate_with_retry(prompt, model, max_retries, expected_type="object")
        return FileTree.from_dict(tree_data)
    except Exception as e:
        log_generation_error("minimal tree creation", "root", e)
        raise ValueError(f"Failed to create minimal tree: {e}")


def extend_depth(tree: FileTree, path: List[str], model: str = "gpt-4o-mini", max_retries: int = 3) -> FileTree:
    """Extend tree depth by adding one child to the specified node."""
    node = tree.get_node_by_path(["root"] + path)
    if not node:
        return tree
    
    path_description = " -> ".join(["root"] + path)
    
    prompt = f"""Given this file tree path: {path_description}

Current node: "{node.name}"
Current node content: "{node.content}"

Add ONE child node that continues deeper into this topic. Requirements:
1. The child should explore a more specific aspect of "{node.name}"
2. Include both name (with file extension) and content
3. Use varied file extensions based on content type (.txt, .md, .json, .py, .js, .html, .css, .yaml, etc.), and content should match the file type (e.g., .py files contain Python code, .json contains valid JSON)
4. Return a single node in JSON format:

{{"name": "specific_topic.txt", "content": "Detailed content about the specific topic", "children": []}}

Filename format requirements:
- Only filenames with extensions (e.g., "report.pdf", "config.json")
- No slashes or special characters

IMPORTANT: Return only ONE child node."""
    
    try:
        child_data = generate_with_retry(prompt, model, max_retries, expected_type="object")
        
        # Add the new child to the node
        child_node = FileTreeNode.from_dict(child_data, depth=node.depth + 1, parent=node)
        node.children.append(child_node)
        
        return tree
    except Exception as e:
        log_generation_error("depth extension", node.name, e)
        return tree


def extend_breadth_related(tree: FileTree, depth: int, nodes_needed: int, node_to_extend: FileTreeNode, model: str = "gpt-4o-mini", max_retries: int = 3) -> FileTree:
    """Extend tree breadth by adding RELATED siblings to match the target."""
    if not node_to_extend.children:
        return tree

    batch_size = 2
    all_siblings = []

    while nodes_needed > 0:
        current_batch = min(nodes_needed, batch_size)

        existing_children = [f"{child.name}: {child.content[:200]}{'...' if len(child.content) > 200 else ''}" for child in node_to_extend.children]
        for s in all_siblings:
            content = s.get('content', '')
            if not isinstance(content, str):
                content_type = type(content).__name__
                content_str = str(content)
                print(f"Warning: Node '{s.get('name', 'unknown')}' has non-string content type: {content_type}")
                print(f"  Content preview (first 100 chars): {content_str[:100]}...")
                content = content_str
            truncated_content = content[:200] + ('...' if len(content) > 200 else '')
            existing_children.append(f"{s.get('name', 'unnamed')}: {truncated_content}")
        children_info = "\n".join(existing_children)
        
        prompt = f"""Given a node "{node_to_extend.name}" with these existing children:
{children_info}

Generate {current_batch} new child node(s) that are RELATED to the existing children's theme. Requirements:
1. New nodes should explore related aspects not yet covered
2. Maintain thematic consistency with existing children
3. Each node needs name (with file extension) and content
4. Use varied file extensions based on content type (.txt, .md, .json, .py, .js, .html, .css, .yaml, etc.), and content should match the file type (e.g., .py files contain Python code, .json contains valid JSON)
5. Return in JSON format:

[
    {{"name": "related_topic1.txt", "content": "Content about related topic 1", "children": []}},
    {{"name": "related_topic2.txt", "content": "Content about related topic 2", "children": []}}
]

Filename format requirements:
- Only filenames with extensions (e.g., "report.pdf", "config.json")
- No slashes or special characters

IMPORTANT: Generate exactly {current_batch} node(s)."""
        
        try:
            print(f"Extending breadth (related) for node '{node_to_extend.name}': requesting {current_batch} nodes (batch mode, need {nodes_needed} total)")
            siblings_data = generate_with_retry(prompt, model, max_retries, expected_type="array", 
                                              expected_count=current_batch, allow_partial_results=True)
            
            if len(siblings_data) > 0:
                all_siblings.extend(siblings_data)
                nodes_needed -= len(siblings_data)
                print(f"Successfully generated {len(siblings_data)} nodes in this batch, {nodes_needed} remaining")
            else:
                if current_batch > 1:
                    batch_size = 1
                    print(f"No nodes generated with batch size {current_batch}, reducing to single node requests")
                else:
                    print(f"Failed to generate even single node for '{node_to_extend.name}', giving up")
                    break
                    
        except Exception as e:
            log_generation_error(f"breadth extension (related, batch size {current_batch})", node_to_extend.name, e)
            if batch_size > 1:
                batch_size = 1
                print(f"Error with batch size {current_batch}, reducing to single node requests")
            else:
                break
    
    print(f"Total generated {len(all_siblings)} related sibling nodes for '{node_to_extend.name}'")
    
    # Add all new children from response
    for child_data in all_siblings:
        child_node = FileTreeNode.from_dict(child_data, depth=node_to_extend.depth + 1, parent=node_to_extend)
        node_to_extend.children.append(child_node)
    
    return tree


def extend_breadth_unrelated(tree: FileTree, depth: int, nodes_needed: int, node_to_extend: FileTreeNode, model: str = "gpt-4o-mini", max_retries: int = 3) -> FileTree:
    """Extend tree breadth by adding UNRELATED siblings to create topic diversity."""
    if not node_to_extend.children:
        return tree

    batch_size = 2
    all_siblings = []

    while nodes_needed > 0:
        current_batch = min(nodes_needed, batch_size)

        existing_children = [f"{child.name}: {child.content[:200]}{'...' if len(child.content) > 200 else ''}" for child in node_to_extend.children]
        for s in all_siblings:
            content = s.get('content', '')
            if not isinstance(content, str):
                content_type = type(content).__name__
                content_str = str(content)
                print(f"Warning: Node '{s.get('name', 'unknown')}' has non-string content type: {content_type}")
                print(f"  Content preview (first 100 chars): {content_str[:100]}...")
                content = content_str
            truncated_content = content[:200] + ('...' if len(content) > 200 else '')
            existing_children.append(f"{s.get('name', 'unnamed')}: {truncated_content}")
        children_info = "\n".join(existing_children)
        
        prompt = f"""Given a node "{node_to_extend.name}" with these existing children:
{children_info}

Generate {current_batch} new sibling node(s) that are COMPLETELY UNRELATED to the existing siblings' theme. Requirements:
1. New nodes should be about entirely different topics
2. Avoid any thematic connection to existing siblings
3. Each node needs name (with file extension) and content
4. Use varied file extensions based on content type (.txt, .md, .json, .py, .js, .html, .css, .yaml, etc.), and content should match the file type (e.g., .py files contain Python code, .json contains valid JSON)
5. Return in JSON format:

[
    {{"name": "unrelated_topic1.txt", "content": "Content about completely different topic 1", "children": []}},
    {{"name": "unrelated_topic2.txt", "content": "Content about completely different topic 2", "children": []}}
]

Filename format requirements:
- Only filenames with extensions (e.g., "report.pdf", "config.json")
- No slashes or special characters

IMPORTANT: Generate exactly {current_batch} node(s) that are UNRELATED to the theme."""
        
        try:
            print(f"Extending breadth (unrelated) for node '{node_to_extend.name}': requesting {current_batch} nodes (batch mode, need {nodes_needed} total)")
            siblings_data = generate_with_retry(prompt, model, max_retries, expected_type="array", 
                                              expected_count=current_batch, allow_partial_results=True)
            
            if len(siblings_data) > 0:
                all_siblings.extend(siblings_data)
                nodes_needed -= len(siblings_data)
                print(f"Successfully generated {len(siblings_data)} nodes in this batch, {nodes_needed} remaining")
            else:
                if current_batch > 1:
                    batch_size = 1
                    print(f"No nodes generated with batch size {current_batch}, reducing to single node requests")
                else:
                    print(f"Failed to generate even single node for '{node_to_extend.name}', giving up")
                    break
                    
        except Exception as e:
            log_generation_error(f"breadth extension (unrelated, batch size {current_batch})", node_to_extend.name, e)
            if batch_size > 1:
                batch_size = 1
                print(f"Error with batch size {current_batch}, reducing to single node requests")
            else:
                break
    
    print(f"Total generated {len(all_siblings)} unrelated sibling nodes for '{node_to_extend.name}'")
    
    # Add all new children from response
    for child_data in all_siblings:
        child_node = FileTreeNode.from_dict(child_data, depth=node_to_extend.depth + 1, parent=node_to_extend)
        node_to_extend.children.append(child_node)
    
    return tree


def check_duplicate_names(tree: FileTree):
    """Print duplicate filenames if they appear in the tree."""
    def collect_all_names(node: FileTreeNode, names: List[str], path: str = ""):
        current_path = f"{path}/{node.name}" if path else node.name
        names.append((node.name, current_path))
        
        for child in node.children:
            collect_all_names(child, names, current_path)
    
    all_names = []
    collect_all_names(tree.root, all_names)
    
    name_counts = {}
    for name, path in all_names:
        if name not in name_counts:
            name_counts[name] = []
        name_counts[name].append(path)
    
    has_duplicates = False
    for name, paths in name_counts.items():
        if len(paths) > 1:
            has_duplicates = True
            print(f"Warning: filename '{name}' appears {len(paths)} times:")
            for path in paths:
                print(f"  - {path}")
    
    if not has_duplicates:
        print("No duplicate filenames found.")
