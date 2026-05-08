#!/usr/bin/env python3
"""Parallel file-tree generation helpers."""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from file.utils.file_tree_dataset import FileTree, FileTreeNode
from file.utils.generate import generate
from file.create.make_file_tree_dataset import (
    create_minimal_tree,
    _collect_nodes_at_depth,
    check_duplicate_names
)
from file.create.tree_generation_utils import generate_with_retry, log_generation_error
from file.create.tree_validation import validate_tree_structure


def create_fan_tree_parallel(depth: int, width: int, breadth_mode: str = "mixed", related_ratio: float = 0.7, 
                           model: str = "gpt-4o-mini", max_workers: int = 8, max_retries: int = 3) -> FileTree:
    """Create a fan-shaped tree using parallel API calls.

    Nodes at the same level are expanded concurrently while preserving the same
    output structure as the sequential generator.

    Args:
        depth: Tree depth.
        width: Number of children for each non-leaf node.
        breadth_mode: "related", "unrelated", or "mixed".
        related_ratio: Probability of related expansion in mixed mode.
        model: Model used for generation.
        max_workers: Maximum number of worker threads.
    """
    tree = create_minimal_tree(model, max_retries)
    
    if depth <= 1:
        return tree
    
    total_api_calls = 1
    for d in range(1, depth + 1):
        nodes_at_depth = width ** (d - 1)
        total_api_calls += nodes_at_depth
        total_api_calls += nodes_at_depth
    
    with tqdm(total=total_api_calls, desc=f"Generating fan tree in parallel (depth={depth}, width={width})") as pbar:
        pbar.update(1)
        
        for current_depth in range(1, depth + 1):
            nodes_to_extend = []
            _collect_nodes_at_depth(tree.root, 0, current_depth - 1, nodes_to_extend)
            
            _parallel_extend_depth_batch(tree, nodes_to_extend, model, pbar, max_workers, max_retries)
            
            _parallel_extend_breadth_batch(tree, nodes_to_extend, breadth_mode, related_ratio, 
                                         width, current_depth, model, pbar, max_workers, max_retries)
    
    # Final validation
    is_valid, error_msg = validate_tree_structure(tree, depth, width)
    if not is_valid:
        print(f"Warning: Generated tree failed validation: {error_msg}")
    
    return tree


def _parallel_extend_depth_batch(tree: FileTree, nodes_to_extend: List[FileTreeNode], 
                               model: str, pbar: tqdm, max_workers: int = 8, max_retries: int = 3) -> None:
    """Extend depth for a batch of nodes in parallel."""
    def extend_single_node(node: FileTreeNode) -> Tuple[FileTreeNode, Optional[Dict]]:
        """Return generated child data for one node."""
        path = tree.get_path_to_node(node)
        
        path_description = " -> ".join(path)
        
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
            return node, child_data
        except Exception as e:
            log_generation_error("parallel depth extension", node.name, e)
            return node, None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extend_single_node, node): node for node in nodes_to_extend}
        
        for future in as_completed(futures):
            node, child_data = future.result()
            if child_data:
                child_node = FileTreeNode.from_dict(child_data, depth=node.depth + 1, parent=node)
                node.children.append(child_node)
            pbar.update(1)


def _parallel_extend_breadth_batch(tree: FileTree, nodes_to_extend: List[FileTreeNode], breadth_mode: str,
                                 related_ratio: float, width: int, current_depth: int, 
                                 model: str, pbar: tqdm, max_workers: int = 8, max_retries: int = 3) -> None:
    """Extend breadth for a batch of nodes in parallel."""
    def extend_single_node_breadth(node: FileTreeNode) -> Tuple[FileTreeNode, List[Dict]]:
        """Return generated sibling data for one node."""
        if len(node.children) >= width:
            return node, []
        
        nodes_needed = width - len(node.children)
        
        batch_size = 2
        all_siblings = []
        
        while nodes_needed > 0:
            current_batch = min(nodes_needed, batch_size)
            
            existing_children = [f"{child.name}: {child.content[:200]}{'...' if len(child.content) > 200 else ''}" for child in node.children]
            for s in all_siblings:
                content = s.get('content', '')
                if not isinstance(content, str):
                    import json
                    content = json.dumps(content, ensure_ascii=False, indent=2)
                truncated_content = content[:200] + ('...' if len(content) > 200 else '')
                existing_children.append(f"{s.get('name', 'unnamed')}: {truncated_content}")
            children_info = "\n".join(existing_children)
            
            if breadth_mode == "related" or (breadth_mode == "mixed" and random.random() < related_ratio):
                prompt = f"""Given a node "{node.name}" with these existing children:
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
            else:
                prompt = f"""Given a node "{node.name}" with these existing children:
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
                print(f"Parallel breadth extension for node '{node.name}': requesting {current_batch} nodes (batch mode, need {nodes_needed} total)")
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
                        print(f"Failed to generate even single node for '{node.name}', giving up")
                        break
                        
            except Exception as e:
                log_generation_error(f"parallel breadth extension (batch size {current_batch})", node.name, e)
                if batch_size > 1:
                    batch_size = 1
                    print(f"Error with batch size {current_batch}, reducing to single node requests")
                else:
                    break
        
        print(f"Total generated {len(all_siblings)} sibling nodes for '{node.name}'")
        return node, all_siblings
    
    nodes_needing_extension = [node for node in nodes_to_extend if len(node.children) < width]
    
    if not nodes_needing_extension:
        return
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extend_single_node_breadth, node): node for node in nodes_needing_extension}
        
        for future in as_completed(futures):
            node, siblings_data = future.result()
            if siblings_data:
                for child_data in siblings_data:
                    child_node = FileTreeNode.from_dict(child_data, depth=node.depth + 1, parent=node)
                    node.children.append(child_node)
            pbar.update(1)
