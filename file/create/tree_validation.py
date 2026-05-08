#!/usr/bin/env python3
"""Validation utilities for file tree structures.

This module provides functions to validate tree structures, clean LLM responses,
and verify node counts.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from file.utils.file_tree_dataset import FileTree, FileTreeNode


def validate_tree_structure(tree: FileTree, expected_depth: int, expected_width: int) -> Tuple[bool, str]:
    """Validate that the tree matches expected depth and width.
    
    Args:
        tree: The file tree to validate
        expected_depth: Expected maximum depth of the tree
        expected_width: Expected width (children per non-leaf node)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    def get_tree_depth(node: FileTreeNode, current_depth: int = 0) -> int:
        """Get the maximum depth of the tree."""
        if not node.children:
            return current_depth
        return max(get_tree_depth(child, current_depth + 1) for child in node.children)
    
    def validate_width(node: FileTreeNode, depth: int, max_depth: int) -> List[str]:
        """Validate width at each non-leaf node."""
        errors = []
        
        # Skip leaf nodes
        if depth >= max_depth - 1:
            return errors
            
        if len(node.children) != expected_width:
            errors.append(f"Node '{node.name}' at depth {depth} has {len(node.children)} children, expected {expected_width}")
            
        for child in node.children:
            errors.extend(validate_width(child, depth + 1, max_depth))
            
        return errors
    
    # Check depth
    actual_depth = get_tree_depth(tree.root)
    if actual_depth != expected_depth:
        return False, f"Tree depth is {actual_depth}, expected {expected_depth}"
    
    # Check width
    width_errors = validate_width(tree.root, 0, expected_depth)
    if width_errors:
        return False, "Width validation failed:\n" + "\n".join(width_errors[:5])  # Limit to first 5 errors
    
    return True, ""


def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from LLM response.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Cleaned JSON string
    """
    content = response_text.strip()
    
    # Remove markdown code blocks
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end != -1:
            content = content[start:end].strip()
    
    # Extract JSON object or array
    if not content.startswith("{") and not content.startswith("["):
        # Try to find JSON object
        obj_start = content.find("{")
        arr_start = content.find("[")
        
        if obj_start == -1 and arr_start == -1:
            return content
            
        # Choose the earlier starting position
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            # Extract JSON object
            brace_count = 0
            end = obj_start
            for i in range(obj_start, len(content)):
                if content[i] == "{":
                    brace_count += 1
                elif content[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            content = content[obj_start:end]
        else:
            # Extract JSON array
            bracket_count = 0
            end = arr_start
            for i in range(arr_start, len(content)):
                if content[i] == "[":
                    bracket_count += 1
                elif content[i] == "]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
            content = content[arr_start:end]
    
    return content


def validate_node_count(nodes: List[Dict[str, Any]], expected_count: int) -> Tuple[bool, str]:
    """Validate that the correct number of nodes were generated.
    
    Args:
        nodes: List of node dictionaries
        expected_count: Expected number of nodes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    actual_count = len(nodes)
    if actual_count != expected_count:
        return False, f"Generated {actual_count} nodes, expected {expected_count}"
    
    # Validate each node has required fields
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            return False, f"Node {i} is not a dictionary"
        
        if "name" not in node:
            return False, f"Node {i} missing 'name' field"
        
        if "content" not in node:
            return False, f"Node {i} missing 'content' field"
        
        if "children" not in node:
            return False, f"Node {i} missing 'children' field"

        # Validate filename
        is_valid, error_msg = validate_filename(node["name"], strict=True)
        if not is_valid:
            return False, f"Node {i} has invalid filename '{node['name']}': {error_msg}"

    return True, ""


def parse_json_response(response_text: str, expected_type: str = "object") -> Tuple[Optional[Any], str]:
    """Parse JSON response with error handling.
    
    Args:
        response_text: Raw response text
        expected_type: "object" for single node, "array" for multiple nodes
        
    Returns:
        Tuple of (parsed_data, error_message)
    """
    try:
        cleaned = clean_json_response(response_text)
        parsed = json.loads(cleaned)
        
        if expected_type == "object" and not isinstance(parsed, dict):
            return None, f"Expected JSON object, got {type(parsed).__name__}"
        elif expected_type == "array" and not isinstance(parsed, list):
            return None, f"Expected JSON array, got {type(parsed).__name__}"
            
        return parsed, ""
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def validate_filename(name: str, strict: bool = True) -> Tuple[bool, str]:
    """Validate filename according to filesystem and security requirements.

    Args:
        name: The filename to validate
        strict: If True, apply strict validation rules for special characters and reserved names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name or not isinstance(name, str):
        return False, "Filename cannot be empty or non-string"

    # Strip leading/trailing spaces for testing
    stripped_name = name.strip()
    if name != stripped_name:
        return False, "Filename cannot have leading or trailing spaces"

    # Check length
    if len(name) >= 255:
        return False, f"Filename too long ({len(name)} characters, max 254)"

    # Check for control characters
    for i, char in enumerate(name):
        char_code = ord(char)
        if char_code < 32 or char_code == 127:
            return False, f"Filename contains control character at position {i} (code {char_code})"

    # Check for path separators
    if "/" in name or "\\" in name:
        return False, "Filename cannot contain path separators (/ or \\)"

    # Check for path traversal sequences
    if name == ".." or name == ".":
        return False, "Filename cannot be '.' or '..'"
    if name.startswith("../") or name.startswith("./") or "/../" in name or "/." in name:
        return False, "Filename cannot contain path traversal sequences"

    if strict:
        # Check for special characters that can cause issues
        special_chars = set('<>:"|?*')
        found_special = [char for char in name if char in special_chars]
        if found_special:
            return False, f"Filename contains invalid characters: {', '.join(found_special)}"

        # Check for Windows reserved names (case-insensitive)
        reserved_names = {
            "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
        }

        # Check the name without extension
        name_base = name.split('.')[0] if '.' in name else name
        if name_base.upper() in reserved_names:
            return False, f"Filename uses reserved name: {name_base}"

    # Check for file extension (except for "root")
    if name != "root":
        if '.' not in name:
            return False, "Filename must have an extension (except for 'root')"

        # Check for consecutive dots or ending with dot
        if ".." in name:
            return False, "Filename cannot contain consecutive dots (..)"
        if name.endswith('.'):
            return False, "Filename cannot end with a dot"

        # Validate extension exists and is reasonable
        parts = name.split('.')
        if len(parts) < 2:
            return False, "Filename must have an extension"
        extension = parts[-1]
        if not extension or len(extension) > 10:
            return False, "Invalid file extension"

    return True, ""


def validate_tree_filenames(tree_dict: dict) -> Tuple[bool, List[str]]:
    """Recursively validate all filenames in a tree structure.

    Args:
        tree_dict: Dictionary representation of the tree

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    def validate_node_recursive(node: dict, path: List[str] = None) -> None:
        if path is None:
            path = []

        # Validate the current node's name
        if "name" in node:
            current_path = path + [node["name"]]
            is_valid, error_msg = validate_filename(node["name"], strict=True)
            if not is_valid:
                path_str = " -> ".join(current_path)
                errors.append(f"Invalid filename at path '{path_str}': {error_msg}")

        # Recursively validate children
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                if isinstance(child, dict):
                    validate_node_recursive(child, current_path)

    # Validate the tree starting from root
    if isinstance(tree_dict, dict):
        validate_node_recursive(tree_dict)
    else:
        errors.append("Tree structure is not a dictionary")

    return len(errors) == 0, errors