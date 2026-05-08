#!/usr/bin/env python3
"""Utilities for tree generation with retry logic.

This module provides retry wrappers and response validation for tree generation.
"""

import time
from functools import wraps
from typing import Callable, TypeVar, Optional, Any, Dict, List

T = TypeVar('T')


def retry_with_backoff(max_retries: int = 3) -> Callable:
    """Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Calculate backoff time: 1s, 2s, 4s
                        wait_time = 2 ** attempt
                        print(f"Retry attempt {attempt + 1}/{max_retries} after error: {e}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed after {max_retries} retries: {e}")
                        
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(f"Failed after {max_retries} retries")
                
        return wrapper
    return decorator


def validate_and_parse_response(response: Any, expected_type: str = "object", expected_count: Optional[int] = None, allow_partial: bool = False) -> Any:
    """Validate and parse LLM response with enhanced error handling.
    
    Args:
        response: Response object from LLM
        expected_type: "object" for single node, "array" for multiple nodes
        expected_count: Expected number of nodes (for array type)
        allow_partial: If True, accept partial results when fewer nodes than expected are returned
        
    Returns:
        For allow_partial=False: Parsed data
        For allow_partial=True: Tuple of (result, is_partial, actual_count)
        
    Raises:
        ValueError: If validation fails
    """
    from file.create.tree_validation import parse_json_response, validate_node_count
    
    if not response or not response.content:
        raise ValueError("Empty response from LLM")
    
    # Parse JSON response
    parsed_data, error = parse_json_response(response.content, expected_type)
    if error:
        raise ValueError(f"Response parsing failed: {error}")
    
    # Validate node structure for arrays (regardless of count)
    if expected_type == "array" and isinstance(parsed_data, list):
        for i, node in enumerate(parsed_data):
            if not isinstance(node, dict):
                raise ValueError(f"Node {i} is not a dictionary")
            if "name" not in node:
                raise ValueError(f"Node {i} missing 'name' field")
            if "content" not in node:
                raise ValueError(f"Node {i} missing 'content' field")
            if "children" not in node:
                raise ValueError(f"Node {i} missing 'children' field")

    # Additional validation for expected count
    if expected_type == "array" and expected_count is not None:
        is_valid, error = validate_node_count(parsed_data, expected_count)
        if not is_valid:
            if allow_partial and len(parsed_data) < expected_count and len(parsed_data) > 0:
                # Return partial result with metadata
                actual_count = len(parsed_data)
                is_partial = actual_count < expected_count
                return (parsed_data, is_partial, actual_count)
            else:
                raise ValueError(f"Node count validation failed: {error}")
    
    # Validate filenames in the parsed data
    from file.create.tree_validation import validate_filename

    def validate_filenames_in_data(data):
        """Validate filenames in parsed data structure."""
        validation_errors = []

        if expected_type == "object" and isinstance(data, dict):
            # Single node validation
            if "name" in data:
                is_valid, error_msg = validate_filename(data["name"], strict=True)
                if not is_valid:
                    validation_errors.append(f"Invalid filename '{data['name']}': {error_msg}")
        elif expected_type == "array" and isinstance(data, list):
            # Multiple nodes validation
            for i, node in enumerate(data):
                if isinstance(node, dict) and "name" in node:
                    is_valid, error_msg = validate_filename(node["name"], strict=True)
                    if not is_valid:
                        validation_errors.append(f"Node {i} invalid filename '{node['name']}': {error_msg}")

        return validation_errors

    # Validate filenames
    filename_errors = validate_filenames_in_data(parsed_data)
    if filename_errors:
        error_summary = "; ".join(filename_errors)
        print(f"Warning: Filename validation failed: {error_summary}")
        # Don't raise an error, just log the warning to allow the generation to continue
        # but still notify about the validation issues

    if allow_partial:
        # For non-array types or when expected_count is None, return with metadata
        return (parsed_data, False, 1 if expected_type == "object" else len(parsed_data))

    return parsed_data


def generate_nodes_incrementally(base_prompt: str, model: str, expected_count: int, 
                                existing_nodes: List[Dict] = None, max_retries: int = 3) -> List[Dict]:
    """Generate nodes incrementally, accumulating partial results across attempts.
    
    Args:
        base_prompt: Base prompt template (should contain placeholders for count)
        model: The model to use
        expected_count: Total number of nodes needed
        existing_nodes: Previously generated nodes to build upon
        max_retries: Maximum number of retry attempts per request
        
    Returns:
        Complete list of nodes
    """
    if existing_nodes is None:
        existing_nodes = []
    
    accumulated_nodes = existing_nodes.copy()
    
    while len(accumulated_nodes) < expected_count:
        remaining_count = expected_count - len(accumulated_nodes)
        
        # Update prompt with current remaining count
        current_prompt = base_prompt.replace(f"Generate {expected_count}", f"Generate {remaining_count}")
        current_prompt = current_prompt.replace(f"exactly {expected_count}", f"exactly {remaining_count}")
        
        # Get existing node names to avoid duplicates
        existing_names = {node["name"] for node in accumulated_nodes}
        
        # Add note about avoiding duplicates if we have existing nodes
        if accumulated_nodes:
            existing_names_list = ", ".join(f'"{name}"' for name in existing_names)
            duplicate_note = f"\n\nIMPORTANT: Avoid duplicate names. Do not use these existing names: {existing_names_list}"
            current_prompt += duplicate_note
        
        print(f"Requesting {remaining_count} more nodes (have {len(accumulated_nodes)}/{expected_count})")
        
        try:
            @retry_with_backoff(max_retries)
            def _generate_partial():
                from file.utils.generate import generate

                response = generate([{"role": "user", "content": current_prompt}], model)
                return validate_and_parse_response(response, "array", remaining_count, allow_partial=True)
            
            result, is_partial, actual_count = _generate_partial()
            
            # Filter out duplicates by name
            new_nodes = []
            for node in result:
                if node["name"] not in existing_names:
                    new_nodes.append(node)
                    existing_names.add(node["name"])
                else:
                    print(f"Warning: Skipping duplicate node name '{node['name']}'")
            
            accumulated_nodes.extend(new_nodes)
            
            if new_nodes:
                print(f"Successfully added {len(new_nodes)} nodes (total: {len(accumulated_nodes)}/{expected_count})")
            else:
                print(f"Warning: No new unique nodes obtained in this attempt")
                break  # Avoid infinite loop if model keeps generating duplicates
            
        except Exception as e:
            print(f"Error generating remaining {remaining_count} nodes: {e}")
            break  # Exit on error to avoid infinite loop
    
    if len(accumulated_nodes) < expected_count:
        print(f"Warning: Only generated {len(accumulated_nodes)} nodes out of {expected_count} expected")
    
    return accumulated_nodes


def generate_with_retry(prompt: str, model: str, max_retries: int = 3, 
                       expected_type: str = "object", expected_count: Optional[int] = None,
                       allow_partial_results: bool = False) -> Any:
    """Generate LLM response with retry logic and validation.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use
        max_retries: Maximum number of retry attempts
        expected_type: "object" or "array"
        expected_count: Expected number of nodes (for array type)
        allow_partial_results: If True, use incremental generation for missing nodes
        
    Returns:
        Validated and parsed response data
    """
    from file.utils.generate import generate
    
    # Use incremental generation for arrays when partial results are allowed
    if allow_partial_results and expected_type == "array" and expected_count is not None:
        return generate_nodes_incrementally(prompt, model, expected_count, None, max_retries)
    
    # Original behavior for other cases
    @retry_with_backoff(max_retries)
    def _generate_and_validate():
        response = generate([{"role": "user", "content": prompt}], model)
        return validate_and_parse_response(response, expected_type, expected_count)
    
    return _generate_and_validate()


def log_generation_error(operation: str, node_name: str, error: Exception) -> None:
    """Log generation errors in a consistent format.
    
    Args:
        operation: The operation that failed (e.g., "depth extension", "breadth extension")
        node_name: Name of the node being processed
        error: The exception that occurred
    """
    print(f"Error during {operation} for node '{node_name}': {error}")
