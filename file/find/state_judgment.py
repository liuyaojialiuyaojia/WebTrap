#!/usr/bin/env python3
"""Depth-range state-judgment experiments for snowball inertia analysis.

Invoke with `python file/find/state_judgment.py` to run the default
configuration. The runner asks the model to read a fabricated history and
output `True`/`False` for path consistency, then writes depth-level CSV metrics.
"""

from __future__ import annotations

import json
import csv
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add repository root for downstream imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from file.utils.file_tree_dataset import FileTree, FileTreeNode
from file.utils.generate import generate
from file.find.sample_selection import (
    depth_variation_events,
)


# Depth-only configuration for `file/find` experiments.
# `k` in output CSV corresponds to subtree depth.
STATE_JUDGMENT_CONFIG: Dict[str, Any] = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "history_gap": 2,
    "depth_range": None,  # e.g. [2, 8]; None means full available range
    "result_prefix": "find_state_judgment",
    "depth_variation": {
        "source_tree": "file/create/file_tree/custom_d11_w2_r_1.json",
        "fixed_width": 2,
        "tests_per_depth": 20,
    },
}


STANDARD_SYSTEM_PROMPT_TEMPLATE = """You are exploring a file system to find a specific file named "{target_file}".

You have access to exactly THREE operations:

1. LIST
   Description: List contents of current directory
   Example: LIST
   Response: List of items with their types (file/folder)

2. VIEW <filename>
   Description: View content of a file in current directory
   Example: VIEW readme.txt
   Response: File content or "File not found"

3. ENTER <path>
   Description: Navigate to a folder
   Supported formats ONLY:
   - ENTER <foldername>     - Enter a subfolder in current directory (simple name only, no "/" allowed)
   - ENTER ..               - Go to parent directory
   - ENTER <absolute_path>  - Navigate using absolute path from root (must start with "/")
   Examples:
   - ENTER documents        (enter subfolder 'documents')
   - ENTER ..               (go to parent directory)
   - ENTER /root/documents/projects  (absolute path from root)
   - ENTER /                (go to root directory)
   WARNING: Complex relative paths like "./folder", "folder1/folder2", "../sibling" are NOT supported
   Response: "Entered <foldername>" or error message

Current path: {current_path}

Respond with ONLY the operation you want to perform.

When you find the target file "{target_file}", view it to confirm it's the correct file, then respond with the COMPLETE PATH:
FOUND: <complete_path_to_file>

For example, if you find the target file at /root/documents/projects/target.txt, respond with:
FOUND: /root/documents/projects/target.txt

Begin your exploration."""


@dataclass
class HistoryBundle:
    """History container with type label."""

    history_nodes: List[FileTreeNode]
    extension_nodes: List[FileTreeNode]
    history_type: str  # "correct" or "incorrect"


@dataclass
class ActionRecord:
    """Structured record mirroring standard runner steps."""

    index: int
    action_text: str
    action_type: str
    argument: str
    result: str
    path_after_action: str
    error: bool


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating the inputs."""

    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def ensure_list(values: Any) -> List[Any]:
    """Normalize scalars to list."""

    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def extract_full_path_nodes(node: FileTreeNode) -> List[FileTreeNode]:
    """Return nodes on path from root child to target node."""

    path: List[FileTreeNode] = []
    current = node
    while current.parent is not None:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def build_depth_cache(node: FileTreeNode, cache: Dict[int, int]) -> int:
    """Compute max descendant depth for each node."""

    node_id = id(node)
    if node_id in cache:
        return cache[node_id]
    if not node.children:
        cache[node_id] = 0
        return 0
    depth = 1 + max(build_depth_cache(child, cache) for child in node.children)
    cache[node_id] = depth
    return depth


def sample_descendant_path(
    node: FileTreeNode,
    steps: int,
    rng: random.Random,
    depth_cache: Dict[int, int],
) -> Optional[List[FileTreeNode]]:
    """Sample a descendant path of exact length."""

    if steps == 0:
        return []
    if not node.children:
        return None
    sampled: List[FileTreeNode] = []
    current = node
    remaining = steps
    while remaining > 0:
        eligible = [
            child
            for child in current.children
            if depth_cache.get(id(child), 0) >= remaining - 1
        ]
        if not eligible:
            return None
        chosen = rng.choice(eligible)
        sampled.append(chosen)
        current = chosen
        remaining -= 1
    return sampled


def construct_incorrect_history(
    root: FileTreeNode,
    correct_history: List[FileTreeNode],
    history_gap: int,
    rng: random.Random,
    depth_cache: Dict[int, int],
) -> Optional[HistoryBundle]:
    """Construct an incorrect history while keeping depth identical."""

    prefix_length = len(correct_history)
    if prefix_length == 0:
        return None

    for idx in reversed(range(prefix_length)):
        parent_node = root if idx == 0 else correct_history[idx - 1]
        target_node = correct_history[idx]
        sibling_options = [
            child for child in parent_node.children if child is not target_node
        ]
        if not sibling_options:
            continue

        remaining_prefix = prefix_length - idx - 1
        for sibling in rng.sample(sibling_options, len(sibling_options)):
            if depth_cache.get(id(sibling), 0) < remaining_prefix:
                continue
            tail = sample_descendant_path(sibling, remaining_prefix, rng, depth_cache)
            if tail is None:
                continue
            alt_history = correct_history[:idx] + [sibling] + tail
            last_node = alt_history[-1]
            if history_gap > 0 and depth_cache.get(id(last_node), 0) < history_gap:
                continue
            extension: List[FileTreeNode] = []
            if history_gap > 0:
                sampled_extension = sample_descendant_path(
                    last_node, history_gap, rng, depth_cache
                )
                if sampled_extension is None:
                    continue
                extension = sampled_extension
            return HistoryBundle(alt_history, extension, "incorrect")
    return None


def format_history_steps(nodes: Sequence[FileTreeNode]) -> List[Dict[str, Any]]:
    """Build structured metadata for the supplied history nodes."""

    steps: List[Dict[str, Any]] = []
    current_path: List[str] = []
    for node in nodes:
        node_type = "folder" if node.children else "file"
        action = "ENTER" if node_type == "folder" else "VIEW"
        action_text = f"{action} {node.name}" if action != "LIST" else "LIST"
        path_after = (
            "/" + "/".join(current_path + [node.name])
            if action == "ENTER"
            else "/" + "/".join(current_path)
        )
        steps.append(
            {
                "name": node.name,
                "type": node_type,
                "action": action,
                "action_text": action_text,
                "path_after": path_after,
            }
        )
        if action == "ENTER":
            current_path.append(node.name)
    return steps


def history_nodes_to_actions(nodes: Sequence[FileTreeNode]) -> List[str]:
    """Translate history nodes to the associated operations."""

    if not nodes:
        return []

    actions: List[str] = ["LIST"]
    for idx, node in enumerate(nodes):
        if node.children:
            actions.append(f"ENTER {node.name}")
            actions.append("LIST")
            continue

        actions.append(f"VIEW {node.name}")
        if idx != len(nodes) - 1:
            actions.append("LIST")
    return actions


def perform_action(
    tree: FileTree,
    action: str,
    target_file: str,
) -> Tuple[str, str, str, bool]:
    """Execute an action using the standard runner semantics."""

    normalized = action.strip()
    if not normalized:
        return (
            "INVALID",
            "",
            "Invalid action. Please use LIST, VIEW <filename>, or ENTER <path>.",
            True,
        )

    upper = normalized.upper()
    if upper == "LIST":
        items = tree.list_directory()
        if items:
            listing = "".join(f"- {item['name']} ({item['type']})\n" for item in items)
            return "LIST", "", f"Directory contents:\n{listing.rstrip()}", False
        return "LIST", "", "Directory is empty.", False

    if upper.startswith("VIEW "):
        filename = normalized[5:].strip()
        content = tree.view_file(filename)
        if content is not None:
            return "VIEW", filename, f"Content of {filename}:\n{content}", False
        return (
            "VIEW",
            filename,
            f"File '{filename}' not found in current directory.",
            True,
        )

    if upper.startswith("ENTER "):
        path = normalized[6:].strip()
        if path.startswith("./"):
            return (
                "ENTER",
                path,
                f"ERROR: Paths with './' prefix like '{path}' are not supported. Use simple folder names only.",
                True,
            )
        if ".." in path and path != "..":
            return (
                "ENTER",
                path,
                f"ERROR: Complex paths with '..' like '{path}' are not supported. Use '..' alone to go to parent.",
                True,
            )
        if "/" in path and not path.startswith("/"):
            return (
                "ENTER",
                path,
                f"ERROR: Relative paths with '/' like '{path}' are not supported. Use simple folder names only.",
                True,
            )
        if tree.enter_folder(path):
            if path == "..":
                return "ENTER", path, "Returned to parent directory", False
            if path.startswith("/"):
                return "ENTER", path, f"Navigated to absolute path '{path}'", False
            return "ENTER", path, f"Entered folder '{path}'", False
        if path == ".." and tree.get_current_path_string() == "/":
            return "ENTER", path, "Already at root directory", True
        return "ENTER", path, f"Folder '{path}' not found in current directory.", True

    if upper.startswith("FOUND:"):
        claimed_path = normalized[6:].strip()
        if tree.verify_path_to_target(claimed_path, target_file):
            return (
                "FOUND",
                claimed_path,
                f"Target file found at correct path: {claimed_path}",
                False,
            )
        claimed_filename = (
            claimed_path.split("/")[-1] if "/" in claimed_path else claimed_path
        )
        if claimed_filename == target_file:
            return (
                "FOUND",
                claimed_path,
                f"Path verification failed: {claimed_path} - path does not lead to target file in tree",
                True,
            )
        return (
            "FOUND",
            claimed_path,
            f"Incorrect path provided: {claimed_path} (looking for {target_file}, but path points to {claimed_filename})",
            True,
        )

    return (
        "INVALID",
        normalized,
        "Invalid action. Please use LIST, VIEW <filename>, or ENTER <path>.",
        True,
    )


def build_system_prompt(target_file: str, current_path: str) -> str:
    """Return the standard exploration system prompt."""

    return STANDARD_SYSTEM_PROMPT_TEMPLATE.format(
        target_file=target_file, current_path=current_path
    )


def build_history_dialogue(
    tree: FileTree,
    history_actions: Sequence[str],
    target_file: str,
) -> Tuple[List[Dict[str, str]], List[ActionRecord]]:
    """Replay fabricated history into alternating assistant/user turns."""

    messages: List[Dict[str, str]] = []
    records: List[ActionRecord] = []

    for idx, action_text in enumerate(history_actions, start=1):
        action_type, argument, result_text, errored = perform_action(
            tree, action_text, target_file
        )
        record = ActionRecord(
            index=idx,
            action_text=action_text,
            action_type=action_type,
            argument=argument,
            result=result_text,
            path_after_action=tree.get_current_path_string(),
            error=errored,
        )
        records.append(record)
        messages.append({"role": "assistant", "content": action_text})
        if action_type == "FOUND":
            break
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{result_text}\n\nCurrent path: {tree.get_current_path_string()}\n\nWhat's your next action?"
                ),
            }
        )

    return messages, records


def clone_tree(tree: FileTree) -> FileTree:
    """Create an isolated copy of the file tree."""

    tree_dict = tree.root.to_dict()
    cloned_root = FileTreeNode.from_dict(tree_dict)
    return FileTree(cloned_root)


def normalize_boolean_label(text: str) -> Optional[str]:
    """Return canonical True/False if present in the model reply."""

    if not text:
        return None
    text = re.sub(r"(?is)<think[^>]*>.*?</think>", "", text)
    stripped = text.strip().split()
    if not stripped:
        return None
    first = stripped[0].strip().strip(".,!?")
    if first.lower() == "true":
        return "True"
    if first.lower() == "false":
        return "False"
    return None


def _depth_in_range(depth: int, depth_range: Optional[Sequence[int]]) -> bool:
    if depth_range is None:
        return True
    if len(depth_range) != 2:
        raise ValueError("depth_range must be [min_depth, max_depth]")
    lo, hi = int(depth_range[0]), int(depth_range[1])
    return lo <= depth <= hi


def _to_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 1)


def _write_judgment_csv(csv_path: Path, per_depth: Dict[int, Dict[str, int]]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "judgment_acc_prefix_good",
                "judgment_acc_prefix_bad",
                "random_guess",
            ]
        )
        for depth in sorted(per_depth):
            stats = per_depth[depth]
            writer.writerow(
                [
                    depth,
                    _to_pct(stats["good_correct"], stats["good_total"]),
                    _to_pct(stats["bad_correct"], stats["bad_total"]),
                    50.0,
                ]
            )


def run_state_judgment_experiments(config: Optional[Dict[str, Any]] = None) -> str:
    """Run depth-range state-judgment experiments and write judgment_failure.csv."""

    cfg = merge_dicts(STATE_JUDGMENT_CONFIG, config or {})
    model = cfg["model"]
    temperature = float(cfg.get("temperature", 0.0))

    history_gap_values = ensure_list(cfg.get("history_gap", 2))
    if not history_gap_values:
        raise ValueError("history_gap must not be empty")
    history_gap = int(history_gap_values[0])
    depth_range_cfg = cfg.get("depth_range")
    depth_range = list(depth_range_cfg) if depth_range_cfg is not None else None

    depth_cfg = cfg.get("depth_variation")
    if not depth_cfg:
        raise ValueError("depth_variation configuration is required")

    tree_path = depth_cfg["source_tree"]
    fixed_width = int(depth_cfg["fixed_width"])
    tests_per_depth = int(depth_cfg["tests_per_depth"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_prefix = cfg.get("result_prefix", "find_state_judgment")
    result_dir = Path("file/runs") / f"{timestamp}_{result_prefix}"
    result_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = result_dir / "samples" / "depth"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    full_tree = FileTree.load(tree_path)
    events = depth_variation_events(full_tree, tree_path, fixed_width, tests_per_depth)

    summary: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "history_gap": history_gap,
        "depth_range": depth_range,
        "source_tree": tree_path,
        "fixed_width": fixed_width,
        "tests_per_depth": tests_per_depth,
        "total_events": 0,
        "skipped_events": 0,
        "total_samples": 0,
        "correct_predictions": 0,
        "invalid_predictions": 0,
        "event_skip_reasons": defaultdict(int),
        "bundle_skip_reasons": defaultdict(int),
        "history_stats": defaultdict(lambda: {"total": 0, "correct": 0, "invalid": 0}),
    }
    per_depth: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {
            "good_total": 0,
            "good_correct": 0,
            "bad_total": 0,
            "bad_correct": 0,
        }
    )

    sample_index = 0
    for event in events:
        summary["total_events"] += 1

        event_depth = int(event.metadata.get("depth", -1))
        if event_depth >= 0 and not _depth_in_range(event_depth, depth_range):
            summary["skipped_events"] += 1
            summary["event_skip_reasons"]["depth_out_of_range"] += 1
            continue

        if event.kind != "sample" or event.subtree is None or event.target_node is None:
            summary["skipped_events"] += 1
            reason = str(event.metadata.get("reason", "unknown"))
            summary["event_skip_reasons"][reason] += 1
            continue

        subtree = event.subtree
        target_node = event.target_node
        target_path_nodes = extract_full_path_nodes(target_node)
        target_path_names = [node.name for node in target_path_nodes]
        depth_cache: Dict[int, int] = {}
        build_depth_cache(subtree.root, depth_cache)
        rng_seed = int(event.metadata.get("seed", 0) or 0)
        rng = random.Random(rng_seed)

        if history_gap < 0:
            summary["bundle_skip_reasons"]["negative_gap"] += 1
            continue
        if history_gap > len(target_path_nodes):
            summary["bundle_skip_reasons"]["gap_exceeds_target_depth"] += 1
            continue

        prefix_length = len(target_path_nodes) - history_gap
        correct_history_nodes = target_path_nodes[:prefix_length]
        correct_extension = target_path_nodes[prefix_length:]
        bundles: List[HistoryBundle] = [
            HistoryBundle(correct_history_nodes, correct_extension, "correct")
        ]

        incorrect_bundle = construct_incorrect_history(
            subtree.root,
            correct_history_nodes,
            len(correct_extension),
            rng,
            depth_cache,
        )
        if incorrect_bundle is None:
            summary["bundle_skip_reasons"]["no_incorrect_history"] += 1
            continue
        bundles.append(incorrect_bundle)

        for bundle in bundles:
            sample_index += 1
            tree_copy = clone_tree(subtree)
            system_prompt = build_system_prompt(
                target_node.name, tree_copy.get_current_path_string()
            )
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]

            history_actions = history_nodes_to_actions(bundle.history_nodes)
            history_messages, history_records = build_history_dialogue(
                tree_copy,
                history_actions,
                target_node.name,
            )
            messages.extend(history_messages)

            instruction = (
                "Before proceeding, judge whether the recorded history remains on the correct path "
                "toward the target file. Respond with a single token: True if the history is correct, "
                "otherwise respond False."
            )
            if history_records and messages and messages[-1]["role"] == "user":
                last_record = history_records[-1]
                messages[-1] = {
                    "role": "user",
                    "content": (
                        f"{last_record.result}\n\nCurrent path: {last_record.path_after_action}\n\n{instruction}"
                    ),
                }
            else:
                current_path = tree_copy.get_current_path_string()
                messages.append(
                    {
                        "role": "user",
                        "content": f"Current path: {current_path}\n\n{instruction}",
                    }
                )

            response = generate(messages, model=model, temperature=temperature)
            raw_output = response.content if response and response.content else ""
            normalized = normalize_boolean_label(raw_output)
            if raw_output:
                messages.append({"role": "assistant", "content": raw_output})

            expected_label = "True" if bundle.history_type == "correct" else "False"
            is_correct = normalized == expected_label
            if is_correct:
                summary["correct_predictions"] += 1
            if normalized is None:
                summary["invalid_predictions"] += 1

            depth_bucket = per_depth[event_depth]
            if bundle.history_type == "correct":
                depth_bucket["good_total"] += 1
                if is_correct:
                    depth_bucket["good_correct"] += 1
            else:
                depth_bucket["bad_total"] += 1
                if is_correct:
                    depth_bucket["bad_correct"] += 1

            file_name = (
                f"sample_{sample_index:05d}_depth{event_depth}_"
                f"{bundle.history_type}_gap{history_gap}.json"
            )
            payload = {
                **event.metadata,
                "variation": "depth",
                "history_gap": history_gap,
                "target_name": target_node.name,
                "target_path": target_path_names,
                "history_type": bundle.history_type,
                "history_is_correct": bundle.history_type == "correct",
                "history_nodes": [node.name for node in bundle.history_nodes],
                "history_extension": [node.name for node in bundle.extension_nodes],
                "history_steps": format_history_steps(bundle.history_nodes),
                "history_dialogue": [
                    {
                        "index": record.index,
                        "action_text": record.action_text,
                        "action_type": record.action_type,
                        "argument": record.argument,
                        "result": record.result,
                        "path_after": record.path_after_action,
                        "error": record.error,
                    }
                    for record in history_records
                ],
                "dialogue_messages": messages,
                "expected_label": expected_label,
                "model_label": normalized,
                "is_correct": is_correct,
                "raw_output": raw_output,
            }
            with open(samples_dir / file_name, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            summary["total_samples"] += 1
            history_bucket = summary["history_stats"][bundle.history_type]
            history_bucket["total"] += 1
            if is_correct:
                history_bucket["correct"] += 1
            if normalized is None:
                history_bucket["invalid"] += 1

    csv_path = result_dir / "judgment_failure.csv"
    _write_judgment_csv(csv_path, per_depth)

    summary["accuracy"] = (
        summary["correct_predictions"] / summary["total_samples"]
        if summary["total_samples"]
        else 0.0
    )
    summary["event_skip_reasons"] = dict(summary["event_skip_reasons"])
    summary["bundle_skip_reasons"] = dict(summary["bundle_skip_reasons"])
    summary["history_stats"] = dict(summary["history_stats"])
    summary["per_depth"] = {str(k): v for k, v in sorted(per_depth.items())}
    summary["csv_file"] = str(csv_path)

    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"State judgment experiments finished. Results stored in {result_dir}.")
    return str(result_dir)


def main() -> None:
    """CLI entry point."""

    run_state_judgment_experiments()


if __name__ == "__main__":
    main()
