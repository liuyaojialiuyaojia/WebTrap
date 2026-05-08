#!/usr/bin/env python3
"""Build exp1 single environment and tasks from top-k attack cases."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.exp1.build_security_tree_from_spec import build_security_tree_from_spec_json
from file.exp.exp1.build_user_tree import build_user_tree_json, parse_root_coord
from file.exp.exp1.make_user_task_question import generate_user_question_for_leaf
from file.exp.exp1.merge_security_tree import merge_security_tree_json
from file.exp.pipeline_common import (
    case_id_from_index,
    to_posix,
    write_json,
    write_jsonl,
)

DEFAULT_DS_JSON_PATH = REPO_ROOT / "file" / "exp" / "exp1" / "data" / "ds_top20.json"
DEFAULT_SECURITY_SPEC_PATH = (
    REPO_ROOT / "file" / "exp" / "exp1" / "data" / "security_spec.filetree_ds_top20.json"
)
DEFAULT_SOURCE_TREE_PATH = REPO_ROOT / "file" / "create" / "file_tree" / "custom_d11_w2_r_1.json"
DEFAULT_RUNS_ROOT = REPO_ROOT / "file" / "runs"


def _load_ds_cases(ds_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(ds_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("ds_top20.json must be a JSON array of sample objects.")
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(payload):
        if not isinstance(sample, dict):
            raise ValueError(f"Invalid sample at index {idx}: expected object.")
        rows.append(sample)
    return rows


def _build_env_path_indexes(
    tree: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, int]]:
    logical_to_index: dict[str, str] = {}
    logical_to_node: dict[str, dict[str, Any]] = {}
    leaf_name_counts: dict[str, int] = {}

    def walk(node: dict[str, Any], logical_path: str, index_path: str) -> None:
        logical_to_index[logical_path] = index_path
        logical_to_node[logical_path] = node

        children = node.get("children") or []
        if not children:
            name = str(node.get("name") or "").strip()
            if name:
                leaf_name_counts[name] = leaf_name_counts.get(name, 0) + 1
            return

        for idx, child in enumerate(children):
            name = str(child.get("name") or "").strip()
            if not name:
                continue
            child_logical = f"{logical_path}/{name}".replace("//", "/")
            child_index_path = f"{index_path}/{idx}" if index_path != "/" else f"/{idx}"
            walk(child, child_logical, child_index_path)

    root_name = str(tree.get("name") or "").strip()
    if not root_name:
        raise ValueError("Env tree root node must have a non-empty name.")
    walk(tree, f"/{root_name}".replace("//", "/"), "/")
    return logical_to_index, logical_to_node, leaf_name_counts


def _validate_node_name_for_filesystem(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError("Tree node name cannot be empty when materializing filesystem.")
    if "/" in value or "\\" in value:
        raise ValueError(
            f"Tree node name contains path separators and cannot be materialized: {value}"
        )
    if value in {".", ".."}:
        raise ValueError(f"Invalid tree node name for filesystem materialization: {value}")
    return value


def _materialize_node_to_filesystem(node: dict[str, Any], destination: Path) -> tuple[int, int]:
    children = node.get("children") or []
    if isinstance(children, list) and children:
        destination.mkdir(parents=True, exist_ok=True)
        dir_count = 1
        file_count = 0
        for child in children:
            if not isinstance(child, dict):
                continue
            child_name = _validate_node_name_for_filesystem(str(child.get("name") or ""))
            child_path = destination / child_name
            child_dirs, child_files = _materialize_node_to_filesystem(child, child_path)
            dir_count += child_dirs
            file_count += child_files
        return dir_count, file_count

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(node.get("content") or ""), encoding="utf-8")
    return 0, 1


def _materialize_env_tree_filesystem(*, env_tree: dict[str, Any], fs_root: Path) -> dict[str, int]:
    root_name = str(env_tree.get("name") or "").strip()
    if root_name != "root":
        raise ValueError(
            f"Expected merged env tree root node to be 'root', got: {root_name or '<empty>'}"
        )

    if fs_root.exists():
        shutil.rmtree(fs_root)
    fs_root.mkdir(parents=True, exist_ok=True)

    children = env_tree.get("children") or []
    if not isinstance(children, list):
        raise ValueError("Merged env tree root children must be a list.")

    dir_count = 1  # include root
    file_count = 0
    for child in children:
        if not isinstance(child, dict):
            continue
        child_name = _validate_node_name_for_filesystem(str(child.get("name") or ""))
        child_path = fs_root / child_name
        child_dirs, child_files = _materialize_node_to_filesystem(child, child_path)
        dir_count += child_dirs
        file_count += child_files

    return {"directories": dir_count, "files": file_count}


def _logical_path_to_filesystem_path(*, logical_path: str, fs_root: Path) -> Path:
    raw = str(logical_path or "").strip()
    if not raw.startswith("/root"):
        raise ValueError(f"Logical path must start with /root: {raw}")

    suffix = raw[len("/root") :]
    if suffix in {"", "/"}:
        return fs_root.resolve()

    relative_text = suffix.lstrip("/")
    relative_path = Path(relative_text)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe logical path for filesystem conversion: {logical_path}")
    return (fs_root / relative_path).resolve()


def _select_single_user_target_logical_path(
    *,
    user_manifest: dict[str, Any],
    env_leaf_name_counts: dict[str, int],
) -> str:
    files = list(user_manifest.get("user_leaf_files", []))
    if not files:
        raise ValueError("User tree has no leaf files; cannot assign user target.")

    # Prefer right-to-left leaves and enforce unique filename in whole env tree.
    for logical_path in reversed(files):
        filename = Path(str(logical_path)).name
        if env_leaf_name_counts.get(filename, 0) == 1:
            return str(logical_path)

    raise ValueError(
        "No eligible user target leaf found: right-to-left scan could not find a "
        "leaf with a unique filename in the current env tree."
    )


def _resolve_target_for_case(
    *,
    merged_security_manifest: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    target_index = merged_security_manifest.get("target_index") or {}
    if (
        isinstance(target_index, dict)
        and case_id in target_index
        and isinstance(target_index[case_id], dict)
    ):
        return dict(target_index[case_id])

    raise KeyError(
        f"No security target found for case_id={case_id}. "
        "Security spec must provide one target per case_id."
    )


def _default_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_file_ds_top20_single_env_tree"


def _init_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    for rel in ["env", "task/user", "task/attacker"]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)


def build_cases_top20(
    *,
    run_dir: Path,
    ds_json_path: Path,
    security_spec_path: Path,
    source_tree: Path,
    top_k: int,
    depth: int,
    width: int,
    seed_base: int,
    root_coord: Optional[tuple[int, int]],
    anchor_index_path: str,
    user_question_model: str,
    user_question_cache_ttl: Optional[int],
    user_question_max_tokens: int,
    user_question_temperature: float,
    user_question_body_max_input_tokens: int,
) -> dict[str, Any]:
    all_cases = _load_ds_cases(ds_json_path)
    selected_cases = all_cases[:top_k]
    if len(selected_cases) < top_k:
        raise ValueError(f"Requested top_k={top_k} but only {len(selected_cases)} cases found.")

    user_result = build_user_tree_json(
        source_tree_path=source_tree,
        depth=depth,
        width=width,
        seed=seed_base,
        root_coord=root_coord,
    )
    user_tree = user_result["tree"]
    user_manifest = user_result["manifest"]

    security_result = build_security_tree_from_spec_json(spec_path=security_spec_path)

    merged_result = merge_security_tree_json(
        user_tree=user_tree,
        security_tree=security_result["tree"],
        security_manifest=security_result["manifest"],
        anchor_index_path=anchor_index_path,
        overwrite=False,
    )
    env_tree = merged_result["tree"]
    security_manifest = merged_result["manifest"]
    env_root_name = str(env_tree.get("name") or "").strip()
    if not env_root_name:
        raise ValueError("Merged env tree root node must have a non-empty name.")
    env_root_logical_path = f"/{env_root_name}".replace("//", "/")
    anchor_logical_resolved = str(security_manifest["anchor_logical_path"])
    anchor_index_resolved = str(security_manifest["anchor_index_path"])

    env_tree_path = run_dir / "env" / "env_pre_injection_tree.json"
    write_json(env_tree_path, env_tree)

    env_logical_to_index, env_logical_to_node, env_leaf_name_counts = _build_env_path_indexes(env_tree)
    security_entry_logical = str(security_manifest["entry_logical_path"])
    security_entry_index = env_logical_to_index.get(security_entry_logical, "")
    security_entry_node = env_logical_to_node.get(security_entry_logical)
    if not isinstance(security_entry_node, dict) or not security_entry_node.get("children"):
        raise ValueError(
            f"Security entry logical path does not resolve to a directory node: {security_entry_logical}"
        )
    if not security_entry_index:
        raise ValueError(
            f"Security entry logical path is missing from env index: {security_entry_logical}"
        )

    user_target_logical_path = _select_single_user_target_logical_path(
        user_manifest=user_manifest,
        env_leaf_name_counts=env_leaf_name_counts,
    )
    user_target_index_path = env_logical_to_index.get(user_target_logical_path, "")
    user_target_node = env_logical_to_node.get(user_target_logical_path)
    if not isinstance(user_target_node, dict) or user_target_node.get("children"):
        raise ValueError(
            f"Resolved user target is not a leaf node: {user_target_logical_path}"
        )
    if not user_target_index_path:
        raise ValueError(
            f"User target logical path is missing from env index: {user_target_logical_path}"
        )

    target_content = str(user_target_node.get("content") or "")

    user_question, _ = generate_user_question_for_leaf(
        target_logical_path=user_target_logical_path,
        target_content=target_content,
        model=user_question_model,
        cache_ttl=user_question_cache_ttl,
        max_tokens=user_question_max_tokens,
        temperature=user_question_temperature,
        body_max_input_tokens=user_question_body_max_input_tokens,
    )

    user_task_payload = {
        "task_id": "user_task_single",
        "question": user_question,
        "target_logical_path": user_target_logical_path,
        "target_index_path": user_target_index_path,
        "target_filename": Path(user_target_logical_path).name,
    }
    user_task_path = run_dir / "task" / "user" / "user_task.json"
    write_json(user_task_path, user_task_payload)

    attack_cases_rows: list[dict[str, Any]] = []

    for idx, sample in enumerate(selected_cases):
        case_id = case_id_from_index(idx)
        target_row = _resolve_target_for_case(
            merged_security_manifest=security_manifest,
            case_id=case_id,
        )

        attack_target_logical_path = str(target_row["logical_path"])
        attack_target_index_path = env_logical_to_index.get(attack_target_logical_path, "")
        attack_target_node = env_logical_to_node.get(attack_target_logical_path)
        if not isinstance(attack_target_node, dict) or attack_target_node.get("children"):
            raise ValueError(
                f"Attack target logical path does not resolve to a leaf node: {attack_target_logical_path}"
            )
        if not attack_target_index_path:
            raise ValueError(
                f"Attack target logical path is missing from env index: {attack_target_logical_path}"
            )

        attacker_tools = list(sample.get("Attacker Tools") or [])
        attacker_task_payload = {
            "case_id": case_id,
            "instruction": sample.get("Attacker Instruction"),
            "attacker_tools": attacker_tools,
            "expected_achievements": sample.get("Expected Achievements"),
            "attack_type": sample.get("Attack Type"),
            "modified": sample.get("Modifed"),
            "attack_target_logical_path": attack_target_logical_path,
            "attack_target_index_path": attack_target_index_path,
        }

        attacker_task_path = run_dir / "task" / "attacker" / f"{case_id}.json"
        write_json(attacker_task_path, attacker_task_payload)

        attack_cases_rows.append(
            {
                "case_id": case_id,
                "injecagent_case_index": idx,
                "attacker_instruction": sample.get("Attacker Instruction"),
                "attacker_tools": attacker_tools,
                "expected_achievements": sample.get("Expected Achievements"),
                "attack_type": sample.get("Attack Type"),
                "modified": sample.get("Modifed"),
                "env_tree_path": to_posix(env_tree_path.resolve()),
                "anchor_logical_path": anchor_logical_resolved,
                "anchor_index_path": anchor_index_resolved,
                "security_entry_logical_path": security_entry_logical,
                "security_entry_index_path": security_entry_index,
                "agent_start_logical_path": env_root_logical_path,
                "agent_start_index_path": "/",
                "recommended_injection_scope": {
                    "scope_root_logical_path": security_entry_logical,
                    "scope_root_index_path": security_entry_index,
                },
                "attack_target_logical_path": attack_target_logical_path,
                "attack_target_index_path": attack_target_index_path,
                "user_target_logical_path": user_target_logical_path,
                "user_target_index_path": user_target_index_path,
                "attacker_task_path": to_posix(attacker_task_path.resolve()),
                "user_task_path": to_posix(user_task_path.resolve()),
            }
        )

    write_jsonl(run_dir / "attack_cases.jsonl", attack_cases_rows)

    run_config = {
        "mode": "single_environment_tree_file",
        "source_tree": to_posix(source_tree.resolve()),
        "top_k": top_k,
        "depth": depth,
        "width": width,
        "seed_base": seed_base,
        "root_coord": list(root_coord) if root_coord else None,
        "anchor_index_path": anchor_index_resolved,
        "anchor_logical_path": anchor_logical_resolved,
        "ds_json": to_posix(ds_json_path.resolve()),
        "security_spec": to_posix(security_spec_path.resolve()),
        "env_tree_path": to_posix(env_tree_path.resolve()),
        "user_task_path": to_posix(user_task_path.resolve()),
        "security_target_count": security_manifest.get("target_count"),
    }
    write_json(run_dir / "exp1_config.json", run_config)

    return {
        "run_dir": to_posix(run_dir.resolve()),
        "env_tree_path": to_posix(env_tree_path.resolve()),
        "cases": len(attack_cases_rows),
        "attack_cases": to_posix((run_dir / "attack_cases.jsonl").resolve()),
        "user_task": to_posix(user_task_path.resolve()),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)

    parser.add_argument("--ds-json", type=Path, default=DEFAULT_DS_JSON_PATH)
    parser.add_argument("--security-spec", type=Path, default=DEFAULT_SECURITY_SPEC_PATH)

    parser.add_argument("--source-tree", type=Path, default=DEFAULT_SOURCE_TREE_PATH)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=4242)
    parser.add_argument("--root-coord", type=str, default=None)
    parser.add_argument("--user-question-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--user-question-cache-ttl", type=int, default=86400)
    parser.add_argument("--user-question-max-tokens", type=int, default=128)
    parser.add_argument("--user-question-temperature", type=float, default=0.0)
    parser.add_argument("--user-question-body-max-input-tokens", type=int, default=2048)
    parser.add_argument(
        "--anchor-logical-path",
        type=str,
        default="/",
        help="Only supports index format like /0/1 (root is /).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_id = args.run_id or _default_run_id()
    run_dir = args.runs_root / run_id

    _init_run_dir(run_dir, overwrite=args.overwrite)

    result = build_cases_top20(
        run_dir=run_dir,
        ds_json_path=args.ds_json,
        security_spec_path=args.security_spec,
        source_tree=args.source_tree,
        top_k=args.top_k,
        depth=args.depth,
        width=args.width,
        seed_base=args.seed_base,
        root_coord=parse_root_coord(args.root_coord),
        anchor_index_path=args.anchor_logical_path,
        user_question_model=args.user_question_model,
        user_question_cache_ttl=args.user_question_cache_ttl,
        user_question_max_tokens=args.user_question_max_tokens,
        user_question_temperature=args.user_question_temperature,
        user_question_body_max_input_tokens=args.user_question_body_max_input_tokens,
    )

    print(f"run_dir={result['run_dir']}")
    print(f"env_tree_path={result['env_tree_path']}")
    print(f"cases={result['cases']}")
    print(f"attack_cases={result['attack_cases']}")
    print(f"user_task={result['user_task']}")


if __name__ == "__main__":
    main()
