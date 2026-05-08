#!/usr/bin/env python3
"""Inject the official Combined Attack string into a file-tree README."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.attack.combinedattack.build_payload import build_combinedattack_text
from file.exp.attack.common import (
    ensure_readme_and_append,
    hash_text,
    load_attack_cases,
    prepare_injection_environment,
    resolve_attacker_instruction,
    resolve_logical_dir_by_index_path,
    resolve_security_entry_dir,
    resolve_user_question,
    write_injection_text_files,
    write_post_injection_environment_artifacts,
)
from file.exp.attack.official_baseline_helpers import (
    build_actionable_goal_text,
    serialize_directory_observation,
)
from file.exp.pipeline_common import (
    append_jsonl,
    to_posix,
    utc_timestamp,
    write_json,
    write_jsonl,
)


def inject_combinedattack(
    *,
    run_dir: Path,
    case_id: str,
    seed: int,
    inject_index_path: str,
    user_goal_str: str,
) -> dict[str, Any]:
    cases = load_attack_cases(run_dir, case_id=case_id)
    rows: list[dict[str, Any]] = []
    case = cases[0]
    cid = str(case.get("case_id") or "")
    active_case, post_tree, pre_tree_path, root_logical = prepare_injection_environment(
        run_dir=run_dir,
        case=case,
    )

    inject_directory_logical, inject_index_norm = resolve_logical_dir_by_index_path(
        post_tree,
        inject_index_path,
    )
    security_entry_dir = resolve_security_entry_dir(
        active_case,
        root_logical=root_logical,
    )
    attacker_instruction = resolve_attacker_instruction(active_case, run_dir=run_dir)
    user_question = resolve_user_question(active_case, run_dir=run_dir)

    goal_text, nav_metadata = build_actionable_goal_text(
        start_directory_logical_path=inject_directory_logical,
        security_entry_directory_logical_path=security_entry_dir,
        attack_target_logical_path=str(active_case.get("attack_target_logical_path") or ""),
        attacker_instruction=attacker_instruction,
        root_logical=root_logical,
        user_question=user_question,
        user_goal_str=user_goal_str,
    )
    directory_observation = serialize_directory_observation(
        tree=post_tree,
        directory_logical_path=inject_directory_logical,
        root_logical=root_logical,
    )
    seed_material = f"{cid}::{inject_index_norm}"
    payload, payload_metadata = build_combinedattack_text(
        goal_text=goal_text,
        base_seed=seed,
        seed_material=seed_material,
    )
    prompt = "\n".join(
        [
            f"case_id: {cid}",
            f"seed: {seed}",
            f"inject_index_path: {inject_index_norm}",
            f"inject_directory: {inject_directory_logical}",
            f"security_entry_directory: {security_entry_dir}",
            f"goal_text: {goal_text}",
        ]
    ).strip()
    prompt_path, payload_path = write_injection_text_files(
        run_dir=run_dir,
        mode="combinedattack",
        case_id=cid,
        stage="official",
        stage_index=0,
        prompt=prompt,
        payload=payload,
    )
    prompt_path_str = to_posix(prompt_path) if prompt_path is not None else None
    payload_path_str = to_posix(payload_path)

    artifact_path = (run_dir / "injection" / f"combinedattack_payload_{cid}.json").resolve()
    artifact = {
        "mode": "combinedattack",
        "baseline_variant": "official",
        "official_variant_name": "completion_realcmb",
        "case_id": cid,
        "seed": int(seed),
        "seed_material": seed_material,
        "inject_index_path": inject_index_norm,
        "inject_directory_logical_path": inject_directory_logical,
        "nav_metadata": nav_metadata,
        "user_question": user_question,
        "directory_observation": directory_observation,
        "goal_text": goal_text,
        "payload": payload,
        "payload_metadata": payload_metadata,
    }
    write_json(artifact_path, artifact)

    marker = f"COMBINEDATTACK:official:{cid}"
    readme_logical_path, before_size, after_size = ensure_readme_and_append(
        post_tree,
        inject_directory_logical,
        root_logical=root_logical,
        marker=marker,
        content=payload,
    )

    row = {
        "timestamp": utc_timestamp(),
        "case_id": cid,
        "mode": "combinedattack",
        "baseline_variant": "official",
        "single_node": True,
        "official_variant_name": "completion_realcmb",
        "directory_logical_path": inject_directory_logical,
        "readme_logical_path": readme_logical_path,
        "payload_hash": hash_text(payload),
        "prompt_hash": hash_text(prompt),
        "prompt_path": prompt_path_str,
        "payload_path": payload_path_str,
        "artifact_path": to_posix(artifact_path),
        "before_size": before_size,
        "after_size": after_size,
        "inject_index_path": inject_index_norm,
        "inject_directory_logical_path": inject_directory_logical,
        "security_entry_directory_logical_path": security_entry_dir,
        "instantiated_goal_text": goal_text,
        "user_goal_str": str(nav_metadata.get("user_goal_str") or ""),
        "user_goal_source": str(nav_metadata.get("user_goal_source") or ""),
        **payload_metadata,
    }
    rows.append(row)

    generations_path = run_dir / "injection" / "injection_generations.jsonl"
    append_jsonl(
        generations_path,
        {
            "timestamp": utc_timestamp(),
            "case_id": cid,
            "mode": "combinedattack",
            "baseline_variant": "official",
            "stage": "official",
            "stage_index": 0,
            "directory_logical_path": inject_directory_logical,
            "readme_logical_path": readme_logical_path,
            "prompt_hash": hash_text(prompt),
            "payload_hash": hash_text(payload),
            "prompt_path": prompt_path_str,
            "payload_path": payload_path_str,
            "artifact_path": to_posix(artifact_path),
        },
    )

    active_cases_path = write_post_injection_environment_artifacts(
        run_dir=run_dir,
        active_case=active_case,
        post_tree=post_tree,
    )
    post_env_tree_path = (run_dir / "env" / "env_post_injection_tree.json").resolve()

    manifest_path = run_dir / "injection" / "injection_manifest.jsonl"
    write_jsonl(manifest_path, rows)
    write_json(
        run_dir / "injection" / "injection_summary.json",
        {
            "mode": "combinedattack",
            "baseline_variant": "official",
            "case_id": cid,
            "cases": 1,
            "writes": len(rows),
            "manifest": to_posix(manifest_path.resolve()),
            "generations_index": to_posix(generations_path.resolve()),
            "payload_artifact": to_posix(artifact_path),
            "inject_index_path": inject_index_norm,
            "inject_directory_logical_path": inject_directory_logical,
            "security_entry_directory_logical_path": security_entry_dir,
            "active_attack_cases": to_posix(active_cases_path),
            "source_env_tree": to_posix(pre_tree_path),
            "post_env_tree": to_posix(post_env_tree_path),
        },
    )
    return {
        "mode": "combinedattack",
        "baseline_variant": "official",
        "case_id": cid,
        "cases": 1,
        "writes": len(rows),
        "manifest": to_posix(manifest_path.resolve()),
        "generations_index": to_posix(generations_path.resolve()),
        "payload_artifact": to_posix(artifact_path),
        "inject_index_path": inject_index_norm,
        "inject_directory_logical_path": inject_directory_logical,
        "active_attack_cases": to_posix(active_cases_path),
        "source_env_tree": to_posix(pre_tree_path),
        "post_env_tree": to_posix(post_env_tree_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inject-index-path",
        dest="inject_index_path",
        type=str,
        default="/",
        help="Directory index path like /0/1 (root is /).",
    )
    parser.add_argument(
        "--user-goal-str",
        type=str,
        default="",
        help="Optional explicit user-goal phrase for hijacking-style goal text.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = inject_combinedattack(
        run_dir=args.run_dir,
        case_id=args.case_id,
        seed=args.seed,
        inject_index_path=args.inject_index_path,
        user_goal_str=args.user_goal_str,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
