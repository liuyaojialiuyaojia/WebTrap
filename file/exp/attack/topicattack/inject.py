#!/usr/bin/env python3
"""Inject the official TopicAttack agent transcript into a file-tree README."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.llm_client import chat as llm_chat
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
from file.exp.attack.topicattack.build_payload import (
    DEFAULT_TOPICATTACK_SYSTEM_MESSAGE,
    build_topicattack_generation_prompt,
    finalize_topicattack_payload,
    render_topicattack_transcript,
)
from file.exp.pipeline_common import (
    append_jsonl,
    to_posix,
    utc_timestamp,
    write_json,
    write_jsonl,
)


def _extract_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("TopicAttack generator returned no choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("TopicAttack generator returned no message.")
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content or "")


def _call_topicattack_generator(
    *,
    prompt_text: str,
    model: str,
    system_message: str,
    temperature: float,
    max_tokens: int,
    cache_ttl: int,
) -> tuple[str, dict[str, Any]]:
    messages: list[dict[str, str]] = []
    if str(system_message or "").strip():
        messages.append({"role": "system", "content": str(system_message)})
    messages.append({"role": "user", "content": prompt_text})

    response = llm_chat(
        messages=messages,
        model=str(model),
        cache_ttl=int(cache_ttl),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    usage = getattr(response, "usage", None)
    raw_generation = _extract_message_text(response).strip()
    diagnostics = {
        "response_id": getattr(response, "id", None),
        "response_model": getattr(response, "model", None),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }
    return raw_generation, diagnostics


def inject_topicattack(
    *,
    run_dir: Path,
    case_id: str,
    model: str,
    num_turns: int,
    temperature: float,
    max_tokens: int,
    cache_ttl: int,
    system_message: str,
    with_reminder: bool,
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
    generation_prompt = build_topicattack_generation_prompt(
        data_text=directory_observation,
        goal_text=goal_text,
        num_turns=num_turns,
    )
    raw_generation, generator_diagnostics = _call_topicattack_generator(
        prompt_text=generation_prompt,
        model=model,
        system_message=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_ttl=cache_ttl,
    )
    transcript = render_topicattack_transcript(
        raw_generation=raw_generation,
        goal_text=goal_text,
    )
    payload = finalize_topicattack_payload(
        transcript=transcript,
        with_reminder=bool(with_reminder),
    )

    prompt_path, payload_path = write_injection_text_files(
        run_dir=run_dir,
        mode="topicattack",
        case_id=cid,
        stage="agent",
        stage_index=0,
        prompt=generation_prompt,
        payload=payload,
    )
    prompt_path_str = to_posix(prompt_path) if prompt_path is not None else None
    payload_path_str = to_posix(payload_path)

    generation_artifact_path = (
        run_dir / "injection" / f"topicattack_generation_{cid}.json"
    ).resolve()
    generation_artifact = {
        "mode": "topicattack",
        "baseline_variant": "agent",
        "official_variant_name": "topicattack_agent",
        "case_id": cid,
        "generator_model": str(model),
        "num_turns": int(num_turns),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "cache_ttl": int(cache_ttl),
        "with_reminder": bool(with_reminder),
        "inject_index_path": inject_index_norm,
        "inject_directory_logical_path": inject_directory_logical,
        "nav_metadata": nav_metadata,
        "user_question": user_question,
        "directory_observation": directory_observation,
        "generation_prompt": generation_prompt,
        "raw_generation": raw_generation,
        "rendered_transcript": transcript,
        "final_payload": payload,
        "generator_response": generator_diagnostics,
    }
    write_json(generation_artifact_path, generation_artifact)

    marker = f"TOPICATTACK:agent:{cid}"
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
        "mode": "topicattack",
        "baseline_variant": "agent",
        "single_node": True,
        "official_variant_name": "topicattack_agent",
        "directory_logical_path": inject_directory_logical,
        "readme_logical_path": readme_logical_path,
        "payload_hash": hash_text(payload),
        "prompt_hash": hash_text(generation_prompt),
        "prompt_path": prompt_path_str,
        "payload_path": payload_path_str,
        "artifact_path": to_posix(generation_artifact_path),
        "before_size": before_size,
        "after_size": after_size,
        "inject_index_path": inject_index_norm,
        "inject_directory_logical_path": inject_directory_logical,
        "security_entry_directory_logical_path": security_entry_dir,
        "generator_model": str(model),
        "instantiated_goal_text": goal_text,
        "user_goal_str": str(nav_metadata.get("user_goal_str") or ""),
        "user_goal_source": str(nav_metadata.get("user_goal_source") or ""),
    }
    rows.append(row)

    generations_path = run_dir / "injection" / "injection_generations.jsonl"
    append_jsonl(
        generations_path,
        {
            "timestamp": utc_timestamp(),
            "case_id": cid,
            "mode": "topicattack",
            "baseline_variant": "agent",
            "stage": "agent",
            "stage_index": 0,
            "directory_logical_path": inject_directory_logical,
            "readme_logical_path": readme_logical_path,
            "prompt_hash": hash_text(generation_prompt),
            "payload_hash": hash_text(payload),
            "prompt_path": prompt_path_str,
            "payload_path": payload_path_str,
            "artifact_path": to_posix(generation_artifact_path),
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
            "mode": "topicattack",
            "baseline_variant": "agent",
            "case_id": cid,
            "cases": 1,
            "writes": len(rows),
            "manifest": to_posix(manifest_path.resolve()),
            "generations_index": to_posix(generations_path.resolve()),
            "generation_artifact": to_posix(generation_artifact_path),
            "inject_index_path": inject_index_norm,
            "inject_directory_logical_path": inject_directory_logical,
            "security_entry_directory_logical_path": security_entry_dir,
            "active_attack_cases": to_posix(active_cases_path),
            "source_env_tree": to_posix(pre_tree_path),
            "post_env_tree": to_posix(post_env_tree_path),
        },
    )
    return {
        "mode": "topicattack",
        "baseline_variant": "agent",
        "case_id": cid,
        "cases": 1,
        "writes": len(rows),
        "manifest": to_posix(manifest_path.resolve()),
        "generations_index": to_posix(generations_path.resolve()),
        "generation_artifact": to_posix(generation_artifact_path),
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
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--num-turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cache-ttl", type=int, default=86400)
    parser.add_argument(
        "--system-message",
        type=str,
        default=DEFAULT_TOPICATTACK_SYSTEM_MESSAGE,
    )
    try:
        action_bool = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        class _BoolAct(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        action_bool = _BoolAct  # type: ignore
    parser.add_argument("--with-reminder", action=action_bool, default=True)
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
    result = inject_topicattack(
        run_dir=args.run_dir,
        case_id=args.case_id,
        model=args.model,
        num_turns=args.num_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_ttl=args.cache_ttl,
        system_message=args.system_message,
        with_reminder=args.with_reminder,
        inject_index_path=args.inject_index_path,
        user_goal_str=args.user_goal_str,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
