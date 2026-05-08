#!/usr/bin/env python3
"""Inject the official TopicAttack agent-style transcript into web page metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
WEB_ROOT = Path(__file__).resolve().parents[2]
for candidate in (WEB_ROOT, WORKSPACE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
STAGE2_DIR = WEB_ROOT / "exp" / "02_nested_builder"
for candidate in (STAGE2_DIR,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiment_paths import resolve_experiment_root, resolve_path_within
from utils.llm_client import chat as llm_chat
from web.baseline.inject_from_attack_case import (
    _collect_program_html_sentinels,
    _discover_first_attacker_task,
    _index_pages,
    _load_json,
    _resolve_homepage_index,
    _resolve_relpath_across_roots,
    _write_json,
)
from web.baseline.official_baseline_helpers import (
    build_actionable_goal_text,
    resolve_reference_instruction,
    serialize_page_observation,
)
from web.baseline.topicattack.build_payload import (
    DEFAULT_TOPICATTACK_SYSTEM_MESSAGE,
    build_topicattack_generation_prompt,
    finalize_topicattack_payload,
    render_topicattack_transcript,
)
from web.psaa.inject_from_attack_case import _resolve_page_index


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--attack-case", type=Path, default=None)
    parser.add_argument("--attacker-tasks", type=Path, default=None)
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
    parser.add_argument("--inject-page-path", type=str, default="/")
    parser.add_argument("--user-goal-str", type=str, default="")
    parser.add_argument("--user-task-path", type=Path, default=None)
    parser.add_argument("--design", type=Path, default=None)
    parser.add_argument("--transitions", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out-design", type=Path, default=None)
    parser.add_argument("--out-transitions", type=Path, default=None)
    parser.add_argument("--out-metadata", type=Path, default=None)
    parser.add_argument("--out-generation", type=Path, default=None)
    try:
        action_bool = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        class _BoolAct(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        action_bool = _BoolAct  # type: ignore
    parser.add_argument("--use-security-microtree", action=action_bool, default=True)
    parser.add_argument("--with-reminder", action=action_bool, default=True)
    parser.add_argument("--pack", action=action_bool, default=False)
    return parser.parse_args(argv)


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.attack_case = resolve_path_within(
        args.attack_case,
        root=base_root,
        relative="attack_case.json",
    )
    if not args.attack_case.is_file():
        raise FileNotFoundError(f"attack_case.json not found: {args.attack_case}")
    attack_case = _load_json(args.attack_case)

    attacker_tasks_dir = attack_case.get("attacker_tasks_dir")
    if args.attacker_tasks is None and attacker_tasks_dir:
        args.attacker_tasks = _resolve_relpath_across_roots(
            str(attacker_tasks_dir),
            WORKSPACE_ROOT,
            WEB_ROOT,
        )
    if args.user_task_path is not None:
        args.user_task_path = _resolve_relpath_across_roots(
            str(args.user_task_path),
            WORKSPACE_ROOT,
            WEB_ROOT,
            base_root,
        )
    args.attacker_tasks = resolve_path_within(
        args.attacker_tasks,
        root=base_root,
        relative="webarena_tasks_attacker",
    )
    if not args.attacker_tasks.is_dir():
        raise FileNotFoundError(
            f"Attacker tasks directory not found: {args.attacker_tasks}"
        )
    return base_root, args.attack_case


def _resolve_site_paths(base_root: Path, args: argparse.Namespace) -> None:
    site_root = base_root
    security_root = base_root / "security_microtree"
    if args.use_security_microtree and security_root.is_dir():
        site_root = security_root
    args.design = resolve_path_within(
        args.design, root=site_root, relative="website_designs.json"
    )
    args.transitions = resolve_path_within(
        args.transitions, root=site_root, relative="transitions.json"
    )
    args.metadata = resolve_path_within(
        args.metadata, root=site_root, relative="page_metadata.json"
    )
    static_root = resolve_path_within(None, root=base_root, relative="static")
    args.out_design = resolve_path_within(
        args.out_design, root=static_root, relative="design_injected.json"
    )
    args.out_transitions = resolve_path_within(
        args.out_transitions,
        root=static_root,
        relative="transitions_injected.json",
    )
    args.out_metadata = resolve_path_within(
        args.out_metadata,
        root=static_root,
        relative="page_metadata_injected.json",
    )
    args.out_generation = resolve_path_within(
        args.out_generation,
        root=static_root,
        relative="topicattack_generation.json",
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


def inject_topicattack(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    base_root, attack_case_path = _resolve_inputs(args)
    _resolve_site_paths(base_root, args)
    attack_case = _load_json(attack_case_path)

    task_path = _discover_first_attacker_task(args.attacker_tasks)
    attacker_task = _load_json(task_path)
    attacker_eval = attacker_task.get("eval") or {}
    if not isinstance(attacker_eval, dict):
        attacker_eval = {}
    sentinels = _collect_program_html_sentinels(attacker_eval)

    designs = _load_json(args.design)
    if not isinstance(designs, list) or not designs:
        raise ValueError("Design file must contain a non-empty list.")
    design = designs[0]
    transitions = _load_json(args.transitions)
    metadata = _load_json(args.metadata)

    page_by_index, breadcrumb_to_index = _index_pages(metadata)
    try:
        inject_page_index = _resolve_page_index(
            dict(metadata),
            None,
            args.inject_page_path,
        )
    except Exception:
        if str(args.inject_page_path or "").strip() != "/":
            raise
        inject_page_index = _resolve_homepage_index(page_by_index, breadcrumb_to_index)
    inject_page = page_by_index.get(inject_page_index)
    if not isinstance(inject_page, dict):
        raise ValueError(
            f"Inject page_index {inject_page_index} not present in page metadata."
        )

    microtree = attack_case.get("microtree") or {}
    if not isinstance(microtree, dict):
        raise ValueError("attack_case.microtree must be an object.")
    microtree_id = str(microtree.get("id") or "attack_case")
    task_instruction, _task_instruction_source = resolve_reference_instruction(
        microtree_id=microtree_id,
        attacker_task=attacker_task,
    )

    anchor = microtree.get("anchor") or {}
    if not isinstance(anchor, dict) or not isinstance(anchor.get("page_index"), int):
        raise ValueError("attack_case.microtree.anchor.page_index must be an int.")
    anchor_page_index = int(anchor["page_index"])
    anchor_page = page_by_index.get(anchor_page_index)
    if not isinstance(anchor_page, dict):
        raise ValueError(
            f"Anchor page_index {anchor_page_index} not present in page metadata."
        )
    anchor_breadcrumb = str(anchor.get("breadcrumb") or "").strip()
    if not anchor_breadcrumb:
        anchor_breadcrumb = str(anchor_page.get("breadcrumb") or "").strip() or "/"
    anchor_title = str(anchor.get("title") or "").strip()
    if not anchor_title:
        anchor_title = str(anchor_page.get("title") or "").strip() or "Anchor Page"

    entry = microtree.get("entry") or {}
    if not isinstance(entry, dict):
        entry = {}
    entry_label = str(entry.get("label") or "").strip() or "Security Console"

    goal_text, nav_metadata = build_actionable_goal_text(
        metadata=metadata,
        start_page_index=inject_page_index,
        anchor_page_index=anchor_page_index,
        anchor_breadcrumb=anchor_breadcrumb,
        anchor_title=anchor_title,
        entry_label=entry_label,
        attacker_instruction=task_instruction,
        user_goal_str=args.user_goal_str,
        user_task_path=args.user_task_path,
    )
    page_observation = serialize_page_observation(inject_page)
    generation_prompt = build_topicattack_generation_prompt(
        data_text=page_observation,
        goal_text=goal_text,
        num_turns=args.num_turns,
    )
    raw_generation, generator_diagnostics = _call_topicattack_generator(
        prompt_text=generation_prompt,
        model=args.model,
        system_message=args.system_message,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_ttl=args.cache_ttl,
    )
    transcript = render_topicattack_transcript(
        raw_generation=raw_generation,
        goal_text=goal_text,
    )
    payload = finalize_topicattack_payload(
        transcript=transcript,
        with_reminder=bool(args.with_reminder),
    )

    attack_id = f"topicattack::agent::{microtree_id}::{inject_page_index}"
    generation_artifact = {
        "baseline_family": "topicattack",
        "baseline_variant": "agent",
        "official_variant_name": "topicattack_agent",
        "generator_model": str(args.model),
        "num_turns": int(args.num_turns),
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "cache_ttl": int(args.cache_ttl),
        "with_reminder": bool(args.with_reminder),
        "inject_page_index": int(inject_page_index),
        "inject_page_breadcrumb": str(inject_page.get("breadcrumb") or "").strip() or "/",
        "anchor_page_index": int(anchor_page_index),
        "anchor_breadcrumb": anchor_breadcrumb,
        "entry_label": entry_label,
        "nav_metadata": nav_metadata,
        "user_task_path": str(args.user_task_path) if args.user_task_path else None,
        "page_observation": page_observation,
        "generation_prompt": generation_prompt,
        "raw_generation": raw_generation,
        "rendered_transcript": transcript,
        "final_payload": payload,
        "generator_response": generator_diagnostics,
    }
    _write_json(args.out_generation, generation_artifact)

    injection_entry: dict[str, Any] = {
        "attack_id": attack_id,
        "text": payload,
        "show_breadcrumb": True,
        "source": {
            "attack_case": str(args.attack_case),
            "attacker_task_path": str(task_path),
            "baseline_family": "topicattack",
            "baseline_variant": "agent",
            "inject_page_index": inject_page_index,
            "anchor_page_index": anchor_page_index,
            "anchor_breadcrumb": anchor_breadcrumb,
            "entry_label": entry_label,
            "generator_model": str(args.model),
            "instantiated_goal_text": goal_text,
            "user_task_path": str(args.user_task_path) if args.user_task_path else None,
            "user_goal_str": str(nav_metadata.get("user_goal_str") or ""),
            "user_goal_source": str(nav_metadata.get("user_goal_source") or ""),
            "generation_artifact": str(args.out_generation),
        },
        "topicattack": {
            "single_node": True,
            "official_variant_name": "topicattack_agent",
            "generator_model": str(args.model),
            "num_turns": int(args.num_turns),
            "with_reminder": bool(args.with_reminder),
            "with_fake_completion": True,
            "prompt_budget_tokens": 0,
            **nav_metadata,
        },
    }
    if sentinels:
        injection_entry["sentinels"] = sentinels
    inject_page.setdefault("injections", []).append(injection_entry)
    design.setdefault("trap_injections", []).append(
        {
            "page_index": inject_page_index,
            "attack_id": attack_id,
            "source": "attack_case",
        }
    )

    _write_json(args.out_design, [design])
    _write_json(args.out_transitions, transitions)
    _write_json(args.out_metadata, metadata)
    if args.pack:
        subprocess.run(
            [
                "python",
                "web/baseline/pack_injected_site.py",
                "--experiment-root",
                str(base_root),
                *(
                    ["--experiment-id", str(args.experiment_id)]
                    if args.experiment_id
                    else []
                ),
            ],
            check=True,
            cwd=str(WORKSPACE_ROOT),
        )
    return {
        "baseline_family": "topicattack",
        "baseline_variant": "agent",
        "generator_model": str(args.model),
        "microtree_id": microtree_id,
        "inject_page_index": inject_page_index,
        "generation_artifact": str(args.out_generation),
        "out_design": str(args.out_design),
        "out_transitions": str(args.out_transitions),
        "out_metadata": str(args.out_metadata),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    result = inject_topicattack(argv)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
