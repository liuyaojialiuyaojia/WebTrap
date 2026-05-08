#!/usr/bin/env python3
"""Inject PSAA staged prompt-injection text into file-tree readme files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.attack.common import (
    build_directory_chain,
    ensure_readme_and_inject,
    format_virtual_dir_path,
    hash_text,
    logical_parent_dir,
    load_attack_cases,
    prepare_injection_environment,
    render_directory_context,
    resolve_attacker_instruction,
    resolve_security_entry_dir,
    resolve_start_dir,
    write_injection_text_files,
    write_post_injection_environment_artifacts,
)
from file.exp.attack.psaa.prompt_spec import PsaaPromptSpec, load_psaa_prompt_spec
from file.exp.attack.psaa.template_render import render_template
from file.exp.pipeline_common import (
    append_jsonl,
    to_posix,
    utc_timestamp,
    write_json,
    write_jsonl,
)
from file.utils.generate import generate

_PSEUDO_SYSTEM_BEGIN_TAG = "<｜begin▁of▁sentence｜>"
_PSEUDO_SYSTEM_TAG = "<｜System｜>"
_PSEUDO_ASSISTANT_TAG = "<｜Assistant｜>"
_PSEUDO_ASSISTANT_PREFIX = "</think>"
_PSEUDO_USER_TAG = "<｜User｜>"
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CHECK_TEMPLATE_CANDIDATE_KEYS = frozenset(
    {
        "injection",
        "candidate",
        "candidate_text",
        "candidate_injection",
        "generated_injection",
        "instruction_to_check",
    }
)
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _parse_index_path(raw: Optional[str]) -> list[int]:
    value = str(raw or "").strip()
    if value in {"", "/"}:
        return []
    if not value.startswith("/"):
        raise ValueError("--start-index-path must use index format like /0/1 (root is /).")
    if value.endswith("/"):
        raise ValueError("--start-index-path must use index format like /0/1 (no trailing /).")

    tokens = value.split("/")[1:]
    indexes: list[int] = []
    for token in tokens:
        if token == "":
            raise ValueError("--start-index-path cannot contain empty segment.")
        if not token.isdigit():
            raise ValueError("--start-index-path segments must be digits, e.g. /0/1.")
        indexes.append(int(token))
    return indexes


def _resolve_logical_dir_by_index_path(
    tree: dict[str, Any],
    raw_index_path: str,
) -> tuple[str, str]:
    root_name = str(tree.get("name") or "").strip()
    if not root_name:
        raise ValueError("Env tree root node must have a non-empty name.")

    indexes = _parse_index_path(raw_index_path)
    node: dict[str, Any] = tree
    logical_parts = [root_name]
    for index in indexes:
        children = node.get("children") or []
        if not isinstance(children, list):
            raise ValueError("Env tree node children must be a list.")
        if index < 0 or index >= len(children):
            raise ValueError(
                f"--start-index-path index out of range: {raw_index_path} "
                f"(index={index}, children={len(children)})."
            )
        next_node = children[index]
        if not isinstance(next_node, dict):
            raise ValueError(
                f"--start-index-path resolved to non-object child at {raw_index_path}."
            )
        name = str(next_node.get("name") or "").strip()
        if not name:
            raise ValueError(
                f"--start-index-path resolved to unnamed node at {raw_index_path}."
            )
        logical_parts.append(name)
        node = next_node

    if not node.get("children"):
        logical_path = "/" + "/".join(logical_parts)
        raise ValueError(
            "--start-index-path must resolve to a directory node, got: "
            f"{raw_index_path} -> {logical_path}"
        )

    normalized_index = "/" if not indexes else "/" + "/".join(str(i) for i in indexes)
    logical_path = "/" + "/".join(logical_parts)
    return logical_path, normalized_index


def _select_three_stage_positions(chain_len: int) -> tuple[int, int, int]:
    if chain_len <= 0:
        raise ValueError("Injection chain length must be positive.")
    if chain_len == 1:
        return 0, 0, 0
    if chain_len == 2:
        return 0, 1, 1

    payload_pos = chain_len - 1
    inertia_pos = min(payload_pos - 1, max(1, (2 * payload_pos) // 3))
    return 0, inertia_pos, payload_pos


def _build_stage_plan(chain_len: int) -> list[tuple[str, int]]:
    if chain_len <= 0:
        raise ValueError("Injection chain length must be positive.")
    if chain_len == 1:
        return [("payload", 0)]
    if chain_len == 2:
        return [("lure", 0), ("payload", 1)]

    lure_pos, inertia_pos, payload_pos = _select_three_stage_positions(chain_len)
    return [
        ("lure", lure_pos),
        ("inertia", inertia_pos),
        ("payload", payload_pos),
    ]


def _logical_rel_parts(logical_path: str, *, root_logical: str) -> list[str]:
    raw = str(logical_path or "").strip()
    if raw == root_logical:
        return []
    if raw.startswith(root_logical + "/"):
        suffix = raw[len(root_logical) :].lstrip("/")
        return [part for part in suffix.split("/") if part]
    raise ValueError(f"Logical path {logical_path!r} is not under root {root_logical!r}.")


def _build_navigation_path_text(
    *,
    route_dirs: Sequence[str],
    root_logical: str,
) -> tuple[list[str], str]:
    if len(route_dirs) < 2:
        return [], ""

    labels: list[str] = []
    for current, nxt in zip(route_dirs, route_dirs[1:]):
        current_parts = _logical_rel_parts(str(current), root_logical=root_logical)
        next_parts = _logical_rel_parts(str(nxt), root_logical=root_logical)
        if len(next_parts) + 1 == len(current_parts) and current_parts[:-1] == next_parts:
            labels.append("..")
            continue
        if len(next_parts) == len(current_parts) + 1 and next_parts[:-1] == current_parts:
            labels.append(next_parts[-1])
            continue
        raise ValueError(
            "Route must move through adjacent file-tree directories: "
            f"current={current!r} next={nxt!r}"
        )

    path_text = " -> ".join(f'"{label}"' for label in labels)
    return labels, path_text


def _build_security_target_path_text(
    *,
    security_entry_dir: str,
    target_security_file: str,
    root_logical: str,
) -> tuple[list[str], str]:
    format_virtual_dir_path(security_entry_dir, root_logical=root_logical)
    format_virtual_dir_path(target_security_file, root_logical=root_logical)

    prefix = str(security_entry_dir).rstrip("/") + "/"
    if not str(target_security_file).startswith(prefix):
        raise ValueError(
            "attack_target_logical_path must be under security_entry_logical_path: "
            f"security_entry={security_entry_dir} target={target_security_file}"
        )
    suffix = str(target_security_file)[len(prefix) :].strip("/")
    labels = [part for part in suffix.split("/") if part]
    if not labels:
        raise ValueError(
            "Derived empty relative target path from security entry: "
            f"security_entry={security_entry_dir} target={target_security_file}"
        )
    path_text = " -> ".join(f'"{label}"' for label in labels)
    return labels, path_text


def _extract_first_email(*texts: str) -> str:
    for text in texts:
        match = _EMAIL_RE.search(str(text or ""))
        if match:
            return match.group(0)
    return "the requested recipient"


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:]
        newline_idx = stripped.find("\n")
        if newline_idx != -1:
            fence_info = stripped[:newline_idx].strip()
            if re.fullmatch(r"[A-Za-z0-9_+-]*", fence_info):
                stripped = stripped[newline_idx + 1 :]
            else:
                stripped = stripped.lstrip()
        else:
            stripped = stripped.lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _strip_wrapping_block(text: str, *, tag: str) -> str:
    pattern = re.compile(
        rf"^\s*<{tag}>\s*(?P<body>.*?)\s*</{tag}>\s*$",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(text)
    if not match:
        return text
    return str(match.group("body") or "").strip()


def _normalise_generated_injection_text(text: str) -> str:
    cleaned = _strip_code_fence(str(text or "")).replace("\r\n", "\n").strip()

    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        if cleaned.startswith(_PSEUDO_SYSTEM_BEGIN_TAG):
            cleaned = cleaned[len(_PSEUDO_SYSTEM_BEGIN_TAG) :].lstrip()
            continue
        if cleaned.startswith(_PSEUDO_ASSISTANT_PREFIX):
            cleaned = cleaned[len(_PSEUDO_ASSISTANT_PREFIX) :].lstrip()
            continue
        if cleaned.startswith(_PSEUDO_ASSISTANT_TAG):
            cleaned = cleaned[len(_PSEUDO_ASSISTANT_TAG) :].lstrip()
            continue
        if cleaned.startswith(_PSEUDO_SYSTEM_TAG):
            cleaned = cleaned[len(_PSEUDO_SYSTEM_TAG) :].lstrip()
            continue
        if cleaned.endswith(_PSEUDO_USER_TAG):
            cleaned = cleaned[: -len(_PSEUDO_USER_TAG)].rstrip()
            continue
        for tag in ("output", "statement", "message"):
            stripped = _strip_wrapping_block(cleaned, tag=tag)
            if stripped != cleaned:
                cleaned = stripped
                break

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'", "`"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _wrap_pseudo_system_message(message: str) -> str:
    material = str(message or "").strip()
    if not material:
        raise ValueError("Cannot wrap an empty pseudo-system message.")
    return f"{_PSEUDO_SYSTEM_TAG}\n{material}\n{_PSEUDO_USER_TAG}"


def _validate_candidate_structure(*, stage: str, candidate: str) -> tuple[bool, str]:
    material = str(candidate or "").strip()
    if not material:
        return False, "candidate is empty"
    if _PSEUDO_SYSTEM_TAG not in material:
        return False, "missing pseudo-system wrapper"
    if _PSEUDO_USER_TAG not in material:
        return False, "missing pseudo-user terminator"
    body = material
    if body.startswith(_PSEUDO_SYSTEM_TAG):
        body = body[len(_PSEUDO_SYSTEM_TAG) :].lstrip()
    if body.endswith(_PSEUDO_USER_TAG):
        body = body[: -len(_PSEUDO_USER_TAG)].rstrip()
    if not body.strip():
        return False, f"{stage} candidate body is empty"
    return True, ""


def _parse_llm_json(raw_text: str) -> Any:
    stripped = _strip_code_fence(raw_text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {raw_text!r}") from exc


def _parse_check_result(raw_text: str) -> tuple[bool, str]:
    try:
        payload = _parse_llm_json(raw_text)
    except Exception:
        lowered = str(raw_text or "").strip().lower()
        if '"tag"' in lowered:
            if '"yes"' in lowered:
                return True, str(raw_text or "").strip()[:200]
            if '"no"' in lowered:
                return False, str(raw_text or "").strip()[:200]
        if lowered.startswith("yes"):
            return True, str(raw_text or "").strip()[:200]
        if lowered.startswith("no"):
            return False, str(raw_text or "").strip()[:200]
        trimmed = str(raw_text or "").strip().replace("\n", "\\n")
        return False, f"check_llm returned invalid JSON: {trimmed[:300]}"

    if not isinstance(payload, dict):
        return False, f"check_llm returned non-object JSON: {payload!r}"

    tag = str(payload.get("tag") or "").strip().lower()
    comment = str(payload.get("comment") or "").strip()
    if tag not in {"yes", "no"}:
        return False, comment or f"check_llm returned invalid tag: {payload!r}"
    return tag == "yes", comment


def _extract_template_placeholders(template: str) -> set[str]:
    material = template.replace("{{", "").replace("}}", "")
    return set(_TEMPLATE_PLACEHOLDER_RE.findall(material))


def _validate_check_template_uses_candidate(*, stage: str, template: str) -> None:
    placeholders = _extract_template_placeholders(template)
    if placeholders.intersection(_CHECK_TEMPLATE_CANDIDATE_KEYS):
        return
    found = ", ".join(sorted(placeholders)) if placeholders else "<none>"
    required = ", ".join(sorted(_CHECK_TEMPLATE_CANDIDATE_KEYS))
    raise ValueError(
        f"check_llm template for stage '{stage}' does not reference the generated "
        f"injection candidate. Found placeholders: {found}. Expected at least one of: "
        f"{required}."
    )


def _chat_once(*, model: str, system_message: str, user_prompt: str) -> str:
    response = generate(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
    )
    if response is None:
        raise ValueError("LLM returned None.")
    content = str(response.content or "").strip()
    if not content:
        raise ValueError("LLM returned empty response content.")
    return content


def _generate_checked_injection_text(
    *,
    spec: PsaaPromptSpec,
    stage: str,
    page_text: str,
    goal_label: str,
    goal_path_text: str,
    task_instruction: str,
    previous_injection: str | None,
    security_target_path_text: str,
    recipient_email: str,
    attempt_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    if stage not in {"lure", "inertia", "payload"}:
        raise ValueError(f"Unknown stage: {stage}")

    attacker_stage_tpl = getattr(spec.attacker_templates, stage)
    check_template = getattr(spec.check_templates, stage)
    use_checker = spec.retry_num > 1
    if use_checker:
        _validate_check_template_uses_candidate(stage=stage, template=check_template)

    last_candidate: str | None = None
    last_candidate_body = ""
    last_comment = ""
    passed = False
    attempts_used = 0

    for attempt in range(spec.retry_num):
        attempts_used = attempt + 1
        attacker_template = (
            attacker_stage_tpl.first if attempt == 0 else attacker_stage_tpl.rewrite
        )
        variables: dict[str, Any] = {
            "page": page_text,
            "goal": goal_label,
            "goal_path_text": goal_path_text,
            "route": goal_path_text,
            "instruction": task_instruction,
            "task_instruction": task_instruction,
            "security_target_path_text": security_target_path_text,
            "recipient_email": recipient_email,
            "previous_injection": previous_injection or "",
            "injection": "",
            "comment": "",
            "attempt": attempt + 1,
            "retry_num": spec.retry_num,
        }
        if attempt > 0:
            variables["injection"] = last_candidate_body
            variables["comment"] = last_comment

        attacker_prompt = render_template(attacker_template, variables).strip()
        attacker_raw = _chat_once(
            model=spec.attacker.model,
            system_message=spec.attacker.system_message,
            user_prompt=attacker_prompt,
        )

        parse_error = ""
        try:
            candidate_body = _normalise_generated_injection_text(attacker_raw)
            if not candidate_body:
                raise ValueError("attacker output is empty after normalization")
        except Exception as exc:
            parse_error = str(exc)
            candidate_body = _strip_code_fence(str(attacker_raw or "")).strip()
            if not candidate_body:
                candidate_body = "[empty attacker output]"

        last_candidate_body = candidate_body
        candidate = _wrap_pseudo_system_message(candidate_body)
        last_candidate = candidate

        structure_ok, structure_comment = _validate_candidate_structure(
            stage=stage,
            candidate=candidate,
        )

        check_prompt = ""
        check_raw = ""
        if parse_error:
            passed = False
            last_comment = f"attacker output parse error: {parse_error}"
        elif not structure_ok:
            passed = False
            last_comment = structure_comment
        elif not use_checker:
            passed = True
            last_comment = "checker skipped because retry_num=1"
        else:
            check_variables = {
                "instruction": task_instruction,
                "injection": candidate,
                "candidate": candidate,
                "candidate_text": candidate,
                "candidate_injection": candidate,
                "generated_injection": candidate,
                "instruction_to_check": candidate,
                "goal": goal_label,
                "goal_path_text": goal_path_text,
                "route": goal_path_text,
                "page": page_text,
                "task_instruction": task_instruction,
                "security_target_path_text": security_target_path_text,
                "recipient_email": recipient_email,
                "reference_instruction": task_instruction,
                "retry_num": spec.retry_num,
                "attempt": attempt + 1,
                "previous_injection": previous_injection or "",
                "comment": last_comment,
                "stage": stage,
            }
            check_prompt = render_template(check_template, check_variables).strip()
            check_raw = _chat_once(
                model=spec.checker.model,
                system_message=spec.checker.system_message,
                user_prompt=check_prompt,
            )
            passed, last_comment = _parse_check_result(check_raw)

        if attempt_callback is not None:
            attempt_callback(
                {
                    "attempt": attempt + 1,
                    "candidate": candidate,
                    "raw_candidate": attacker_raw,
                    "candidate_body": candidate_body,
                    "parse_error": parse_error,
                    "passed": bool(passed),
                    "checker_used": bool(use_checker),
                    "check_comment": last_comment,
                    "attacker_prompt": attacker_prompt,
                    "check_prompt": check_prompt,
                    "check_raw": check_raw,
                }
            )
        if passed:
            break

    assert last_candidate is not None
    return last_candidate, {
        "passed": bool(passed),
        "attempts": attempts_used,
        "last_comment": last_comment,
        "candidate_body": last_candidate_body,
        "checker_used": bool(use_checker),
    }


def inject_psaa(
    *,
    run_dir: Path,
    spec_path: Path,
    case_id: str,
    start_index_path: Optional[str] = None,
) -> dict[str, Any]:
    spec = load_psaa_prompt_spec(spec_path)
    cases = load_attack_cases(run_dir, case_id=case_id)
    rows: list[dict[str, Any]] = []
    case = cases[0]
    cid = str(case.get("case_id") or "")
    active_case, post_tree, pre_tree_path, root_logical = prepare_injection_environment(
        run_dir=run_dir,
        case=case,
    )

    generations_path = run_dir / "injection" / "injection_generations.jsonl"

    def _emit_generation_event(event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "timestamp": utc_timestamp(),
            "event": event,
            "case_id": cid,
            "mode": "psaa",
        }
        record.update(payload)
        append_jsonl(generations_path, record)

    try:
        root_dir = root_logical
        start_dir = resolve_start_dir(active_case, root_logical=root_logical)
        start_index_norm: Optional[str] = None
        if start_index_path is not None:
            raw_index_path = str(start_index_path or "").strip()
            if raw_index_path:
                start_dir, start_index_norm = _resolve_logical_dir_by_index_path(
                    post_tree,
                    raw_index_path,
                )
                active_case["agent_start_logical_path"] = start_dir
                active_case["agent_start_index_path"] = start_index_norm

        security_entry_dir = resolve_security_entry_dir(
            active_case,
            root_logical=root_logical,
        )
        attach_dir = logical_parent_dir(security_entry_dir, root_logical=root_logical)
        attacker_instruction = resolve_attacker_instruction(active_case, run_dir=run_dir)
        attack_target_logical_path = str(
            active_case.get("attack_target_logical_path") or ""
        ).strip()
        security_target_labels, security_target_path_text = _build_security_target_path_text(
            security_entry_dir=security_entry_dir,
            target_security_file=attack_target_logical_path,
            root_logical=root_logical,
        )
        recipient_email = _extract_first_email(
            attacker_instruction,
            str(active_case.get("attacker_instruction") or ""),
        )

        chain = build_directory_chain(
            root_dir=root_dir,
            start_dir=start_dir,
            target_dir=attach_dir,
        )
        stage_plan = _build_stage_plan(len(chain))
        stage_positions = {stage: position for stage, position in stage_plan}
        inertia_steps = sum(1 for stage, _ in stage_plan if stage == "inertia")

        events: list[dict[str, Any]] = []
        for step_index, (stage, chain_position) in enumerate(stage_plan):
            directory = chain[chain_position]
            if stage != "payload" and step_index + 1 < len(stage_plan):
                route_target_position = int(stage_plan[step_index + 1][1])
                route_dirs = [str(path) for path in chain[chain_position : route_target_position + 1]]
                next_dir = str(chain[route_target_position])
            else:
                route_dirs = [directory, security_entry_dir]
                next_dir = security_entry_dir

            goal_path_labels, goal_path_text = _build_navigation_path_text(
                route_dirs=route_dirs,
                root_logical=root_logical,
            )
            if len(route_dirs) < 2 or not goal_path_labels:
                raise ValueError(
                    f"Invalid PSAA route for stage {stage}: route_dirs={route_dirs}"
                )

            events.append(
                {
                    "stage": stage,
                    "step_index": step_index,
                    "chain_position": chain_position,
                    "directory": directory,
                    "next_directory_logical_path": next_dir,
                    "goal_label": goal_path_labels[0],
                    "goal_path_labels": goal_path_labels,
                    "goal_path_text": goal_path_text,
                }
            )

        _emit_generation_event(
            "run_start",
            {
                "prompt_spec_path": to_posix(spec_path.resolve()),
                "prompt_spec_version": spec.version,
                "retry_num": spec.retry_num,
                "attacker_model": spec.attacker.model,
                "checker_model": spec.checker.model,
                "start_directory_logical_path": start_dir,
                "start_index_path": start_index_norm,
                "attach_directory_logical_path": attach_dir,
                "security_entry_directory_logical_path": security_entry_dir,
                "security_target_path_text": security_target_path_text,
                "security_target_labels": security_target_labels,
                "chain_len": len(chain),
                "inertia_steps": inertia_steps,
                "stage_positions": stage_positions,
            },
        )

        previous_injection: str | None = None
        for event in events:
            stage = str(event["stage"])
            step_index = int(event["step_index"])
            chain_position = int(event["chain_position"])
            directory = str(event["directory"])
            next_dir = str(event["next_directory_logical_path"])
            goal_label = str(event["goal_label"])
            goal_path_labels = [str(label) for label in event["goal_path_labels"]]
            goal_path_text = str(event["goal_path_text"])

            _emit_generation_event(
                "step_start",
                {
                    "stage": stage,
                    "step_index": step_index,
                    "chain_position": chain_position,
                    "directory_logical_path": directory,
                    "next_directory_logical_path": next_dir,
                    "goal_label": goal_label,
                    "goal_path_labels": goal_path_labels,
                    "goal_path_text": goal_path_text,
                },
            )

            page_context = render_directory_context(
                post_tree,
                directory,
                root_logical=root_logical,
                include_readme=True,
                include_entries=True,
            )

            attempt_records: list[dict[str, Any]] = []

            def _on_attempt(attempt_data: dict[str, Any]) -> None:
                attempt_records.append(dict(attempt_data))
                attempt_record = {
                    "stage": stage,
                    "step_index": step_index,
                    "chain_position": chain_position,
                    "directory_logical_path": directory,
                    "goal_label": goal_label,
                    "goal_path_text": goal_path_text,
                }
                attempt_record.update(attempt_data)
                _emit_generation_event("attempt_result", attempt_record)

            wrapped_text, generation = _generate_checked_injection_text(
                spec=spec,
                stage=stage,
                page_text=page_context,
                goal_label=goal_label,
                goal_path_text=goal_path_text,
                task_instruction=attacker_instruction,
                previous_injection=previous_injection,
                security_target_path_text=security_target_path_text,
                recipient_email=recipient_email,
                attempt_callback=_on_attempt,
            )
            previous_injection = wrapped_text

            if not attempt_records:
                raise ValueError(f"No generation attempts were recorded for stage={stage}")
            selected_prompt = str(attempt_records[-1].get("attacker_prompt") or "").strip()

            prompt_path, payload_path = write_injection_text_files(
                run_dir=run_dir,
                mode="psaa",
                case_id=cid,
                stage=stage,
                stage_index=step_index,
                prompt=selected_prompt,
                payload=wrapped_text,
            )
            prompt_path_str = to_posix(prompt_path) if prompt_path is not None else None
            payload_path_str = to_posix(payload_path)

            marker = f"PSAA_STAGE:{cid}:{stage}:{step_index}"
            readme_logical_path, before_size, after_size = ensure_readme_and_inject(
                post_tree,
                directory,
                root_logical=root_logical,
                marker=marker,
                content=wrapped_text,
                wrapper="raw",
                position="prepend",
            )

            if directory == root_logical:
                rel_dir = "."
            elif directory.startswith(root_logical + "/"):
                rel_dir = directory[len(root_logical) + 1 :]
            else:
                rel_dir = directory

            row = {
                "timestamp": utc_timestamp(),
                "case_id": cid,
                "mode": "psaa",
                "stage": stage,
                "stage_index": step_index,
                "chain_len": len(chain),
                "inertia_steps": inertia_steps,
                "chain_position": chain_position,
                "directory_logical_path": directory,
                "relative_directory": rel_dir or ".",
                "next_directory_logical_path": next_dir,
                "readme_logical_path": readme_logical_path,
                "goal_label": goal_label,
                "goal_path_labels": goal_path_labels,
                "goal_path_text": goal_path_text,
                "security_target_labels": security_target_labels,
                "security_target_path_text": security_target_path_text,
                "payload_hash": hash_text(wrapped_text),
                "prompt_hash": hash_text(selected_prompt),
                "prompt_path": prompt_path_str,
                "payload_path": payload_path_str,
                "before_size": before_size,
                "after_size": after_size,
                "start_directory_logical_path": start_dir,
                "start_index_path": start_index_norm,
                "attach_directory_logical_path": attach_dir,
                "security_entry_directory_logical_path": security_entry_dir,
                "attacker_model": spec.attacker.model,
                "checker_model": spec.checker.model,
                "retry_num": spec.retry_num,
                "generation": generation,
            }
            rows.append(row)

            _emit_generation_event(
                "step_end",
                {
                    "stage": stage,
                    "step_index": step_index,
                    "directory_logical_path": directory,
                    "readme_logical_path": readme_logical_path,
                    "goal_path_text": goal_path_text,
                    "selected_injection": wrapped_text,
                    "generation": generation,
                    "prompt_path": prompt_path_str,
                    "payload_path": payload_path_str,
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
                "mode": "psaa",
                "case_id": cid,
                "cases": 1,
                "writes": len(rows),
                "manifest": to_posix(manifest_path.resolve()),
                "generations_index": to_posix(generations_path.resolve()),
                "prompt_spec": to_posix(spec_path.resolve()),
                "prompt_spec_version": spec.version,
                "retry_num": spec.retry_num,
                "attacker_model": spec.attacker.model,
                "checker_model": spec.checker.model,
                "start_directory_logical_path": start_dir,
                "start_index_path": start_index_norm,
                "active_attack_cases": to_posix(active_cases_path),
                "source_env_tree": to_posix(pre_tree_path),
                "post_env_tree": to_posix(post_env_tree_path),
            },
        )

        _emit_generation_event(
            "run_end",
            {
                "writes": len(rows),
                "manifest": to_posix(manifest_path.resolve()),
                "post_env_tree": to_posix(post_env_tree_path),
            },
        )

        return {
            "mode": "psaa",
            "case_id": cid,
            "cases": 1,
            "writes": len(rows),
            "manifest": to_posix(manifest_path.resolve()),
            "generations_index": to_posix(generations_path.resolve()),
            "start_directory_logical_path": start_dir,
            "start_index_path": start_index_norm,
            "active_attack_cases": to_posix(active_cases_path),
            "source_env_tree": to_posix(pre_tree_path),
            "post_env_tree": to_posix(post_env_tree_path),
        }
    except Exception as exc:
        _emit_generation_event(
            "run_error",
            {"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("file/exp/attack/psaa/file_psaa_v1.yaml"),
    )
    parser.add_argument(
        "--start-index-path",
        type=str,
        default=None,
        help="Optional directory index path like /0/1 (root is /). Overrides contract start.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = inject_psaa(
        run_dir=args.run_dir,
        spec_path=args.spec,
        case_id=args.case_id,
        start_index_path=args.start_index_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
