#!/usr/bin/env python3
"""Sample real shared histories that reach the pilot anchor under path hints."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import re
from pathlib import Path
from typing import Any

from common import (
    LocalStaticServer,
    REPO_ROOT,
    ensure_openai_compatible_env,
    ensure_visualwebarena_site_env,
    find_click_target,
    index_pages,
    load_json,
    load_module_from_path,
    normalize_breadcrumb,
    write_json,
)

NAV_STATUS_RE = re.compile(r"Information navigation → page\s+(\d+)")
HASH_URL_RE = re.compile(r"url:\s*[^\s]+#p(\d+)", re.IGNORECASE)
INDEX_ROOT_RE = re.compile(r"url:\s*[^\s]*index\.html(?!#p)", re.IGNORECASE)
TRACE_RE = re.compile(r"trace_sampling_r(?P<trial>\d+)\.jsonl$")
EXPLICIT_HINT_PATTERNS = (
    re.compile(r"\bnavigation hint\b", re.IGNORECASE),
    re.compile(r"\bpath hint\b", re.IGNORECASE),
    re.compile(r"\bspecified path\b", re.IGNORECASE),
    re.compile(r"\bbreadcrumb\b", re.IGNORECASE),
    re.compile(r"\baccording to (?:the )?(?:hint|path)\b", re.IGNORECASE),
    re.compile(r"\bfollow(?:ing)? (?:the )?(?:hint|path|specified path)\b", re.IGNORECASE),
    re.compile(r"\bthe task (?:says|indicates|mentions|states)\b", re.IGNORECASE),
    re.compile(r"\blikely related to content around\b", re.IGNORECASE),
    re.compile(r"\bas indicated in the task\b", re.IGNORECASE),
    re.compile(r"\bas described in the task\b", re.IGNORECASE),
)


def _load_stage05_agent_module() -> Any:
    path = REPO_ROOT / "web" / "exp" / "05_evaluate" / "tool_calling_agent_defended.py"
    return load_module_from_path("pilot_stage05_agent_sampling", path)


def _build_sampling_agent(base_module: Any) -> type:
    class SamplingAgent(base_module.GPTWebAgent):
        def loop_until_anchor(
            self,
            *,
            start_url: str,
            user_objective: str,
            anchor_page_index: int,
            max_actions: int,
            max_observations_to_keep: int,
        ) -> None:
            self.current_user_objective = user_objective
            messages: list[dict[str, Any]] = []
            system_message = {
                "role": "system",
                "content": base_module.apply_system_prompt_defense(
                    self.system_prompt,
                    self.defense_mode,
                    medium="screen",
                ),
            }
            messages.append(system_message)

            user_intent_message = {
                "role": "user",
                "content": base_module.apply_step_wise_defense(
                    f"Start on {start_url} {user_objective}",
                    self.defense_mode,
                    medium="screen",
                ),
            }
            messages.append(user_intent_message)

            for action_number in range(max_actions):
                model_response_message = self._call_model(messages)

                print(
                    f"Model [{self.model}] response {action_number} {model_response_message['content']}"
                )

                result_of_execution = self._execute_requested_tool_calls(
                    model_response_message["tool_calls"]
                )

                messages.append(model_response_message)

                if len(result_of_execution) < 1:
                    print("Agent did not call any tools; exiting.")
                    break

                messages.extend(result_of_execution)
                if result_of_execution[0]["role"] != "stop":
                    messages.extend(
                        base_module.build_goal_reinforcement_messages(
                            self.defense_mode,
                            trusted_goal=user_objective,
                            medium="screen",
                        )
                    )

                self._log_messages(messages)

                if result_of_execution[0]["role"] == "stop":
                    print(
                        f"Agent finished with stop action and answer {result_of_execution[0]['answer']}"
                    )
                    break

                current_page_index = _detect_page_index(
                    str(result_of_execution[0].get("content") or "")
                )
                if current_page_index == int(anchor_page_index):
                    print(f"Sampling reached anchor page {anchor_page_index}; stopping.")
                    break

                base_module._maybe_filter_tool_call_results(messages, max_observations_to_keep)

    return SamplingAgent


def _detect_page_index(content: str) -> int | None:
    if not content:
        return None
    status_matches = NAV_STATUS_RE.findall(content)
    if status_matches:
        return int(status_matches[-1])
    hash_matches = HASH_URL_RE.findall(content)
    if hash_matches:
        return int(hash_matches[-1])
    if INDEX_ROOT_RE.search(content):
        return 0
    return None


def _normalize_tool_arguments(args_payload: object) -> dict[str, object]:
    if isinstance(args_payload, str):
        try:
            parsed = json.loads(args_payload)
        except json.JSONDecodeError:
            return {}
    else:
        parsed = args_payload
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    return dict(parsed) if isinstance(parsed, dict) else {}


def _assistant_mentions_hint(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        if any(pattern.search(content) for pattern in EXPLICIT_HINT_PATTERNS):
            return True
    return False


def _build_sampling_intent(case_payload: dict[str, Any]) -> str:
    hint_text = str(case_payload["sampling"]["hint_text"]).strip()
    original_intent = str(case_payload["task"]["intent"]).strip()
    return f"{hint_text}\n\n{original_intent}".strip()


def _action_signature(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(action.get("tool") or ""),
        normalize_breadcrumb(action.get("source_breadcrumb")),
        normalize_breadcrumb(action.get("target_breadcrumb")),
        str(action.get("element_id") or "").strip(),
    )


def _analyze_snapshot(
    messages: list[dict[str, Any]],
    *,
    pages_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    current_page_index: int | None = None
    breadcrumbs: list[str] = []
    raw_actions: list[dict[str, Any]] = []
    pending_action: dict[str, Any] | None = None
    had_tool_error = False
    unsupported_action = False
    saw_initial_goto = False
    normalized_clicks: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if len(tool_calls) > 1:
                unsupported_action = True
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                tool_name = str(function.get("name") or "").strip()
                arguments = _normalize_tool_arguments(function.get("arguments", "{}"))
                action_entry = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "page_before": current_page_index,
                }
                raw_actions.append(action_entry)
                pending_action = action_entry
                if tool_name not in {"goto", "click"}:
                    unsupported_action = True
        elif role == "tool":
            content = str(message.get("content") or "")
            if content.lstrip().startswith("ERROR:"):
                had_tool_error = True
                if pending_action is not None:
                    pending_action["tool_error"] = content
                pending_action = None
                continue
            new_page_index = _detect_page_index(content)
            if new_page_index is None:
                pending_action = None
                continue
            if current_page_index != new_page_index:
                current_page_index = int(new_page_index)
                breadcrumb = normalize_breadcrumb(
                    (pages_by_index.get(current_page_index) or {}).get("breadcrumb")
                )
                if not breadcrumbs or breadcrumbs[-1] != breadcrumb:
                    breadcrumbs.append(breadcrumb)
            if pending_action is not None:
                pending_action["page_after"] = current_page_index
                pending_action = None

    for index, action in enumerate(raw_actions):
        tool_name = str(action.get("tool") or "")
        page_before = action.get("page_before")
        page_after = action.get("page_after")
        if tool_name == "goto":
            if index != 0 or page_after != 0:
                unsupported_action = True
            saw_initial_goto = True
            continue
        if tool_name != "click":
            unsupported_action = True
            continue
        if page_before is None or page_after is None or int(page_before) == int(page_after):
            unsupported_action = True
            continue
        source_page = pages_by_index.get(int(page_before))
        if source_page is None:
            unsupported_action = True
            continue
        try:
            stable_click = find_click_target(source_page, target_page_index=int(page_after))
        except Exception:
            unsupported_action = True
            continue
        normalized_clicks.append(
            {
                "tool": "click",
                "source_page_index": int(page_before),
                "source_breadcrumb": normalize_breadcrumb(source_page.get("breadcrumb")),
                "target_page_index": int(page_after),
                "target_breadcrumb": normalize_breadcrumb(stable_click.get("target_breadcrumb")),
                "element_id": str(stable_click.get("element_id") or "").strip(),
                "label": str(stable_click.get("label") or "").strip(),
            }
        )

    return {
        "current_page_index": current_page_index,
        "breadcrumbs": breadcrumbs,
        "raw_actions": raw_actions,
        "normalized_clicks": normalized_clicks,
        "had_tool_error": had_tool_error,
        "unsupported_action": unsupported_action,
        "saw_initial_goto": saw_initial_goto,
    }


def _first_anchor_candidate(
    trace_path: Path,
    *,
    case_payload: dict[str, Any],
    base_module: Any,
    max_observations: int,
    start_url: str,
) -> dict[str, Any] | None:
    lines = trace_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None

    metadata = load_json(Path(case_payload["inputs"]["page_metadata"]))
    pages_by_index, _pages_by_breadcrumb = index_pages(metadata)
    anchor_page_index = int(case_payload["anchor"]["page_index"])
    expected_breadcrumbs = list(case_payload["prefix"]["breadcrumbs"])
    expected_clicks = list(case_payload["prefix"]["actions"])
    sampling_intent = _build_sampling_intent(case_payload)

    for line_number, line in enumerate(lines, start=1):
        try:
            messages = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(messages, list):
            continue
        snapshot = _analyze_snapshot(messages, pages_by_index=pages_by_index)
        if int(snapshot["current_page_index"] or -1) != anchor_page_index:
            continue

        issues: list[str] = []
        if not snapshot["saw_initial_goto"]:
            issues.append("missing_initial_goto")
        if snapshot["had_tool_error"]:
            issues.append("tool_error_before_anchor")
        if snapshot["unsupported_action"]:
            issues.append("unsupported_action_before_anchor")
        if list(snapshot["breadcrumbs"]) != expected_breadcrumbs:
            issues.append("sampled_path_mismatch")
        if len(snapshot["normalized_clicks"]) != len(expected_clicks):
            issues.append("sampled_click_count_mismatch")
        else:
            sampled_sig = [_action_signature(action) for action in snapshot["normalized_clicks"]]
            expected_sig = [_action_signature(action) for action in expected_clicks]
            if sampled_sig != expected_sig:
                issues.append("sampled_click_sequence_mismatch")
        if _assistant_mentions_hint(messages):
            issues.append("assistant_explicitly_mentions_hint")

        messages_for_resume = copy.deepcopy(messages)
        base_module._maybe_filter_tool_call_results(messages_for_resume, int(max_observations))

        return {
            "accepted": not issues,
            "issues": issues,
            "line_number": line_number,
            "messages_for_resume": messages_for_resume,
            "breadcrumbs": list(snapshot["breadcrumbs"]),
            "replay_clicks": list(snapshot["normalized_clicks"]),
            "raw_actions": list(snapshot["raw_actions"]),
            "sampling_intent": sampling_intent,
            "start_url": start_url,
        }
    return None


def _sample_case(
    *,
    base_module: Any,
    case_payload: dict[str, Any],
    site_dir: Path,
    output_dir: Path,
    port: int,
    model: str,
    temperature: float | None,
    seed_base: int | None,
    trials: int,
    max_actions: int,
    max_observations: int,
    defense_mode: str,
    resume: bool,
) -> dict[str, Any]:
    SamplingAgent = _build_sampling_agent(base_module)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "trace_logs"
    trace_dir.mkdir(parents=True, exist_ok=True)
    shared_history_path = output_dir / "shared_history.json"
    manifest_path = output_dir / "sampling_manifest.json"
    if resume and shared_history_path.is_file() and manifest_path.is_file():
        return load_json(manifest_path)

    start_url = f"http://127.0.0.1:{port}{case_payload['task']['start_path']}"
    sampling_intent = _build_sampling_intent(case_payload)
    server_log = output_dir / "static_server.log"
    manifest = {
        "case_id": case_payload["case_id"],
        "anchor_breadcrumb": case_payload["anchor"]["breadcrumb"],
        "anchor_page_index": case_payload["anchor"]["page_index"],
        "hint_text": case_payload["sampling"]["hint_text"],
        "sampling_intent": sampling_intent,
        "original_intent": case_payload["task"]["intent"],
        "trace_dir": str(trace_dir.resolve()),
        "trials": [],
        "selected_trial_index": None,
        "shared_history_path": str(shared_history_path.resolve()),
    }

    with LocalStaticServer(site_dir=site_dir, port=port, log_path=server_log):
        for trial_index in range(1, int(trials) + 1):
            trace_path = trace_dir / f"trace_sampling_r{trial_index:02d}.jsonl"
            trial_seed = None if seed_base is None else int(seed_base) + (trial_index - 1)
            trial_row = {
                "trial_index": trial_index,
                "trace_path": str(trace_path.resolve()),
                "seed": trial_seed,
                "status": "pending",
            }
            manifest["trials"].append(trial_row)

            if resume and trace_path.exists() and trace_path.stat().st_size > 0:
                candidate = _first_anchor_candidate(
                    trace_path,
                    case_payload=case_payload,
                    base_module=base_module,
                    max_observations=max_observations,
                    start_url=start_url,
                )
                if candidate is None:
                    trial_row["status"] = "rejected_missing_anchor"
                    trial_row["issues"] = ["anchor_not_reached"]
                    continue
                trial_row["status"] = "accepted" if candidate["accepted"] else "rejected"
                trial_row["issues"] = list(candidate["issues"])
                trial_row["anchor_line_number"] = int(candidate["line_number"])
                trial_row["sampled_breadcrumbs"] = list(candidate["breadcrumbs"])
                if candidate["accepted"]:
                    shared_history_payload = {
                        "case_id": case_payload["case_id"],
                        "trial_index": trial_index,
                        "trace_path": str(trace_path.resolve()),
                        "anchor_breadcrumb": case_payload["anchor"]["breadcrumb"],
                        "anchor_page_index": case_payload["anchor"]["page_index"],
                        "breadcrumbs": list(candidate["breadcrumbs"]),
                        "replay_clicks": list(candidate["replay_clicks"]),
                        "messages_for_resume": list(candidate["messages_for_resume"]),
                        "sampling_intent": sampling_intent,
                        "original_intent": case_payload["task"]["intent"],
                        "hint_text": case_payload["sampling"]["hint_text"],
                        "hint_display_path": case_payload["sampling"]["hint_display_path"],
                        "sampled_start_url": start_url,
                        "max_observations_to_keep": int(max_observations),
                    }
                    write_json(shared_history_path, shared_history_payload)
                    manifest["selected_trial_index"] = trial_index
                    break
                continue

            try:
                with SamplingAgent(
                    model=model,
                    filepath_to_trace_log=str(trace_path),
                    temperature=temperature,
                    seed=trial_seed,
                    defense_mode=defense_mode,
                ) as agent:
                    agent.loop_until_anchor(
                        start_url=start_url,
                        user_objective=sampling_intent,
                        anchor_page_index=int(case_payload["anchor"]["page_index"]),
                        max_actions=max_actions,
                        max_observations_to_keep=max_observations,
                    )
                trial_row["status"] = "completed"
            except Exception as exc:
                trial_row["status"] = "failed"
                trial_row["issues"] = [f"{type(exc).__name__}: {exc}"]
                break

            candidate = _first_anchor_candidate(
                trace_path,
                case_payload=case_payload,
                base_module=base_module,
                max_observations=max_observations,
                start_url=start_url,
            )
            if candidate is None:
                trial_row["status"] = "rejected_missing_anchor"
                trial_row["issues"] = ["anchor_not_reached"]
                continue

            trial_row["status"] = "accepted" if candidate["accepted"] else "rejected"
            trial_row["issues"] = list(candidate["issues"])
            trial_row["anchor_line_number"] = int(candidate["line_number"])
            trial_row["sampled_breadcrumbs"] = list(candidate["breadcrumbs"])
            if not candidate["accepted"]:
                continue

            shared_history_payload = {
                "case_id": case_payload["case_id"],
                "trial_index": trial_index,
                "trace_path": str(trace_path.resolve()),
                "anchor_breadcrumb": case_payload["anchor"]["breadcrumb"],
                "anchor_page_index": case_payload["anchor"]["page_index"],
                "breadcrumbs": list(candidate["breadcrumbs"]),
                "replay_clicks": list(candidate["replay_clicks"]),
                "messages_for_resume": list(candidate["messages_for_resume"]),
                "sampling_intent": sampling_intent,
                "original_intent": case_payload["task"]["intent"],
                "hint_text": case_payload["sampling"]["hint_text"],
                "hint_display_path": case_payload["sampling"]["hint_display_path"],
                "sampled_start_url": start_url,
                "max_observations_to_keep": int(max_observations),
            }
            write_json(shared_history_path, shared_history_payload)
            manifest["selected_trial_index"] = trial_index
            break

    if manifest["selected_trial_index"] is None:
        write_json(manifest_path, manifest)
        raise RuntimeError(
            f"Unable to sample an acceptable shared history for case {case_payload['case_id']}"
        )

    write_json(manifest_path, manifest)
    return manifest


def _sample_case_worker(
    *,
    case_path: str,
    site_manifest_path: str,
    port: int,
    model: str,
    temperature: float | None,
    seed_base: int | None,
    trials: int,
    max_actions: int,
    max_observations: int,
    defense_mode: str,
    resume: bool,
) -> dict[str, Any]:
    ensure_openai_compatible_env()
    ensure_visualwebarena_site_env()
    base_module = _load_stage05_agent_module()
    case_path_obj = Path(case_path)
    case_payload = load_json(case_path_obj)
    site_manifest = load_json(Path(site_manifest_path))
    return _sample_case(
        base_module=base_module,
        case_payload=case_payload,
        site_dir=Path(site_manifest["site_dir"]),
        output_dir=case_path_obj.parent / "shared_history",
        port=int(port),
        model=model,
        temperature=temperature,
        seed_base=seed_base,
        trials=int(trials),
        max_actions=int(max_actions),
        max_observations=int(max_observations),
        defense_mode=defense_mode,
        resume=bool(resume),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--max-observations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--defense-mode", type=str, default="default_attack")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_openai_compatible_env()
    ensure_visualwebarena_site_env()
    case_jobs: list[dict[str, Any]] = []
    next_port = int(args.port)
    for case_dir in sorted((args.run_root / "cases").glob("*")):
        case_path = case_dir / "case.json"
        site_manifest_path = case_dir / "no_injection" / "site_manifest.json"
        if not case_path.is_file() or not site_manifest_path.is_file():
            continue
        case_jobs.append(
            {
                "case_path": str(case_path.resolve()),
                "site_manifest_path": str(site_manifest_path.resolve()),
                "port": next_port,
            }
        )
        next_port += 1

    if not case_jobs:
        return

    if max(1, int(args.concurrency)) <= 1 or len(case_jobs) == 1:
        for job in case_jobs:
            _sample_case_worker(
                case_path=job["case_path"],
                site_manifest_path=job["site_manifest_path"],
                port=int(job["port"]),
                model=args.model,
                temperature=args.temperature,
                seed_base=args.seed_base,
                trials=int(args.trials),
                max_actions=int(args.max_actions),
                max_observations=int(args.max_observations),
                defense_mode=args.defense_mode,
                resume=bool(args.resume),
            )
        return

    max_workers = min(max(1, int(args.concurrency)), len(case_jobs))
    spawn_ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=spawn_ctx,
    ) as executor:
        futures = [
            executor.submit(
                _sample_case_worker,
                case_path=job["case_path"],
                site_manifest_path=job["site_manifest_path"],
                port=int(job["port"]),
                model=args.model,
                temperature=args.temperature,
                seed_base=args.seed_base,
                trials=int(args.trials),
                max_actions=int(args.max_actions),
                max_observations=int(args.max_observations),
                defense_mode=args.defense_mode,
                resume=bool(args.resume),
            )
            for job in case_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
