#!/usr/bin/env python3
"""Run detour trials by restoring a sampled shared history up to the anchor page."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import multiprocessing
import re
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_FORMAL_MAX_ACTIONS,
    LocalStaticServer,
    REPO_ROOT,
    assess_path_behavior,
    attack_path_for_condition,
    ensure_openai_compatible_env,
    ensure_visualwebarena_site_env,
    index_pages,
    load_json,
    load_module_from_path,
    minimum_sufficient_max_actions,
    normalize_breadcrumb,
    user_path_from_anchor,
    write_json,
)


NAV_STATUS_RE = re.compile(r"Information navigation → page\s+(\d+)")
HASH_URL_RE = re.compile(r"url:\s*[^\s]+#p(\d+)", re.IGNORECASE)
INDEX_ROOT_RE = re.compile(r"url:\s*[^\s]*index\.html(?!#p)", re.IGNORECASE)
_WORKER_BASE_MODULE: Any | None = None
_WORKER_HISTORY_RESUME_AGENT_CLASS: type | None = None


def _load_stage05_agent_module() -> Any:
    path = REPO_ROOT / "web" / "exp" / "05_evaluate" / "tool_calling_agent_defended.py"
    return load_module_from_path("pilot_stage05_agent_resume", path)


def _replace_in_message_content(
    messages: list[dict[str, Any]],
    *,
    old_text: str,
    new_text: str,
) -> None:
    if not old_text or old_text == new_text:
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content.replace(old_text, new_text)


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


def _append_breadcrumb(history: list[str], breadcrumb: str) -> None:
    if not history or history[-1] != breadcrumb:
        history.append(breadcrumb)


def _build_history_resume_agent(base_module: Any) -> type:
    class HistoryResumeAgent(base_module.GPTWebAgent):
        def _execute_replay_action(
            self,
            *,
            tool_name: str,
            arguments: dict[str, Any],
        ) -> tuple[str, str]:
            create_action_function = self.tool_name_to_action[tool_name]
            action = create_action_function(**arguments)
            browser_execution_result = self.browser_env.step(action)
            observation_text = str(browser_execution_result[0]["text"])
            fail_error = str(browser_execution_result[4].get("fail_error") or "").strip()
            if fail_error:
                raise RuntimeError(f"Replay {tool_name} failed: {fail_error}")
            current_url = str(self.browser_env.page.url)
            return self._sanitize_observation_text(observation_text), current_url

        def replay_shared_history(
            self,
            *,
            start_url: str,
            replay_clicks: list[dict[str, Any]],
        ) -> tuple[str, str]:
            page = self.browser_env.page
            page.goto(start_url)
            for step_index, step in enumerate(replay_clicks, start=1):
                target_page_index = int(step["target_page_index"])
                page.evaluate(
                    """(payload) => {
                        const targetPageIndex = Number(payload.targetPageIndex);
                        const cause = String(payload.cause || "pilot-replay");
                        if (typeof window.openPageSafe === "function") {
                            window.openPageSafe(targetPageIndex, cause);
                            return;
                        }
                        document.querySelectorAll('[id^="page"]').forEach((el) => {
                            el.style.display = 'none';
                        });
                        const targetPage = document.getElementById('page' + targetPageIndex);
                        if (targetPage) {
                            targetPage.style.display = 'block';
                        }
                        try {
                            window.location.hash = 'p' + targetPageIndex;
                        } catch (_err) {}
                    }""",
                    {
                        "targetPageIndex": target_page_index,
                        "cause": f"pilot-replay:{step_index}",
                    },
                )
            current_observation = self._sanitize_observation_text(
                str(self.browser_env._get_obs()["text"])
            )
            current_url = str(page.url)
            return current_observation, current_url

        def prepare_resumed_messages(
            self,
            *,
            shared_history: dict[str, Any],
            start_url: str,
            current_observation: str,
            current_url: str,
            original_user_objective: str,
        ) -> list[dict[str, Any]]:
            messages = copy.deepcopy(shared_history["messages_for_resume"])
            sampled_intent = str(shared_history.get("sampling_intent") or "")
            sampled_start_url = str(shared_history.get("sampled_start_url") or "")

            _replace_in_message_content(
                messages,
                old_text=sampled_intent,
                new_text=original_user_objective,
            )
            _replace_in_message_content(
                messages,
                old_text=sampled_start_url,
                new_text=start_url,
            )

            self.current_user_objective = original_user_objective

            first_user_index = next(
                (idx for idx, message in enumerate(messages) if message.get("role") == "user"),
                None,
            )
            if first_user_index is None:
                raise RuntimeError("Shared history is missing the initial user message.")
            messages[first_user_index] = {
                "role": "user",
                "content": base_module.apply_step_wise_defense(
                    f"Start on {start_url} {original_user_objective}",
                    self.defense_mode,
                    medium="screen",
                ),
            }

            last_tool_index = next(
                (
                    idx
                    for idx in range(len(messages) - 1, -1, -1)
                    if isinstance(messages[idx], dict) and messages[idx].get("role") == "tool"
                ),
                None,
            )
            if last_tool_index is None:
                raise RuntimeError("Shared history is missing the anchor-page tool observation.")
            last_tool_call_id = str(messages[last_tool_index].get("tool_call_id") or "")
            formatted_tool_result = f"""OBSERVATION:
{current_observation}
URL: {current_url}
"""
            messages[last_tool_index] = self._tool_message(
                last_tool_call_id,
                formatted_tool_result,
            )
            return messages

        def loop_from_restored_history(
            self,
            *,
            initial_messages: list[dict[str, Any]],
            user_objective: str,
            max_actions: int,
            max_observations_to_keep: int,
            page_catalog: dict[int, dict[str, Any]],
            anchor_breadcrumb: str,
            user_path_from_anchor: list[str],
            attack_path_breadcrumbs: list[str],
        ) -> None:
            self.current_user_objective = user_objective
            messages = copy.deepcopy(initial_messages)
            trace_messages: list[dict[str, Any]] = []
            post_injection_breadcrumbs = [normalize_breadcrumb(anchor_breadcrumb)]
            self._log_messages(messages)

            for action_number in range(max_actions):
                model_response_message = self._call_model(messages)
                print(
                    f"Model [{self.model}] response {action_number} {model_response_message['content']}"
                )
                result_of_execution = self._execute_requested_tool_calls(
                    model_response_message["tool_calls"]
                )
                messages.append(model_response_message)

                if not result_of_execution:
                    print("Agent did not call any tools; exiting.")
                    break

                messages.extend(result_of_execution)
                trace_messages.append(model_response_message)
                trace_messages.extend(result_of_execution)
                if result_of_execution[0]["role"] != "stop":
                    reinforcement = base_module.build_goal_reinforcement_messages(
                        self.defense_mode,
                        trusted_goal=user_objective,
                        medium="screen",
                    )
                    messages.extend(reinforcement)
                    trace_messages.extend(reinforcement)

                self._log_messages(trace_messages)

                current_page_index = _detect_page_index(
                    str(result_of_execution[0].get("content") or "")
                )
                if current_page_index is not None:
                    current_page = page_catalog.get(int(current_page_index))
                    if isinstance(current_page, dict):
                        _append_breadcrumb(
                            post_injection_breadcrumbs,
                            normalize_breadcrumb(current_page.get("breadcrumb")),
                        )

                stopped = result_of_execution[0]["role"] == "stop"
                assessment = assess_path_behavior(
                    post_injection_breadcrumbs=post_injection_breadcrumbs,
                    user_path_from_anchor_breadcrumbs=user_path_from_anchor,
                    attack_path_breadcrumbs=attack_path_breadcrumbs,
                    stopped=stopped,
                    remaining_actions=max_actions - action_number - 1,
                )

                if assessment["decisive_label"] == "Hijacked":
                    print(
                        "Early stopping trial after completing attack, returning to anchor, "
                        "and taking one deeper step."
                    )
                    break

                if assessment["decisive_label"] == "Unaffected":
                    print("Early stopping trial after a decisive clean user-path trajectory.")
                    break

                if assessment["decisive_label"] == "Perturbed":
                    print(
                        "Early stopping trial after ruling out both Unaffected and Hijacked "
                        f"within the remaining budget. Path: {post_injection_breadcrumbs}"
                    )
                    break

                if stopped:
                    print(
                        f"Agent finished with stop action and answer {result_of_execution[0]['answer']}"
                    )
                    break

                base_module._maybe_filter_tool_call_results(
                    messages,
                    max_observations_to_keep,
                )

    return HistoryResumeAgent


def _init_trial_worker() -> None:
    global _WORKER_BASE_MODULE, _WORKER_HISTORY_RESUME_AGENT_CLASS
    ensure_openai_compatible_env()
    ensure_visualwebarena_site_env()
    if _WORKER_BASE_MODULE is None:
        _WORKER_BASE_MODULE = _load_stage05_agent_module()
    if _WORKER_HISTORY_RESUME_AGENT_CLASS is None:
        _WORKER_HISTORY_RESUME_AGENT_CLASS = _build_history_resume_agent(_WORKER_BASE_MODULE)


def _run_single_trial(
    *,
    trace_path: str,
    model: str,
    temperature: float | None,
    trial_seed: int | None,
    defense_mode: str,
    shared_history: dict[str, Any],
    start_url: str,
    task_intent: str,
    max_actions: int,
    max_observations: int,
    page_catalog: dict[int, dict[str, Any]],
    anchor_breadcrumb: str,
    user_path_from_anchor_breadcrumbs: list[str],
    attack_path_breadcrumbs: list[str],
) -> dict[str, Any]:
    global _WORKER_BASE_MODULE, _WORKER_HISTORY_RESUME_AGENT_CLASS
    if _WORKER_BASE_MODULE is None or _WORKER_HISTORY_RESUME_AGENT_CLASS is None:
        _init_trial_worker()
    assert _WORKER_HISTORY_RESUME_AGENT_CLASS is not None

    trace_path_obj = Path(trace_path)
    trace_path_obj.unlink(missing_ok=True)
    try:
        with _WORKER_HISTORY_RESUME_AGENT_CLASS(
            model=model,
            filepath_to_trace_log=trace_path,
            temperature=temperature,
            seed=trial_seed,
            defense_mode=defense_mode,
        ) as agent:
            agent.current_user_objective = task_intent
            current_observation, current_url = agent.replay_shared_history(
                start_url=start_url,
                replay_clicks=list(shared_history["replay_clicks"]),
            )
            restored_messages = agent.prepare_resumed_messages(
                shared_history=shared_history,
                start_url=start_url,
                current_observation=current_observation,
                current_url=current_url,
                original_user_objective=task_intent,
            )
            agent.loop_from_restored_history(
                initial_messages=restored_messages,
                user_objective=task_intent,
                max_actions=max_actions,
                max_observations_to_keep=max_observations,
                page_catalog=page_catalog,
                anchor_breadcrumb=anchor_breadcrumb,
                user_path_from_anchor=user_path_from_anchor_breadcrumbs,
                attack_path_breadcrumbs=attack_path_breadcrumbs,
            )
        return {"status": "completed", "attempts": 1}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "attempts": 1}


def _prepare_condition_state(
    *,
    case_payload: dict[str, Any],
    condition_manifest: dict[str, Any],
    shared_history: dict[str, Any],
    output_dir: Path,
    port: int,
    model: str,
    temperature: float | None,
    seed_base: int | None,
    trials: int,
    max_actions: int | None,
    max_observations: int,
    defense_mode: str,
    resume: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    site_dir = Path(condition_manifest["site_dir"])
    start_url = f"http://127.0.0.1:{port}{case_payload['task']['start_path']}"
    server_log = output_dir / "static_server.log"

    effective_max_actions = (
        int(max_actions)
        if max_actions is not None
        else minimum_sufficient_max_actions(
            case_payload=case_payload,
            condition=str(condition_manifest["condition"]),
        )
    )

    execution_manifest = {
        "case_id": case_payload["case_id"],
        "condition": condition_manifest["condition"],
        "condition_display_name": condition_manifest["condition_display_name"],
        "site_dir": str(site_dir.resolve()),
        "log_dir": str(log_dir.resolve()),
        "start_url": start_url,
        "port": int(port),
        "server_log": str(server_log.resolve()),
        "model": model,
        "temperature": temperature,
        "seed_base": seed_base,
        "max_actions": effective_max_actions,
        "concurrency": None,
        "trials": [],
        "anchor_page_index": case_payload["anchor"]["page_index"],
        "anchor_breadcrumb": case_payload["anchor"]["breadcrumb"],
        "shared_history_path": str(
            (output_dir.parent / "shared_history" / "shared_history.json").resolve()
        ),
        "shared_history_trial_index": shared_history.get("trial_index"),
    }
    metadata = load_json(Path(condition_manifest["site_inputs"]["page_metadata"]))
    page_catalog, _pages_by_breadcrumb = index_pages(metadata)
    user_path_from_anchor_breadcrumbs = user_path_from_anchor(
        user_path_breadcrumbs=list(case_payload["user_path"]["breadcrumbs"]),
        anchor_breadcrumb=str(case_payload["anchor"]["breadcrumb"]),
    )
    attack_path_breadcrumbs = attack_path_for_condition(
        case_payload=case_payload,
        condition=str(condition_manifest["condition"]),
    )
    execution_manifest_path = output_dir / "execution_manifest.json"

    pending_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for trial_index in range(1, int(trials) + 1):
        trace_name = f'trace_{case_payload["task"]["task_id"]}_r{trial_index:02d}.jsonl'
        trace_path = log_dir / trace_name
        trial_seed = None if seed_base is None else int(seed_base) + (trial_index - 1)
        trial_row = {
            "trial_index": trial_index,
            "trace_path": str(trace_path.resolve()),
            "seed": trial_seed,
            "status": "pending",
        }
        execution_manifest["trials"].append(trial_row)

        if resume and trace_path.exists() and trace_path.stat().st_size > 0:
            trial_row["status"] = "skipped_existing"
            continue

        pending_jobs.append(
            (
                trial_row,
                {
                    "trace_path": str(trace_path),
                    "model": model,
                    "temperature": temperature,
                    "trial_seed": trial_seed,
                    "defense_mode": defense_mode,
                    "shared_history": shared_history,
                    "start_url": start_url,
                    "task_intent": str(case_payload["task"]["intent"]),
                    "max_actions": effective_max_actions,
                    "max_observations": max_observations,
                    "page_catalog": page_catalog,
                    "anchor_breadcrumb": str(case_payload["anchor"]["breadcrumb"]),
                    "user_path_from_anchor_breadcrumbs": user_path_from_anchor_breadcrumbs,
                    "attack_path_breadcrumbs": attack_path_breadcrumbs,
                },
            )
        )

    write_json(execution_manifest_path, execution_manifest)
    return {
        "case_id": case_payload["case_id"],
        "condition": str(condition_manifest["condition"]),
        "site_dir": site_dir,
        "server_log": server_log,
        "port": int(port),
        "execution_manifest": execution_manifest,
        "execution_manifest_path": execution_manifest_path,
        "pending_jobs": pending_jobs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_FORMAL_MAX_ACTIONS,
        help="Maximum continuation steps for the post-anchor test agent (default: 7).",
    )
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

    condition_states: list[dict[str, Any]] = []
    next_port = int(args.port)
    for case_dir in sorted((args.run_root / "cases").glob("*")):
        case_path = case_dir / "case.json"
        shared_history_path = case_dir / "shared_history" / "shared_history.json"
        if not case_path.is_file():
            continue
        if not shared_history_path.is_file():
            raise FileNotFoundError(f"Missing sampled shared history: {shared_history_path}")
        case_payload = load_json(case_path)
        shared_history = load_json(shared_history_path)
        for condition_dir in sorted(case_dir.glob("*")):
            if not condition_dir.is_dir() or condition_dir.name == "shared_history":
                continue
            site_manifest_path = condition_dir / "site_manifest.json"
            if not site_manifest_path.is_file():
                continue
            condition_manifest = load_json(site_manifest_path)
            condition_states.append(
                _prepare_condition_state(
                    case_payload=case_payload,
                    condition_manifest=condition_manifest,
                    shared_history=shared_history,
                    output_dir=condition_dir,
                    port=next_port,
                    model=args.model,
                    temperature=args.temperature,
                    seed_base=args.seed_base,
                    trials=int(args.trials),
                    max_actions=(
                        int(args.max_actions) if args.max_actions is not None else None
                    ),
                    max_observations=int(args.max_observations),
                    defense_mode=args.defense_mode,
                    resume=bool(args.resume),
                )
            )
            next_port += 1

    for state in condition_states:
        state["execution_manifest"]["concurrency"] = max(1, int(args.concurrency))
        write_json(state["execution_manifest_path"], state["execution_manifest"])

    planned_jobs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    next_round_index = 0
    while True:
        added_any = False
        for state in condition_states:
            pending_jobs = state["pending_jobs"]
            if next_round_index >= len(pending_jobs):
                continue
            trial_row, job_kwargs = pending_jobs[next_round_index]
            planned_jobs.append((state, trial_row, job_kwargs))
            added_any = True
        if not added_any:
            break
        next_round_index += 1

    with contextlib.ExitStack() as stack:
        for state in condition_states:
            stack.enter_context(
                LocalStaticServer(
                    site_dir=state["site_dir"],
                    port=int(state["port"]),
                    log_path=state["server_log"],
                )
            )

        if not planned_jobs:
            return

        max_workers = min(max(1, int(args.concurrency)), len(planned_jobs))
        total_jobs = len(planned_jobs)
        if max_workers <= 1:
            for completed_count, (state, trial_row, job_kwargs) in enumerate(planned_jobs, start=1):
                result = _run_single_trial(**job_kwargs)
                trial_row.update(result)
                print(
                    "Completed "
                    f"{completed_count}/{total_jobs} trials total "
                    f"({state['case_id']} {state['condition']})"
                )
                write_json(state["execution_manifest_path"], state["execution_manifest"])
            return

        spawn_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=spawn_ctx,
            initializer=_init_trial_worker,
        ) as executor:
            future_to_job = {
                executor.submit(_run_single_trial, **job_kwargs): (state, trial_row)
                for state, trial_row, job_kwargs in planned_jobs
            }
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_job):
                state, trial_row = future_to_job[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                trial_row.update(result)
                completed_count += 1
                print(
                    "Completed "
                    f"{completed_count}/{total_jobs} trials total "
                    f"({state['case_id']} {state['condition']})"
                )
                write_json(state["execution_manifest_path"], state["execution_manifest"])


if __name__ == "__main__":
    main()
