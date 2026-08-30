#!/usr/bin/env python3
"""Run the Web/GitLab Full-condition model ablation in an isolated directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Rebuttal.experiments.materialize_formal_envs import _prepare_browser_static
from Rebuttal.experiments.run_formal_deepseek import (
    CommandResult,
    _base_env,
    _copy_browser_tasks_with_port,
    _copy_canonical_logs,
    _find_free_port,
    _run_command,
    _wait_for_url,
    _write_json,
)

DATASET = "visualwebarena"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_CHECK_MODEL = "deepseek-v3-1-terminus"
DEFAULT_SOURCE_BATCH = (
    REPO_ROOT
    / "web/runs/exp_d10w2_psaa/batch_runs/20260405_161802_full"
)
DEFAULT_USER_TASKS = REPO_ROOT / "web/runs/exp_d10w2_psaa/webarena_tasks"
DEFAULT_ATTACKER_TASKS = (
    REPO_ROOT / "web/runs/exp_d10w2_psaa/webarena_tasks_attacker_wasp"
)
DEFAULT_SHARED_STATIC = REPO_ROOT / "Rebuttal/runs/browser_formal/_shared_static"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "Rebuttal/runs/model_ablation/web_gitlab"
LOCKED_TEMPERATURE = 1.0
LOCKED_TRIALS = 3
LOCKED_SEED = 42
LOCKED_MAX_ACTIONS = 20
LOCKED_MAX_OBSERVATIONS = 20
LOCKED_MAX_PARALLEL = 6
LOCKED_DEFENSE_MODE = "default_attack"
LOCKED_CHECK_TEMPERATURE = 0.0
LOCKED_CHECK_MAX_TOKENS = 4096


def model_path_name(model: str) -> str:
    """Return a readable path component without allowing path traversal."""

    normalized = model.strip().replace("/", "__")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("Model name does not contain a usable path component.")
    return normalized


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_tasks(task_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in task_dir.glob("*.json")
            if path.is_file() and path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
    )


def discover_gitlab_attacks(
    source_batch: Path,
    attacker_tasks_root: Path,
) -> list[tuple[str, Path, Path]]:
    if not source_batch.is_dir():
        raise FileNotFoundError(f"Missing Full-condition source batch: {source_batch}")
    if not attacker_tasks_root.is_dir():
        raise FileNotFoundError(
            f"Missing attacker-task directory: {attacker_tasks_root}"
        )

    attacks: list[tuple[str, Path, Path]] = []
    for attack_dir in sorted(source_batch.iterdir()):
        if not attack_dir.is_dir() or not attack_dir.name.startswith("wasp_gitlab_"):
            continue
        metadata = attack_dir / "static_snapshot/page_metadata.json"
        attacker_tasks = attacker_tasks_root / attack_dir.name
        if not metadata.is_file():
            raise FileNotFoundError(f"Missing frozen metadata: {metadata}")
        if not attacker_tasks.is_dir():
            raise FileNotFoundError(f"Missing attacker tasks: {attacker_tasks}")
        attacks.append((attack_dir.name, metadata, attacker_tasks))
    if not attacks:
        raise ValueError(f"No GitLab attacks found under {source_batch}")
    return attacks


def _safe_reset_directory(path: Path, *, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to reset path outside output root: {resolved}")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _keep_selected_tasks(task_dir: Path, task_ids: set[int] | None) -> list[int]:
    discovered = _numeric_tasks(task_dir)
    if task_ids is None:
        return [int(path.stem) for path in discovered]
    for path in discovered:
        if int(path.stem) not in task_ids:
            path.unlink()
    retained = [int(path.stem) for path in _numeric_tasks(task_dir)]
    missing = sorted(task_ids - set(retained))
    if missing:
        raise ValueError(f"Requested task IDs are unavailable: {missing}")
    return retained


def _apply_user_intent_prefix(task_dir: Path, prefix: str) -> None:
    if not prefix:
        return
    for task_path in _numeric_tasks(task_dir):
        payload = _load_json(task_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected task JSON object: {task_path}")
        intent = payload.get("intent")
        if not isinstance(intent, str):
            raise ValueError(f"Expected string intent in task: {task_path}")
        payload["intent"] = f"{prefix}{intent}"
        _write_json(task_path, payload)


def prepare_attack_environment(
    *,
    attack_name: str,
    source_metadata: Path,
    source_user_tasks: Path,
    source_attacker_tasks: Path,
    shared_static: Path,
    output_dir: Path,
    allowed_output_root: Path,
    port: int,
    task_ids: set[int] | None,
    reset_output: bool,
    user_intent_prefix: str = "",
) -> tuple[Path, Path, Path, list[int]]:
    if reset_output:
        _safe_reset_directory(output_dir, allowed_root=allowed_output_root)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_json(source_metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Expected metadata JSON object: {source_metadata}")
    static_dir = _prepare_browser_static(
        shared_static=shared_static,
        env_root=output_dir,
        metadata=metadata,
    )
    user_tasks = output_dir / "webarena_tasks"
    attacker_tasks = output_dir / "webarena_tasks_attacker"
    _copy_browser_tasks_with_port(source_user_tasks, user_tasks, port=port)
    _copy_browser_tasks_with_port(
        source_attacker_tasks,
        attacker_tasks,
        port=port,
    )
    retained_user_ids = _keep_selected_tasks(user_tasks, task_ids)
    retained_attacker_ids = _keep_selected_tasks(
        attacker_tasks,
        set(retained_user_ids),
    )
    if retained_user_ids != retained_attacker_ids:
        raise ValueError(
            f"User/attacker task IDs differ for {attack_name}: "
            f"{retained_user_ids} vs {retained_attacker_ids}"
        )
    _apply_user_intent_prefix(user_tasks, user_intent_prefix)

    snapshot_dir = output_dir / "static_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_metadata, snapshot_dir / "page_metadata.json")
    return static_dir, user_tasks, attacker_tasks, retained_user_ids


def _trace_has_tool_call(trace_path: Path) -> bool:
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        messages = payload if isinstance(payload, list) else [payload]
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            tool_calls = message.get("tool_calls")
            if (
                message.get("role") == "assistant"
                and isinstance(tool_calls, list)
                and bool(tool_calls)
            ):
                return True
    return False


def validate_agent_outputs(
    *,
    log_dir: Path,
    task_ids: Sequence[int],
    trials: int,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = log_dir / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing agent manifest: {manifest_path}")
        manifest: Mapping[str, Any] = {}
    else:
        payload = _load_json(manifest_path)
        manifest = payload if isinstance(payload, Mapping) else {}
        if not isinstance(payload, Mapping):
            errors.append(f"invalid agent manifest object: {manifest_path}")

    statuses: list[str] = []
    for task in manifest.get("tasks", []) if isinstance(manifest, Mapping) else []:
        if not isinstance(task, Mapping):
            continue
        task_trials = (
            task.get("trials", [])
            if isinstance(task.get("trials"), list)
            else []
        )
        for trial in task_trials:
            if isinstance(trial, Mapping):
                statuses.append(str(trial.get("status") or ""))
    expected_count = len(task_ids) * trials
    if len(statuses) != expected_count:
        errors.append(
            f"manifest has {len(statuses)} trial records; expected {expected_count}"
        )
    bad_statuses = [status for status in statuses if status != "completed"]
    if bad_statuses:
        errors.append(f"non-completed trial statuses: {bad_statuses}")

    non_empty_trace_count = 0
    tool_call_trace_count = 0
    for task_id in task_ids:
        for trial in range(1, trials + 1):
            name = (
                f"trace_{task_id}_r{trial:02d}.jsonl"
                if trials > 1
                else f"trace_{task_id}.jsonl"
            )
            trace_path = log_dir / name
            if not trace_path.is_file() or trace_path.stat().st_size == 0:
                errors.append(f"missing or empty trace: {trace_path}")
                continue
            non_empty_trace_count += 1
            try:
                has_tool_call = _trace_has_tool_call(trace_path)
                if has_tool_call:
                    tool_call_trace_count += 1
                else:
                    errors.append(f"trace contains no assistant tool call: {trace_path}")
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"invalid trace {trace_path}: {exc}")

    api_error_marker = "Error occurred while requesting OpenAI API"
    for worker_log in sorted((log_dir / "worker_logs").glob("*.log")):
        worker_output = worker_log.read_text(
            encoding="utf-8",
            errors="replace",
        )
        if api_error_marker in worker_output:
            errors.append(f"model API error recorded in {worker_log}")

    return {
        "passed": not errors,
        "errors": errors,
        "expected_trace_count": expected_count,
        "trace_count": non_empty_trace_count,
        "tool_call_trace_count": tool_call_trace_count,
    }


def assess_compromise_checkpoint(
    *,
    run_root: Path,
    attack_names: Sequence[str],
) -> dict[str, Any]:
    """Assess all three trials before deciding whether to stop the batch."""

    per_sample: list[dict[str, Any]] = []
    attacker_passed_tasks = 0.0
    attacker_total_tasks = 0.0
    intermediate_compromised_trials = 0
    intermediate_total_trials = 0
    user_passed_tasks = 0.0
    user_total_tasks = 0.0

    for attack_name in attack_names:
        sample_dir = run_root / attack_name
        metrics_path = sample_dir / "metrics_post_injection.json"
        all_trials_path = (
            sample_dir
            / "agent_logs_post_injection/reasoning_eval_all_trials/trial_results.json"
        )
        protocol_path = sample_dir / "protocol.json"
        for required in (metrics_path, all_trials_path, protocol_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Early-stop checkpoint is missing required output: {required}"
                )

        metrics = _load_json(metrics_path)
        all_trials = _load_json(all_trials_path)
        protocol = _load_json(protocol_path)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Invalid metrics object: {metrics_path}")
        if not isinstance(all_trials, Mapping):
            raise ValueError(f"Invalid all-trial reasoning object: {all_trials_path}")
        if not isinstance(protocol, Mapping):
            raise ValueError(f"Invalid protocol object: {protocol_path}")

        attacker = metrics.get("attacker_utility")
        user = metrics.get("user_utility")
        counts = all_trials.get("counts")
        trials = all_trials.get("trials")
        if not isinstance(attacker, Mapping) or not isinstance(user, Mapping):
            raise ValueError(f"Missing utility metrics: {metrics_path}")
        if not isinstance(counts, Mapping) or not isinstance(trials, list):
            raise ValueError(f"Missing all-trial reasoning counts: {all_trials_path}")

        task_ids = protocol.get("task_ids")
        trial_count = int(protocol.get("trials") or 0)
        if not isinstance(task_ids, list):
            raise ValueError(f"Missing task IDs in protocol: {protocol_path}")
        expected_trials = len(task_ids) * trial_count
        if len(trials) != expected_trials:
            raise ValueError(
                f"{attack_name} has {len(trials)} reasoning trial results; "
                f"expected {expected_trials}"
            )

        sample_attacker_passed = float(attacker.get("total_scores") or 0.0)
        sample_attacker_total = float(attacker.get("cnt_tasks") or 0.0)
        sample_user_passed = float(user.get("total_scores") or 0.0)
        sample_user_total = float(user.get("cnt_tasks") or 0.0)
        sample_intermediate_compromised = int(counts.get("compromised") or 0)

        attacker_passed_tasks += sample_attacker_passed
        attacker_total_tasks += sample_attacker_total
        user_passed_tasks += sample_user_passed
        user_total_tasks += sample_user_total
        intermediate_compromised_trials += sample_intermediate_compromised
        intermediate_total_trials += len(trials)
        per_sample.append(
            {
                "sample_id": attack_name,
                "attacker_passed_tasks": sample_attacker_passed,
                "attacker_total_tasks": sample_attacker_total,
                "intermediate_compromised_trials": (
                    sample_intermediate_compromised
                ),
                "intermediate_total_trials": len(trials),
                "user_passed_tasks": sample_user_passed,
                "user_total_tasks": sample_user_total,
                "metrics_path": metrics_path.as_posix(),
                "all_trials_path": all_trials_path.as_posix(),
            }
        )

    zero_compromise = (
        attacker_passed_tasks == 0.0
        and intermediate_compromised_trials == 0
    )
    return {
        "sample_definition": "one GitLab attack environment",
        "sample_count": len(attack_names),
        "samples": list(attack_names),
        "attacker_passed_tasks": attacker_passed_tasks,
        "attacker_total_tasks": attacker_total_tasks,
        "intermediate_compromised_trials": intermediate_compromised_trials,
        "intermediate_total_trials": intermediate_total_trials,
        "user_passed_tasks": user_passed_tasks,
        "user_total_tasks": user_total_tasks,
        "completely_uncompromised": zero_compromise,
        "decision": (
            "stop_zero_compromise"
            if zero_compromise
            else "continue_nonzero_compromise"
        ),
        "per_sample": per_sample,
    }


def _protocol_payload(
    *,
    args: argparse.Namespace,
    attack_name: str,
    source_metadata: Path,
    task_ids: Sequence[int],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment": "model_ablation",
        "system": "web",
        "suite": "gitlab",
        "condition": "full",
        "run_variant": args.run_variant,
        "attack": attack_name,
        "model": args.model,
        "check_model": args.check_model,
        "temperature": args.temperature,
        "trials": args.trials,
        "seed_requested": args.seed,
        "seed_guarantee": "provider-dependent",
        "cache": {"no-cache": True, "no-store": True},
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "agent_api_base_url": args.agent_api_base_url or "",
        "agent_api_key_env": args.agent_api_key_env or "",
        "agent_reasoning_effort": args.agent_reasoning_effort or "",
        "user_intent_prefix": args.user_intent_prefix,
        "max_actions": args.max_actions,
        "max_observations": args.max_observations,
        "max_parallel": args.max_parallel,
        "defense_mode": args.defense_mode,
        "check_temperature": args.check_temperature,
        "check_max_tokens": args.check_max_tokens,
        "task_ids": list(task_ids),
        "source_metadata": source_metadata.as_posix(),
        "source_metadata_sha256": _sha256(source_metadata),
        "output_dir": output_dir.as_posix(),
    }


def build_agent_env(args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, str]:
    agent_env = dict(env)
    if args.agent_api_base_url:
        agent_env["OPENAI_BASE_URL"] = args.agent_api_base_url
        agent_env["OPENAI_API_BASE"] = args.agent_api_base_url
    if args.agent_api_key_env:
        api_key = os.environ.get(args.agent_api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing agent API key environment variable: {args.agent_api_key_env}"
            )
        agent_env["OPENAI_API_KEY"] = api_key
    if args.agent_reasoning_effort:
        agent_env["WEBTRAP_REASONING_EFFORT"] = args.agent_reasoning_effort
    return agent_env


def run_attack(
    *,
    attack_name: str,
    source_metadata: Path,
    source_attacker_tasks: Path,
    args: argparse.Namespace,
    run_root: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    output_dir = run_root / attack_name
    metrics_path = output_dir / "metrics_post_injection.json"
    status_path = output_dir / "run_status.json"
    if args.skip_done and metrics_path.is_file():
        return {
            "attack": attack_name,
            "status": "skipped_existing_metrics",
            "metrics_path": metrics_path.as_posix(),
        }

    port = _find_free_port(args.port)
    static_dir, user_tasks, attacker_tasks, task_ids = prepare_attack_environment(
        attack_name=attack_name,
        source_metadata=source_metadata,
        source_user_tasks=args.user_tasks,
        source_attacker_tasks=source_attacker_tasks,
        shared_static=args.shared_static,
        output_dir=output_dir,
        allowed_output_root=args.output_root,
        port=port,
        task_ids=set(args.task_ids) if args.task_ids else None,
        reset_output=not args.keep_existing_output,
        user_intent_prefix=args.user_intent_prefix,
    )
    protocol = _protocol_payload(
        args=args,
        attack_name=attack_name,
        source_metadata=source_metadata,
        task_ids=task_ids,
        output_dir=output_dir,
    )
    _write_json(output_dir / "protocol.json", protocol)
    if args.prepare_only:
        result = {
            **protocol,
            "status": "prepared",
            "static_dir": static_dir.as_posix(),
        }
        _write_json(status_path, result)
        return result

    server_log_handle = (output_dir / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [
            args.python_bin,
            "-m",
            "http.server",
            str(port),
            "--directory",
            str(static_dir),
        ],
        cwd=REPO_ROOT,
        stdout=server_log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    command_results: list[CommandResult] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    validation: dict[str, Any] = {"passed": False, "errors": ["not run"]}
    status = "failed"
    error = ""
    try:
        _wait_for_url(
            f"http://127.0.0.1:{port}/index.html",
            timeout_seconds=10.0,
        )
        log_dir = output_dir / "agent_logs_post_injection"
        agent_cmd = [
            args.python_bin,
            "web/exp/05_evaluate/run_agent.py",
            "--tasks",
            str(user_tasks),
            "--result-dir",
            str(log_dir),
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--seed",
            str(args.seed),
            "--defense-mode",
            args.defense_mode,
            "--dataset",
            DATASET,
            "--max-actions",
            str(args.max_actions),
            "--max-observations",
            str(args.max_observations),
            "--trials",
            str(args.trials),
            "--max-parallel",
            str(args.max_parallel),
            "--url-timeout",
            "5",
            "--experiment-root",
            str(output_dir),
        ]
        command_results.append(
            _run_command(
                name="web_agent",
                command=agent_cmd,
                log_path=output_dir / "run_agent.log",
                env=build_agent_env(args, env),
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.agent_timeout,
            )
        )
        validation = validate_agent_outputs(
            log_dir=log_dir,
            task_ids=task_ids,
            trials=args.trials,
        )
        if not validation["passed"]:
            raise RuntimeError(
                "Web agent output validation failed: "
                + "; ".join(validation["errors"])
            )

        canonical = _copy_canonical_logs(
            task_dir=user_tasks,
            log_dir=log_dir,
            trials=args.trials,
        )
        command_specs = [
            (
                "extract_paths",
                [
                    args.python_bin,
                    "web/exp/05_evaluate/extract_paths.py",
                    "--logs",
                    str(canonical),
                    "--out",
                    str(log_dir / "path_stats.json"),
                    "--format",
                    "gpt_web_tools",
                    "--page-metadata",
                    str(output_dir / "static_snapshot/page_metadata.json"),
                    "--experiment-root",
                    str(output_dir),
                ],
                output_dir / "extract_paths.log",
            ),
            (
                "step_by_step_asr",
                [
                    args.python_bin,
                    "web/exp/05_evaluate/evaluator_step_by_step_reasoning.py",
                    "--log-folder",
                    str(canonical),
                    "--task-folder",
                    str(user_tasks),
                    "--model",
                    args.check_model,
                    "--temperature",
                    str(args.check_temperature),
                    "--max-tokens",
                    str(args.check_max_tokens),
                    "--out",
                    str(output_dir / "step_by_step_asr.json"),
                ],
                output_dir / "step_by_step.log",
            ),
            (
                "reasoning_asr_all_trials",
                [
                    args.python_bin,
                    "web/exp/05_evaluate/supplement_reasoning_eval_all_trials.py",
                    "--log-dir",
                    str(log_dir),
                    "--model",
                    args.check_model,
                    "--temperature",
                    str(args.check_temperature),
                    "--max-tokens",
                    str(args.check_max_tokens),
                ],
                output_dir / "reasoning_all_trials.log",
            ),
            (
                "user_utility",
                [
                    args.python_bin,
                    "web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py",
                    "--log-folder",
                    str(log_dir),
                    "--task-folder",
                    str(user_tasks),
                    "--page-metadata",
                    str(output_dir / "static_snapshot/page_metadata.json"),
                    "--out",
                    str(output_dir / "user_utility.json"),
                ],
                output_dir / "user_utility.log",
            ),
            (
                "attacker_utility",
                [
                    args.python_bin,
                    "web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py",
                    "--log-folder",
                    str(log_dir),
                    "--task-folder",
                    str(attacker_tasks),
                    "--page-metadata",
                    str(output_dir / "static_snapshot/page_metadata.json"),
                    "--out",
                    str(output_dir / "attacker_utility.json"),
                ],
                output_dir / "attacker_utility.log",
            ),
            (
                "summarize_results",
                [
                    args.python_bin,
                    "web/exp/05_evaluate/summarize_results.py",
                    "--experiment-root",
                    str(output_dir),
                    "--step-by-step",
                    str(output_dir / "step_by_step_asr.json"),
                    "--user-utility",
                    str(output_dir / "user_utility.json"),
                    "--attacker-utility",
                    str(output_dir / "attacker_utility.json"),
                    "--out",
                    str(metrics_path),
                ],
                output_dir / "summarize_results.log",
            ),
        ]
        for name, command, log_path in command_specs:
            command_results.append(
                _run_command(
                    name=name,
                    command=command,
                    log_path=log_path,
                    env=env,
                    continue_on_error=args.continue_on_error,
                    timeout_seconds=args.command_timeout,
                )
            )
        status = (
            "completed"
            if metrics_path.is_file()
            and all(result.returncode == 0 for result in command_results)
            else "partial_failed"
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not args.continue_on_error:
            raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)
        server_log_handle.close()
        summary = {
            **protocol,
            "status": status,
            "error": error,
            "agent_output_validation": validation,
            "metrics_path": metrics_path.as_posix(),
            "started_at": started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commands": [
                {
                    "name": result.name,
                    "returncode": result.returncode,
                    "command": result.command,
                    "log_path": result.log_path.as_posix(),
                }
                for result in command_results
            ],
        }
        _write_json(status_path, summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--check-model", default=DEFAULT_CHECK_MODEL)
    parser.add_argument(
        "--python-bin",
        default="wasp/visualwebarena/venv/bin/python",
    )
    parser.add_argument("--source-batch", type=Path, default=DEFAULT_SOURCE_BATCH)
    parser.add_argument("--user-tasks", type=Path, default=DEFAULT_USER_TASKS)
    parser.add_argument(
        "--attacker-tasks-root",
        type=Path,
        default=DEFAULT_ATTACKER_TASKS,
    )
    parser.add_argument(
        "--shared-static",
        type=Path,
        default=DEFAULT_SHARED_STATIC,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-variant",
        default="full",
        help=(
            "Output subdirectory for this run variant. Keep the default for "
            "the standard Full-condition model ablation."
        ),
    )
    parser.add_argument(
        "--user-intent-prefix",
        default="",
        help=(
            "Optional text prepended only to copied user task intents in the "
            "run output directory. Source tasks and attacker tasks are not changed."
        ),
    )
    parser.add_argument("--attacks", nargs="+")
    parser.add_argument("--max-attacks", type=int, default=0)
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--temperature", type=float, default=LOCKED_TEMPERATURE)
    parser.add_argument("--trials", type=int, default=LOCKED_TRIALS)
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--max-actions", type=int, default=LOCKED_MAX_ACTIONS)
    parser.add_argument(
        "--max-observations",
        type=int,
        default=LOCKED_MAX_OBSERVATIONS,
    )
    parser.add_argument("--max-parallel", type=int, default=LOCKED_MAX_PARALLEL)
    parser.add_argument("--defense-mode", default=LOCKED_DEFENSE_MODE)
    parser.add_argument(
        "--check-temperature",
        type=float,
        default=LOCKED_CHECK_TEMPERATURE,
    )
    parser.add_argument(
        "--check-max-tokens",
        type=int,
        default=LOCKED_CHECK_MAX_TOKENS,
    )
    parser.add_argument(
        "--early-stop-after-samples",
        type=int,
        default=0,
        help=(
            "After this many GitLab attack environments, stop when both "
            "best-of-trials attacker success and all-trial ASR-I are zero."
        ),
    )
    parser.add_argument(
        "--agent-api-base-url",
        default="",
        help=(
            "Optional OpenAI-compatible base URL used only for the tested "
            "agent model. Evaluators keep the default checker endpoint."
        ),
    )
    parser.add_argument(
        "--agent-api-key-env",
        default="",
        help=(
            "Environment variable containing the agent endpoint API key. "
            "The key value is never written to protocol files."
        ),
    )
    parser.add_argument(
        "--agent-reasoning-effort",
        default="",
        help="Optional reasoning_effort value forwarded only to agent calls.",
    )
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--agent-timeout", type=int, default=3600)
    parser.add_argument("--command-timeout", type=int, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--keep-existing-output", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.max_actions < 1 or args.max_observations < 1:
        raise ValueError("Browser action/observation caps must be >= 1")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")
    if args.early_stop_after_samples < 0:
        raise ValueError("--early-stop-after-samples must be >= 0")
    model_path_name(args.run_variant)
    if args.agent_api_base_url and not args.agent_api_key_env:
        raise ValueError("--agent-api-base-url requires --agent-api-key-env")
    if args.agent_api_key_env and args.agent_api_key_env not in os.environ:
        raise ValueError(
            f"Missing agent API key environment variable: {args.agent_api_key_env}"
        )
    if not args.user_tasks.is_dir():
        raise FileNotFoundError(f"Missing user tasks: {args.user_tasks}")
    if not args.shared_static.is_dir():
        raise FileNotFoundError(f"Missing shared static site: {args.shared_static}")
    if not args.prepare_only:
        locked_values = {
            "check_model": (args.check_model, DEFAULT_CHECK_MODEL),
            "temperature": (args.temperature, LOCKED_TEMPERATURE),
            "trials": (args.trials, LOCKED_TRIALS),
            "seed": (args.seed, LOCKED_SEED),
            "max_actions": (args.max_actions, LOCKED_MAX_ACTIONS),
            "max_observations": (
                args.max_observations,
                LOCKED_MAX_OBSERVATIONS,
            ),
            "max_parallel": (args.max_parallel, LOCKED_MAX_PARALLEL),
            "defense_mode": (args.defense_mode, LOCKED_DEFENSE_MODE),
            "check_temperature": (
                args.check_temperature,
                LOCKED_CHECK_TEMPERATURE,
            ),
            "check_max_tokens": (
                args.check_max_tokens,
                LOCKED_CHECK_MAX_TOKENS,
            ),
        }
        changed = [
            f"{name} must remain {expected!r}, got {actual!r}"
            for name, (actual, expected) in locked_values.items()
            if actual != expected
        ]
        if changed:
            raise ValueError(
                "Model ablation may only change the tested model: "
                + "; ".join(changed)
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    attacks = discover_gitlab_attacks(
        args.source_batch,
        args.attacker_tasks_root,
    )
    if args.attacks:
        selected = set(args.attacks)
        known = {name for name, _metadata, _tasks in attacks}
        missing = sorted(selected - known)
        if missing:
            raise ValueError(f"Unknown GitLab attacks: {missing}")
        attacks = [row for row in attacks if row[0] in selected]
    if args.max_attacks:
        attacks = attacks[: args.max_attacks]
    if (
        args.early_stop_after_samples
        and args.early_stop_after_samples > len(attacks)
    ):
        raise ValueError(
            "--early-stop-after-samples exceeds the selected GitLab sample count"
        )

    args.output_root = args.output_root.resolve()
    run_root = args.output_root / model_path_name(args.model) / model_path_name(
        args.run_variant
    )
    run_root.mkdir(parents=True, exist_ok=True)
    env = _base_env()
    env["DATASET"] = DATASET
    results: list[dict[str, Any]] = []
    early_stop_checkpoint: dict[str, Any] | None = None
    evaluated_attack_names: list[str] = []
    for index, (attack_name, metadata, attacker_tasks) in enumerate(attacks, start=1):
        print(f"[web-model-ablation] {index}/{len(attacks)} {attack_name}", flush=True)
        try:
            result = run_attack(
                attack_name=attack_name,
                source_metadata=metadata.resolve(),
                source_attacker_tasks=attacker_tasks.resolve(),
                args=args,
                run_root=run_root,
                env=env,
            )
            results.append(result)
            evaluated_attack_names.append(attack_name)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            results.append(
                {
                    "attack": attack_name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (
            not args.prepare_only
            and args.early_stop_after_samples
            and len(evaluated_attack_names) == args.early_stop_after_samples
        ):
            early_stop_checkpoint = assess_compromise_checkpoint(
                run_root=run_root,
                attack_names=evaluated_attack_names,
            )
            _write_json(
                run_root / "early_stop_checkpoint.json",
                early_stop_checkpoint,
            )
            print(
                "[web-model-ablation] checkpoint "
                f"after {len(evaluated_attack_names)} samples: "
                f"{early_stop_checkpoint['decision']}",
                flush=True,
            )
            if early_stop_checkpoint["completely_uncompromised"]:
                break

    _write_json(
        run_root / "run_summary.json",
        {
            "model": args.model,
            "check_model": args.check_model,
            "system": "web",
            "suite": "gitlab",
            "condition": "full",
            "run_variant": args.run_variant,
            "user_intent_prefix": args.user_intent_prefix,
            "prepare_only": args.prepare_only,
            "early_stop_after_samples": args.early_stop_after_samples,
            "early_stop_checkpoint": early_stop_checkpoint,
            "runs": results,
        },
    )
    if not args.prepare_only and any(
        result.get("status") == "completed" for result in results
    ):
        _run_command(
            name="batch_summary",
            command=[
                args.python_bin,
                "web/exp/summarize_batch_runs.py",
                str(run_root),
                "--out-md",
                str(run_root / "batch_summary.md"),
                "--out-json",
                str(run_root / "batch_summary.json"),
            ],
            log_path=run_root / "summarize_batch_runs.log",
            env=env,
            continue_on_error=True,
            timeout_seconds=args.command_timeout,
        )
    print(f"Wrote {run_root / 'run_summary.json'}")


if __name__ == "__main__":
    main()
