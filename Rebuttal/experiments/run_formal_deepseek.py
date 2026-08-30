#!/usr/bin/env python3
"""Run formal DeepSeek v3.1 Terminal experiments on Rebuttal environments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from Rebuttal.experiments.contracts import (
    BROWSER_MAX_ACTIONS,
    BROWSER_MAX_OBSERVATIONS,
    BROWSER_CHECK_MAX_TOKENS,
    BROWSER_MAX_PARALLEL,
    BROWSER_SEED,
    BROWSER_TEMPERATURE,
    BROWSER_TRIALS,
    CHECK_MODEL_ID,
    CHECK_TEMPERATURE,
    DEFENSE_MODE,
    FILE_MAX_STEPS,
    FILE_NUM_SAMPLES,
    FILE_AGENT_MAX_TOKENS,
    FILE_ASR_MAX_TOKENS,
    FORMAL_VARIANTS,
    MODEL_ID,
    browser_budgets_for_variant,
    environment_fingerprint,
    file_max_steps_for_variant,
    runtime_contract_errors,
    validate_environment_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = "visualwebarena"


@dataclass(frozen=True)
class CommandResult:
    name: str
    returncode: int
    command: list[str]
    log_path: Path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OPENAI_API_BASE", "http://localhost:4000/v1")
    env.setdefault("OPENAI_BASE_URL", "http://localhost:4000/v1")
    env.setdefault(
        "OPENAI_API_KEY",
        env.get("LITELLM_MASTER_KEY", "sk-yaojia-get-ccfa"),
    )
    env.setdefault("DATASET", DATASET)
    env.setdefault("NO_PROXY", "127.0.0.1,localhost,::1,192.168.11.218")
    env.setdefault("no_proxy", "127.0.0.1,localhost,::1,192.168.11.218")
    env.setdefault("REDDIT", "http://127.0.0.1:8001")
    env.setdefault("SHOPPING", "http://127.0.0.1:8002")
    env.setdefault("GITLAB", "http://127.0.0.1:8003")
    env.setdefault("WIKIPEDIA", "http://127.0.0.1:8004")
    env.setdefault("HOMEPAGE", "http://127.0.0.1:8005")
    env.setdefault("CLASSIFIEDS", "http://127.0.0.1:8006")
    env.setdefault("CLASSIFIEDS_RESET_TOKEN", "dummy")
    env.setdefault("REDDIT_RESET_URL", "http://127.0.0.1:8001/reset")
    return env


def _run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    env: Mapping[str, str],
    continue_on_error: bool,
    timeout_seconds: int | None = None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            handle.write(f"\n[TIMEOUT] {name} exceeded {timeout_seconds} seconds; terminating process group.\n")
            handle.flush()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait(timeout=10)
            returncode = 124
    result = CommandResult(
        name=name,
        returncode=int(returncode),
        command=command,
        log_path=log_path,
    )
    if result.returncode != 0 and not continue_on_error:
        raise RuntimeError(
            f"{name} failed with rc={result.returncode}; see {log_path}"
        )
    return result


def _find_free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free local port found from {start}.")


def _rewrite_start_url(url: str, *, port: int) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return url
    netloc = f"127.0.0.1:{port}"
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))


def _copy_browser_tasks_with_port(src: Path, dst: Path, *, port: int) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for path in sorted(dst.glob("*.json")):
        payload = _load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("start_url"), str):
            payload["start_url"] = _rewrite_start_url(str(payload["start_url"]), port=port)
            _write_json(path, payload)


def _wait_for_url(url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=2.0):  # noqa: S310 - local benchmark URL
                return
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def _copy_canonical_logs(*, task_dir: Path, log_dir: Path, trials: int) -> Path:
    canonical = log_dir / "canonical_logs"
    if canonical.exists():
        shutil.rmtree(canonical)
    canonical.mkdir(parents=True, exist_ok=True)

    for task_json in sorted(task_dir.glob("*.json")):
        if not task_json.stem.isdigit():
            continue
        task_id = task_json.stem
        candidates: list[Path] = []
        if trials > 1:
            preferred = log_dir / f"trace_{task_id}_r01.jsonl"
            if preferred.is_file():
                candidates.append(preferred)
            candidates.extend(
                path
                for path in sorted(log_dir.glob(f"trace_{task_id}_r*.jsonl"))
                if path != preferred
            )
        plain = log_dir / f"trace_{task_id}.jsonl"
        if plain.is_file():
            candidates.append(plain)

        picked = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
        if picked is None:
            picked = next((path for path in candidates if path.is_file()), None)
        if picked is not None:
            shutil.copy2(picked, canonical / f"trace_{task_id}.jsonl")
    return canonical


def _non_empty_trace_count(log_dir: Path) -> int:
    return sum(
        1
        for path in log_dir.glob("trace_*.jsonl")
        if path.is_file() and path.stat().st_size > 0
    )


def _browser_rows(manifest_path: Path, variants: set[str]) -> list[dict[str, Any]]:
    payload = _load_json(manifest_path)
    rows = payload.get("environments") if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        raise ValueError(f"Invalid Browser manifest: {manifest_path}")
    result = [row for row in rows if isinstance(row, dict) and str(row.get("variant")) in variants]
    return sorted(result, key=lambda row: (str(row.get("variant")), str(row.get("sample_id"))))


def _environment_contract_errors(row: Mapping[str, Any]) -> list[str]:
    if not bool(row.get("ready_for_deepseek_v3_1_terminal", False)):
        return [
            str(
                row.get("model_run_status")
                or row.get("note")
                or "environment is not marked ready"
            )
        ]
    root = Path(str(row.get("environment_root") or ""))
    return validate_environment_contract(root / "environment_manifest.json")


def _is_environment_ready(row: Mapping[str, Any]) -> bool:
    return not _environment_contract_errors(row)


def _blocked_environment_result(system: str, row: Mapping[str, Any]) -> dict[str, Any]:
    errors = _environment_contract_errors(row)
    return {
        "system": system,
        "variant": str(row.get("variant") or ""),
        "sample_id": str(row.get("sample_id") or ""),
        "status": "blocked_not_ready",
        "reason": "; ".join(errors),
        "contract_errors": errors,
        "source_environment_root": str(row.get("environment_root") or ""),
    }


def _runtime_contract_for_args(
    system: str,
    args: argparse.Namespace,
    *,
    variant: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "system": system,
        "variant": variant,
        "model": args.model,
        "check_model": args.check_model,
        "temperature": args.temperature,
        "defense_mode": args.defense_mode,
        "check_temperature": args.check_temperature,
    }
    if system.lower() == "browser":
        max_actions, max_observations = (
            browser_budgets_for_variant(variant)
            if variant is not None
            else (args.max_actions, args.max_observations)
        )
        kwargs.update(
            {
                "trials": args.trials,
                "seed": args.seed,
                "max_actions": max_actions,
                "max_observations": max_observations,
                "max_parallel": args.max_parallel,
                "check_max_tokens": args.check_max_tokens,
            }
        )
    else:
        max_steps = (
            file_max_steps_for_variant(variant)
            if variant is not None
            else args.file_max_steps
        )
        kwargs.update(
            {
                "num_samples": args.num_samples,
                "max_steps": max_steps,
                "agent_max_tokens": args.file_max_tokens,
                "asr_max_tokens": args.file_asr_max_tokens,
            }
        )
    errors = runtime_contract_errors(**kwargs)
    return {"passed": not errors, "errors": errors}


def _require_runtime_contract(system: str, args: argparse.Namespace) -> None:
    contract = _runtime_contract_for_args(system, args)
    if not contract["passed"]:
        raise ValueError(
            f"{system} formal runtime contract failed: "
            + "; ".join(contract["errors"])
        )


def run_browser_env(
    *,
    row: Mapping[str, Any],
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> dict[str, Any]:
    if not _is_environment_ready(row):
        return _blocked_environment_result("Browser", row)

    variant = str(row["variant"])
    max_actions, max_observations = browser_budgets_for_variant(variant)
    sample_id = str(row["sample_id"]).split(":", 1)[-1]
    env_root = Path(str(row["environment_root"])).resolve()
    manifest_path = env_root / "environment_manifest.json"
    source_fingerprint = environment_fingerprint(manifest_path)
    static_dir = env_root / "static"
    page_metadata = Path(str(row["active_environment_file"])).resolve()
    output_dir = (
        REPO_ROOT
        / "Rebuttal/runs/browser_formal/deepseek_v3_1_terminal"
        / variant
        / sample_id
    )
    metrics_path = output_dir / "metrics_post_injection.json"
    status_path = output_dir / "run_status.json"
    if args.skip_done and metrics_path.is_file():
        return {
            "system": "Browser",
            "variant": variant,
            "sample_id": sample_id,
            "status": "skipped_existing_metrics",
            "metrics_path": metrics_path.as_posix(),
        }

    if output_dir.exists() and not args.keep_existing_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    port = _find_free_port(args.port)
    user_tasks = output_dir / "webarena_tasks"
    attacker_tasks = output_dir / "webarena_tasks_attacker"
    _copy_browser_tasks_with_port(Path(str(row["user_tasks"])), user_tasks, port=port)
    _copy_browser_tasks_with_port(Path(str(row["attacker_tasks"])), attacker_tasks, port=port)
    (output_dir / "static_snapshot").mkdir(parents=True, exist_ok=True)
    shutil.copy2(page_metadata, output_dir / "static_snapshot" / "page_metadata.json")

    server_log = output_dir / "server.log"
    server = subprocess.Popen(
        [args.python_bin, "-m", "http.server", str(port), "--directory", str(static_dir)],
        cwd=REPO_ROOT,
        stdout=server_log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    command_results: list[CommandResult] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        _wait_for_url(f"http://127.0.0.1:{port}/index.html", timeout_seconds=10.0)
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
            str(max_actions),
            "--max-observations",
            str(max_observations),
            "--trials",
            str(args.trials),
            "--max-parallel",
            str(args.max_parallel),
            "--url-timeout",
            "5",
            "--experiment-root",
            str(env_root),
        ]
        command_results.append(
            _run_command(
                name="browser_run_agent",
                command=agent_cmd,
                log_path=output_dir / "run_agent.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.browser_agent_timeout,
            )
        )
        if _non_empty_trace_count(log_dir) == 0:
            raise RuntimeError(f"No non-empty Browser traces were produced under {log_dir}")

        canonical = _copy_canonical_logs(task_dir=user_tasks, log_dir=log_dir, trials=args.trials)
        command_results.append(
            _run_command(
                name="browser_extract_paths",
                command=[
                    args.python_bin,
                    "web/exp/05_evaluate/extract_paths.py",
                    "--logs",
                    str(canonical),
                    "--out",
                    str(log_dir / "path_stats.json"),
                    "--format",
                    "gpt_web_tools",
                    "--page-metadata",
                    str(page_metadata),
                    "--experiment-root",
                    str(env_root),
                ],
                log_path=output_dir / "extract_paths.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.command_timeout,
            )
        )
        step_out = output_dir / "step_by_step_asr.json"
        user_out = output_dir / "user_utility.json"
        attacker_out = output_dir / "attacker_utility.json"
        command_results.append(
            _run_command(
                name="browser_step_by_step",
                command=[
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
                    str(step_out),
                ],
                log_path=output_dir / "step_by_step.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.command_timeout,
            )
        )
        command_results.append(
            _run_command(
                name="browser_user_utility",
                command=[
                    args.python_bin,
                    "web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py",
                    "--log-folder",
                    str(log_dir),
                    "--task-folder",
                    str(user_tasks),
                    "--page-metadata",
                    str(page_metadata),
                    "--out",
                    str(user_out),
                ],
                log_path=output_dir / "user_utility.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.command_timeout,
            )
        )
        command_results.append(
            _run_command(
                name="browser_attacker_utility",
                command=[
                    args.python_bin,
                    "web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py",
                    "--log-folder",
                    str(log_dir),
                    "--task-folder",
                    str(attacker_tasks),
                    "--page-metadata",
                    str(page_metadata),
                    "--out",
                    str(attacker_out),
                ],
                log_path=output_dir / "attacker_utility.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.command_timeout,
            )
        )
        command_results.append(
            _run_command(
                name="browser_summarize_results",
                command=[
                    args.python_bin,
                    "web/exp/05_evaluate/summarize_results.py",
                    "--experiment-root",
                    str(output_dir),
                    "--step-by-step",
                    str(step_out),
                    "--user-utility",
                    str(user_out),
                    "--attacker-utility",
                    str(attacker_out),
                    "--out",
                    str(metrics_path),
                ],
                log_path=output_dir / "summarize_results.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.command_timeout,
            )
        )
        status = "completed" if all(result.returncode == 0 for result in command_results) else "partial_failed"
    except Exception as exc:
        status = "failed"
        _write_json(
            status_path,
            {
                "status": status,
                "error": f"{type(exc).__name__}: {exc}",
                "variant": variant,
                "sample_id": sample_id,
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "commands": [result.__dict__ | {"log_path": result.log_path.as_posix()} for result in command_results],
            },
        )
        if not args.continue_on_error:
            raise
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    summary = {
        "status": status,
        "system": "Browser",
        "variant": variant,
        "sample_id": sample_id,
        "model": args.model,
        "check_model": args.check_model,
        "temperature": args.temperature,
        "trials": args.trials,
        "max_actions": max_actions,
        "max_observations": max_observations,
        "max_parallel": args.max_parallel,
        "seed": args.seed,
        "defense_mode": args.defense_mode,
        "check_temperature": args.check_temperature,
        "check_max_tokens": args.check_max_tokens,
        "runtime_contract": _runtime_contract_for_args(
            "Browser", args, variant=variant
        ),
        "source_environment_fingerprint": source_fingerprint,
        "metrics_path": metrics_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "source_environment_root": env_root.as_posix(),
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


def run_browser_batch(args: argparse.Namespace, env: Mapping[str, str]) -> list[dict[str, Any]]:
    _require_runtime_contract("Browser", args)
    variants = {value.strip() for value in args.browser_variants.split(",") if value.strip()}
    rows = _browser_rows(
        REPO_ROOT / "Rebuttal/results/materialized_envs/browser_formal_envs.json",
        variants,
    )
    if args.max_browser_envs:
        rows = rows[: int(args.max_browser_envs)]

    blocked_rows = [row for row in rows if not _is_environment_ready(row)]
    ready_rows = [row for row in rows if _is_environment_ready(row)]
    results: list[dict[str, Any]] = [
        _blocked_environment_result("Browser", row) for row in blocked_rows
    ]
    for row in blocked_rows:
        print(
            f"[browser] blocked_not_ready variant={row.get('variant')}: "
            f"{row.get('model_run_status') or row.get('note')}",
            flush=True,
        )
    for idx, row in enumerate(ready_rows, start=1):
        print(
            f"[browser] {idx}/{len(ready_rows)} variant={row['variant']}",
            flush=True,
        )
        results.append(run_browser_env(row=row, args=args, env=env))

    out_root = REPO_ROOT / "Rebuttal/runs/browser_formal/deepseek_v3_1_terminal"
    _write_json(out_root / "run_summary.json", {"runs": results})
    _write_csv(out_root / "run_summary.csv", results)
    for variant in sorted(variants):
        variant_dir = out_root / variant
        if variant_dir.is_dir():
            _run_command(
                name=f"browser_batch_summary_{variant}",
                command=[
                    args.python_bin,
                    "web/exp/summarize_batch_runs.py",
                    str(variant_dir),
                    "--out-md",
                    str(variant_dir / "batch_summary.md"),
                    "--out-json",
                    str(variant_dir / "batch_summary.json"),
                ],
                log_path=variant_dir / "summarize_batch_runs.log",
                env=env,
                continue_on_error=True,
                timeout_seconds=args.command_timeout,
            )
    return results


def _file_rows(manifest_path: Path, variants: set[str]) -> list[dict[str, Any]]:
    payload = _load_json(manifest_path)
    rows = payload.get("environments") if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        raise ValueError(f"Invalid File manifest: {manifest_path}")
    return [row for row in rows if isinstance(row, dict) and str(row.get("variant")) in variants]


def run_file_dir(
    *,
    row: Mapping[str, Any],
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> dict[str, Any]:
    if not _is_environment_ready(row):
        return _blocked_environment_result("File", row)

    run_dir = Path(str(row["environment_root"])).resolve()
    manifest_path = run_dir / "environment_manifest.json"
    source_fingerprint = environment_fingerprint(manifest_path)
    variant = str(row["variant"])
    max_steps = file_max_steps_for_variant(variant)
    status_path = run_dir / "deepseek_v3_1_terminal" / "run_status.json"
    metrics_path = run_dir / "eval" / "metrics.json"
    if args.skip_done and metrics_path.is_file():
        previous_status = (
            _load_json(status_path)
            if status_path.is_file()
            else {}
        )
        previous_commands = (
            previous_status.get("commands")
            if isinstance(previous_status, Mapping)
            else None
        )
        summary = {
            "system": "File",
            "variant": variant,
            "status": "completed",
            "model": args.model,
            "check_model": args.check_model,
            "temperature": args.temperature,
            "num_samples": args.num_samples,
            "max_steps": max_steps,
            "defense_mode": args.defense_mode,
            "check_temperature": args.check_temperature,
            "agent_max_tokens": args.file_max_tokens,
            "asr_max_tokens": args.file_asr_max_tokens,
            "runtime_contract": _runtime_contract_for_args(
                "File", args, variant=variant
            ),
            "source_environment_fingerprint": source_fingerprint,
            "metrics_path": metrics_path.as_posix(),
            "run_dir": run_dir.as_posix(),
            "started_at": (
                previous_status.get("started_at")
                if isinstance(previous_status, Mapping)
                else time.strftime("%Y-%m-%dT%H:%M:%S%z")
            ),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commands": previous_commands if isinstance(previous_commands, list) else [],
            "reused_existing_metrics": True,
        }
        _write_json(status_path, summary)
        return summary

    if (run_dir / "logs").exists() and not args.keep_existing_output:
        shutil.rmtree(run_dir / "logs")
    if (run_dir / "eval").exists() and not args.keep_existing_output:
        shutil.rmtree(run_dir / "eval")
    (run_dir / "deepseek_v3_1_terminal").mkdir(parents=True, exist_ok=True)

    command_results: list[CommandResult] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        agent_cmd = [
            args.python_bin,
            "file/exp/exp2/agent_runner.py",
            "--run-dir",
            str(run_dir),
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.file_max_tokens),
            "--max-steps",
            str(max_steps),
            "--num-samples",
            str(args.num_samples),
            "--defense-mode",
            args.defense_mode,
        ]
        if args.case_id:
            agent_cmd.extend(["--case-id", args.case_id])
        command_results.append(
            _run_command(
                name="file_agent_runner",
                command=agent_cmd,
                log_path=run_dir / "deepseek_v3_1_terminal" / "agent_runner.log",
                env=env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.file_agent_timeout,
            )
        )
        exp3_env = dict(env)
        exp3_env["PYTHON_BIN"] = args.python_bin
        exp3_env["ASR_MODEL"] = args.check_model
        exp3_env["ASR_TEMPERATURE"] = str(args.check_temperature)
        exp3_env["ASR_MAX_TOKENS"] = str(args.file_asr_max_tokens)
        command_results.append(
            _run_command(
                name="file_exp3",
                command=[
                    "bash",
                    "file/exp/exp3/run_exp3.sh",
                    "--run-dir",
                    str(run_dir),
                ],
                log_path=run_dir / "deepseek_v3_1_terminal" / "exp3.log",
                env=exp3_env,
                continue_on_error=args.continue_on_error,
                timeout_seconds=args.file_eval_timeout,
            )
        )
        status = "completed" if metrics_path.is_file() and all(result.returncode == 0 for result in command_results) else "partial_failed"
    except Exception as exc:
        status = "failed"
        _write_json(
            status_path,
            {
                "status": status,
                "error": f"{type(exc).__name__}: {exc}",
                "variant": variant,
                "run_dir": run_dir.as_posix(),
                "started_at": started_at,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        if not args.continue_on_error:
            raise

    summary = {
        "status": status,
        "system": "File",
        "variant": variant,
        "model": args.model,
        "check_model": args.check_model,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "max_steps": max_steps,
        "defense_mode": args.defense_mode,
        "check_temperature": args.check_temperature,
        "agent_max_tokens": args.file_max_tokens,
        "asr_max_tokens": args.file_asr_max_tokens,
        "runtime_contract": _runtime_contract_for_args(
            "File", args, variant=variant
        ),
        "source_environment_fingerprint": source_fingerprint,
        "metrics_path": metrics_path.as_posix(),
        "run_dir": run_dir.as_posix(),
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


def run_file_batch(args: argparse.Namespace, env: Mapping[str, str]) -> list[dict[str, Any]]:
    _require_runtime_contract("File", args)
    variants = {value.strip() for value in args.file_variants.split(",") if value.strip()}
    rows = _file_rows(
        REPO_ROOT / "Rebuttal/results/materialized_envs/file_formal_envs.json",
        variants,
    )
    if args.max_file_runs:
        rows = rows[: int(args.max_file_runs)]
    blocked_rows = [row for row in rows if not _is_environment_ready(row)]
    ready_rows = [row for row in rows if _is_environment_ready(row)]
    results: list[dict[str, Any]] = [
        _blocked_environment_result("File", row) for row in blocked_rows
    ]
    for row in blocked_rows:
        print(
            f"[file] blocked_not_ready {row.get('variant')}: "
            f"{row.get('model_run_status') or row.get('note')}",
            flush=True,
        )
    for idx, row in enumerate(ready_rows, start=1):
        print(f"[file] {idx}/{len(ready_rows)} {row['variant']}", flush=True)
        results.append(run_file_dir(row=row, args=args, env=env))

    out_root = REPO_ROOT / "Rebuttal/runs/file_formal/deepseek_v3_1_terminal"
    _write_json(out_root / "run_summary.json", {"runs": results})
    _write_csv(out_root / "run_summary.csv", results)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("browser", "file", "all"), default="all")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--check-model", default=CHECK_MODEL_ID)
    parser.add_argument(
        "--python-bin",
        default="wasp/visualwebarena/venv/bin/python",
        help="Python executable with the web/file experiment dependencies.",
    )
    parser.add_argument("--temperature", type=float, default=BROWSER_TEMPERATURE)
    parser.add_argument("--check-temperature", type=float, default=CHECK_TEMPERATURE)
    parser.add_argument(
        "--check-max-tokens", type=int, default=BROWSER_CHECK_MAX_TOKENS
    )
    parser.add_argument("--seed", type=int, default=BROWSER_SEED)
    parser.add_argument("--defense-mode", default=DEFENSE_MODE)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--keep-existing-output", action="store_true")
    parser.add_argument("--command-timeout", type=int, default=3600)

    parser.add_argument(
        "--browser-variants",
        default=",".join(FORMAL_VARIANTS),
    )
    parser.add_argument("--max-browser-envs", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trials", type=int, default=BROWSER_TRIALS)
    parser.add_argument("--max-actions", type=int, default=BROWSER_MAX_ACTIONS)
    parser.add_argument("--max-observations", type=int, default=BROWSER_MAX_OBSERVATIONS)
    parser.add_argument("--max-parallel", type=int, default=BROWSER_MAX_PARALLEL)
    parser.add_argument("--browser-agent-timeout", type=int, default=3600)

    parser.add_argument(
        "--file-variants",
        default=",".join(FORMAL_VARIANTS),
    )
    parser.add_argument("--max-file-runs", type=int, default=0)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--num-samples", type=int, default=FILE_NUM_SAMPLES)
    parser.add_argument("--file-max-steps", type=int, default=FILE_MAX_STEPS)
    parser.add_argument(
        "--file-max-tokens", type=int, default=FILE_AGENT_MAX_TOKENS
    )
    parser.add_argument(
        "--file-asr-max-tokens", type=int, default=FILE_ASR_MAX_TOKENS
    )
    parser.add_argument("--file-agent-timeout", type=int, default=14400)
    parser.add_argument("--file-eval-timeout", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    env = _base_env()
    all_results: list[dict[str, Any]] = []
    if args.target in {"browser", "all"}:
        all_results.extend(run_browser_batch(args, env))
    if args.target in {"file", "all"}:
        all_results.extend(run_file_batch(args, env))
    out_dir = REPO_ROOT / "Rebuttal/results/deepseek_formal"
    _write_json(
        out_dir / "run_summary.json",
        {
            "model": args.model,
            "check_model": args.check_model,
            "target": args.target,
            "results": all_results,
        },
    )
    _write_csv(out_dir / "run_summary.csv", all_results)
    print(f"Wrote {out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
