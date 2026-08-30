#!/usr/bin/env python3
"""Helper script that iterates over WebArena tasks and invokes the tool-calling agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

WEB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = WEB_ROOT / "exp" / "02_nested_builder"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiment_paths import resolve_experiment_root, resolve_path_within
from experiment_defenses import ALLOWED_DEFENSE_MODES


def _discover_tasks(task_dir: Path) -> List[Path]:
    tasks = []
    for path in task_dir.glob("*.json"):
        if not path.is_file():
            continue
        stem = path.stem
        if stem.isdigit():
            tasks.append(path)
    return sorted(tasks, key=lambda p: int(p.stem))


def _build_command(
    task_path: Path,
    model: str,
    temperature: float | None,
    seed: int | None,
    defense_mode: str,
    max_actions: int,
    max_observations: int,
    log_path: Path,
) -> List[str]:
    command = [
        sys.executable,
        "web/exp/05_evaluate/tool_calling_agent_defended.py",
        "--webarena_config_path",
        str(task_path),
        "--model",
        model,
    ]
    if temperature is not None:
        command.extend(["--temperature", str(float(temperature))])
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    command.extend(["--defense-mode", defense_mode])
    command.extend(
        [
            "--trace-log-filepath",
            str(log_path),
            "--max_actions",
            str(max_actions),
            "--max_observations_to_keep",
            str(max_observations),
        ]
    )
    return command


def _load_task_payload(task_path: Path) -> Dict[str, object]:
    try:
        return json.loads(task_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse task file {task_path}: {exc}") from exc


def _collect_start_urls(task_paths: List[Path]) -> List[str]:
    start_urls: List[str] = []
    for path in task_paths:
        payload = _load_task_payload(path)
        url = payload.get("start_url") if isinstance(payload, dict) else None
        if isinstance(url, str) and url:
            start_urls.append(url)
    return sorted({url for url in start_urls})


def _check_start_urls(start_urls: List[str], timeout: float) -> None:
    for url in start_urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=timeout):  # noqa: S310 - local URLs allowed
                pass
        except URLError as exc:  # pragma: no cover - network failures are surfaced
            raise ConnectionError(f"Unable to reach start URL {url}: {exc}") from exc


def _log_filename(task_id: int, trial_index: int, total_trials: int) -> str:
    if total_trials > 1:
        return f"trace_{task_id}_r{trial_index:02d}.jsonl"
    return f"trace_{task_id}.jsonl"


def _worker_log_filename(task_id: int, trial_index: int, total_trials: int) -> str:
    if total_trials > 1:
        return f"task_{task_id}_r{trial_index:02d}.log"
    return f"task_{task_id}.log"


@dataclass
class PlannedRun:
    task_id: int
    trial_index: int
    command: List[str]
    display_cmd: str
    run_record: Dict[str, object]
    worker_log_path: Path
    sleep_after: float = 0.0
    process: subprocess.Popen[object] | None = None
    log_handle: object | None = None
    cancel_requested: bool = False
    cancel_deadline: float | None = None


def _execute_planned_runs(
    *,
    planned_runs: List[PlannedRun],
    max_parallel: int,
    env: Dict[str, str],
    repo_root: Path,
) -> tuple[int, List[str]] | None:
    queue = list(planned_runs)
    running: Dict[int, PlannedRun] = {}
    first_failed_run: PlannedRun | None = None
    first_failed_returncode = 0

    try:
        while queue or running:
            while first_failed_run is None and queue and len(running) < max_parallel:
                run = queue.pop(0)
                run.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
                run.log_handle = open(run.worker_log_path, "w", encoding="utf-8")
                run.run_record["status"] = "running"
                run.run_record["started_at"] = datetime.now().isoformat()
                run.process = subprocess.Popen(
                    run.command,
                    env=env,
                    cwd=repo_root,
                    stdout=run.log_handle,
                    stderr=subprocess.STDOUT,
                )
                running[int(run.process.pid)] = run
                print(
                    f"[run_agent] Started task {run.task_id} trial {run.trial_index}"
                    f" (pid={run.process.pid})"
                )

            if not running:
                break

            made_progress = False
            now = time.monotonic()
            for pid, run in list(running.items()):
                assert run.process is not None
                returncode = run.process.poll()
                if returncode is None:
                    if (
                        run.cancel_requested
                        and run.cancel_deadline is not None
                        and now > run.cancel_deadline
                    ):
                        run.process.kill()
                        run.cancel_deadline = None
                    continue

                made_progress = True
                running.pop(pid, None)
                if run.log_handle is not None:
                    run.log_handle.close()
                    run.log_handle = None
                run.run_record["finished_at"] = datetime.now().isoformat()
                run.run_record["returncode"] = int(returncode)

                if returncode == 0:
                    run.run_record["status"] = "completed"
                    print(
                        f"[run_agent] Completed task {run.task_id} trial {run.trial_index}"
                    )
                    if (
                        first_failed_run is None
                        and max_parallel == 1
                        and run.sleep_after > 0.0
                    ):
                        time.sleep(run.sleep_after)
                    continue

                if first_failed_run is None and not run.cancel_requested:
                    first_failed_run = run
                    first_failed_returncode = int(returncode)
                    run.run_record["status"] = "failed"
                    print(
                        f"[run_agent] FAILED task {run.task_id} trial {run.trial_index}"
                        f" (rc={returncode}); fail-fast cancelling {len(running)} running job(s)."
                    )
                    for other in running.values():
                        if other.cancel_requested:
                            continue
                        other.cancel_requested = True
                        other.run_record["status"] = "cancelling"
                        other.run_record["cancel_reason"] = "fail_fast"
                        other.cancel_deadline = time.monotonic() + 3.0
                        assert other.process is not None
                        other.process.terminate()
                    continue

                if run.cancel_requested:
                    run.run_record["status"] = "cancelled_running"
                    run.run_record["cancel_reason"] = "fail_fast"
                    print(
                        f"[run_agent] Cancelled running task {run.task_id} trial {run.trial_index}"
                        f" (rc={returncode})"
                    )
                else:
                    run.run_record["status"] = "failed_after_fail_fast"
                    print(
                        f"[run_agent] Additional failure task {run.task_id} trial {run.trial_index}"
                        f" (rc={returncode}) after fail-fast trigger."
                    )

            if not made_progress:
                time.sleep(0.1)

        if first_failed_run is not None:
            for run in queue:
                run.run_record["status"] = "cancelled_not_started"
                run.run_record["cancel_reason"] = "fail_fast"
            return first_failed_returncode, first_failed_run.command

        return None
    except KeyboardInterrupt:
        for run in queue:
            run.run_record["status"] = "cancelled_not_started"
            run.run_record["cancel_reason"] = "keyboard_interrupt"
        for run in running.values():
            run.cancel_requested = True
            run.run_record["status"] = "cancelled_running"
            run.run_record["cancel_reason"] = "keyboard_interrupt"
            if run.process is not None and run.process.poll() is None:
                run.process.terminate()
                try:
                    run.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    run.process.kill()
            if run.log_handle is not None:
                run.log_handle.close()
                run.log_handle = None
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=None,
        help="Base directory for experiment artefacts (default: web/runs/trap_site_mvp).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        help="Optional experiment identifier appended under --experiment-root.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Directory containing WebArena task JSON files. Defaults to experiment_root/webarena_tasks.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory to store agent traces and logs. Defaults to experiment_root/agent_logs_pre_injection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name passed through to the agent runner.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature passed to the agent (default: use server-side default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional base seed for deterministic sampling. When --trials > 1, each trial "
            "uses seed + (trial_index - 1)."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="visualwebarena",
        help="DATASET environment value required by VisualWebArena.",
    )
    parser.add_argument(
        "--defense-mode",
        type=str,
        choices=ALLOWED_DEFENSE_MODES,
        default="default_attack",
        help="Defense mode for agent prompts.",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=20,
        help="Maximum number of actions the agent may perform per task.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=3,
        help="Maximum number of tool-call observations retained in the agent context.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of repeated runs per task for robustness checks.",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="Optional list of task IDs to execute (others will be skipped).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose log files already exist.",
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="Seconds to sleep between repeated trials of the same task.",
    )
    parser.add_argument(
        "--skip-url-check",
        action="store_true",
        help="Disable reachability check for task start URLs.",
    )
    parser.add_argument(
        "--url-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for the start URL reachability probe.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "Maximum number of task-trial runs executed in parallel within one attack"
            " environment. Any failed run triggers fail-fast cancellation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.tasks = resolve_path_within(args.tasks, root=base_root, relative="webarena_tasks")
    args.result_dir = resolve_path_within(args.result_dir, root=base_root, relative="agent_logs_pre_injection")

    task_dir = args.tasks
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    task_paths = _discover_tasks(task_dir)
    if args.task_ids:
        selected = {int(task_id) for task_id in args.task_ids}
        task_paths = [path for path in task_paths if int(path.stem) in selected]
    if not task_paths:
        print(f"[run_agent] No tasks discovered in {task_dir}")
        return

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")
    if args.max_observations < 1:
        raise ValueError("--max-observations must be >= 1")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    if not args.skip_url_check:
        start_urls = _collect_start_urls(task_paths)
        if start_urls:
            print(f"[run_agent] Probing reachability for {len(start_urls)} start URLs…")
            _check_start_urls(start_urls, timeout=args.url_timeout)

    env = os.environ.copy()
    env.setdefault("DATASET", args.dataset)

    manifest: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(),
        "args": {
            "tasks": str(args.tasks),
            "result_dir": str(args.result_dir),
            "model": args.model,
            "dataset": args.dataset,
            "defense_mode": args.defense_mode,
            "max_actions": args.max_actions,
            "max_observations": args.max_observations,
            "trials": args.trials,
            "resume": args.resume,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_parallel": args.max_parallel,
            "sleep_between": args.sleep_between,
        },
        "tasks": [],
    }

    repo_root = WORKSPACE_ROOT
    worker_log_dir = args.result_dir / "worker_logs"
    planned_runs: List[PlannedRun] = []

    if args.sleep_between > 0 and args.max_parallel > 1:
        print(
            "[run_agent] Warning: --sleep-between is ignored when --max-parallel > 1."
        )

    for task_path in task_paths:
        task_id = int(task_path.stem)
        task_entry: Dict[str, object] = {
            "task_id": task_id,
            "task_path": str(task_path),
            "trials": [],
        }

        for trial_index in range(1, args.trials + 1):
            trial_seed = None
            if args.seed is not None:
                trial_seed = int(args.seed) + (trial_index - 1)
            log_filename = _log_filename(task_id, trial_index, args.trials)
            log_path = args.result_dir / log_filename
            command = _build_command(
                task_path,
                args.model,
                args.temperature,
                trial_seed,
                args.defense_mode,
                args.max_actions,
                args.max_observations,
                log_path,
            )
            display_cmd = " ".join(command)
            run_record = {
                "trial": trial_index,
                "log": str(log_path),
                "status": "pending",
                "command": display_cmd,
            }

            if args.resume and log_path.exists():
                run_record["status"] = "skipped_existing"
                print(f"[run_agent] Skipping task {task_id} trial {trial_index} (log exists).")
                task_entry["trials"].append(run_record)
                continue

            if args.dry_run:
                run_record["status"] = "dry_run"
                print(
                    f"[run_agent] Dry run task {task_id} trial {trial_index}: {display_cmd}"
                )
                task_entry["trials"].append(run_record)
                continue

            worker_log_path = worker_log_dir / _worker_log_filename(
                task_id, trial_index, args.trials
            )
            run_record["worker_log"] = str(worker_log_path)
            task_entry["trials"].append(run_record)
            planned_runs.append(
                PlannedRun(
                    task_id=task_id,
                    trial_index=trial_index,
                    command=command,
                    display_cmd=display_cmd,
                    run_record=run_record,
                    worker_log_path=worker_log_path,
                    sleep_after=(
                        float(args.sleep_between) if trial_index != args.trials else 0.0
                    ),
                )
            )

        manifest["tasks"].append(task_entry)

    failure: tuple[int, List[str]] | None = None
    if not args.dry_run and planned_runs:
        failure = _execute_planned_runs(
            planned_runs=planned_runs,
            max_parallel=int(args.max_parallel),
            env=env,
            repo_root=repo_root,
        )

    args.result_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if failure is not None:
        returncode, command = failure
        raise subprocess.CalledProcessError(returncode, command)


if __name__ == "__main__":
    main()
