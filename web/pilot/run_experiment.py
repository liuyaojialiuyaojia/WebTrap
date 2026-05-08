#!/usr/bin/env python3
"""Orchestrate the pilot detour experiment entirely under web/pilot."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import (
    DEFAULT_FORMAL_MAX_ACTIONS,
    DEFAULT_CASES_ROOT,
    DEFAULT_RUNS_ROOT,
    PILOT_ROOT,
    REPO_ROOT,
    load_json,
    minimum_sufficient_max_actions,
    write_json,
)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _run_step(args: list[str]) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--defense-mode", type=str, default="default_attack")
    parser.add_argument("--sampling-trials", type=int, default=16)
    parser.add_argument("--sampling-max-actions", type=int, default=None)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=DEFAULT_FORMAL_MAX_ACTIONS,
        help="Maximum continuation steps for the post-anchor test agent (default: 7).",
    )
    parser.add_argument("--max-observations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root or (DEFAULT_RUNS_ROOT / f"{_timestamp()}_motivation_detour")
    run_root.mkdir(parents=True, exist_ok=True)

    _run_step(
        [
            sys.executable,
            str(PILOT_ROOT / "build_cases.py"),
            "--output-root",
            str(args.cases_root.resolve()),
        ]
    )
    _run_step(
        [
            sys.executable,
            str(PILOT_ROOT / "materialize_sites.py"),
            "--cases-root",
            str(args.cases_root.resolve()),
            "--output-root",
            str(run_root.resolve()),
        ]
    )

    case_payloads = [
        load_json(case_path)
        for case_path in sorted(args.cases_root.glob("*/case.json"))
        if case_path.is_file()
    ]
    if not case_payloads:
        raise RuntimeError(f"No pilot cases were built under {args.cases_root}")

    required_min_max_actions = max(
        minimum_sufficient_max_actions(case_payload=case_payload, condition=condition)
        for case_payload in case_payloads
        for condition in case_payload["conditions"].keys()
    )
    effective_max_actions = int(args.max_actions)
    if effective_max_actions < required_min_max_actions:
        raise ValueError(
            f"--max-actions={effective_max_actions} is smaller than the required minimum "
            f"{required_min_max_actions} for the current pilot cases."
        )
    sampling_max_actions = (
        int(args.sampling_max_actions)
        if args.sampling_max_actions is not None
        else effective_max_actions
    )

    write_json(
        run_root / "run_manifest.json",
        {
            "version": 1,
            "status": "initializing",
            "cases_root": str(args.cases_root.resolve()),
            "run_root": str(run_root.resolve()),
            "model": args.model,
            "temperature": args.temperature,
            "defense_mode": args.defense_mode,
            "sampling_trials": args.sampling_trials,
            "sampling_concurrency": min(max(1, int(args.concurrency)), len(case_payloads)),
            "sampling_max_actions": sampling_max_actions,
            "trials": args.trials,
            "max_actions": effective_max_actions,
            "required_min_max_actions": required_min_max_actions,
            "concurrency": max(1, int(args.concurrency)),
            "port_base": args.port,
            "site_variants": sum(len(case_payload["conditions"]) for case_payload in case_payloads),
        },
    )
    sample_cmd = [
        sys.executable,
        str(PILOT_ROOT / "sample_histories.py"),
        "--run-root",
        str(run_root.resolve()),
        "--model",
        args.model,
        "--trials",
        str(args.sampling_trials),
        "--max-actions",
        str(sampling_max_actions),
        "--max-observations",
        str(args.max_observations),
        "--concurrency",
        str(min(max(1, int(args.concurrency)), len(case_payloads))),
        "--defense-mode",
        args.defense_mode,
        "--port",
        str(args.port),
    ]
    if args.temperature is not None:
        sample_cmd.extend(["--temperature", str(args.temperature)])
    if args.seed_base is not None:
        sample_cmd.extend(["--seed-base", str(args.seed_base)])
    if args.resume:
        sample_cmd.append("--resume")
    _run_step(sample_cmd)

    run_cmd = [
        sys.executable,
        str(PILOT_ROOT / "run_warm_start_trials.py"),
        "--run-root",
        str(run_root.resolve()),
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
        "--trials",
        str(args.trials),
        "--max-actions",
        str(effective_max_actions),
        "--max-observations",
        str(args.max_observations),
        "--concurrency",
        str(max(1, int(args.concurrency))),
        "--defense-mode",
        args.defense_mode,
        "--port",
        str(args.port),
    ]
    if args.seed_base is not None:
        run_cmd.extend(["--seed-base", str(args.seed_base)])
    if args.resume:
        run_cmd.append("--resume")
    _run_step(run_cmd)

    _run_step(
        [
            sys.executable,
            str(PILOT_ROOT / "classify_trials.py"),
            "--run-root",
            str(run_root.resolve()),
        ]
    )
    _run_step(
        [
            sys.executable,
            str(PILOT_ROOT / "summarize_results.py"),
            "--run-root",
            str(run_root.resolve()),
        ]
    )

    write_json(
        run_root / "run_manifest.json",
        {
            "version": 1,
            "status": "completed",
            "cases_root": str(args.cases_root.resolve()),
            "run_root": str(run_root.resolve()),
            "model": args.model,
            "temperature": args.temperature,
            "defense_mode": args.defense_mode,
            "sampling_trials": args.sampling_trials,
            "sampling_concurrency": min(max(1, int(args.concurrency)), len(case_payloads)),
            "sampling_max_actions": sampling_max_actions,
            "trials": args.trials,
            "max_actions": effective_max_actions,
            "required_min_max_actions": required_min_max_actions,
            "concurrency": max(1, int(args.concurrency)),
            "port_base": args.port,
            "site_variants": sum(len(case_payload["conditions"]) for case_payload in case_payloads),
            "outputs": {
                "classification_summary": str((run_root / "classification_summary.json").resolve()),
                "figure1_table_json": str((run_root / "figure1_table.json").resolve()),
                "figure1_table_csv": str((run_root / "figure1_table.csv").resolve()),
            },
        },
    )


if __name__ == "__main__":
    main()
