#!/usr/bin/env python3
"""Run clean File node-frequency cases without exposing attacker tools."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.exp2 import agent_runner as shared
from file.exp.osr.no_cache import wrap_generate
from file.exp.pipeline_common import read_jsonl, write_json


def _run_one(
    *,
    run_dir: Path,
    case_row: dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    defense_mode: str,
) -> dict[str, Any]:
    return shared.run_case(
        run_dir=run_dir,
        case_row=case_row,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_steps=max_steps,
        num_samples=1,
        defense_mode=defense_mode,
        injec_tools={},
    )


def run_clean_cases(
    *,
    run_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    max_steps: int,
    defense_mode: str,
    max_parallel: int,
    expected_cases: int,
) -> dict[str, object]:
    cases = read_jsonl(run_dir / "attack_cases.jsonl")
    if len(cases) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} clean cases, found {len(cases)}"
        )
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    if any(
        row.get("attacker_instruction")
        or row.get("attacker_tools")
        or row.get("attack_target_logical_path")
        for row in cases
    ):
        raise ValueError("Clean File cases must not contain attacker objectives or tools")

    shared.generate = wrap_generate(shared.generate)
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_case = {
            executor.submit(
                _run_one,
                run_dir=run_dir,
                case_row=case,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_steps=max_steps,
                defense_mode=defense_mode,
            ): str(case["case_id"])
            for case in cases
        }
        for future in as_completed(future_to_case):
            case_id = future_to_case[future]
            summary = future.result()
            summaries.append(summary)
            print(
                f"[file-clean] completed {case_id}: "
                f"status={summary.get('status')} steps={summary.get('steps')}"
            )

    summaries.sort(key=lambda row: str(row.get("case_id") or ""))
    nonempty_trace_count = sum(
        Path(str(row.get("trace_path") or "")).is_file()
        and Path(str(row.get("trace_path") or "")).stat().st_size > 0
        for row in summaries
    )
    aggregate: dict[str, object] = {
        "condition": "clean_no_attack",
        "run_dir": str(run_dir.resolve()),
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "num_samples_per_target": 1,
        "max_parallel": max_parallel,
        "cases": len(summaries),
        "nonempty_traces": nonempty_trace_count,
        "final_answer": sum(
            row.get("status") == "final_answer" for row in summaries
        ),
        "model_errors": sum(
            row.get("status") == "model_error" for row in summaries
        ),
        "attacker_tools_exposed": False,
        "case_statuses": [
            {
                "case_id": row.get("case_id"),
                "status": row.get("status"),
                "steps": row.get("steps"),
                "trace_path": row.get("trace_path"),
            }
            for row in summaries
        ],
    }
    if nonempty_trace_count != expected_cases:
        raise RuntimeError(
            f"Only {nonempty_trace_count}/{expected_cases} File traces are non-empty"
        )
    write_json(run_dir / "logs" / "clean_run_summary.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("Rebuttal/runs/node_frequency_rerun/file"),
    )
    parser.add_argument("--model", default="deepseek-v3-1-terminus")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--defense-mode", default="default_attack")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--expected-cases", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_clean_cases(
        run_dir=args.run_dir,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        defense_mode=args.defense_mode,
        max_parallel=args.max_parallel,
        expected_cases=args.expected_cases,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
