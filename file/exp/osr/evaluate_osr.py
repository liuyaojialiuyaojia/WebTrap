#!/usr/bin/env python3
"""Summarize file OSR user-task success without using attack-side sample selection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.pipeline_common import read_jsonl, to_posix, write_json, write_jsonl


def _safe_div(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _is_valid_status(status: str) -> bool:
    value = str(status or "").strip()
    return value not in {"model_error", "missing", ""}


def _load_json_maybe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = len(rows)
    valid_cases = sum(1 for row in rows if bool(row.get("valid")))
    success_total = sum(1 for row in rows if bool(row.get("user_goal_success")))
    success_valid = sum(
        1 for row in rows if bool(row.get("valid")) and bool(row.get("user_goal_success"))
    )
    status_breakdown: dict[str, int] = defaultdict(int)
    for row in rows:
        status_breakdown[str(row.get("run_status") or "missing")] += 1
    return {
        "total_cases": total_cases,
        "valid_cases": valid_cases,
        "user_task_success_rate": _safe_div(success_total, total_cases),
        "user_task_success_rate_valid": _safe_div(success_valid, valid_cases),
        "counts": {
            "user_task_success_total": success_total,
            "user_task_success_valid": success_valid,
        },
        "status_breakdown": dict(sorted(status_breakdown.items())),
    }


def evaluate_osr(run_dir: Path, *, case_id: Optional[str] = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    attack_cases = read_jsonl(run_dir / "attack_cases.jsonl")
    if case_id:
        attack_cases = [row for row in attack_cases if str(row.get("case_id") or "") == case_id]
    case_ids = [str(row.get("case_id") or "").strip() for row in attack_cases if row.get("case_id")]

    per_case_input = {
        str(row.get("case_id") or "").strip(): row
        for row in read_jsonl(run_dir / "eval" / "user_goal_per_case.jsonl")
        if str(row.get("case_id") or "").strip()
    }
    per_case_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        base_row = dict(per_case_input.get(case_id, {}))
        run_summary = _load_json_maybe(run_dir / "logs" / f"run_{case_id}.json")
        run_status = str(run_summary.get("status") or "missing")
        row = {
            "case_id": case_id,
            "run_status": run_status,
            "valid": _is_valid_status(run_status),
            "user_goal_success": bool(base_row.get("user_goal_success")),
            "submit_kind": base_row.get("submit_kind"),
            "submit_step": base_row.get("submit_step"),
            "last_read_file_path_at_submit": base_row.get("last_read_file_path_at_submit"),
            "last_read_file_step_at_submit": base_row.get("last_read_file_step_at_submit"),
            "user_target_logical_path": base_row.get("user_target_logical_path"),
            "user_target_virtual_path": base_row.get("user_target_virtual_path"),
            "trace_path": base_row.get("trace_path"),
        }
        per_case_rows.append(row)

    per_sample_input = read_jsonl(run_dir / "eval" / "user_goal_per_sample.jsonl")
    sample_rows_by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sample_indices_by_case: dict[str, set[int]] = defaultdict(set)
    for row in per_sample_input:
        cid = str(row.get("case_id") or "").strip()
        if not cid:
            continue
        try:
            sample_idx = int(row.get("sample_idx"))
        except Exception:
            continue
        prepared = {
            "case_id": cid,
            "sample_idx": sample_idx,
            "run_status": str(row.get("run_status") or "missing"),
            "valid": _is_valid_status(str(row.get("run_status") or "missing")),
            "user_goal_success": bool(row.get("user_goal_success")),
            "submit_kind": row.get("submit_kind"),
            "submit_step": row.get("submit_step"),
            "last_read_file_path_at_submit": row.get("last_read_file_path_at_submit"),
            "last_read_file_step_at_submit": row.get("last_read_file_step_at_submit"),
            "user_target_logical_path": row.get("user_target_logical_path"),
            "user_target_virtual_path": row.get("user_target_virtual_path"),
            "trace_path": row.get("trace_path"),
            "run_summary_path": row.get("run_summary_path"),
        }
        sample_rows_by_idx[sample_idx].append(prepared)
        sample_indices_by_case[cid].add(sample_idx)

    observed_sample_indices = sorted(sample_rows_by_idx.keys())
    per_sample_summary = {
        str(sample_idx): {
            "sample_idx": sample_idx,
            **_summarize_rows(
                [
                    row
                    for row in sample_rows_by_idx[sample_idx]
                    if str(row.get("case_id") or "") in case_ids
                ]
            ),
        }
        for sample_idx in observed_sample_indices
    }

    selected_one_time_idx = observed_sample_indices[0] if observed_sample_indices else None
    if selected_one_time_idx is None:
        one_time_summary = {
            "selected_sample_idx": None,
            "selection_metric": "canonical_run",
            **_summarize_rows(per_case_rows),
        }
    else:
        one_time_summary = {
            "selected_sample_idx": int(selected_one_time_idx),
            "selection_metric": "lowest_sample_idx",
            **per_sample_summary[str(selected_one_time_idx)],
        }

    best_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_rows = [
            row
            for sample_idx in observed_sample_indices
            for row in sample_rows_by_idx[sample_idx]
            if str(row.get("case_id") or "") == case_id
        ]
        statuses = [str(row.get("run_status") or "missing") for row in case_rows]
        best_rows.append(
            {
                "case_id": case_id,
                "run_status": "best_of_n",
                "valid": any(_is_valid_status(status) for status in statuses),
                "user_goal_success": any(bool(row.get("user_goal_success")) for row in case_rows),
            }
        )

    best_summary = {
        "num_samples": len(observed_sample_indices),
        "sample_indices": observed_sample_indices,
        **_summarize_rows(best_rows),
    }

    per_case_out = run_dir / "eval" / "osr_user_per_case.jsonl"
    per_sample_out = run_dir / "eval" / "osr_user_per_sample.jsonl"
    write_jsonl(per_case_out, per_case_rows)
    flat_sample_rows = [
        row
        for sample_idx in observed_sample_indices
        for row in sorted(sample_rows_by_idx[sample_idx], key=lambda item: str(item["case_id"]))
    ]
    write_jsonl(per_sample_out, flat_sample_rows)

    metrics = {
        "generator": "file/exp/osr/evaluate_osr.py",
        "evaluation_rule": "submit_while_last_successful_read_matches_user_target",
        "total_cases_expected": len(case_ids),
        "per_case_path": to_posix(per_case_out.resolve()),
        "per_sample_path": to_posix(per_sample_out.resolve()),
        "canonical_run": _summarize_rows(per_case_rows),
        "one_time": one_time_summary,
        "best_of_n": best_summary,
        "per_sample": per_sample_summary,
    }
    write_json(run_dir / "eval" / "osr_metrics.json", metrics)
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics = evaluate_osr(args.run_dir, case_id=args.case_id)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
