#!/usr/bin/env python3
"""Merge exp3 per-case signals and compute aggregate metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
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


def _load_json_maybe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_valid_status(status: str) -> bool:
    value = str(status or "").strip()
    return value not in {"model_error", "missing", ""}


def _index_sample_rows(
    rows: list[dict[str, Any]],
    *,
    success_key: str,
    value_out: dict[tuple[str, int], bool],
    status_out: dict[tuple[str, int], str],
    indices_out: dict[str, set[int]],
) -> None:
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            continue
        if "sample_idx" not in row:
            continue
        try:
            sample_idx = int(row.get("sample_idx"))
        except Exception:
            continue
        key = (case_id, sample_idx)
        value_out[key] = bool(row.get(success_key))
        if "run_status" in row:
            status_out[key] = str(row.get("run_status") or "missing")
        indices_out[case_id].add(sample_idx)


def summarize_metrics(run_dir: Path) -> dict[str, Any]:
    cases = read_jsonl(run_dir / "attack_cases.jsonl")
    user_rows = {
        str(row.get("case_id")): row
        for row in read_jsonl(run_dir / "eval" / "user_goal_per_case.jsonl")
    }
    end2end_rows = {
        str(row.get("case_id")): row
        for row in read_jsonl(run_dir / "eval" / "asr_end_to_end_per_case.jsonl")
    }
    intermediate_rows = {
        str(row.get("case_id")): row
        for row in read_jsonl(run_dir / "eval" / "asr_intermediate_per_case.jsonl")
    }

    user_sample_rows = read_jsonl(run_dir / "eval" / "user_goal_per_sample.jsonl")
    end2end_sample_rows = read_jsonl(run_dir / "eval" / "asr_end_to_end_per_sample.jsonl")
    intermediate_sample_rows = read_jsonl(
        run_dir / "eval" / "asr_intermediate_per_sample.jsonl"
    )

    sample_user_success: dict[tuple[str, int], bool] = {}
    sample_end2end_success: dict[tuple[str, int], bool] = {}
    sample_intermediate_success: dict[tuple[str, int], bool] = {}
    sample_run_status: dict[tuple[str, int], str] = {}
    case_sample_indices: dict[str, set[int]] = defaultdict(set)

    _index_sample_rows(
        user_sample_rows,
        success_key="user_goal_success",
        value_out=sample_user_success,
        status_out=sample_run_status,
        indices_out=case_sample_indices,
    )
    _index_sample_rows(
        end2end_sample_rows,
        success_key="asr_end_to_end_success",
        value_out=sample_end2end_success,
        status_out=sample_run_status,
        indices_out=case_sample_indices,
    )
    _index_sample_rows(
        intermediate_sample_rows,
        success_key="asr_intermediate_success",
        value_out=sample_intermediate_success,
        status_out=sample_run_status,
        indices_out=case_sample_indices,
    )

    per_case_rows: list[dict[str, Any]] = []
    status_breakdown: dict[str, int] = {}
    intermediate_breakdown: dict[str, int] = {}

    for case in cases:
        case_id = str(case.get("case_id") or "").strip()
        if not case_id:
            continue

        user_eval = user_rows.get(case_id, {})
        end2end_eval = end2end_rows.get(case_id, {})
        intermediate_eval = intermediate_rows.get(case_id, {})

        run_summary = _load_json_maybe(run_dir / "logs" / f"run_{case_id}.json")
        run_status = str(run_summary.get("status") or "missing")
        status_breakdown[run_status] = status_breakdown.get(run_status, 0) + 1
        valid = run_status not in {"model_error", "missing"}

        intermediate_label = str(intermediate_eval.get("final_classification") or "missing")
        intermediate_breakdown[intermediate_label] = (
            intermediate_breakdown.get(intermediate_label, 0) + 1
        )

        per_case_rows.append(
            {
                "case_id": case_id,
                "run_status": run_status,
                "valid": valid,
                "user_goal_success": bool(user_eval.get("user_goal_success")),
                "asr_end_to_end_success": bool(end2end_eval.get("asr_end_to_end_success")),
                "gmail_called": bool(end2end_eval.get("gmail_called")),
                "asr_intermediate_success": bool(intermediate_eval.get("asr_intermediate_success")),
                "asr_intermediate_final_classification": intermediate_label,
                "trace_path": run_summary.get("trace_path")
                or end2end_eval.get("trace_path")
                or user_eval.get("trace_path")
                or intermediate_eval.get("trace_path"),
            }
        )

    total = len(per_case_rows)
    valid_count = sum(1 for row in per_case_rows if row["valid"])

    user_success_total = sum(1 for row in per_case_rows if row["user_goal_success"])
    user_success_valid = sum(
        1 for row in per_case_rows if row["valid"] and row["user_goal_success"]
    )

    end2end_success_total = sum(1 for row in per_case_rows if row["asr_end_to_end_success"])
    end2end_success_valid = sum(
        1 for row in per_case_rows if row["valid"] and row["asr_end_to_end_success"]
    )

    intermediate_success_total = sum(
        1 for row in per_case_rows if row["asr_intermediate_success"]
    )
    intermediate_success_valid = sum(
        1 for row in per_case_rows if row["valid"] and row["asr_intermediate_success"]
    )

    metrics = {
        "run_dir": to_posix(run_dir.resolve()),
        "total_cases": total,
        "valid_cases": valid_count,
        "user_task_success_rate": _safe_div(user_success_total, total),
        "user_task_success_rate_valid": _safe_div(user_success_valid, valid_count),
        "asr_end_to_end_rate": _safe_div(end2end_success_total, total),
        "asr_end_to_end_rate_valid": _safe_div(end2end_success_valid, valid_count),
        "asr_intermediate_rate": _safe_div(intermediate_success_total, total),
        "asr_intermediate_rate_valid": _safe_div(intermediate_success_valid, valid_count),
        "counts": {
            "user_task_success_total": user_success_total,
            "user_task_success_valid": user_success_valid,
            "asr_end_to_end_success_total": end2end_success_total,
            "asr_end_to_end_success_valid": end2end_success_valid,
            "asr_intermediate_success_total": intermediate_success_total,
            "asr_intermediate_success_valid": intermediate_success_valid,
        },
        "status_breakdown": status_breakdown,
        "asr_intermediate_breakdown": intermediate_breakdown,
    }

    case_ids = [str(row.get("case_id") or "").strip() for row in cases]
    case_ids = [cid for cid in case_ids if cid]

    sample_metrics_ready = bool(
        user_sample_rows and end2end_sample_rows and intermediate_sample_rows
    )
    if sample_metrics_ready:
        user_idx = {idx for (_cid, idx) in sample_user_success.keys()}
        end2end_idx = {idx for (_cid, idx) in sample_end2end_success.keys()}
        intermediate_idx = {idx for (_cid, idx) in sample_intermediate_success.keys()}
        observed_sample_indices = sorted(user_idx & end2end_idx & intermediate_idx)
    else:
        observed_sample_indices = []
    observed_num_samples = (
        (max(observed_sample_indices) + 1) if observed_sample_indices else 0
    )

    if sample_metrics_ready and observed_sample_indices:
        best_user_total = 0
        best_end2end_total = 0
        best_intermediate_total = 0
        best_valid_cases = 0
        best_user_valid = 0
        best_end2end_valid = 0
        best_intermediate_valid = 0

        for case_id in case_ids:
            indices = sorted(
                idx
                for idx in case_sample_indices.get(case_id, set())
                if idx in observed_sample_indices
            )
            if not indices:
                continue

            statuses = [sample_run_status.get((case_id, idx), "missing") for idx in indices]
            is_valid = any(_is_valid_status(s) for s in statuses)
            if is_valid:
                best_valid_cases += 1

            user_any = any(sample_user_success.get((case_id, idx), False) for idx in indices)
            end2end_any = any(
                sample_end2end_success.get((case_id, idx), False) for idx in indices
            )
            intermediate_any = any(
                sample_intermediate_success.get((case_id, idx), False) for idx in indices
            )

            if user_any:
                best_user_total += 1
                if is_valid:
                    best_user_valid += 1
            if end2end_any:
                best_end2end_total += 1
                if is_valid:
                    best_end2end_valid += 1
            if intermediate_any:
                best_intermediate_total += 1
                if is_valid:
                    best_intermediate_valid += 1

        metrics["best_of_n"] = {
            "num_samples": observed_num_samples,
            "sample_indices": observed_sample_indices,
            "total_cases": len(case_ids),
            "valid_cases": best_valid_cases,
            "user_task_success_rate": _safe_div(best_user_total, len(case_ids)),
            "user_task_success_rate_valid": _safe_div(best_user_valid, best_valid_cases),
            "asr_end_to_end_rate": _safe_div(best_end2end_total, len(case_ids)),
            "asr_end_to_end_rate_valid": _safe_div(best_end2end_valid, best_valid_cases),
            "asr_intermediate_rate": _safe_div(best_intermediate_total, len(case_ids)),
            "asr_intermediate_rate_valid": _safe_div(best_intermediate_valid, best_valid_cases),
            "counts": {
                "user_task_success_total": best_user_total,
                "user_task_success_valid": best_user_valid,
                "asr_end_to_end_success_total": best_end2end_total,
                "asr_end_to_end_success_valid": best_end2end_valid,
                "asr_intermediate_success_total": best_intermediate_total,
                "asr_intermediate_success_valid": best_intermediate_valid,
            },
        }

        per_idx: dict[int, dict[str, Any]] = {}
        for sample_idx in observed_sample_indices:
            idx_valid_cases = 0
            user_total_idx = 0
            end2end_total_idx = 0
            intermediate_total_idx = 0
            user_valid_idx = 0
            end2end_valid_idx = 0
            intermediate_valid_idx = 0

            for case_id in case_ids:
                status = sample_run_status.get((case_id, sample_idx), "missing")
                is_valid = _is_valid_status(status)
                if is_valid:
                    idx_valid_cases += 1

                user_ok = sample_user_success.get((case_id, sample_idx), False)
                end2end_ok = sample_end2end_success.get((case_id, sample_idx), False)
                intermediate_ok = sample_intermediate_success.get((case_id, sample_idx), False)

                if user_ok:
                    user_total_idx += 1
                    if is_valid:
                        user_valid_idx += 1
                if end2end_ok:
                    end2end_total_idx += 1
                    if is_valid:
                        end2end_valid_idx += 1
                if intermediate_ok:
                    intermediate_total_idx += 1
                    if is_valid:
                        intermediate_valid_idx += 1

            per_idx[sample_idx] = {
                "sample_idx": int(sample_idx),
                "total_cases": len(case_ids),
                "valid_cases": idx_valid_cases,
                "user_task_success_rate": _safe_div(user_total_idx, len(case_ids)),
                "user_task_success_rate_valid": _safe_div(user_valid_idx, idx_valid_cases),
                "asr_end_to_end_rate": _safe_div(end2end_total_idx, len(case_ids)),
                "asr_end_to_end_rate_valid": _safe_div(end2end_valid_idx, idx_valid_cases),
                "asr_intermediate_rate": _safe_div(intermediate_total_idx, len(case_ids)),
                "asr_intermediate_rate_valid": _safe_div(
                    intermediate_valid_idx, idx_valid_cases
                ),
                "counts": {
                    "user_task_success_total": user_total_idx,
                    "user_task_success_valid": user_valid_idx,
                    "asr_end_to_end_success_total": end2end_total_idx,
                    "asr_end_to_end_success_valid": end2end_valid_idx,
                    "asr_intermediate_success_total": intermediate_total_idx,
                    "asr_intermediate_success_valid": intermediate_valid_idx,
                },
            }

        selected_one_time_idx: Optional[int] = None
        selected_score: Optional[tuple[float, int, float, int, int]] = None
        for sample_idx, row in per_idx.items():
            end2end_rate = float(row.get("asr_end_to_end_rate") or 0.0)
            end2end_total = int(
                row.get("counts", {}).get("asr_end_to_end_success_total") or 0
            )
            intermediate_rate = float(row.get("asr_intermediate_rate") or 0.0)
            intermediate_total = int(
                row.get("counts", {}).get("asr_intermediate_success_total") or 0
            )
            score = (
                end2end_rate,
                end2end_total,
                intermediate_rate,
                intermediate_total,
                -int(sample_idx),
            )
            if selected_score is None or score > selected_score:
                selected_score = score
                selected_one_time_idx = int(sample_idx)

        selected_row = per_idx.get(selected_one_time_idx or 0, {})
        metrics["one_time"] = {
            "num_samples": observed_num_samples,
            "sample_indices": observed_sample_indices,
            "selected_sample_idx": selected_one_time_idx,
            "selection_metric": "asr_end_to_end_rate",
            "tie_breaker_metric": "asr_intermediate_rate",
            "total_cases": int(selected_row.get("total_cases") or len(case_ids)),
            "valid_cases": int(selected_row.get("valid_cases") or 0),
            "user_task_success_rate": float(selected_row.get("user_task_success_rate") or 0.0),
            "user_task_success_rate_valid": float(
                selected_row.get("user_task_success_rate_valid") or 0.0
            ),
            "asr_end_to_end_rate": float(selected_row.get("asr_end_to_end_rate") or 0.0),
            "asr_end_to_end_rate_valid": float(
                selected_row.get("asr_end_to_end_rate_valid") or 0.0
            ),
            "asr_intermediate_rate": float(selected_row.get("asr_intermediate_rate") or 0.0),
            "asr_intermediate_rate_valid": float(
                selected_row.get("asr_intermediate_rate_valid") or 0.0
            ),
            "counts": dict(selected_row.get("counts") or {}),
            "candidate_asr_end_to_end_rate": {
                str(idx): float(per_idx[idx]["asr_end_to_end_rate"])
                for idx in observed_sample_indices
            },
            "candidate_asr_intermediate_rate": {
                str(idx): float(per_idx[idx]["asr_intermediate_rate"])
                for idx in observed_sample_indices
            },
        }

    write_jsonl(run_dir / "eval" / "per_case.jsonl", per_case_rows)
    write_json(run_dir / "eval" / "metrics.json", metrics)
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    metrics = summarize_metrics(args.run_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
