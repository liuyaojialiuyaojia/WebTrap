#!/usr/bin/env python3
"""Summarize reasoning-evaluator all-trials artefacts at the method level."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]


def _iter_log_dirs_from_args(
    log_dirs: Iterable[str], log_dir_list: Optional[Path]
) -> list[Path]:
    resolved: list[Path] = [Path(item).resolve() for item in log_dirs]
    if log_dir_list is not None:
        for line in log_dir_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            resolved.append(Path(line).resolve())
    unique = sorted({path for path in resolved})
    if not unique:
        raise ValueError("No log directories provided.")
    return unique


def _method_of(path: Path) -> str:
    parts = path.parts
    for name in (
        "generic_plain_text",
        "generic_url_injection",
        "goal_hijacking_plain_text",
        "goal_hijacking_url_injection",
        "topicattack",
        "combinedattack",
    ):
        if name in parts:
            return name
    return "psaa"


def _empty_bucket() -> dict[str, Any]:
    return {
        "method": "",
        "trial_count": 0,
        "compromised_count": 0,
        "not_compromised_count": 0,
        "first_compromised_step_sum": 0,
        "first_compromised_step_den": 0,
        "compromised_step_sum": 0,
        "compromised_step_den": 0,
        "compromised_stop_num": 0,
        "compromised_stop_den": 0,
        "not_compromised_step_sum": 0,
        "not_compromised_step_den": 0,
        "not_compromised_stop_num": 0,
        "not_compromised_stop_den": 0,
    }


def _safe_div(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _format_fraction(num: int, den: int) -> str:
    value = _safe_div(num, den)
    if value is None:
        return f"{num}/{den}"
    return f"{num}/{den} ({value:.6f})"


def summarize(log_dirs: list[Path], out_name: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(_empty_bucket)

    for log_dir in log_dirs:
        trial_results_path = log_dir / out_name / "trial_results.json"
        if not trial_results_path.exists():
            raise FileNotFoundError(f"Missing trial results: {trial_results_path}")
        payload = json.loads(trial_results_path.read_text(encoding="utf-8"))
        for trial in payload.get("trials") or []:
            method = _method_of(log_dir)
            bucket = buckets[method]
            bucket["method"] = method
            bucket["trial_count"] += 1
            label = str(trial.get("binary_final_label") or "")
            total_steps = int(trial.get("total_tool_action_count") or 0)
            is_stop = bool(trial.get("final_action_is_stop"))
            if label == "compromised":
                bucket["compromised_count"] += 1
                bucket["compromised_step_sum"] += total_steps
                bucket["compromised_step_den"] += 1
                bucket["compromised_stop_num"] += int(is_stop)
                bucket["compromised_stop_den"] += 1
                first_step = trial.get("first_compromised_step_number")
                if first_step is not None:
                    bucket["first_compromised_step_sum"] += int(first_step)
                    bucket["first_compromised_step_den"] += 1
            else:
                bucket["not_compromised_count"] += 1
                bucket["not_compromised_step_sum"] += total_steps
                bucket["not_compromised_step_den"] += 1
                bucket["not_compromised_stop_num"] += int(is_stop)
                bucket["not_compromised_stop_den"] += 1

    summary_rows = []
    for method in sorted(buckets):
        bucket = buckets[method]
        row = {
            **bucket,
            "first_compromised_step_avg": _safe_div(
                bucket["first_compromised_step_sum"],
                bucket["first_compromised_step_den"],
            ),
            "compromised_avg_step": _safe_div(
                bucket["compromised_step_sum"],
                bucket["compromised_step_den"],
            ),
            "compromised_stop_ratio": _safe_div(
                bucket["compromised_stop_num"],
                bucket["compromised_stop_den"],
            ),
            "not_compromised_avg_step": _safe_div(
                bucket["not_compromised_step_sum"],
                bucket["not_compromised_step_den"],
            ),
            "not_compromised_stop_ratio": _safe_div(
                bucket["not_compromised_stop_num"],
                bucket["not_compromised_stop_den"],
            ),
            "display": {
                "first_compromised_step": _format_fraction(
                    bucket["first_compromised_step_sum"],
                    bucket["first_compromised_step_den"],
                ),
                "compromised_avg_step": _format_fraction(
                    bucket["compromised_step_sum"],
                    bucket["compromised_step_den"],
                ),
                "compromised_stop_ratio": _format_fraction(
                    bucket["compromised_stop_num"],
                    bucket["compromised_stop_den"],
                ),
                "not_compromised_avg_step": _format_fraction(
                    bucket["not_compromised_step_sum"],
                    bucket["not_compromised_step_den"],
                ),
                "not_compromised_stop_ratio": _format_fraction(
                    bucket["not_compromised_stop_num"],
                    bucket["not_compromised_stop_den"],
                ),
            },
        }
        summary_rows.append(row)

    return {
        "out_subdir": out_name,
        "log_dir_count": len(log_dirs),
        "methods": summary_rows,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", action="append", default=[], help="Target agent_logs_post_injection directory.")
    parser.add_argument("--log-dir-list", type=Path, help="File containing one log directory per line.")
    parser.add_argument("--out-subdir", type=str, default="reasoning_eval_all_trials")
    parser.add_argument("--out", type=Path, help="Optional output JSON path.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    log_dirs = _iter_log_dirs_from_args(args.log_dir, args.log_dir_list)
    payload = summarize(log_dirs, args.out_subdir)
    if args.out is not None:
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
