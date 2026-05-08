#!/usr/bin/env python3
"""Summarize navigation-guided web OSR metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--user-utility", type=Path, required=True)
    parser.add_argument("--path-stats", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_manifest = _load_json(args.task_manifest)
    user_utility = _load_json(args.user_utility)
    path_stats = _load_json(args.path_stats)

    if not isinstance(task_manifest, dict):
        raise ValueError("task manifest must be a JSON object.")
    if not isinstance(user_utility, dict):
        raise ValueError("user utility payload must be a JSON object.")

    total_scores = float(user_utility.get("total_scores") or 0.0)
    cnt_tasks = float(user_utility.get("cnt_tasks") or 0.0)
    payload = {
        "generator": "web/exp/osr/summarize_osr_metrics.py",
        "task_manifest": task_manifest,
        "user_utility": user_utility,
        "path_stats": path_stats,
        "summary": {
            "base_user_task_count": int(task_manifest.get("base_user_task_count") or 0),
            "selected_microtree_count": int(task_manifest.get("selected_microtree_count") or 0),
            "expanded_task_count": int(task_manifest.get("expanded_task_count") or 0),
            "passed_tasks": total_scores,
            "total_tasks": cnt_tasks,
            "osr_score": _safe_div(total_scores, cnt_tasks),
            "evaluation_rule": str(user_utility.get("rule") or ""),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
