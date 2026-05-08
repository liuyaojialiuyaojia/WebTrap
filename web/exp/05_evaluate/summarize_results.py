#!/usr/bin/env python3
"""Combine evaluation artefacts into a single metrics report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = REPO_ROOT / "exp" / "02_nested_builder"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from experiment_paths import resolve_experiment_root, resolve_path_within


def _safe_load(path: Path | None) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse JSON file {path}: {exc}") from exc


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
        "--step-by-step",
        type=Path,
        default=Path("/tmp/run_step_by_step_asr.json"),
        help="JSON file emitted by evaluator_step_by_step.py",
    )
    parser.add_argument(
        "--user-utility",
        type=Path,
        default=Path("/tmp/run_user_utility.json"),
        help="JSON file emitted by evaluator_final_step.py for user tasks.",
    )
    parser.add_argument(
        "--attacker-utility",
        type=Path,
        default=Path("/tmp/run_attacker_utility.json"),
        help="JSON file emitted by evaluator_final_step.py for attacker tasks.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination for the combined metrics report. Defaults to experiment_root/metrics.json.",
    )
    return parser.parse_args()


def _summarise(step_by_step: Any, user: Any, attacker: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if step_by_step is not None:
        payload["step_by_step"] = step_by_step
    if user is not None:
        payload["user_utility"] = user
    if attacker is not None:
        payload["attacker_utility"] = attacker
    compromised_count = None
    if isinstance(step_by_step, dict):
        compromised_count = step_by_step.get("total_compromised")
        if compromised_count is None:
            compromised_count = step_by_step.get("compromised")
    payload["summary"] = {
        "intermediate_compromised": compromised_count,
        "user_score": user.get("total_scores", None) if isinstance(user, dict) else None,
        "attack_score": attacker.get("total_scores", None) if isinstance(attacker, dict) else None,
    }
    return payload


def main() -> None:
    args = parse_args()
    base_root = resolve_experiment_root(args.experiment_root, args.experiment_id)
    args.out = resolve_path_within(args.out, root=base_root, relative="metrics.json")

    step_by_step = _safe_load(args.step_by_step)
    user = _safe_load(args.user_utility)
    attacker = _safe_load(args.attacker_utility)

    combined = _summarise(step_by_step, user, attacker)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
