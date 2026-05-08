#!/usr/bin/env python3
"""Unified entry for depth-range inertia experiments in file/find."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add repository root to import path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from file.find.next_step import NEXT_STEP_CONFIG, run_next_step_experiments
from file.find.state_judgment import (
    STATE_JUDGMENT_CONFIG,
    run_state_judgment_experiments,
)


def _build_override(args: argparse.Namespace) -> Dict[str, Any]:
    override: Dict[str, Any] = {}
    if args.model:
        override["model"] = args.model
    if args.temperature is not None:
        override["temperature"] = args.temperature
    if args.history_gap is not None:
        override["history_gap"] = args.history_gap
    if args.depth_min is not None and args.depth_max is not None:
        override["depth_range"] = [args.depth_min, args.depth_max]
    if args.source_tree:
        override.setdefault("depth_variation", {})["source_tree"] = args.source_tree
    if args.fixed_width is not None:
        override.setdefault("depth_variation", {})["fixed_width"] = args.fixed_width
    if args.tests_per_depth is not None:
        override.setdefault("depth_variation", {})[
            "tests_per_depth"
        ] = args.tests_per_depth
    return override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=["state_judgment", "next_step"],
        default="state_judgment",
        help="Experiment type to run.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--history-gap", type=int, default=None)
    parser.add_argument("--depth-min", type=int, default=None)
    parser.add_argument("--depth-max", type=int, default=None)
    parser.add_argument("--source-tree", type=str, default=None)
    parser.add_argument("--fixed-width", type=int, default=None)
    parser.add_argument("--tests-per-depth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    override = _build_override(args)

    if args.task == "state_judgment":
        cfg = dict(STATE_JUDGMENT_CONFIG)
        cfg.update(override)
        run_state_judgment_experiments(cfg)
        return

    cfg = dict(NEXT_STEP_CONFIG)
    cfg.update(override)
    run_next_step_experiments(cfg)


if __name__ == "__main__":
    main()
