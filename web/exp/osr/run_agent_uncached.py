#!/usr/bin/env python3
"""OSR-only wrapper for Stage 05 web runner that forces uncached fresh trials."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List


def _load_shared_module():
    module_path = Path(__file__).resolve().parents[1] / "05_evaluate" / "run_agent.py"
    spec = importlib.util.spec_from_file_location("web_osr_run_agent_shared", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_module()


def _build_command_uncached(
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
        "python",
        "web/exp/osr/tool_calling_agent_uncached.py",
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


def main() -> None:
    args = shared.parse_args()
    if args.seed is not None:
        raise ValueError(
            "--seed is disabled for OSR fresh sampling. Remove --seed to force fresh trials."
        )
    shared._build_command = _build_command_uncached
    shared.parse_args = lambda: args
    shared.main()


if __name__ == "__main__":
    main()
