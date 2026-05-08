#!/usr/bin/env python3
"""Shared helpers for resolving experiment-specific output directories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike[str], Path]

DEFAULT_EXPERIMENT_ROOT = Path("web/runs/trap_site_mvp")


def resolve_experiment_root(
    base: Optional[PathLike],
    experiment_id: Optional[str],
    *,
    default: Path = DEFAULT_EXPERIMENT_ROOT,
) -> Path:
    """Return the directory containing artefacts for a specific experiment.

    When ``base`` is None, ``default`` is used. If ``experiment_id`` is provided,
    it is appended as a child directory.
    """

    root = Path(base) if base is not None else default
    if experiment_id:
        root = root / experiment_id
    return root


def resolve_path_within(
    candidate: Optional[PathLike],
    *,
    root: Path,
    relative: str,
) -> Path:
    """Return ``candidate`` as a ``Path`` or fallback to ``root / relative``."""

    if candidate is not None:
        return Path(candidate)
    return root / relative
