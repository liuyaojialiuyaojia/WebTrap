from __future__ import annotations


def build_stage_plan(chain_len: int) -> list[tuple[str, int]]:
    """Retain Lure at its Full-condition location."""

    if chain_len < 2:
        raise ValueError("Lure-only requires a Full condition that contains Lure.")
    return [("lure", 0)]
