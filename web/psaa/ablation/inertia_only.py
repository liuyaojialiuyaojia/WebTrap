from __future__ import annotations


def _full_inertia_position(chain_len: int) -> int:
    if chain_len < 3:
        raise ValueError(
            "Inertia-only requires a Full condition that contains Inertia."
        )
    payload_position = chain_len - 1
    return min(payload_position - 1, max(1, (2 * payload_position) // 3))


def build_stage_plan(chain_len: int) -> list[tuple[str, int]]:
    """Retain Inertia at its Full-condition location."""

    return [("inertia", _full_inertia_position(chain_len))]
