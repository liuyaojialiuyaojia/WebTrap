from __future__ import annotations


def build_stage_plan(chain_len: int) -> list[tuple[str, int]]:
    """Retain Payload at its Full-condition location."""

    if chain_len <= 0:
        raise ValueError("Injection chain length must be positive.")
    return [("payload", chain_len - 1)]
