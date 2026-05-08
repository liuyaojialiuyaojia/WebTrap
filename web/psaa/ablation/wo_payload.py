from __future__ import annotations


def build_stage_plan(chain_len: int) -> list[tuple[str, int]]:
    if chain_len <= 0:
        raise ValueError("Injection chain length must be positive.")
    if chain_len == 1:
        return [("inertia", 0)]
    return [
        ("lure", 0),
        ("inertia", chain_len - 1),
    ]
