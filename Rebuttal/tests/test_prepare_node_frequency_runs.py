from collections import Counter

from Rebuttal.experiments.prepare_node_frequency_runs import (
    branch_coverage,
    select_prefix_balanced,
)


def test_prefix_balanced_selection_spreads_binary_leaves() -> None:
    leaves = [
        tuple(int(bit) for bit in f"{index:09b}")
        for index in range(2**9)
    ]
    selected = select_prefix_balanced(
        leaves,
        count=16,
        path_parts=lambda row: row,
    )
    coverage = branch_coverage(selected, path_parts=lambda row: row, max_depth=5)
    assert len(selected) == 16
    assert len(set(selected)) == 16
    assert coverage == {"1": 2, "2": 4, "3": 8, "4": 16, "5": 16}
    assert set(Counter(row[:1] for row in selected).values()) == {8}
    assert set(Counter(row[:2] for row in selected).values()) == {4}
    assert set(Counter(row[:3] for row in selected).values()) == {2}
    assert set(Counter(row[:4] for row in selected).values()) == {1}


def test_prefix_balanced_selection_rejects_duplicate_paths() -> None:
    rows = [("a",), ("a",), ("b",)]
    try:
        select_prefix_balanced(rows, count=2, path_parts=lambda row: row)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate paths must be rejected")
