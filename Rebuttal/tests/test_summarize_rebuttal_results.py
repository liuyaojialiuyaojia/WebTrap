from __future__ import annotations

import json
from pathlib import Path

from Rebuttal.experiments.summarize_rebuttal_results import (
    _load_coverage_rows,
    _load_supplemental_coverage_rows,
    _metric_totals,
)


def test_metric_totals_require_and_sum_all_formal_counts(tmp_path: Path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"metrics-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "summary": {"intermediate_compromised": 1},
                    "user_utility": {"total_scores": 1, "cnt_tasks": 2},
                    "attacker_utility": {"total_scores": 1, "cnt_tasks": 2},
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    assert _metric_totals(paths) == {
        "user_pass": 2,
        "user_total": 4,
        "attack_pass": 2,
        "attack_total": 4,
        "intermediate_compromised": 2,
        "intermediate_total": 4,
    }


def test_coverage_summary_keeps_primary_and_supplemental_results() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    primary = _load_coverage_rows(repo_root)
    supplemental = _load_supplemental_coverage_rows(repo_root)

    assert len(primary) == 4
    assert len(supplemental) == 4
    assert {row["trajectories"] for row in primary} == {60, 72}
    assert {row["trajectories"] for row in supplemental} == {16}
    assert all(
        "nodes_er_ge_10_ratio_upper_bound" in row for row in primary
    )
    assert all(
        row["top1_er_upper_bound"]
        == row["top2_er_upper_bound"]
        == row["top3_er_upper_bound"]
        == 1.0
        for row in primary
    )
    assert all("nodes_er_ge_10_ratio" in row for row in supplemental)
