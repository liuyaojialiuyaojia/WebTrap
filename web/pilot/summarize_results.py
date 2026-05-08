#!/usr/bin/env python3
"""Summarize pilot path-behavior results into a figure-friendly table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_json(args.run_root / "classification_rows.json")
    if not isinstance(rows, list):
        raise ValueError("classification_rows.json must be a list")

    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_trials = int(row.get("n_trials") or 0)
        unaffected = int(row.get("Unaffected") or 0)
        hijacked = int(row.get("Hijacked") or 0)
        perturbed = int(row.get("Perturbed") or 0)
        denom = float(n_trials) if n_trials > 0 else 1.0
        summary_rows.append(
            {
                "setting_label": row.get("setting_label"),
                "source_tree_depth": row.get("source_tree_depth"),
                "condition": row.get("condition"),
                "condition_display_name": row.get("condition_display_name"),
                "n_trials": n_trials,
                "Unaffected": unaffected,
                "Hijacked": hijacked,
                "Perturbed": perturbed,
                "Unaffected_ratio": unaffected / denom,
                "Hijacked_ratio": hijacked / denom,
                "Perturbed_ratio": perturbed / denom,
            }
        )

    write_json(args.run_root / "figure1_table.json", summary_rows)
    fieldnames = [
        "setting_label",
        "source_tree_depth",
        "condition",
        "condition_display_name",
        "n_trials",
        "Unaffected",
        "Hijacked",
        "Perturbed",
        "Unaffected_ratio",
        "Hijacked_ratio",
        "Perturbed_ratio",
    ]
    csv_path = args.run_root / "figure1_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
