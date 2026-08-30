#!/usr/bin/env python3
"""Aggregate the original clean OSR trajectories for node exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from Rebuttal.experiments.analyze_node_frequency import (
    CoverageRow,
    TraceRecord,
    _normalize_file_path,
    _walk_file_nodes,
    _write_csv,
    _write_markdown,
    _write_node_rates_csv,
    browser_candidate_sets,
    parse_browser_trace,
    parse_file_trace,
    summarize_coverage,
)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[object]:
    rows: list[object] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_original_browser_records(
    run_root: Path,
    *,
    expected_tasks: int = 24,
    expected_trials_per_task: int = 3,
) -> list[TraceRecord]:
    """Load every trial from the original clean Browser OSR run."""

    task_root = run_root / "webarena_tasks_osr"
    trace_root = run_root / "agent_logs_osr"
    task_paths = sorted(
        (path for path in task_root.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if len(task_paths) != expected_tasks:
        raise ValueError(
            f"Browser: expected {expected_tasks} tasks, found {len(task_paths)}"
        )

    records: list[TraceRecord] = []
    for task_path in task_paths:
        task = _read_json(task_path)
        if not isinstance(task, Mapping):
            raise ValueError(f"Browser task must be an object: {task_path}")
        task_id = int(task.get("task_id", task_path.stem))
        metadata = task.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Browser task missing metadata: {task_path}")
        target = int(metadata["page_index"])
        trace_paths = sorted(trace_root.glob(f"trace_{task_id}_r*.jsonl"))
        if len(trace_paths) != expected_trials_per_task:
            raise ValueError(
                f"Browser task {task_id}: expected {expected_trials_per_task} "
                f"trials, found {len(trace_paths)}"
            )
        for trace_path in trace_paths:
            if trace_path.stat().st_size == 0:
                raise ValueError(f"Browser trace is empty: {trace_path}")
            visited = parse_browser_trace(trace_path)
            if not visited:
                raise ValueError(
                    f"Browser trace has no parseable page visits: {trace_path}"
                )
            records.append(
                TraceRecord(
                    trace_id=trace_path.stem,
                    target_node=target,
                    visited_nodes=visited,
                )
            )
    return records


def load_original_file_records(
    run_root: Path,
    *,
    expected_cases: int = 20,
    expected_samples_per_case: int = 3,
) -> list[TraceRecord]:
    """Load every sample from the original clean File OSR run."""

    task_root = run_root / "tasks" / "user"
    trace_root = run_root / "logs" / "samples"
    task_paths = sorted(task_root.glob("case_*.json"))
    if len(task_paths) != expected_cases:
        raise ValueError(
            f"File: expected {expected_cases} cases, found {len(task_paths)}"
        )

    records: list[TraceRecord] = []
    for task_path in task_paths:
        task = _read_json(task_path)
        if not isinstance(task, Mapping):
            raise ValueError(f"File task must be an object: {task_path}")
        case_id = str(task["case_id"])
        target = _normalize_file_path(str(task["user_target_logical_path"]))
        trace_paths = sorted(
            (trace_root / case_id).glob("trace_sample_*.jsonl")
        )
        if len(trace_paths) != expected_samples_per_case:
            raise ValueError(
                f"File {case_id}: expected {expected_samples_per_case} "
                f"samples, found {len(trace_paths)}"
            )
        for trace_path in trace_paths:
            if trace_path.stat().st_size == 0:
                raise ValueError(f"File trace is empty: {trace_path}")
            visited = parse_file_trace(trace_path)
            if visited == frozenset({"/root"}):
                raise ValueError(
                    f"File trace has no parseable non-root visits: {trace_path}"
                )
            records.append(
                TraceRecord(
                    trace_id=f"{case_id}:{trace_path.stem}",
                    target_node=target,
                    visited_nodes=visited,
                )
            )
    return records


def _file_security_prefixes(run_root: Path) -> set[str]:
    prefixes: set[str] = set()
    for row in _read_jsonl(run_root / "attack_cases.jsonl"):
        if not isinstance(row, Mapping):
            continue
        material = str(row.get("security_entry_logical_path") or "").strip()
        if material:
            prefixes.add(_normalize_file_path(material))
    if not prefixes:
        raise ValueError("File OSR run has no security-entry prefixes")
    return prefixes


def original_file_candidate_sets(
    run_root: Path,
) -> tuple[dict[str, set[str]], set[str]]:
    """Return user-tree candidates after removing neutral security subtrees."""

    tree_path = run_root / "env" / "env_pre_injection_tree.json"
    tree = _read_json(tree_path)
    if not isinstance(tree, Mapping):
        raise ValueError(f"File environment tree must be an object: {tree_path}")
    directories, files = _walk_file_nodes(tree)
    security_prefixes = _file_security_prefixes(run_root)

    def is_security_node(node: str) -> bool:
        return any(
            node == prefix or node.startswith(f"{prefix}/")
            for prefix in security_prefixes
        )

    user_directories = {
        node for node in directories if not is_security_node(node)
    }
    user_files = {node for node in files if not is_security_node(node)}
    candidates = {
        "All non-trivial nodes": (user_directories | user_files) - {"/root"},
        "Attacker-writable public nodes": user_directories - {"/root"},
    }
    return candidates, security_prefixes


def _browser_security_nodes(run_root: Path, user_nodes: set[int]) -> set[int]:
    payload = _read_json(run_root / "static" / "page_metadata.json")
    pages = payload.get("pages") if isinstance(payload, Mapping) else None
    if not isinstance(pages, list):
        raise ValueError("Browser static metadata must contain pages")
    rendered_nodes = {
        int(page["page_index"])
        for page in pages
        if isinstance(page, Mapping) and "page_index" in page
    }
    return rendered_nodes - user_nodes


def _validate_non_user_visits(
    *,
    system: str,
    records: Sequence[TraceRecord],
    user_nodes: set[object],
    permitted_non_user_nodes: set[object],
) -> set[object]:
    visited = set().union(*(record.visited_nodes for record in records))
    non_user = visited - user_nodes
    unexpected = non_user - permitted_non_user_nodes
    if unexpected:
        preview = ", ".join(
            map(str, sorted(unexpected, key=str)[:5])
        )
        raise ValueError(
            f"{system}: trace contains unexpected non-user nodes: {preview}"
        )
    return non_user


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--browser-root",
        type=Path,
        default=Path("web/runs/exp_d10w2_osr_gitlab"),
    )
    parser.add_argument(
        "--file-root",
        type=Path,
        default=Path("file/runs/exp_d10w2_osr"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Rebuttal/results/node_frequency_original_clean"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    browser_records = load_original_browser_records(args.browser_root)
    file_records = load_original_file_records(args.file_root)

    browser_candidates = browser_candidate_sets(args.browser_root)
    browser_user_nodes = {
        0, *browser_candidates["All non-trivial nodes"]
    }
    browser_security_nodes = _browser_security_nodes(
        args.browser_root, browser_user_nodes
    )
    browser_non_user_visits = _validate_non_user_visits(
        system="Browser",
        records=browser_records,
        user_nodes=set(browser_user_nodes),
        permitted_non_user_nodes=set(browser_security_nodes),
    )

    file_candidates, file_security_prefixes = original_file_candidate_sets(
        args.file_root
    )
    file_user_nodes = {
        "/root", *file_candidates["All non-trivial nodes"]
    }
    all_file_visits = set().union(
        *(record.visited_nodes for record in file_records)
    )
    file_permitted_non_user_nodes = {
        node
        for node in all_file_visits
        if isinstance(node, str)
        and any(
            node == prefix or node.startswith(f"{prefix}/")
            for prefix in file_security_prefixes
        )
    }
    file_non_user_visits = _validate_non_user_visits(
        system="File",
        records=file_records,
        user_nodes=set(file_user_nodes),
        permitted_non_user_nodes=set(file_permitted_non_user_nodes),
    )

    systems = (
        ("Browser", browser_candidates, browser_records),
        ("File", file_candidates, file_records),
    )
    rows: list[CoverageRow] = []
    for system, candidates, records in systems:
        for label in (
            "All non-trivial nodes",
            "Attacker-writable public nodes",
        ):
            rows.append(
                summarize_coverage(
                    system=system,
                    candidate_label=label,
                    candidates=candidates[label],
                    records=records,
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "EXP-COVER-001-ORIGINAL-CLEAN",
        "protocol": {
            "condition": "clean_no_attack",
            "source_runs": "original OSR runs",
            "trajectories": {
                "Browser": len(browser_records),
                "File": len(file_records),
            },
            "task_or_case_counts": {
                "Browser": 24,
                "File": 20,
            },
            "trials_or_samples_per_task": 3,
            "unique_target_nodes": {
                "Browser": len(
                    {record.target_node for record in browser_records}
                ),
                "File": len({record.target_node for record in file_records}),
            },
            "node_counting": "one visit per node per trajectory",
            "scope": "user-tree nodes only in numerator and denominator",
            "root": "excluded as the mandatory trivial node",
            "target_nodes": "included",
            "top1_er": "maximum empirical node visit frequency",
            "random1_er": (
                "exact expected ER for one uniformly sampled candidate node"
            ),
            "threshold_ratios": (
                "candidate nodes meeting threshold / all candidate nodes"
            ),
            "browser_attacker_writable_public": (
                "all non-root Browser user-tree pages"
            ),
            "file_attacker_writable_public": (
                "all non-root File user-tree directories"
            ),
            "neutral_security_nodes": (
                "excluded from numerator and denominator under the existing "
                "user-tree-only table definition"
            ),
        },
        "rows": [asdict(row) for row in rows],
    }
    json_path = args.out_dir / "node_frequency.json"
    csv_path = args.out_dir / "node_frequency.csv"
    node_rates_path = args.out_dir / "node_rates.csv"
    table_path = args.out_dir / "table.md"
    validation_path = args.out_dir / "validation.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    _write_node_rates_csv(node_rates_path, systems)
    _write_markdown(table_path, rows)

    validation = {
        "condition": "clean_no_attack",
        "trajectory_counts": {
            "Browser": len(browser_records),
            "File": len(file_records),
        },
        "unique_target_counts": {
            "Browser": len({record.target_node for record in browser_records}),
            "File": len({record.target_node for record in file_records}),
        },
        "candidate_counts": {
            system: {
                row.candidate_nodes: row.candidate_count
                for row in rows
                if row.system == system
            }
            for system in ("Browser", "File")
        },
        "excluded_neutral_security_nodes_visited": {
            "Browser": sorted(browser_non_user_visits),
            "File": sorted(file_non_user_visits),
        },
        "output_sha256": {
            "node_frequency.json": _sha256(json_path),
            "node_frequency.csv": _sha256(csv_path),
            "node_rates.csv": _sha256(node_rates_path),
            "table.md": _sha256(table_path),
        },
    }
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        print(
            f"{row.system}\t{row.candidate_nodes}\t"
            f"Top-1={row.top1_er:.4f}\tRandom-1={row.random1_er:.4f}\t"
            f"nodes>=10%={row.nodes_er_ge_10_ratio:.4f}\t"
            f"nodes>=30%={row.nodes_er_ge_30_ratio:.4f}"
        )
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
