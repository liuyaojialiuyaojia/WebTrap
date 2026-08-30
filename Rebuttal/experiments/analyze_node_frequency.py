#!/usr/bin/env python3
"""Aggregate the clean 16-target user-tree node-frequency rerun."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

Node = Hashable

_BROWSER_URL_RE = re.compile(
    r"https?://127\.0\.0\.1:\d+/index\.html(?:#p(?P<page>\d+))?"
)


@dataclass(frozen=True)
class TraceRecord:
    """One clean trajectory and the user-tree nodes it visited."""

    trace_id: str
    target_node: Node
    visited_nodes: frozenset[Node]


@dataclass(frozen=True)
class CoverageRow:
    """Requested paper-facing aggregate for one candidate-node policy."""

    system: str
    candidate_nodes: str
    trajectories: int
    candidate_count: int
    top1_node: Node
    top1_er: float
    random1_er: float
    nodes_er_ge_10_count: int
    nodes_er_ge_10_ratio: float
    nodes_er_ge_30_count: int
    nodes_er_ge_30_ratio: float


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_browser_trace(path: Path) -> frozenset[int]:
    """Extract unique rendered page indices from one Browser trace."""

    text = path.read_text(encoding="utf-8", errors="replace")
    visited: set[int] = set()
    for match in _BROWSER_URL_RE.finditer(text):
        page = match.group("page")
        visited.add(int(page) if page is not None else 0)
    return frozenset(visited)


def _normalize_file_path(path: str) -> str:
    material = str(path or "").strip()
    if material in {"", "/"}:
        return "/root"
    if material == "/root" or material.startswith("/root/"):
        return material.rstrip("/")
    return f"/root/{material.lstrip('/')}".rstrip("/")


def parse_file_trace(path: Path) -> frozenset[str]:
    """Extract unique directory and read-file nodes from one File trace."""

    visited: set[str] = {"/root"}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        observation = event.get("tool_observation")
        if not isinstance(observation, Mapping) or not observation.get("success"):
            continue
        current_path = observation.get("current_path")
        if isinstance(current_path, str):
            visited.add(_normalize_file_path(current_path))
        tool_call = event.get("tool_call")
        tool_name = (
            str(tool_call.get("name") or "")
            if isinstance(tool_call, Mapping)
            else ""
        )
        observed_path = observation.get("path")
        if tool_name == "read_file" and isinstance(observed_path, str):
            visited.add(_normalize_file_path(observed_path))
    return frozenset(visited)


def load_browser_records(run_root: Path) -> list[TraceRecord]:
    task_root = run_root / "tasks"
    trace_root = run_root / "logs"
    task_paths = sorted(
        (path for path in task_root.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
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
        trace_path = trace_root / f"trace_{task_id}.jsonl"
        if not trace_path.is_file() or trace_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Browser trace: {trace_path}")
        records.append(
            TraceRecord(
                trace_id=trace_path.stem,
                target_node=target,
                visited_nodes=parse_browser_trace(trace_path),
            )
        )
    return records


def browser_candidate_sets(run_root: Path) -> dict[str, set[int]]:
    payload = _read_json(run_root / "page_metadata.json")
    pages = payload.get("pages") if isinstance(payload, Mapping) else None
    if not isinstance(pages, list):
        raise ValueError("Browser user-tree metadata must contain pages")
    user_nodes = {
        int(page["page_index"])
        for page in pages
        if isinstance(page, Mapping) and "page_index" in page
    }
    nontrivial = user_nodes - {0}
    return {
        "All non-trivial nodes": set(nontrivial),
        # In the synthetic Browser user site every user-tree page is public and
        # admits content injection. Restricted security-microtree pages are not
        # present in this clean run or either numerator/denominator.
        "Attacker-writable public nodes": set(nontrivial),
    }


def load_file_records(run_root: Path) -> list[TraceRecord]:
    task_root = run_root / "tasks" / "user"
    trace_root = run_root / "logs" / "samples"
    task_paths = sorted(task_root.glob("case_*.json"))
    records: list[TraceRecord] = []
    for task_path in task_paths:
        task = _read_json(task_path)
        if not isinstance(task, Mapping):
            raise ValueError(f"File task must be an object: {task_path}")
        case_id = str(task["case_id"])
        target = _normalize_file_path(str(task["user_target_logical_path"]))
        trace_path = trace_root / case_id / "trace_sample_000.jsonl"
        if not trace_path.is_file() or trace_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing File trace: {trace_path}")
        records.append(
            TraceRecord(
                trace_id=f"{case_id}:sample_000",
                target_node=target,
                visited_nodes=parse_file_trace(trace_path),
            )
        )
    return records


def _walk_file_nodes(
    tree: Mapping[str, object],
    *,
    prefix: str = "",
) -> tuple[set[str], set[str]]:
    name = str(tree.get("name") or "").strip()
    current = f"{prefix}/{name}".replace("//", "/") if name else prefix or "/"
    current = _normalize_file_path(current)
    children = tree.get("children")
    child_rows = children if isinstance(children, list) else []
    is_directory = bool(child_rows)
    directories = {current} if is_directory else set()
    files = set() if is_directory else {current}
    for child in child_rows:
        if not isinstance(child, Mapping):
            continue
        child_directories, child_files = _walk_file_nodes(child, prefix=current)
        directories.update(child_directories)
        files.update(child_files)
    return directories, files


def file_candidate_sets(run_root: Path) -> dict[str, set[str]]:
    payload = _read_json(run_root / "env" / "user_tree.json")
    if not isinstance(payload, Mapping):
        raise ValueError("File user tree must be an object")
    directories, files = _walk_file_nodes(payload)
    return {
        "All non-trivial nodes": (directories | files) - {"/root"},
        "Attacker-writable public nodes": directories - {"/root"},
    }


def _node_rates(
    candidates: Iterable[Node],
    records: Sequence[TraceRecord],
) -> dict[Node, float]:
    if not records:
        raise ValueError("Node rates require at least one trajectory")
    candidate_set = set(candidates)
    counts: Counter[Node] = Counter()
    for record in records:
        counts.update(candidate_set.intersection(record.visited_nodes))
    return {
        node: counts[node] / len(records)
        for node in candidate_set
    }


def summarize_coverage(
    *,
    system: str,
    candidate_label: str,
    candidates: set[Node],
    records: Sequence[TraceRecord],
) -> CoverageRow:
    if not candidates:
        raise ValueError(f"Empty candidate set: {system}/{candidate_label}")
    rates = _node_rates(candidates, records)
    ranked = sorted(rates, key=lambda node: (-rates[node], str(node)))
    top1 = ranked[0]
    ge_10 = sum(rate >= 0.10 for rate in rates.values())
    ge_30 = sum(rate >= 0.30 for rate in rates.values())
    return CoverageRow(
        system=system,
        candidate_nodes=candidate_label,
        trajectories=len(records),
        candidate_count=len(candidates),
        top1_node=top1,
        top1_er=rates[top1],
        # Exact expectation under a uniform draw of one node from this same
        # candidate set. This is equivalent to infinitely many Random-1 draws.
        random1_er=sum(rates.values()) / len(rates),
        nodes_er_ge_10_count=ge_10,
        nodes_er_ge_10_ratio=ge_10 / len(rates),
        nodes_er_ge_30_count=ge_30,
        nodes_er_ge_30_ratio=ge_30 / len(rates),
    )


def _validate_records(
    *,
    system: str,
    records: Sequence[TraceRecord],
    user_nodes: set[Node],
    expected_samples: int,
) -> None:
    if len(records) != expected_samples:
        raise ValueError(
            f"{system}: expected {expected_samples} trajectories, got {len(records)}"
        )
    targets = {record.target_node for record in records}
    if len(targets) != expected_samples:
        raise ValueError(
            f"{system}: targets must be unique ({len(targets)}/{expected_samples})"
        )
    unknown = set().union(*(set(record.visited_nodes) for record in records)) - user_nodes
    if unknown:
        preview = ", ".join(map(str, sorted(unknown, key=str)[:5]))
        raise ValueError(f"{system}: trace contains non-user-tree nodes: {preview}")


def _write_csv(path: Path, rows: Sequence[CoverageRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_node_rates_csv(
    path: Path,
    systems: Sequence[
        tuple[str, Mapping[str, set[Node]], Sequence[TraceRecord]]
    ],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "candidate_nodes",
        "node",
        "visit_count",
        "trajectories",
        "er",
        "er_ge_10",
        "er_ge_30",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for system, candidate_sets, records in systems:
            for label in (
                "All non-trivial nodes",
                "Attacker-writable public nodes",
            ):
                rates = _node_rates(candidate_sets[label], records)
                for node in sorted(rates, key=str):
                    rate = rates[node]
                    writer.writerow(
                        {
                            "system": system,
                            "candidate_nodes": label,
                            "node": node,
                            "visit_count": round(rate * len(records)),
                            "trajectories": len(records),
                            "er": rate,
                            "er_ge_10": rate >= 0.10,
                            "er_ge_30": rate >= 0.30,
                        }
                    )


def _write_markdown(path: Path, rows: Sequence[CoverageRow]) -> None:
    lines = [
        "| System | Candidate nodes | Top-1 ER ↑ | Random-1 ER | nodes ER ≥ 10% ↑ | nodes ER ≥ 30% ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.system} | {row.candidate_nodes} | "
            f"{row.top1_er:.2%} | {row.random1_er:.2%} | "
            f"{row.nodes_er_ge_10_ratio:.2%} | "
            f"{row.nodes_er_ge_30_ratio:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("Rebuttal/runs/node_frequency_rerun"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Rebuttal/results/node_frequency_rerun"),
    )
    parser.add_argument("--expected-samples", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    browser_root = args.run_root / "browser"
    file_root = args.run_root / "file"

    browser_records = load_browser_records(browser_root)
    browser_candidates = browser_candidate_sets(browser_root)
    browser_user_nodes = {0} | set(browser_candidates["All non-trivial nodes"])
    _validate_records(
        system="Browser",
        records=browser_records,
        user_nodes=browser_user_nodes,
        expected_samples=args.expected_samples,
    )

    file_records = load_file_records(file_root)
    file_candidates = file_candidate_sets(file_root)
    file_user_nodes = {"/root"} | set(file_candidates["All non-trivial nodes"])
    _validate_records(
        system="File",
        records=file_records,
        user_nodes=file_user_nodes,
        expected_samples=args.expected_samples,
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
        "experiment": "EXP-COVER-001-RERUN",
        "protocol": {
            "condition": "clean_no_attack",
            "fresh_trajectories": True,
            "samples_per_system": args.expected_samples,
            "unique_target_leaf_per_sample": True,
            "target_selection": "prefix-balanced across user-tree leaves",
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
        },
        "rows": [asdict(row) for row in rows],
    }
    json_path = args.out_dir / "node_frequency.json"
    csv_path = args.out_dir / "node_frequency.csv"
    node_rates_path = args.out_dir / "node_rates.csv"
    table_path = args.out_dir / "table.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, rows)
    _write_node_rates_csv(node_rates_path, systems)
    _write_markdown(table_path, rows)

    for row in rows:
        print(
            f"{row.system}\t{row.candidate_nodes}\t"
            f"Top-1={row.top1_er:.4f}\tRandom-1={row.random1_er:.4f}\t"
            f"nodes>=10%={row.nodes_er_ge_10_ratio:.4f}\t"
            f"nodes>=30%={row.nodes_er_ge_30_ratio:.4f}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {node_rates_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
