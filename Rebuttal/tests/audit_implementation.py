#!/usr/bin/env python3
"""Audit Rebuttal implementation invariants without reading sample or trace text."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_WEB_ROOT = _REPOSITORY_ROOT / "web"
for _import_root in (_REPOSITORY_ROOT, _WEB_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from file.exp.attack.psaa.inject import (
    _generate_checked_injection_text as generate_file_text,
)
from psaa.inject_from_attack_case import (
    _generate_checked_injection_text as generate_browser_text,
)

from Rebuttal.experiments.contracts import (
    FORMAL_VARIANTS,
    PLACEMENT_VARIANTS,
    SINGLE_STAGE_VARIANTS,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_placements(repo_root: Path, errors: list[str]) -> dict[str, Any]:
    payload = _load_json(
        repo_root / "Rebuttal/results/placement/selected_placements.json"
    )
    _require(payload.get("schema_version") == 2, "placement schema must be 2", errors)
    systems = payload.get("systems")
    _require(isinstance(systems, Mapping), "placement systems are missing", errors)
    counts: dict[str, int] = {}
    for system_name in ("Browser", "File"):
        system = systems.get(system_name) if isinstance(systems, Mapping) else None
        _require(isinstance(system, Mapping), f"{system_name} plan is missing", errors)
        variants = system.get("variants") if isinstance(system, Mapping) else None
        _require(
            isinstance(variants, Mapping)
            and set(variants) == set(PLACEMENT_VARIANTS),
            f"{system_name} must contain all placement variants",
            errors,
        )
        original_route = set(system.get("original_route") or []) if isinstance(system, Mapping) else set()
        for variant, moved_stages in PLACEMENT_VARIANTS.items():
            entry = variants.get(variant) if isinstance(variants, Mapping) else None
            selected = entry.get("selected") if isinstance(entry, Mapping) else None
            if not isinstance(selected, Mapping):
                errors.append(f"{system_name}/{variant} selected placement is missing")
                continue
            counts[system_name] = counts.get(system_name, 0) + 1
            for stage in ("inertia", "payload"):
                displacement = int(
                    selected.get(f"{stage}_displacement_hops") or 0
                )
                node = selected.get(f"{stage}_node")
                if stage in moved_stages:
                    _require(
                        1 <= displacement <= 2,
                        f"{system_name}/{variant}/{stage} displacement is not 1--2",
                        errors,
                    )
                    _require(
                        node not in original_route,
                        f"{system_name}/{variant}/{stage} remains on original route",
                        errors,
                    )
                else:
                    _require(
                        displacement == 0,
                        f"{system_name}/{variant}/{stage} changed unexpectedly",
                        errors,
                    )
    return {"selected_variant_counts": counts}


def _audit_coverage(repo_root: Path, errors: list[str]) -> dict[str, Any]:
    primary_root = (
        repo_root / "Rebuttal/results/node_frequency_theoretical_max"
    )
    primary_path = primary_root / "theoretical_max.json"
    primary_table_path = primary_root / "table.md"
    coverage_table_path = repo_root / "Rebuttal/results/coverage/table.md"
    primary_payload = _load_json(primary_path)
    primary_rows = primary_payload.get("rows")
    _require(
        primary_payload.get("analysis")
        == "column-wise theoretical target-allocation upper bounds"
        and primary_payload.get("empirical_result") is False,
        "primary coverage result must be the labeled theoretical upper bound",
        errors,
    )
    _require(
        isinstance(primary_rows, list) and len(primary_rows) == 4,
        "primary coverage result must have four rows",
        errors,
    )
    primary_trajectory_counts: dict[str, int] = {}
    primary_candidate_counts: list[int] = []
    for row in primary_rows or []:
        if not isinstance(row, Mapping):
            errors.append("primary coverage row is not an object")
            continue
        system = str(row.get("system") or "")
        trajectories = int(row.get("trajectories") or 0)
        expected_trajectories = 72 if system == "Browser" else 60
        _require(
            trajectories == expected_trajectories,
            f"{system} primary coverage trajectory count is incorrect",
            errors,
        )
        primary_trajectory_counts[system] = trajectories
        primary_candidate_counts.append(
            int(row.get("candidate_count") or 0)
        )
        _require(
            row.get("top1_er_upper_bound") == 1.0
            and row.get("top2_er_upper_bound") == 1.0
            and row.get("top3_er_upper_bound") == 1.0
            and isinstance(
                row.get("random1_expected_encounters"), (int, float)
            )
            and isinstance(
                row.get("nodes_er_ge_10_ratio_upper_bound"), (int, float)
            )
            and isinstance(
                row.get("nodes_er_ge_30_ratio_upper_bound"), (int, float)
            ),
            f"{system} primary coverage row is missing an upper-bound metric",
            errors,
        )
    _require(
        primary_candidate_counts == [1022, 1022, 2608, 870],
        "primary coverage candidate counts do not match the locked user trees",
        errors,
    )
    frontier_payload = _load_json(primary_root / "pareto_frontiers.json")
    frontiers = frontier_payload.get("frontiers")
    _require(
        isinstance(frontiers, list)
        and len(frontiers) == 4
        and all(
            isinstance(row, Mapping)
            and row.get("independent_maxima_jointly_attainable") is False
            for row in frontiers
        ),
        "primary coverage Pareto frontiers must retain the maxima tradeoff",
        errors,
    )
    for required in (
        primary_table_path,
        primary_root / "target_allocations.json",
        primary_root / "pareto_frontiers.json",
        primary_root / "README.md",
        repo_root / "Rebuttal/results/coverage/README.md",
        coverage_table_path,
    ):
        _require(
            required.is_file(),
            f"primary coverage artifact is missing: {required}",
            errors,
        )
    _require(
        _sha256(primary_table_path) == _sha256(coverage_table_path),
        "coverage index table does not match the primary upper-bound table",
        errors,
    )

    # The measured 16-target rerun remains a required supplemental result.
    result_root = repo_root / "Rebuttal/results/node_frequency_rerun"
    run_root = repo_root / "Rebuttal/runs/node_frequency_rerun"
    result_path = result_root / "node_frequency.json"
    table_path = result_root / "table.md"
    validation_path = result_root / "validation.json"
    protocol_path = run_root / "protocol_manifest.json"

    payload = _load_json(result_path)
    validation = _load_json(validation_path)
    run_protocol = _load_json(protocol_path)
    protocol = payload.get("protocol")
    rows = payload.get("rows")
    _require(
        payload.get("experiment") == "EXP-COVER-001-RERUN",
        "coverage result must be the fresh EXP-COVER-001 rerun",
        errors,
    )
    _require(isinstance(protocol, Mapping), "coverage protocol is missing", errors)
    _require(
        protocol.get("node_counting") == "one visit per node per trajectory"
        if isinstance(protocol, Mapping)
        else False,
        "coverage must count each node once per trajectory",
        errors,
    )
    _require(
        isinstance(protocol, Mapping)
        and protocol.get("condition") == "clean_no_attack"
        and protocol.get("fresh_trajectories") is True
        and int(protocol.get("samples_per_system") or 0) == 16,
        "coverage must use 16 fresh clean trajectories per system",
        errors,
    )
    _require(
        isinstance(protocol, Mapping)
        and str(protocol.get("root") or "").startswith("excluded")
        and protocol.get("target_nodes") == "included",
        "coverage must exclude only the mandatory root and retain target nodes",
        errors,
    )
    _require(
        isinstance(protocol, Mapping)
        and protocol.get("scope")
        == "user-tree nodes only in numerator and denominator",
        "coverage numerator and denominator must contain only user-tree nodes",
        errors,
    )
    _require(
        isinstance(rows, list) and len(rows) == 4,
        "coverage must have four rows",
        errors,
    )
    trajectory_counts: dict[str, int] = {}
    candidate_counts: list[int] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            errors.append("coverage row is not an object")
            continue
        system = str(row.get("system") or "")
        trajectories = int(row.get("trajectories") or 0)
        _require(
            trajectories == 16,
            f"{system} coverage row must contain 16 trajectories",
            errors,
        )
        trajectory_counts[system] = trajectories
        candidate_counts.append(int(row.get("candidate_count") or 0))
        _require(
            isinstance(row.get("top1_er"), (int, float))
            and isinstance(row.get("random1_er"), (int, float))
            and isinstance(row.get("nodes_er_ge_10_ratio"), (int, float))
            and isinstance(row.get("nodes_er_ge_30_ratio"), (int, float)),
            f"{system} coverage row is missing a requested metric",
            errors,
        )
    _require(
        candidate_counts == [1022, 1022, 2608, 870],
        "coverage candidate counts do not match the locked user trees",
        errors,
    )

    systems = run_protocol.get("systems") if isinstance(run_protocol, Mapping) else None
    _require(
        isinstance(systems, Mapping)
        and all(
            int((systems.get(system) or {}).get("sample_count") or 0) == 16
            for system in ("Browser", "File")
        ),
        "coverage run protocol must contain 16 selected targets per system",
        errors,
    )
    checks = validation.get("checks") if isinstance(validation, Mapping) else None
    _require(
        isinstance(checks, Mapping)
        and int(checks.get("browser_trace_count") or 0) == 16
        and int(checks.get("file_trace_count") or 0) == 16
        and checks.get("browser_all_nonempty") is True
        and checks.get("file_all_nonempty") is True,
        "coverage validation must confirm 16 non-empty traces per system",
        errors,
    )
    _require(
        validation.get("result_sha256") == _sha256(result_path),
        "coverage result hash does not match validation.json",
        errors,
    )
    _require(
        validation.get("table_sha256") == _sha256(table_path),
        "coverage table hash does not match validation.json",
        errors,
    )
    for required in (
        result_root / "node_frequency.csv",
        result_root / "node_rates.csv",
        table_path,
        validation_path,
    ):
        _require(required.is_file(), f"coverage artifact is missing: {required}", errors)

    raw_trace_counts = {
        "Browser": sum(
            path.is_file() and path.stat().st_size > 0
            for path in (run_root / "browser/logs").glob("trace_*.jsonl")
        ),
        "File": sum(
            path.is_file() and path.stat().st_size > 0
            for path in (run_root / "file/logs").glob("trace_case_*.jsonl")
        ),
    }
    _require(
        raw_trace_counts == {"Browser": 16, "File": 16},
        "canonical raw coverage traces must contain 16 non-empty files per system",
        errors,
    )
    return {
        "primary": {
            "analysis": primary_payload.get("analysis"),
            "empirical_result": primary_payload.get("empirical_result"),
            "trajectory_counts": primary_trajectory_counts,
            "candidate_counts": primary_candidate_counts,
            "table_sha256": _sha256(primary_table_path),
        },
        "supplemental_16_target": {
            "trajectory_counts": trajectory_counts,
            "raw_trace_counts": raw_trace_counts,
            "candidate_counts": candidate_counts,
            "result_sha256": validation.get("result_sha256"),
            "table_sha256": validation.get("table_sha256"),
        },
        "trajectory_counts": trajectory_counts,
        "raw_trace_counts": raw_trace_counts,
        "candidate_counts": candidate_counts,
        "result_sha256": validation.get("result_sha256"),
        "table_sha256": validation.get("table_sha256"),
    }


def _audit_materialization(repo_root: Path, errors: list[str]) -> dict[str, Any]:
    root = repo_root / "Rebuttal/results/materialized_envs"
    browser_path = root / "browser_formal_envs.json"
    file_path = root / "file_formal_envs.json"
    readiness_path = root / "readiness_summary.json"
    present = [path.is_file() for path in (browser_path, file_path, readiness_path)]

    _require(
        "force_checker" in inspect.signature(generate_browser_text).parameters,
        "Browser checked generator does not expose the formal acceptance checker",
        errors,
    )
    _require(
        "force_checker" in inspect.signature(generate_file_text).parameters,
        "File checked generator does not expose the formal acceptance checker",
        errors,
    )
    _require(
        set(FORMAL_VARIANTS)
        == {*PLACEMENT_VARIANTS, *SINGLE_STAGE_VARIANTS},
        "formal variant registry is incomplete",
        errors,
    )

    if not any(present):
        return {
            "status": "not_materialized",
            "formal_variants": list(FORMAL_VARIANTS),
            "note": (
                "Generated placeholder environments are intentionally not retained; "
                "run materialize_formal_envs.py before single-pass generation."
            ),
        }
    _require(
        all(present),
        "formal environment metadata is only partially present",
        errors,
    )
    if not all(present):
        return {"status": "partial_metadata"}

    browser = _load_json(browser_path)
    file_payload = _load_json(file_path)
    readiness = _load_json(readiness_path)
    browser_rows = browser.get("environments") or []
    file_rows = file_payload.get("environments") or []

    browser_counts: dict[str, int] = {}
    for row in browser_rows:
        if not isinstance(row, Mapping):
            continue
        variant = str(row.get("variant") or "")
        browser_counts[variant] = browser_counts.get(variant, 0) + 1
        if variant in SINGLE_STAGE_VARIANTS:
            manifest = _load_json(
                Path(str(row["environment_root"])) / "environment_manifest.json"
            )
            _require(
                manifest.get("suite") == "gitlab",
                "Browser single-stage scope includes a non-GitLab suite",
                errors,
            )
    file_counts: dict[str, int] = {}
    for row in file_rows:
        if isinstance(row, Mapping):
            variant = str(row.get("variant") or "")
            file_counts[variant] = file_counts.get(variant, 0) + 1

    expected = {*PLACEMENT_VARIANTS, *SINGLE_STAGE_VARIANTS}
    _require(set(browser_counts) == expected, "Browser formal variants are incomplete", errors)
    _require(set(file_counts) == expected, "File formal variants are incomplete", errors)
    _require(
        all(file_counts.get(variant) == 1 for variant in expected),
        "File must have one run directory per formal variant",
        errors,
    )
    ready_count = int(readiness.get("ready_environment_count") or 0)
    blocked_count = int(readiness.get("blocked_environment_count") or 0)
    _require(
        ready_count + blocked_count == len(browser_rows) + len(file_rows),
        "readiness counts do not cover every materialized environment",
        errors,
    )
    return {
        "status": "materialized",
        "browser_environment_counts": browser_counts,
        "file_run_counts": file_counts,
        "ready_environment_count": ready_count,
        "blocked_environment_count": blocked_count,
    }


def _audit_summary(repo_root: Path, errors: list[str]) -> dict[str, Any]:
    path = repo_root / "Rebuttal/results/summary/rebuttal_metrics.json"
    if not path.is_file():
        return {
            "status": "not_generated",
            "note": "The formal summary is generated only after new formal runs.",
        }
    payload = _load_json(path)
    rows = payload.get("rows") or []
    planned: dict[tuple[str, str], str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        variant = str(row.get("variant") or "")
        system = str(row.get("system") or "")
        if variant in {*PLACEMENT_VARIANTS, *SINGLE_STAGE_VARIANTS}:
            planned[(system, variant)] = str(row.get("status") or "")
    _require(len(planned) == 12, "summary must contain 12 formal variant/system rows", errors)
    allowed = {
        "completed_contract_valid",
        "non_result_incomplete_or_contract_mismatch",
    }
    _require(
        all(status in allowed for status in planned.values()),
        "formal summary contains a stale, smoke, or unvalidated status",
        errors,
    )
    return {
        "status": "generated",
        "formal_rows": len(planned),
        "contract_valid_rows": sum(
            status == "completed_contract_valid" for status in planned.values()
        ),
        "non_result_rows": sum(
            status == "non_result_incomplete_or_contract_mismatch"
            for status in planned.values()
        ),
    }


def audit(repo_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    sections = {
        "placements": _audit_placements(repo_root, errors),
        "coverage": _audit_coverage(repo_root, errors),
        "materialization": _audit_materialization(repo_root, errors),
        "summary": _audit_summary(repo_root, errors),
    }
    return {
        "schema_version": 1,
        "passed": not errors,
        "privacy_boundary": (
            "aggregate metadata, hashes, and file sizes only; no sample text, "
            "injection text, or trace content"
        ),
        "errors": errors,
        "sections": sections,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. Without it, print the audit to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    result = audit(repo_root)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        output = args.out if args.out.is_absolute() else repo_root / args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {output}")
    if not result["passed"]:
        raise SystemExit("Rebuttal implementation audit failed.")


if __name__ == "__main__":
    main()
