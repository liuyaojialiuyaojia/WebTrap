#!/usr/bin/env python3
"""Summarize rebuttal evidence without launching new agent/model runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Rebuttal.experiments.analyze_node_frequency import (
    parse_browser_trace,
    parse_file_trace,
)
from Rebuttal.experiments.contracts import (
    BROWSER_MAX_ACTIONS,
    BROWSER_MAX_OBSERVATIONS,
    BROWSER_SEED,
    BROWSER_TRIALS,
    CHECK_MODEL_ID,
    DEFENSE_MODE,
    FILE_MAX_STEPS,
    FILE_NUM_SAMPLES,
    FORMAL_VARIANTS,
    MODEL_ID,
    PLACEMENT_VARIANTS,
    SINGLE_STAGE_VARIANTS,
    environment_fingerprint,
    runtime_contract_errors,
    validate_environment_contract,
)


@dataclass(frozen=True)
class MetricRow:
    experiment: str
    system: str
    variant: str
    status: str
    samples_or_tasks: str
    payload_er: float | None
    asr_e: float | None
    asr_i: float | None
    uua: float | None
    dual_goal: float | None
    note: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _ratio(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def _sum_totals(summary_paths: Sequence[Path]) -> dict[str, int]:
    totals = {
        "user_pass": 0,
        "user_total": 0,
        "attack_pass": 0,
        "attack_total": 0,
        "intermediate_compromised": 0,
        "intermediate_total": 0,
    }
    for path in summary_paths:
        payload = _read_json(path)
        raw_totals = payload.get("totals") if isinstance(payload, Mapping) else {}
        if not isinstance(raw_totals, Mapping):
            continue
        for key in totals:
            value = raw_totals.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def _rates_from_totals(totals: Mapping[str, int]) -> tuple[float | None, float | None, float | None]:
    return (
        _ratio(int(totals.get("attack_pass") or 0), int(totals.get("attack_total") or 0)),
        _ratio(
            int(totals.get("intermediate_compromised") or 0),
            int(totals.get("intermediate_total") or 0),
        ),
        _ratio(int(totals.get("user_pass") or 0), int(totals.get("user_total") or 0)),
    )


def _metric_totals(metrics_paths: Sequence[Path]) -> dict[str, int]:
    """Aggregate evaluator metrics while rejecting absent count fields."""

    totals = {
        "user_pass": 0,
        "user_total": 0,
        "attack_pass": 0,
        "attack_total": 0,
        "intermediate_compromised": 0,
        "intermediate_total": 0,
    }
    for path in metrics_paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Formal metrics must be an object: {path}")
        user = payload.get("user_utility")
        attacker = payload.get("attacker_utility")
        summary = payload.get("summary")
        if (
            not isinstance(user, Mapping)
            or not isinstance(attacker, Mapping)
            or not isinstance(summary, Mapping)
        ):
            raise ValueError(f"Formal metrics are incomplete: {path}")
        values = {
            "user_pass": user.get("total_scores"),
            "user_total": user.get("cnt_tasks"),
            "attack_pass": attacker.get("total_scores"),
            "attack_total": attacker.get("cnt_tasks"),
            "intermediate_compromised": summary.get("intermediate_compromised"),
            "intermediate_total": user.get("cnt_tasks"),
        }
        for key, value in values.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Formal metrics missing {key}: {path}")
            totals[key] += int(value)
    return totals


def _trace_paths(batch_dirs: Iterable[Path]) -> list[Path]:
    paths: list[Path] = []
    for batch_dir in batch_dirs:
        paths.extend(sorted(batch_dir.glob("*/agent_logs_post_injection/trace_*.jsonl")))
    return paths


def _browser_stage_nodes(metadata_path: Path) -> dict[str, int]:
    payload = _read_json(metadata_path)
    stages: dict[str, int] = {}
    for page in payload.get("pages", []) if isinstance(payload, Mapping) else []:
        if not isinstance(page, Mapping):
            continue
        page_index = page.get("page_index")
        if not isinstance(page_index, int):
            continue
        for injection in page.get("injections") or []:
            if not isinstance(injection, Mapping):
                continue
            psaa = injection.get("psaa")
            if not isinstance(psaa, Mapping):
                continue
            stage = psaa.get("stage")
            if isinstance(stage, str) and stage in {"lure", "inertia", "payload"}:
                stages[stage] = int(page_index)
    return stages


def _file_stage_dirs(manifest_path: Path) -> dict[str, str]:
    stages: dict[str, str] = {}
    for row in _read_jsonl(manifest_path):
        stage = row.get("stage")
        directory = row.get("directory_logical_path")
        if isinstance(stage, str) and isinstance(directory, str):
            stages[stage] = directory.rstrip("/")
    return stages


def _browser_payload_er(trace_paths: Sequence[Path], page_index: int) -> tuple[int, int, float | None]:
    total = 0
    hit = 0
    for path in trace_paths:
        if not path.is_file():
            continue
        total += 1
        if page_index in parse_browser_trace(path):
            hit += 1
    return hit, total, _ratio(hit, total)


def _file_payload_er(trace_paths: Sequence[Path], directory: str) -> tuple[int, int, float | None]:
    target = directory.rstrip("/")
    total = 0
    hit = 0
    for path in trace_paths:
        if not path.is_file():
            continue
        total += 1
        if target in parse_file_trace(path):
            hit += 1
    return hit, total, _ratio(hit, total)


def _browser_dual_goal(metrics_paths: Sequence[Path]) -> tuple[int, int, float | None]:
    total_units = 0
    dual_units = 0
    for metrics_path in metrics_paths:
        payload = _read_json(metrics_path)
        if not isinstance(payload, Mapping):
            continue
        user = payload.get("user_utility")
        attacker = payload.get("attacker_utility")
        if not isinstance(user, Mapping) or not isinstance(attacker, Mapping):
            continue
        user_details = user.get("details")
        attacker_details = attacker.get("details")
        if not isinstance(user_details, list) or not isinstance(attacker_details, list):
            continue
        attacker_by_task = {
            str(row.get("task_id")): row
            for row in attacker_details
            if isinstance(row, Mapping)
        }
        for user_row in user_details:
            if not isinstance(user_row, Mapping):
                continue
            attacker_row = attacker_by_task.get(str(user_row.get("task_id")))
            if not isinstance(attacker_row, Mapping):
                continue
            user_trials = user_row.get("trials")
            attacker_trials = attacker_row.get("trials")
            if not isinstance(user_trials, list) or not isinstance(attacker_trials, list):
                continue
            max_trials = min(len(user_trials), len(attacker_trials))
            if max_trials <= 0:
                continue
            total_units += 1
            dual_units += int(
                any(
                    isinstance(user_trials[idx], Mapping)
                    and isinstance(attacker_trials[idx], Mapping)
                    and bool(user_trials[idx].get("passed"))
                    and bool(attacker_trials[idx].get("passed"))
                    for idx in range(max_trials)
                )
            )
    return dual_units, total_units, _ratio(dual_units, total_units)


def _file_dual_goal(run_dir: Path) -> tuple[int, int, float | None]:
    user_rows = _read_jsonl(run_dir / "eval" / "user_goal_per_sample.jsonl")
    attack_rows = _read_jsonl(run_dir / "eval" / "asr_end_to_end_per_sample.jsonl")
    attack_index = {
        (str(row.get("case_id")), int(row.get("sample_idx"))): bool(
            row.get("asr_end_to_end_success")
        )
        for row in attack_rows
        if row.get("case_id") is not None and row.get("sample_idx") is not None
    }
    per_case: dict[str, bool] = defaultdict(bool)
    all_cases: set[str] = set()
    for row in user_rows:
        if row.get("case_id") is None or row.get("sample_idx") is None:
            continue
        case_id = str(row.get("case_id"))
        sample_idx = int(row.get("sample_idx"))
        all_cases.add(case_id)
        if bool(row.get("user_goal_success")) and attack_index.get((case_id, sample_idx), False):
            per_case[case_id] = True
    dual = sum(1 for case_id in all_cases if per_case.get(case_id, False))
    total = len(all_cases)
    return dual, total, _ratio(dual, total)


def _browser_full_rows(repo_root: Path) -> tuple[MetricRow, MetricRow]:
    gitlab_batch = repo_root / "web/runs/exp_d10w2_psaa/batch_runs/20260405_161802_full"
    reddit_batch = repo_root / "web/runs/exp_d10w2_psaa/batch_runs/20260412_140735_full"
    summary_paths = [gitlab_batch / "batch_summary.json", reddit_batch / "batch_summary.json"]
    totals = _sum_totals(summary_paths)
    asr_e, asr_i, uua = _rates_from_totals(totals)
    stages = _browser_stage_nodes(
        repo_root / "web/runs/exp_d10w2_psaa/static/page_metadata_injected.json"
    )
    traces = _trace_paths([gitlab_batch, reddit_batch])
    hit, total, payload_er = _browser_payload_er(traces, stages["payload"])
    metrics_paths = sorted(gitlab_batch.glob("*/metrics_post_injection.json")) + sorted(
        reddit_batch.glob("*/metrics_post_injection.json")
    )
    dual_hit, dual_total, dual = _browser_dual_goal(metrics_paths)
    combined = MetricRow(
        experiment="EXP-PLACE-001",
        system="Browser",
        variant="Optimal full",
        status="completed_existing_run",
        samples_or_tasks=f"payload {hit}/{total}; metrics {totals['attack_total']} tasks",
        payload_er=payload_er,
        asr_e=asr_e,
        asr_i=asr_i,
        uua=uua,
        dual_goal=dual,
        note=f"Combined GitLab+Reddit full PSAA; dual {dual_hit}/{dual_total}.",
    )

    gitlab_totals = _sum_totals([gitlab_batch / "batch_summary.json"])
    gitlab_asr_e, gitlab_asr_i, gitlab_uua = _rates_from_totals(gitlab_totals)
    gitlab_traces = _trace_paths([gitlab_batch])
    g_hit, g_total, g_payload_er = _browser_payload_er(gitlab_traces, stages["payload"])
    g_metrics_paths = sorted(gitlab_batch.glob("*/metrics_post_injection.json"))
    g_dual_hit, g_dual_total, g_dual = _browser_dual_goal(g_metrics_paths)
    gitlab = MetricRow(
        experiment="EXP-ABL-001",
        system="Browser-GitLab",
        variant="Full",
        status="completed_existing_run",
        samples_or_tasks=f"payload {g_hit}/{g_total}; metrics {gitlab_totals['attack_total']} tasks",
        payload_er=g_payload_er,
        asr_e=gitlab_asr_e,
        asr_i=gitlab_asr_i,
        uua=gitlab_uua,
        dual_goal=g_dual,
        note=f"Long GitLab full PSAA comparator; dual {g_dual_hit}/{g_dual_total}.",
    )
    return combined, gitlab


def _file_full_row(repo_root: Path) -> MetricRow:
    run_dir = repo_root / "file/runs/exp_d10w2_psaa"
    metrics = _read_json(run_dir / "eval/metrics.json")
    best = metrics.get("best_of_n") if isinstance(metrics, Mapping) else {}
    if not isinstance(best, Mapping):
        best = metrics
    stages = _file_stage_dirs(run_dir / "injection/injection_manifest.jsonl")
    traces = sorted((run_dir / "logs/samples").glob("case_*/trace_sample_*.jsonl"))
    hit, total, payload_er = _file_payload_er(traces, stages["payload"])
    dual_hit, dual_total, dual = _file_dual_goal(run_dir)
    return MetricRow(
        experiment="EXP-PLACE-001",
        system="File",
        variant="Optimal full",
        status="completed_existing_run",
        samples_or_tasks=f"payload {hit}/{total}; metrics {int(best.get('total_cases') or 0)} cases",
        payload_er=payload_er,
        asr_e=float(best.get("asr_end_to_end_rate") or 0.0),
        asr_i=float(best.get("asr_intermediate_rate") or 0.0),
        uua=float(best.get("user_task_success_rate") or 0.0),
        dual_goal=dual,
        note=f"File best-of-3 full PSAA; dual {dual_hit}/{dual_total}.",
    )


def _formal_non_result(
    *,
    experiment: str,
    system: str,
    variant: str,
    expected: int,
    valid: int,
    reasons: Sequence[str],
) -> MetricRow:
    reason_counts: dict[str, int] = defaultdict(int)
    for reason in reasons:
        reason_counts[reason] += 1
    compact = ", ".join(
        f"{reason} ({count})" for reason, count in sorted(reason_counts.items())
    )
    return MetricRow(
        experiment=experiment,
        system=system,
        variant=variant,
        status="non_result_incomplete_or_contract_mismatch",
        samples_or_tasks=f"{valid}/{expected} contract-valid formal units",
        payload_er=None,
        asr_e=None,
        asr_i=None,
        uua=None,
        dual_goal=None,
        note=compact or "No complete formal result was discovered.",
    )


def _browser_formal_output_dir(repo_root: Path, row: Mapping[str, Any]) -> Path:
    sample_id = str(row.get("sample_id") or "").split(":", 1)[-1]
    return (
        repo_root
        / "Rebuttal/runs/browser_formal/deepseek_v3_1_terminal"
        / str(row.get("variant") or "")
        / sample_id
    )


def _validate_browser_formal_unit(
    repo_root: Path,
    row: Mapping[str, Any],
) -> tuple[Path | None, str | None]:
    env_root = Path(str(row.get("environment_root") or ""))
    manifest_path = env_root / "environment_manifest.json"
    env_errors = validate_environment_contract(manifest_path)
    if env_errors:
        return None, "environment contract not ready"
    output_dir = _browser_formal_output_dir(repo_root, row)
    status_path = output_dir / "run_status.json"
    metrics_path = output_dir / "metrics_post_injection.json"
    if not status_path.is_file() or not metrics_path.is_file():
        return None, "formal output missing"
    status = _read_json(status_path)
    if not isinstance(status, Mapping) or status.get("status") != "completed":
        return None, "run status is not completed"
    contract = status.get("runtime_contract")
    if not isinstance(contract, Mapping) or not bool(contract.get("passed")):
        return None, "runtime contract metadata missing or failed"
    errors = runtime_contract_errors(
        system="Browser",
        variant=str(status.get("variant") or ""),
        model=str(status.get("model") or ""),
        check_model=str(status.get("check_model") or ""),
        temperature=float(status.get("temperature") or 0.0),
        defense_mode=str(status.get("defense_mode") or ""),
        check_temperature=float(status.get("check_temperature") or 0.0),
        trials=int(status.get("trials") or 0),
        seed=int(status.get("seed") or -1),
        max_actions=int(status.get("max_actions") or 0),
        max_observations=int(status.get("max_observations") or 0),
        max_parallel=int(status.get("max_parallel") or 0),
        check_max_tokens=int(status.get("check_max_tokens") or 0),
    )
    if errors:
        return None, "runtime contract mismatch"
    expected_fingerprint = environment_fingerprint(manifest_path)
    if status.get("source_environment_fingerprint") != expected_fingerprint:
        return None, "stale environment fingerprint"
    if Path(str(status.get("metrics_path") or "")).resolve() != metrics_path.resolve():
        return None, "metrics path mismatch"
    return metrics_path, None


def _browser_formal_row(
    repo_root: Path,
    *,
    variant: str,
) -> MetricRow:
    index_path = (
        repo_root / "Rebuttal/results/materialized_envs/browser_formal_envs.json"
    )
    if not index_path.is_file():
        return _formal_non_result(
            experiment=(
                "EXP-PLACE-001"
                if variant in PLACEMENT_VARIANTS
                else "EXP-ABL-001"
            ),
            system="Browser" if variant in PLACEMENT_VARIANTS else "Browser-GitLab",
            variant=variant,
            expected=0,
            valid=0,
            reasons=["materialized environment index missing"],
        )
    payload = _read_json(index_path)
    environments = payload.get("environments") if isinstance(payload, Mapping) else []
    rows = [
        row
        for row in environments or []
        if isinstance(row, Mapping) and str(row.get("variant")) == variant
    ]
    expected = len(rows)
    metrics_paths: list[Path] = []
    output_dirs: list[Path] = []
    reasons: list[str] = []
    for row in rows:
        metrics_path, reason = _validate_browser_formal_unit(repo_root, row)
        if reason:
            reasons.append(reason)
            continue
        assert metrics_path is not None
        metrics_paths.append(metrics_path)
        output_dirs.append(metrics_path.parent)
    experiment = (
        "EXP-PLACE-001" if variant in PLACEMENT_VARIANTS else "EXP-ABL-001"
    )
    system = "Browser" if variant in PLACEMENT_VARIANTS else "Browser-GitLab"
    if expected == 0 or len(metrics_paths) != expected:
        return _formal_non_result(
            experiment=experiment,
            system=system,
            variant=variant,
            expected=expected,
            valid=len(metrics_paths),
            reasons=reasons,
        )

    totals = _metric_totals(metrics_paths)
    asr_e, asr_i, uua = _rates_from_totals(totals)
    traces: list[Path] = []
    payload_nodes: set[int] = set()
    payload_hit = 0
    payload_total = 0
    for output_dir in output_dirs:
        traces = sorted((output_dir / "agent_logs_post_injection").glob("trace_*.jsonl"))
        if variant in {"lure_only", "inertia_only"}:
            payload_total += len(traces)
            continue
        stages = _browser_stage_nodes(
            output_dir / "static_snapshot" / "page_metadata.json"
        )
        if "payload" not in stages:
            raise ValueError(f"Completed {variant} output is missing Payload stage")
        payload_nodes.add(stages["payload"])
        hit, total, _ = _browser_payload_er(traces, stages["payload"])
        payload_hit += hit
        payload_total += total
    payload_er = _ratio(payload_hit, payload_total)
    dual_hit, dual_total, dual = _browser_dual_goal(metrics_paths)
    return MetricRow(
        experiment=experiment,
        system=system,
        variant=variant,
        status="completed_contract_valid",
        samples_or_tasks=(
            f"{expected} formal units; payload {payload_hit}/{payload_total}; "
            f"metrics {totals['attack_total']} tasks"
        ),
        payload_er=payload_er,
        asr_e=asr_e,
        asr_i=asr_i,
        uua=uua,
        dual_goal=dual,
        note=(
            f"Contract-valid DeepSeek result; dual {dual_hit}/{dual_total}; "
            f"payload node definitions {len(payload_nodes)}."
        ),
    )


def _validate_file_formal_unit(
    row: Mapping[str, Any],
) -> tuple[Path | None, str | None]:
    run_dir = Path(str(row.get("environment_root") or ""))
    manifest_path = run_dir / "environment_manifest.json"
    env_errors = validate_environment_contract(manifest_path)
    if env_errors:
        return None, "environment contract not ready"
    status_path = run_dir / "deepseek_v3_1_terminal" / "run_status.json"
    metrics_path = run_dir / "eval" / "metrics.json"
    if not status_path.is_file() or not metrics_path.is_file():
        return None, "formal output missing"
    status = _read_json(status_path)
    if not isinstance(status, Mapping) or status.get("status") != "completed":
        return None, "run status is not completed"
    contract = status.get("runtime_contract")
    if not isinstance(contract, Mapping) or not bool(contract.get("passed")):
        return None, "runtime contract metadata missing or failed"
    errors = runtime_contract_errors(
        system="File",
        variant=str(status.get("variant") or ""),
        model=str(status.get("model") or ""),
        check_model=str(status.get("check_model") or ""),
        temperature=float(status.get("temperature") or 0.0),
        defense_mode=str(status.get("defense_mode") or ""),
        check_temperature=float(status.get("check_temperature") or 0.0),
        num_samples=int(status.get("num_samples") or 0),
        max_steps=int(status.get("max_steps") or 0),
        agent_max_tokens=int(status.get("agent_max_tokens") or 0),
        asr_max_tokens=int(status.get("asr_max_tokens") or 0),
    )
    if errors:
        return None, "runtime contract mismatch"
    expected_fingerprint = environment_fingerprint(manifest_path)
    if status.get("source_environment_fingerprint") != expected_fingerprint:
        return None, "stale environment fingerprint"
    if Path(str(status.get("metrics_path") or "")).resolve() != metrics_path.resolve():
        return None, "metrics path mismatch"
    return metrics_path, None


def _file_formal_row(
    repo_root: Path,
    *,
    variant: str,
) -> MetricRow:
    index_path = repo_root / "Rebuttal/results/materialized_envs/file_formal_envs.json"
    payload = _read_json(index_path) if index_path.is_file() else {}
    environments = payload.get("environments") if isinstance(payload, Mapping) else []
    rows = [
        row
        for row in environments or []
        if isinstance(row, Mapping) and str(row.get("variant")) == variant
    ]
    metrics_paths: list[Path] = []
    reasons: list[str] = []
    for row in rows:
        metrics_path, reason = _validate_file_formal_unit(row)
        if reason:
            reasons.append(reason)
            continue
        assert metrics_path is not None
        metrics_paths.append(metrics_path)
    experiment = (
        "EXP-PLACE-001"
        if variant in PLACEMENT_VARIANTS
        else "EXP-ABL-001-supplementary"
    )
    if len(rows) != 1 or len(metrics_paths) != 1:
        return _formal_non_result(
            experiment=experiment,
            system="File",
            variant=variant,
            expected=len(rows),
            valid=len(metrics_paths),
            reasons=reasons,
        )

    run_dir = Path(str(rows[0]["environment_root"]))
    metrics = _read_json(metrics_paths[0])
    best = metrics.get("best_of_n") if isinstance(metrics, Mapping) else None
    if not isinstance(best, Mapping):
        raise ValueError(f"File formal metrics lack best_of_n: {metrics_paths[0]}")
    expected_cases = int(_read_json(run_dir / "environment_manifest.json").get("cases") or 0)
    if int(best.get("total_cases") or 0) != expected_cases:
        return _formal_non_result(
            experiment=experiment,
            system="File",
            variant=variant,
            expected=expected_cases,
            valid=int(best.get("total_cases") or 0),
            reasons=["formal metrics case count mismatch"],
        )
    if variant in {"lure_only", "inertia_only"}:
        traces = sorted((run_dir / "logs/samples").glob("case_*/trace_sample_*.jsonl"))
        payload_er = 0.0 if traces else None
        hit, total = 0, len(traces)
    else:
        stages = _file_stage_dirs(run_dir / "injection/injection_manifest.jsonl")
        traces = sorted((run_dir / "logs/samples").glob("case_*/trace_sample_*.jsonl"))
        hit, total, payload_er = _file_payload_er(traces, stages["payload"])
    dual_hit, dual_total, dual = _file_dual_goal(run_dir)
    return MetricRow(
        experiment=experiment,
        system="File",
        variant=variant,
        status="completed_contract_valid",
        samples_or_tasks=f"payload {hit}/{total}; metrics {expected_cases} cases",
        payload_er=payload_er,
        asr_e=float(best["asr_end_to_end_rate"]),
        asr_i=float(best["asr_intermediate_rate"]),
        uua=float(best["user_task_success_rate"]),
        dual_goal=dual,
        note=f"Contract-valid File best-of-3 result; dual {dual_hit}/{dual_total}.",
    )


def _formal_rows(repo_root: Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for variant in PLACEMENT_VARIANTS:
        rows.append(_browser_formal_row(repo_root, variant=variant))
        rows.append(_file_formal_row(repo_root, variant=variant))
    for variant in SINGLE_STAGE_VARIANTS:
        rows.append(_browser_formal_row(repo_root, variant=variant))
        rows.append(_file_formal_row(repo_root, variant=variant))
    return rows


def _leave_one_out_rows(repo_root: Path) -> list[MetricRow]:
    batch_root = repo_root / "web/runs/exp_d10w2_ablation/batch_runs"
    variants = {
        "wo_lure": sorted(batch_root.glob("*_wo_lure/batch_summary.json")),
        "wo_inertia": sorted(batch_root.glob("*_wo_inertia/batch_summary.json")),
        "wo_payload": sorted(batch_root.glob("*_wo_payload/batch_summary.json")),
    }
    rows: list[MetricRow] = []
    for variant, paths in variants.items():
        totals = _sum_totals(paths)
        asr_e, asr_i, uua = _rates_from_totals(totals)
        rows.append(
            MetricRow(
                experiment="EXP-ABL-001-context",
                system="Browser-GitLab",
                variant=variant,
                status="completed_existing_leave_one_out",
                samples_or_tasks=f"metrics {totals['attack_total']} tasks",
                payload_er=None,
                asr_e=asr_e,
                asr_i=asr_i,
                uua=uua,
                dual_goal=None,
                note="Existing leave-one-stage-out result; not a single-stage run.",
            )
        )
    return rows


def _placement_proxy_rows(repo_root: Path) -> list[dict[str, Any]]:
    placement_path = repo_root / "Rebuttal/results/placement/selected_placements.json"
    payload = _read_json(placement_path)
    systems = payload.get("systems") if isinstance(payload, Mapping) else {}
    browser_batch_dirs = [
        repo_root / "web/runs/exp_d10w2_psaa/batch_runs/20260405_161802_full",
        repo_root / "web/runs/exp_d10w2_psaa/batch_runs/20260412_140735_full",
    ]
    browser_traces = _trace_paths(browser_batch_dirs)
    file_traces = sorted(
        (repo_root / "file/runs/exp_d10w2_psaa/logs/samples").glob(
            "case_*/trace_sample_*.jsonl"
        )
    )
    rows: list[dict[str, Any]] = []
    for system_name, trace_kind, traces in [
        ("Browser", "browser", browser_traces),
        ("File", "file", file_traces),
    ]:
        system = systems.get(system_name) if isinstance(systems, Mapping) else {}
        if not isinstance(system, Mapping):
            continue
        original = system.get("original_stage_nodes") or {}
        if not isinstance(original, Mapping):
            original = {}
        planned_variants = system.get("variants")
        if isinstance(planned_variants, Mapping):
            conditions: list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]] = [
                ("Optimal", original, None)
            ]
            for variant in PLACEMENT_VARIANTS:
                entry = planned_variants.get(variant)
                selected = (
                    entry.get("selected") if isinstance(entry, Mapping) else None
                )
                if isinstance(selected, Mapping):
                    conditions.append(
                        (
                            variant,
                            selected.get("stage_nodes") or {},
                            selected,
                        )
                    )
        else:
            # Legacy schema remains readable but cannot masquerade as the full
            # three-condition Rebuttal implementation.
            selected = system.get("selected")
            conditions = [("Optimal", original, None)]
            if isinstance(selected, Mapping):
                conditions.append(
                    (
                        "shift_s2s3",
                        selected.get("stage_nodes") or {},
                        selected,
                    )
                )
        for placement, stage_nodes, candidate in conditions:
            if not isinstance(stage_nodes, Mapping):
                continue
            inertia = stage_nodes.get("inertia")
            payload_node = stage_nodes.get("payload")
            if trace_kind == "browser":
                i_hit, i_total, i_er = _browser_payload_er(traces, int(inertia))
                p_hit, p_total, p_er = _browser_payload_er(traces, int(payload_node))
            else:
                i_hit, i_total, i_er = _file_payload_er(traces, str(inertia))
                p_hit, p_total, p_er = _file_payload_er(traces, str(payload_node))
            rows.append(
                {
                    "system": system_name,
                    "placement": placement,
                    "inertia_node": inertia,
                    "payload_node": payload_node,
                    "total_hops": (
                        (len(system.get("original_route") or []) - 1)
                        if candidate is None
                        else candidate.get("total_hops")
                    ),
                    "added_hops": 0 if candidate is None else candidate.get("added_hops"),
                    "historical_inertia_er": i_er,
                    "historical_inertia_hits": f"{i_hit}/{i_total}",
                    "historical_payload_er": p_er,
                    "historical_payload_hits": f"{p_hit}/{p_total}",
                    "status": (
                        "completed_existing_run"
                        if placement == "Optimal"
                        else "structural_proxy_only"
                    ),
                    "note": (
                        "Historical ER is measured on existing full-run traces and is a "
                        "placement proxy, not a rerun under moved injections."
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.2f}%"


def _load_readiness(repo_root: Path) -> Mapping[str, Any] | None:
    path = repo_root / "Rebuttal/results/materialized_envs/readiness_summary.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    return payload if isinstance(payload, Mapping) else None


def _load_coverage_rows(repo_root: Path) -> list[Mapping[str, Any]]:
    path = (
        repo_root
        / "Rebuttal/results/node_frequency_theoretical_max/theoretical_max.json"
    )
    if not path.is_file():
        return []
    payload = _read_json(path)
    rows = payload.get("rows") if isinstance(payload, Mapping) else []
    return [row for row in rows or [] if isinstance(row, Mapping)]


def _load_supplemental_coverage_rows(
    repo_root: Path,
) -> list[Mapping[str, Any]]:
    path = (
        repo_root
        / "Rebuttal/results/node_frequency_rerun/node_frequency.json"
    )
    if not path.is_file():
        return []
    payload = _read_json(path)
    rows = payload.get("rows") if isinstance(payload, Mapping) else []
    return [row for row in rows or [] if isinstance(row, Mapping)]


def _render_readiness_lines(readiness: Mapping[str, Any] | None) -> list[str]:
    if not readiness:
        return []
    counts = readiness.get("environment_counts")
    if not isinstance(counts, Mapping):
        return []
    browser_placements = counts.get("browser_placement_samples")
    file_placements = counts.get("file_placement_runs")
    if not isinstance(browser_placements, Mapping):
        browser_placements = {}
    if not isinstance(file_placements, Mapping):
        file_placements = {}
    ready_count = int(readiness.get("ready_environment_count") or 0)
    blocked_count = int(readiness.get("blocked_environment_count") or 0)
    status = (
        "ready for formal run"
        if blocked_count == 0
        else "blocked until single-pass generation and contract validation"
    )
    lines = [
        "## Formal environment readiness",
        "",
        "| Experiment | System | Variant group | Materialized environments | Status |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for variant in PLACEMENT_VARIANTS:
        lines.extend(
            [
                (
                    f"| EXP-PLACE-001 | Browser | {variant} | "
                    f"{int(browser_placements.get(variant) or 0)} environments | {status} |"
                ),
                (
                    f"| EXP-PLACE-001 | File | {variant} | "
                    f"{int(file_placements.get(variant) or 0)} run directories | {status} |"
                ),
            ]
        )
    lines.extend(
        [
        (
            "| EXP-ABL-001 | Browser | Lure/Inertia/Payload-only | "
            f"{counts.get('browser_single_stage_samples', 0)} attack-specific environments "
            f"| {status} |"
        ),
        (
            "| EXP-ABL-001 | File | Lure/Inertia/Payload-only | "
            f"{counts.get('file_single_stage_runs', 0)} supplementary run directories "
            f"| {status} |"
        ),
        "",
        f"The index currently contains {ready_count} ready and {blocked_count} blocked "
        "environments. Readiness requires checked text, exact stage/node alignment, "
        "valid navigation hops, generation metadata, and the locked runtime contract.",
        "",
        ]
    )
    return lines


def _render_markdown(
    rows: Sequence[MetricRow],
    placement_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any] | None,
    supplemental_coverage_rows: Sequence[Mapping[str, Any]] = (),
) -> str:
    completed_formal = any(
        row.status == "completed_contract_valid" for row in rows
    )
    completed_coverage_primary = bool(coverage_rows) and all(
        "nodes_er_ge_10_ratio_upper_bound" in row
        for row in coverage_rows
    )
    provenance_line = (
        "Completed formal results were discovered and admitted only after contract validation."
        if completed_formal
        else (
            "The primary EXP-COVER-001 target-allocation upper-bound analysis "
            "and supplemental 16-target empirical rerun are complete. "
            "EXP-PLACE-001 and EXP-ABL-001 have not been run, so their planned "
            "rows remain non-results."
            if completed_coverage_primary
            else (
                "No contract-valid new formal result was discovered; incomplete "
                "or stale outputs remain non-results."
            )
        )
    )
    lines = [
        "# Rebuttal Evidence Update",
        "",
        provenance_line,
        "",
    ]
    lines.extend(_render_readiness_lines(readiness))
    if coverage_rows:
        is_theoretical = all(
            "nodes_er_ge_10_ratio_upper_bound" in row
            for row in coverage_rows
        )
        is_rerun = all("nodes_er_ge_10_ratio" in row for row in coverage_rows)
        lines.extend(
            [
                "## EXP-COVER-001 coverage",
                "",
                (
                    "| System | Candidate nodes | Top-1 ER upper bound ↑ | "
                    "Top-2 ER upper bound ↑ | Top-3 ER upper bound ↑ | "
                    "Random-1 expected encounters (batch) | "
                    "max nodes ER ≥ 10% ↑ | max nodes ER ≥ 30% ↑ |"
                    if is_theoretical
                    else
                    "| System | Candidate nodes | Top-1 ER ↑ | Random-1 ER | "
                    "nodes ER ≥ 10% ↑ | nodes ER ≥ 30% ↑ |"
                    if is_rerun
                    else "| System | Candidate universe | Calibration trajectories | Held-out trajectories | Top-1 ER | Top-3 union ER | Random-1 ER | Random-3 union ER |"
                ),
                (
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
                    if is_theoretical
                    else "| --- | --- | ---: | ---: | ---: | ---: |"
                    if is_rerun
                    else "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
                ),
            ]
        )
        for row in coverage_rows:
            if is_theoretical:
                values = [
                    str(row.get("system")),
                    str(row.get("candidate_nodes")),
                    _fmt_pct(row.get("top1_er_upper_bound")),
                    _fmt_pct(row.get("top2_er_upper_bound")),
                    _fmt_pct(row.get("top3_er_upper_bound")),
                    f"{float(row.get('random1_expected_encounters') or 0):.3f}",
                    _fmt_pct(row.get("nodes_er_ge_10_ratio_upper_bound")),
                    _fmt_pct(row.get("nodes_er_ge_30_ratio_upper_bound")),
                ]
            elif is_rerun:
                values = [
                    str(row.get("system")),
                    str(row.get("candidate_nodes")),
                    _fmt_pct(row.get("top1_er")),
                    _fmt_pct(row.get("random1_er")),
                    _fmt_pct(row.get("nodes_er_ge_10_ratio")),
                    _fmt_pct(row.get("nodes_er_ge_30_ratio")),
                ]
            else:
                values = [
                    str(row.get("system")),
                    str(row.get("candidate_nodes")),
                    str(row.get("calibration_trajectories")),
                    str(row.get("heldout_trajectories")),
                    _fmt_pct(row.get("top1_er")),
                    _fmt_pct(row.get("top3_union_er")),
                    _fmt_pct(row.get("random1_er")),
                    _fmt_pct(row.get("random3_union_er")),
                ]
            lines.append("| " + " | ".join(values) + " |")
        if is_theoretical:
            note = (
                "Primary result: column-wise theoretical upper bounds for 72 "
                "Browser and 60 File ideal root-to-target trajectories. "
                "Top-k is the kth-highest individual node ER, not a union; "
                "the reported Top-1/2/3 maxima are jointly attainable here. "
                "Random-1 is an expected total encounter count across the full "
                "batch. The 10% and 30% maxima are optimized independently and "
                "are not jointly attainable by one target allocation."
            )
        elif is_rerun:
            note = (
                "Both systems use 16 fresh clean trajectories with 16 unique, "
                "prefix-balanced user-tree leaf targets. Numerators and "
                "denominators contain user-tree nodes only; Random-1 is the exact "
                "uniform-candidate expectation."
            )
        else:
            note = (
                "Browser uses a task-disjoint split and a common GitLab/Reddit "
                "candidate-universe intersection. File is trajectory-disjoint only "
                "because the available clean set has one unique benign task."
            )
        lines.extend(["", note, ""])
    if supplemental_coverage_rows:
        lines.extend(
            [
                "### Supplemental 16-target empirical rerun",
                "",
                "| System | Candidate nodes | Top-1 ER ↑ | Random-1 ER | "
                "nodes ER ≥ 10% ↑ | nodes ER ≥ 30% ↑ |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in supplemental_coverage_rows:
            values = [
                str(row.get("system")),
                str(row.get("candidate_nodes")),
                _fmt_pct(row.get("top1_er")),
                _fmt_pct(row.get("random1_er")),
                _fmt_pct(row.get("nodes_er_ge_10_ratio")),
                _fmt_pct(row.get("nodes_er_ge_30_ratio")),
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "This supplemental table remains the measured 16+16 clean "
                "rerun over distinct prefix-balanced targets.",
                "",
            ]
        )
    lines.extend(
        [
            "## Main metrics",
        "",
        "| Experiment | System | Variant | Payload ER | ASR-E | ASR-I | UUA | Dual-goal | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.experiment,
                    row.system,
                    row.variant,
                    _fmt_pct(row.payload_er),
                    _fmt_pct(row.asr_e),
                    _fmt_pct(row.asr_i),
                    _fmt_pct(row.uua),
                    _fmt_pct(row.dual_goal),
                    row.status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Suboptimal placement proxy",
            "",
            "| System | Placement | Inertia node | Payload node | Added hops | Historical inertia ER | Historical payload ER | Status |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in placement_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("system")),
                    str(row.get("placement")),
                    str(row.get("inertia_node")),
                    str(row.get("payload_node")),
                    str(row.get("added_hops")),
                    _fmt_pct(row.get("historical_inertia_er")),
                    _fmt_pct(row.get("historical_payload_er")),
                    str(row.get("status")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Rows with `non_result_incomplete_or_contract_mismatch` are not evidence and "
            "must not populate the rebuttal table. Structural placement proxies are kept "
            "separate from rerun metrics. File single-stage results are supplementary; "
            "the paper-facing ablation comparison is Browser-GitLab.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Rebuttal/results/summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    browser_combined, browser_gitlab = _browser_full_rows(repo_root)
    file_full = _file_full_row(repo_root)
    rows = [
        browser_combined,
        file_full,
        browser_gitlab,
        *_formal_rows(repo_root),
        *_leave_one_out_rows(repo_root),
    ]
    placement_rows = _placement_proxy_rows(repo_root)
    coverage_rows = _load_coverage_rows(repo_root)
    supplemental_coverage_rows = _load_supplemental_coverage_rows(repo_root)
    readiness = _load_readiness(repo_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rebuttal_metrics.csv", [asdict(row) for row in rows])
    _write_csv(out_dir / "placement_proxy.csv", placement_rows)
    (out_dir / "rebuttal_metrics.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "formal_results_discovered_dynamically": True,
                    "contract_valid_formal_rows": sum(
                        row.status == "completed_contract_valid" for row in rows
                    ),
                    "incomplete_stale_or_mismatched_outputs_are_non_results": True,
                    "paper_facing_single_stage_scope": "Browser-GitLab",
                    "file_single_stage_scope": "supplementary_only",
                    "formal_environment_readiness": (
                        readiness.get("ready_for_deepseek_v3_1_terminal")
                        if isinstance(readiness, Mapping)
                        else None
                    ),
                },
                "rows": [asdict(row) for row in rows],
                "coverage_rows": coverage_rows,
                "coverage_supplemental_rows": supplemental_coverage_rows,
                "placement_proxy_rows": placement_rows,
                "readiness": readiness,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "evidence_update.md").write_text(
        _render_markdown(
            rows,
            placement_rows,
            coverage_rows,
            readiness,
            supplemental_coverage_rows,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_dir / 'rebuttal_metrics.csv'}")
    print(f"Wrote {out_dir / 'placement_proxy.csv'}")
    print(f"Wrote {out_dir / 'rebuttal_metrics.json'}")
    print(f"Wrote {out_dir / 'evidence_update.md'}")


if __name__ == "__main__":
    main()
