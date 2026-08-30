#!/usr/bin/env python3
"""Generate single-pass, route-aligned text for formal Rebuttal environments.

Materialization deliberately uses copied Full-condition text as a non-runnable
placeholder. This command is the only transition that can mark a Rebuttal
environment ready for formal model evaluation. It reuses the repository's
existing PSAA prompt specifications, generator functions, and task
instructions. The LLM checker is intentionally disabled to match the main
PSAA experiments, whose prompt specs use retry_num=1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "web"
for import_root in (REPO_ROOT, WEB_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from file.exp.attack.common import render_directory_context
from file.exp.attack.psaa.inject import (
    _build_navigation_path_text as build_file_navigation_text,
    _build_security_target_path_text,
    _extract_first_email,
    _generate_checked_injection_text as generate_file_text,
)
from file.exp.attack.psaa.prompt_spec import load_psaa_prompt_spec as load_file_spec
from psaa.inject_from_attack_case import (
    _build_navigation_path_text as build_browser_navigation_text,
    _discover_first_attacker_task,
    _extract_safety_instruction,
    _find_click_target_label,
    _generate_checked_injection_text as generate_browser_text,
    _render_page_for_prompt,
    _render_selections_for_prompt,
)
from psaa.prompt_spec import load_psaa_prompt_spec as load_browser_spec

from Rebuttal.experiments.materialize_formal_envs import (
    INJECTION_GENERATION_BLOCKER,
    _hash_text,
    _load_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    browser_stage_records,
    build_file_post_tree,
)
from Rebuttal.experiments.contracts import (
    FORMAL_VARIANTS,
    PLACEMENT_VARIANTS,
    SINGLE_STAGE_VARIANTS,
    STAGE_ORDER,
    validate_environment_contract,
)
from Rebuttal.experiments.semantic_prompts import (
    SEMANTIC_CHECKER_MODE,
    apply_semantic_prompt_policy,
    semantic_policy_version_for_variant,
    stage_semantic_contract,
)
ALL_VARIANTS = FORMAL_VARIANTS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _generation_contract(
    spec: Any,
    prompt_spec: Path,
    *,
    variant: str,
) -> dict[str, Any]:
    return {
        "prompt_spec_path": prompt_spec.resolve().as_posix(),
        "prompt_spec_sha256": _sha256(prompt_spec),
        "prompt_spec_version": str(spec.version),
        "attacker_model": str(spec.attacker.model),
        "checker_model": str(spec.checker.model),
        "retry_num": int(spec.retry_num),
        "checker_enabled": False,
        "checker_mode": SEMANTIC_CHECKER_MODE,
        "max_generation_attempts": int(spec.retry_num),
        "semantic_policy_version": semantic_policy_version_for_variant(variant),
    }


def _finalize_ready_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Only mark an environment ready after the complete evidence gate passes."""

    manifest["ready_for_deepseek_v3_1_terminal"] = True
    manifest["model_run_status"] = "not_started"
    _write_json(manifest_path, manifest)
    errors = validate_environment_contract(manifest_path)
    manifest["contract_validation"] = {
        "passed": not errors,
        "validated_at": _utc_now(),
        "errors": errors,
    }
    if errors:
        manifest["ready_for_deepseek_v3_1_terminal"] = False
        manifest["model_run_status"] = INJECTION_GENERATION_BLOCKER
    _write_json(manifest_path, manifest)
    if errors:
        raise ValueError(
            f"Formal environment contract failed for {manifest_path}: "
            + "; ".join(errors)
        )


def route_segment(
    route: Sequence[Any],
    start: Any,
    goal: Any,
    *,
    start_at: int = 0,
) -> tuple[list[Any], int]:
    """Return the forward route segment and the goal index used."""

    try:
        start_index = next(
            index
            for index in range(start_at, len(route))
            if route[index] == start
        )
    except StopIteration as exc:
        raise ValueError(f"Stage node {start!r} is absent from planned route") from exc
    try:
        goal_index = next(
            index
            for index in range(start_index + 1, len(route))
            if route[index] == goal
        )
    except StopIteration as exc:
        raise ValueError(
            f"Route goal {goal!r} does not occur after stage node {start!r}"
        ) from exc
    return list(route[start_index : goal_index + 1]), goal_index


def build_stage_routes(
    *,
    planned_route: Sequence[Any],
    planned_stage_nodes: Mapping[str, Any],
    retained_stages: Sequence[str],
) -> dict[str, list[Any]]:
    """Split one planned route into the route assigned to each retained stage."""

    stages = [stage for stage in STAGE_ORDER if stage in set(retained_stages)]
    if not stages:
        raise ValueError("At least one retained stage is required")
    anchor = planned_route[-1]
    result: dict[str, list[Any]] = {}
    search_from = 0
    for index, stage in enumerate(stages):
        start = planned_stage_nodes[stage]
        goal = (
            planned_stage_nodes[stages[index + 1]]
            if index + 1 < len(stages)
            else anchor
        )
        segment, goal_index = route_segment(
            planned_route,
            start,
            goal,
            start_at=search_from,
        )
        if len(segment) < 2:
            raise ValueError(f"Stage {stage} has an empty navigation segment")
        result[stage] = segment
        search_from = max(0, goal_index)
    return result


def _remove_all_browser_psaa(metadata: dict[str, Any]) -> None:
    for page in metadata.get("pages", []):
        if not isinstance(page, dict):
            continue
        injections = page.get("injections")
        if not isinstance(injections, list):
            continue
        page["injections"] = [
            injection
            for injection in injections
            if not (
                isinstance(injection, Mapping)
                and isinstance(injection.get("psaa"), Mapping)
                and injection["psaa"].get("stage") in STAGE_ORDER
            )
        ]


def _browser_pages(metadata: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(page["page_index"]): page
        for page in metadata.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("page_index"), int)
    }


def generate_browser_environment(
    env_root: Path,
    *,
    prompt_spec: Path,
) -> dict[str, Any]:
    manifest_path = env_root / "environment_manifest.json"
    manifest = _load_json(manifest_path)
    source_metadata = Path(str(manifest["source_metadata"]))
    source_payload = _load_json(source_metadata)
    if not isinstance(source_payload, Mapping):
        raise ValueError(f"Invalid source Browser metadata: {source_metadata}")

    source_records = browser_stage_records(source_payload)
    metadata = copy.deepcopy(dict(source_payload))
    _remove_all_browser_psaa(metadata)
    pages = _browser_pages(metadata)

    retained_stages = tuple(str(stage) for stage in manifest["retained_stages"])
    planned_nodes = {
        str(stage): int(node)
        for stage, node in manifest["planned_stage_nodes"].items()
    }
    planned_route = [int(node) for node in manifest["planned_route"]]
    stage_routes = build_stage_routes(
        planned_route=planned_route,
        planned_stage_nodes=planned_nodes,
        retained_stages=retained_stages,
    )

    attack_case = _load_json(env_root / "attack_case.json")
    attacker_tasks = Path(str(manifest["attacker_tasks"]))
    task_path = _discover_first_attacker_task(attacker_tasks)
    attacker_task = _load_json(task_path)
    task_instruction = _extract_safety_instruction(attacker_task)
    microtree = attack_case.get("microtree") if isinstance(attack_case, Mapping) else {}
    microtree_id = (
        str(microtree.get("id") or manifest["sample_id"])
        if isinstance(microtree, Mapping)
        else str(manifest["sample_id"])
    )

    spec = apply_semantic_prompt_policy(
        load_browser_spec(prompt_spec),
        system="browser",
        variant=str(manifest["variant"]),
    )
    generation_log = env_root / "injection_generation.jsonl"
    generation_log.write_text("", encoding="utf-8")
    previous_injection: str | None = None
    generated: list[dict[str, Any]] = []

    for step, stage in enumerate(
        stage for stage in STAGE_ORDER if stage in set(retained_stages)
    ):
        page_index = planned_nodes[stage]
        page = pages.get(page_index)
        if page is None:
            raise KeyError(f"Browser planned stage page is missing: {page_index}")
        route = stage_routes[stage]
        next_page_index = int(route[1])
        goal_label = _find_click_target_label(page, target_page=next_page_index)
        goal_path_labels, goal_path_text = build_browser_navigation_text(
            page_by_index=pages,
            page_sequence=[int(node) for node in route],
        )
        attempts: list[dict[str, Any]] = []

        def on_attempt(record: dict[str, Any]) -> None:
            attempts.append(dict(record))
            _append_jsonl(
                generation_log,
                {
                    "timestamp": _utc_now(),
                    "event": "attempt_result",
                    "variant": manifest["variant"],
                    "stage": stage,
                    "page_index": page_index,
                    "goal_path_text": goal_path_text,
                    **record,
                },
            )

        text, generation = generate_browser_text(
            spec=spec,
            stage=stage,
            page_text=_render_page_for_prompt(page),
            goal_label=goal_label,
            goal_path_text=goal_path_text,
            task_instruction=task_instruction,
            selections_text=_render_selections_for_prompt(page),
            previous_injection=previous_injection,
            force_checker=False,
            attempt_callback=on_attempt,
        )
        if not generation.get("passed"):
            raise ValueError(
                f"Browser single-pass generation rejected "
                f"{manifest['variant']}:{stage}"
            )
        previous_injection = text
        semantic_contract = stage_semantic_contract(
            variant=str(manifest["variant"]),
            stage=stage,
            route=route,
            retained_stages=retained_stages,
        )

        injection = copy.deepcopy(source_records[stage]["injection"])
        injection["attack_id"] = (
            f"psaa::rebuttal::{manifest['variant']}::{microtree_id}::{stage}::{step}"
        )
        injection["text"] = text
        source = injection.setdefault("source", {})
        source.update(
            {
                "variant": manifest["variant"],
                "rebuttal_experiment": manifest["experiment"],
                "anchor_page_index": planned_route[-1],
                "generated_from_full_template": True,
                "copied_stage_text_from_full": False,
            }
        )
        psaa = injection.setdefault("psaa", {})
        psaa.update(
            {
                "variant": manifest["variant"],
                "rebuttal_variant": manifest["variant"],
                "stage": stage,
                "step": step,
                "chain_len": len(planned_route) - 1,
                "chain_position": planned_route.index(page_index),
                "next_page_index": next_page_index,
                "goal_label": goal_label,
                "goal_path_labels": goal_path_labels,
                "goal_path_text": goal_path_text,
                "stage_route": route,
                "semantic_contract": semantic_contract,
                "generation": generation,
                "copied_stage_text_from_full": False,
                "checked_stage_text_generated": True,
                "single_pass_stage_text_generated": True,
            }
        )
        page.setdefault("injections", []).append(injection)
        generated.append(
            {
                "stage": stage,
                "page_index": page_index,
                "route": route,
                "semantic_contract": semantic_contract,
                "payload_hash": _hash_text(text),
                "generation": generation,
                "attempts": len(attempts),
            }
        )

    _write_json(Path(str(manifest["active_page_metadata"])), metadata)
    manifest.update(
        {
            "copied_stage_text_from_full": False,
            "checked_stage_text_generated": True,
            "checked_stage_text_required": True,
            "single_pass_stage_text_generated": True,
            "llm_checker_used": False,
            "route_specific_stage_text_ready": True,
            "generation_completed_at": _utc_now(),
            "generation_prompt_spec": prompt_spec.resolve().as_posix(),
            "generation_prompt_spec_sha256": _sha256(prompt_spec),
            "generation_contract": _generation_contract(
                spec,
                prompt_spec,
                variant=str(manifest["variant"]),
            ),
            "generated_stages": generated,
        }
    )
    _finalize_ready_manifest(manifest_path, manifest)
    return {
        "system": "Browser",
        "variant": manifest["variant"],
        "status": "generated_ready",
        "generated_stages": len(generated),
    }


def _file_case_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["case_id"]): row
        for row in _read_jsonl(run_dir / "attack_cases_pre_injection.jsonl")
    }


def generate_file_environment(
    run_dir: Path,
    *,
    prompt_spec: Path,
    max_cases: int = 0,
) -> dict[str, Any]:
    manifest_path = run_dir / "environment_manifest.json"
    manifest = _load_json(manifest_path)
    retained_stages = tuple(str(stage) for stage in manifest["retained_stages"])
    planned_nodes = {
        str(stage): str(node)
        for stage, node in manifest["planned_stage_nodes"].items()
    }
    planned_route = [str(node) for node in manifest["planned_route"]]
    stage_routes = build_stage_routes(
        planned_route=planned_route,
        planned_stage_nodes=planned_nodes,
        retained_stages=retained_stages,
    )
    cases = _file_case_index(run_dir)
    case_ids = sorted(cases)
    selected_case_ids = case_ids[:max_cases] if max_cases else case_ids
    spec = apply_semantic_prompt_policy(
        load_file_spec(prompt_spec),
        system="file",
        variant=str(manifest["variant"]),
    )
    generation_log = run_dir / "injection" / "injection_generations.jsonl"
    generation_log.write_text("", encoding="utf-8")
    all_manifest_rows: list[dict[str, Any]] = []

    for case_id in selected_case_ids:
        case = cases[case_id]
        pre_tree_path = run_dir / "case_envs" / case_id / "env" / "env_pre_injection_tree.json"
        post_tree_path = run_dir / "case_envs" / case_id / "env" / "env_post_injection_tree.json"
        pre_tree = _load_json(pre_tree_path)
        if not isinstance(pre_tree, Mapping):
            raise ValueError(f"Invalid File pre-tree for {case_id}")
        root_logical = f"/{str(pre_tree.get('name') or 'root').strip()}"
        security_entry = str(case["security_entry_logical_path"])
        target_path = str(case["attack_target_logical_path"])
        _, security_target_text = _build_security_target_path_text(
            security_entry_dir=security_entry,
            target_security_file=target_path,
            root_logical=root_logical,
        )
        instruction = str(case["attacker_instruction"])
        recipient_email = _extract_first_email(instruction)
        previous_injection: str | None = None
        generated_texts: dict[str, str] = {}

        for step, stage in enumerate(
            stage for stage in STAGE_ORDER if stage in set(retained_stages)
        ):
            directory = planned_nodes[stage]
            route = [str(node) for node in stage_routes[stage]]
            labels, goal_path_text = build_file_navigation_text(
                route_dirs=route,
                root_logical=root_logical,
            )
            if not labels:
                raise ValueError(f"File route for {case_id}:{stage} is empty")
            attempts: list[dict[str, Any]] = []

            def on_attempt(record: dict[str, Any]) -> None:
                attempts.append(dict(record))
                _append_jsonl(
                    generation_log,
                    {
                        "timestamp": _utc_now(),
                        "event": "attempt_result",
                        "case_id": case_id,
                        "variant": manifest["variant"],
                        "stage": stage,
                        "directory_logical_path": directory,
                        "goal_path_text": goal_path_text,
                        **record,
                    },
                )

            text, generation = generate_file_text(
                spec=spec,
                stage=stage,
                page_text=render_directory_context(
                    dict(pre_tree),
                    directory,
                    root_logical=root_logical,
                    include_readme=True,
                    include_entries=True,
                ),
                goal_label=labels[0],
                goal_path_text=goal_path_text,
                task_instruction=instruction,
                previous_injection=previous_injection,
                security_target_path_text=security_target_text,
                recipient_email=recipient_email,
                force_checker=False,
                attempt_callback=on_attempt,
            )
            if not generation.get("passed"):
                raise ValueError(
                    f"File single-pass generation rejected "
                    f"{case_id}:{manifest['variant']}:{stage}"
                )
            previous_injection = text
            generated_texts[stage] = text
            semantic_contract = stage_semantic_contract(
                variant=str(manifest["variant"]),
                stage=stage,
                route=route,
                retained_stages=retained_stages,
            )
            payload_path = (
                run_dir
                / "injection"
                / "payloads"
                / f"rebuttal_{case_id}_{manifest['variant']}_{stage}_{step}.txt"
            )
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(text.rstrip() + "\n", encoding="utf-8")
            all_manifest_rows.append(
                {
                    "timestamp": _utc_now(),
                    "case_id": case_id,
                    "mode": "rebuttal_single_pass",
                    "variant": manifest["variant"],
                    "stage": stage,
                    "stage_index": step,
                    "directory_logical_path": directory,
                    "next_directory_logical_path": route[1],
                    "goal_path_labels": labels,
                    "goal_path_text": goal_path_text,
                    "route": route,
                    "semantic_contract": semantic_contract,
                    "payload_hash": _hash_text(text),
                    "payload_path": payload_path.resolve().as_posix(),
                    "generation": generation,
                    "attempts": len(attempts),
                    "copied_stage_text_from_full": False,
                    "checked_stage_text_generated": True,
                    "single_pass_stage_text_generated": True,
                }
            )

        post_tree, _ = build_file_post_tree(
            pre_tree,
            stage_dirs={
                stage: planned_nodes[stage] for stage in retained_stages
            },
            payload_texts=generated_texts,
        )
        _write_json(post_tree_path, post_tree)

    complete = len(selected_case_ids) == len(case_ids)
    _write_jsonl(run_dir / "injection" / "injection_manifest.jsonl", all_manifest_rows)
    manifest.update(
        {
            "copied_stage_text_from_full": not complete,
            "checked_stage_text_generated": complete,
            "checked_stage_text_required": True,
            "single_pass_stage_text_generated": complete,
            "llm_checker_used": False,
            "route_specific_stage_text_ready": complete,
            "generation_completed_at": _utc_now() if complete else None,
            "generation_prompt_spec": prompt_spec.resolve().as_posix(),
            "generation_prompt_spec_sha256": _sha256(prompt_spec),
            "generation_contract": _generation_contract(
                spec,
                prompt_spec,
                variant=str(manifest["variant"]),
            ),
            "generated_case_count": len(selected_case_ids),
            "expected_case_count": len(case_ids),
            "model_run_status": "not_started" if complete else INJECTION_GENERATION_BLOCKER,
            "ready_for_deepseek_v3_1_terminal": complete,
        }
    )
    _write_json(manifest_path, manifest)
    if complete:
        _finalize_ready_manifest(manifest_path, manifest)
    return {
        "system": "File",
        "variant": manifest["variant"],
        "status": "generated_ready" if complete else "partial_generation",
        "generated_stages": len(all_manifest_rows),
        "generated_cases": len(selected_case_ids),
        "expected_cases": len(case_ids),
    }


def aggregate_generation_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate generation health without retaining sample-level identifiers."""

    buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in results:
        system = str(result.get("system") or "")
        variant = str(result.get("variant") or "")
        status = str(result.get("status") or "failed")
        error_type = str(result.get("error_type") or "")
        key = (system, variant, status, error_type)
        bucket = buckets.setdefault(
            key,
            {
                "system": system,
                "variant": variant,
                "status": status,
                "error_type": error_type,
                "environment_count": 0,
                "generated_stages": 0,
                "generated_cases": 0,
                "expected_cases": 0,
            },
        )
        bucket["environment_count"] += 1
        bucket["generated_stages"] += int(result.get("generated_stages") or 0)
        bucket["generated_cases"] += int(result.get("generated_cases") or 0)
        bucket["expected_cases"] += int(result.get("expected_cases") or 0)
    return [
        buckets[key]
        for key in sorted(buckets)
    ]


def classify_generation_error(exc: Exception) -> str:
    """Map an exception to a privacy-safe operational category."""

    message = str(exc).lower()
    if "single-pass generation rejected" in message:
        return "single_pass_rejected"
    if "checker rejected" in message:
        return "checker_rejected"
    if "formal environment contract failed" in message:
        return "contract_validation_failed"
    if any(
        token in message
        for token in (
            "connection",
            "connect",
            "timeout",
            "timed out",
            "apierror",
            "internalservererror",
        )
    ):
        return "transport_failure"
    if "empty response" in message or "returned none" in message:
        return "empty_model_response"
    return "generation_error"


def _write_index_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            for key in ("retained_stages", "moved_stages"):
                if isinstance(payload.get(key), list):
                    payload[key] = ";".join(str(value) for value in payload[key])
            writer.writerow(payload)


def refresh_materialized_indexes(repo_root: Path) -> None:
    results_root = repo_root / "Rebuttal/results/materialized_envs"
    all_rows: list[dict[str, Any]] = []
    for system in ("browser", "file"):
        path = results_root / f"{system}_formal_envs.json"
        payload = _load_json(path)
        rows = payload.get("environments", [])
        for row in rows:
            env_manifest_path = Path(str(row["environment_root"])) / "environment_manifest.json"
            env_manifest = _load_json(env_manifest_path)
            contract_errors = validate_environment_contract(env_manifest_path)
            ready = not contract_errors
            model_run_status = (
                str(env_manifest.get("model_run_status") or "not_started")
                if ready
                else "blocked_contract_validation_failed"
            )
            row.update(
                {
                    "copied_stage_text_from_full": bool(
                        env_manifest.get("copied_stage_text_from_full", True)
                    ),
                    "ready_for_deepseek_v3_1_terminal": ready,
                    "model_run_status": model_run_status,
                    "note": (
                        "Single-pass stage text generated and environment validated."
                        if ready
                        else "Single-pass stage text generation is incomplete."
                    ),
                }
            )
        payload["generated_at"] = _utc_now()
        _write_json(path, payload)
        _write_index_csv(results_root / f"{system}_formal_envs.csv", rows)
        all_rows.extend(rows)

    blocked = [
        row for row in all_rows if not row["ready_for_deepseek_v3_1_terminal"]
    ]
    _write_json(
        results_root / "readiness_summary.json",
        {
            "schema_version": 2,
            "generated_at": _utc_now(),
            "ready_for_deepseek_v3_1_terminal": not blocked,
            "model_run_status": "not_started" if not blocked else "blocked",
            "blocked_environment_count": len(blocked),
            "ready_environment_count": len(all_rows) - len(blocked),
            "environment_counts": {
                "browser_placement_samples": {
                    variant: sum(
                        row["system"] == "Browser" and row["variant"] == variant
                        for row in all_rows
                    )
                    for variant in PLACEMENT_VARIANTS
                },
                "browser_single_stage_samples": sum(
                    row["system"] == "Browser"
                    and row["variant"] in SINGLE_STAGE_VARIANTS
                    for row in all_rows
                ),
                "file_placement_runs": {
                    variant: sum(
                        row["system"] == "File" and row["variant"] == variant
                        for row in all_rows
                    )
                    for variant in PLACEMENT_VARIANTS
                },
                "file_single_stage_runs": sum(
                    row["system"] == "File"
                    and row["variant"] in SINGLE_STAGE_VARIANTS
                    for row in all_rows
                ),
            },
            "notes": [
                "Readiness is derived from each environment_manifest.json.",
                "Only complete single-pass route-specific generation can transition an environment to ready.",
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--systems",
        default="browser,file",
        help="Comma-separated subset of browser,file.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(ALL_VARIANTS),
        help=f"Comma-separated variants from: {','.join(ALL_VARIANTS)}",
    )
    parser.add_argument(
        "--browser-prompt-spec",
        type=Path,
        default=Path("web/psaa/psaa_v1.yaml"),
    )
    parser.add_argument(
        "--file-prompt-spec",
        type=Path,
        default=Path("file/exp/attack/psaa/file_psaa_v1.yaml"),
    )
    parser.add_argument("--max-browser-environments", type=int, default=0)
    parser.add_argument("--max-file-cases", type=int, default=0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later environments after recording an aggregate failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    systems = {value.strip().lower() for value in args.systems.split(",") if value.strip()}
    variants = {value.strip() for value in args.variants.split(",") if value.strip()}
    unknown_systems = systems - {"browser", "file"}
    unknown_variants = variants - set(ALL_VARIANTS)
    if unknown_systems or unknown_variants:
        raise ValueError(
            f"Unknown systems={sorted(unknown_systems)} variants={sorted(unknown_variants)}"
        )

    results: list[dict[str, Any]] = []
    stop_requested = False
    if "browser" in systems:
        index = _load_json(
            repo_root / "Rebuttal/results/materialized_envs/browser_formal_envs.json"
        )
        rows = [
            row
            for row in index["environments"]
            if str(row["variant"]) in variants
        ]
        if args.max_browser_environments:
            rows = rows[: args.max_browser_environments]
        browser_spec = (
            args.browser_prompt_spec
            if args.browser_prompt_spec.is_absolute()
            else repo_root / args.browser_prompt_spec
        )
        for idx, row in enumerate(rows, start=1):
            print(
                f"[generation/browser] {idx}/{len(rows)} "
                f"variant={row['variant']}",
                flush=True,
            )
            try:
                result = generate_browser_environment(
                    Path(str(row["environment_root"])),
                    prompt_spec=browser_spec,
                )
            except Exception as exc:
                failure_code = classify_generation_error(exc)
                results.append(
                    {
                        "system": "Browser",
                        "variant": str(row["variant"]),
                        "status": "failed",
                        "error_type": failure_code,
                    }
                )
                print(
                    "[generation/browser] failed "
                    f"variant={row['variant']} error_type={failure_code}",
                    flush=True,
                )
                if not args.continue_on_error:
                    stop_requested = True
                    break
            else:
                results.append(result)

    if "file" in systems and not stop_requested:
        index = _load_json(
            repo_root / "Rebuttal/results/materialized_envs/file_formal_envs.json"
        )
        file_spec = (
            args.file_prompt_spec
            if args.file_prompt_spec.is_absolute()
            else repo_root / args.file_prompt_spec
        )
        file_rows = [
            row
            for row in index["environments"]
            if str(row["variant"]) in variants
        ]
        for idx, row in enumerate(file_rows, start=1):
            print(
                f"[generation/file] {idx}/{len(file_rows)} "
                f"variant={row['variant']}",
                flush=True,
            )
            try:
                result = generate_file_environment(
                    Path(str(row["environment_root"])),
                    prompt_spec=file_spec,
                    max_cases=args.max_file_cases,
                )
            except Exception as exc:
                failure_code = classify_generation_error(exc)
                results.append(
                    {
                        "system": "File",
                        "variant": str(row["variant"]),
                        "status": "failed",
                        "error_type": failure_code,
                    }
                )
                print(
                    "[generation/file] failed "
                    f"variant={row['variant']} error_type={failure_code}",
                    flush=True,
                )
                if not args.continue_on_error:
                    stop_requested = True
                    break
            else:
                results.append(result)

    refresh_materialized_indexes(repo_root)
    aggregates = aggregate_generation_results(results)
    failed_count = sum(
        int(row["environment_count"])
        for row in aggregates
        if row["status"] == "failed"
    )
    attempted_count = sum(int(row["environment_count"]) for row in aggregates)
    output = repo_root / "Rebuttal/results/materialized_envs/generation_summary.json"
    _write_json(
        output,
        {
            "schema_version": 2,
            "generated_at": _utc_now(),
            "privacy_mode": "aggregate_only",
            "checker_mode": SEMANTIC_CHECKER_MODE,
            "attempted_environment_count": attempted_count,
            "failed_environment_count": failed_count,
            "passed_environment_count": attempted_count - failed_count,
            "aggregates": aggregates,
        },
    )
    print(f"Wrote {output}")
    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
