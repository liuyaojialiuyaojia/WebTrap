#!/usr/bin/env python3
"""Validate the locked Rebuttal experiment contract without reading traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from Rebuttal.experiments.structural_variants import (
    PLACEMENT_VARIANTS,
    load_file_graph,
)
from Rebuttal.experiments.semantic_prompts import (
    SEMANTIC_CHECKER_MODE,
    SINGLE_STAGE_VARIANTS,
    semantic_policy_version_for_variant,
    semantic_contract_errors,
)


MODEL_ID = "deepseek-v3-1-terminus"
CHECK_MODEL_ID = MODEL_ID
CHECK_TEMPERATURE = 0.0
BROWSER_CHECK_MAX_TOKENS = 4096
BROWSER_TEMPERATURE = 1.0
BROWSER_TRIALS = 3
BROWSER_SEED = 42
BROWSER_MAX_PARALLEL = 6
BROWSER_MAX_ACTIONS = 20
BROWSER_MAX_OBSERVATIONS = 20
FILE_TEMPERATURE = 1.0
FILE_NUM_SAMPLES = 3
FILE_MAX_STEPS = 25
FILE_AGENT_MAX_TOKENS = 512
FILE_ASR_MAX_TOKENS = 512
DEFENSE_MODE = "default_attack"

STAGE_ORDER = ("lure", "inertia", "payload")
FORMAL_VARIANTS = (*PLACEMENT_VARIANTS, *SINGLE_STAGE_VARIANTS)

# Placement routes deliberately leave the original shortest path and then
# return to it. Preserve the same post-route action/step slack as Full by
# adding the deterministic route overhead to both Browser caps and the File
# step cap. Single-stage variants retain the original route and base budgets.
BROWSER_BUDGETS_BY_VARIANT: dict[str, tuple[int, int]] = {
    "shift_s2": (22, 22),
    "shift_s3": (22, 22),
    "shift_s2s3": (24, 24),
    "lure_only": (BROWSER_MAX_ACTIONS, BROWSER_MAX_OBSERVATIONS),
    "inertia_only": (BROWSER_MAX_ACTIONS, BROWSER_MAX_OBSERVATIONS),
    "payload_only": (BROWSER_MAX_ACTIONS, BROWSER_MAX_OBSERVATIONS),
}
FILE_MAX_STEPS_BY_VARIANT: dict[str, int] = {
    "shift_s2": 27,
    "shift_s3": 27,
    "shift_s2s3": 29,
    "lure_only": FILE_MAX_STEPS,
    "inertia_only": FILE_MAX_STEPS,
    "payload_only": FILE_MAX_STEPS,
}


def browser_budgets_for_variant(variant: str) -> tuple[int, int]:
    try:
        return BROWSER_BUDGETS_BY_VARIANT[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown formal Rebuttal variant: {variant}") from exc


def file_max_steps_for_variant(variant: str) -> int:
    try:
        return FILE_MAX_STEPS_BY_VARIANT[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown formal Rebuttal variant: {variant}") from exc


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def expected_stages(variant: str) -> tuple[str, ...]:
    if variant in PLACEMENT_VARIANTS:
        return STAGE_ORDER
    retained = SINGLE_STAGE_VARIANTS.get(variant)
    if retained is None:
        raise ValueError(f"Unknown formal Rebuttal variant: {variant}")
    return (retained,)


def expected_moved_stages(variant: str) -> tuple[str, ...]:
    if variant in PLACEMENT_VARIANTS:
        return tuple(PLACEMENT_VARIANTS[variant])
    if variant in SINGLE_STAGE_VARIANTS:
        return ()
    raise ValueError(f"Unknown formal Rebuttal variant: {variant}")


def suite_allowed_for_variant(variant: str, suite: str) -> bool:
    """Placement uses GitLab+Reddit; paper-facing single-stage uses GitLab only."""

    if variant in PLACEMENT_VARIANTS:
        return suite in {"gitlab", "reddit"}
    if variant in SINGLE_STAGE_VARIANTS:
        return suite == "gitlab"
    return False


def runtime_contract_errors(
    *,
    system: str,
    variant: str | None = None,
    model: str,
    check_model: str,
    temperature: float,
    defense_mode: str,
    trials: int | None = None,
    seed: int | None = None,
    max_actions: int | None = None,
    max_observations: int | None = None,
    max_parallel: int | None = None,
    check_temperature: float | None = None,
    check_max_tokens: int | None = None,
    num_samples: int | None = None,
    max_steps: int | None = None,
    agent_max_tokens: int | None = None,
    asr_max_tokens: int | None = None,
) -> list[str]:
    errors: list[str] = []
    if model != MODEL_ID:
        errors.append(f"model must be {MODEL_ID}, got {model}")
    if check_model != CHECK_MODEL_ID:
        errors.append(f"check_model must be {CHECK_MODEL_ID}, got {check_model}")
    if temperature != 1.0:
        errors.append(f"temperature must be 1.0, got {temperature}")
    if defense_mode != DEFENSE_MODE:
        errors.append(f"defense_mode must be {DEFENSE_MODE}, got {defense_mode}")
    if check_temperature != CHECK_TEMPERATURE:
        errors.append(
            f"check_temperature must be {CHECK_TEMPERATURE}, got {check_temperature}"
        )

    normalized = system.strip().lower()
    if normalized == "browser":
        if variant is None:
            expected_max_actions = BROWSER_MAX_ACTIONS
            expected_max_observations = BROWSER_MAX_OBSERVATIONS
        elif variant in BROWSER_BUDGETS_BY_VARIANT:
            expected_max_actions, expected_max_observations = (
                browser_budgets_for_variant(variant)
            )
        else:
            errors.append(f"unknown formal variant: {variant}")
            expected_max_actions = BROWSER_MAX_ACTIONS
            expected_max_observations = BROWSER_MAX_OBSERVATIONS
        expected = {
            "trials": (trials, BROWSER_TRIALS),
            "seed": (seed, BROWSER_SEED),
            "max_actions": (max_actions, expected_max_actions),
            "max_observations": (
                max_observations,
                expected_max_observations,
            ),
            "max_parallel": (max_parallel, BROWSER_MAX_PARALLEL),
            "check_max_tokens": (check_max_tokens, BROWSER_CHECK_MAX_TOKENS),
        }
    elif normalized == "file":
        if variant is None:
            expected_max_steps = FILE_MAX_STEPS
        elif variant in FILE_MAX_STEPS_BY_VARIANT:
            expected_max_steps = file_max_steps_for_variant(variant)
        else:
            errors.append(f"unknown formal variant: {variant}")
            expected_max_steps = FILE_MAX_STEPS
        expected = {
            "num_samples": (num_samples, FILE_NUM_SAMPLES),
            "max_steps": (max_steps, expected_max_steps),
            "agent_max_tokens": (agent_max_tokens, FILE_AGENT_MAX_TOKENS),
            "asr_max_tokens": (asr_max_tokens, FILE_ASR_MAX_TOKENS),
        }
    else:
        return [f"unknown system: {system}"]

    for field, (actual, wanted) in expected.items():
        if actual != wanted:
            errors.append(f"{field} must be {wanted}, got {actual}")
    return errors


def _hash_paths(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({candidate.resolve() for candidate in paths}, key=str):
        if not path.is_file():
            continue
        try:
            label = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def environment_fingerprint(manifest_path: Path) -> str:
    """Hash formal inputs while excluding model logs, traces, and evaluator output."""

    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Invalid environment manifest: {manifest_path}")

    paths: list[Path] = [manifest_path]
    system = str(manifest.get("system") or "")
    if system == "Browser":
        for key in ("active_page_metadata",):
            value = manifest.get(key)
            if isinstance(value, str):
                paths.append(Path(value))
        paths.extend(sorted((root / "webarena_tasks").glob("*.json")))
        paths.extend(sorted((root / "webarena_tasks_attacker").glob("*.json")))
        paths.append(root / "attack_case.json")
    elif system == "File":
        paths.extend(
            [
                root / "attack_cases_pre_injection.jsonl",
                root / "exp1_config.json",
                root / "injection" / "injection_manifest.jsonl",
            ]
        )
        paths.extend(
            sorted(root.glob("case_envs/*/env/env_post_injection_tree.json"))
        )
    else:
        raise ValueError(f"Unknown environment system in {manifest_path}: {system}")
    return _hash_paths(root, paths)


def _browser_stage_rows(metadata: Mapping[str, Any]) -> list[tuple[str, int, Mapping[str, Any]]]:
    rows: list[tuple[str, int, Mapping[str, Any]]] = []
    for page in metadata.get("pages", []):
        if not isinstance(page, Mapping) or not isinstance(page.get("page_index"), int):
            continue
        for injection in page.get("injections") or []:
            if not isinstance(injection, Mapping):
                continue
            psaa = injection.get("psaa")
            if not isinstance(psaa, Mapping):
                continue
            stage = psaa.get("stage")
            if isinstance(stage, str) and stage in STAGE_ORDER:
                rows.append((stage, int(page["page_index"]), psaa))
    return rows


def _browser_adjacency(metadata: Mapping[str, Any]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for page in metadata.get("pages", []):
        if not isinstance(page, Mapping) or not isinstance(page.get("page_index"), int):
            continue
        node = int(page["page_index"])
        adjacency.setdefault(node, set())
        for target in page.get("click_targets") or []:
            if isinstance(target, Mapping) and isinstance(target.get("target_page"), int):
                adjacency[node].add(int(target["target_page"]))
    return adjacency


def _invalid_hops(route: Sequence[Any], adjacency: Mapping[Any, set[Any]]) -> list[str]:
    return [
        f"{start!r}->{goal!r}"
        for start, goal in zip(route, route[1:])
        if goal not in adjacency.get(start, set())
    ]


def _generation_contract_errors(manifest: Mapping[str, Any]) -> list[str]:
    contract = manifest.get("generation_contract")
    if not isinstance(contract, Mapping):
        return ["generation_contract metadata is missing"]
    errors: list[str] = []
    for field in (
        "prompt_spec_path",
        "prompt_spec_sha256",
        "prompt_spec_version",
        "attacker_model",
        "checker_model",
        "retry_num",
        "checker_enabled",
        "checker_mode",
        "max_generation_attempts",
        "semantic_policy_version",
    ):
        if contract.get(field) in (None, ""):
            errors.append(f"generation_contract.{field} is missing")
    if contract.get("checker_enabled") is not False:
        errors.append("generation_contract.checker_enabled must be false")
    if contract.get("checker_mode") != SEMANTIC_CHECKER_MODE:
        errors.append(
            "generation_contract.checker_mode must be "
            f"{SEMANTIC_CHECKER_MODE}"
        )
    if contract.get("retry_num") != 1:
        errors.append("generation_contract.retry_num must be 1")
    if contract.get("max_generation_attempts") != 1:
        errors.append("generation_contract.max_generation_attempts must be 1")
    expected_policy_version = semantic_policy_version_for_variant(
        str(manifest.get("variant") or "")
    )
    if contract.get("semantic_policy_version") != expected_policy_version:
        errors.append(
            "generation_contract.semantic_policy_version does not match "
            f"{expected_policy_version}"
        )
    return errors


def _generation_record_errors(generation: Any) -> list[str]:
    if not isinstance(generation, Mapping):
        return ["generation metadata is missing"]
    errors: list[str] = []
    if not bool(generation.get("passed")):
        errors.append("generation was not accepted")
    if generation.get("checker_used") is not False:
        errors.append("acceptance checker must be disabled")
    if generation.get("checker_mode") != SEMANTIC_CHECKER_MODE:
        errors.append(
            f"checker_mode must be {SEMANTIC_CHECKER_MODE}"
        )
    if generation.get("attempts") != 1:
        errors.append("generation must contain exactly one attacker attempt")
    return errors


def _validate_browser_environment(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    suite = str(manifest.get("suite") or "")
    variant = str(manifest.get("variant") or "")
    if not suite_allowed_for_variant(variant, suite):
        errors.append(f"suite {suite!r} is outside the planned scope for {variant}")

    metadata_path = Path(str(manifest.get("active_page_metadata") or ""))
    if not metadata_path.is_file():
        return [*errors, f"active Browser metadata is missing: {metadata_path}"]
    metadata = _load_json(metadata_path)
    if not isinstance(metadata, Mapping):
        return [*errors, "active Browser metadata is not an object"]

    rows = _browser_stage_rows(metadata)
    expected = expected_stages(variant)
    actual = tuple(stage for stage, _node, _psaa in rows)
    if len(rows) != len(expected) or set(actual) != set(expected):
        errors.append(
            f"generated stage set must be {list(expected)}, got {list(actual)}"
        )

    planned_nodes = manifest.get("planned_stage_nodes")
    if not isinstance(planned_nodes, Mapping):
        errors.append("planned_stage_nodes is missing")
        planned_nodes = {}
    route = manifest.get("planned_route")
    normalized_route = (
        [int(node) for node in route]
        if isinstance(route, list)
        else []
    )
    for stage, node, psaa in rows:
        planned = planned_nodes.get(stage)
        if planned is None or int(planned) != node:
            errors.append(
                f"stage {stage} is at node {node}, expected {planned!r}"
            )
        errors.extend(
            f"stage {stage}: {error}"
            for error in _generation_record_errors(psaa.get("generation"))
        )
        if not bool(psaa.get("checked_stage_text_generated")):
            errors.append(f"stage {stage} is not marked as checked text")
        stage_route = psaa.get("stage_route")
        if not isinstance(stage_route, list) or len(stage_route) < 2:
            errors.append(f"stage {stage} lacks complete stage_route metadata")
        else:
            normalized_stage_route = [int(value) for value in stage_route]
            expected_index = expected.index(stage)
            expected_goal = (
                int(planned_nodes[expected[expected_index + 1]])
                if expected_index + 1 < len(expected)
                else (normalized_route[-1] if normalized_route else None)
            )
            if normalized_stage_route[0] != node:
                errors.append(f"stage {stage} route does not start at its stage node")
            if expected_goal is None or normalized_stage_route[-1] != expected_goal:
                errors.append(
                    f"stage {stage} route does not end at the next retained stage/anchor"
                )
            invalid_stage_hops = _invalid_hops(
                normalized_stage_route,
                _browser_adjacency(metadata),
            )
            if invalid_stage_hops:
                errors.append(
                    f"stage {stage} route contains invalid hops: {invalid_stage_hops}"
                )
            semantic = psaa.get("semantic_contract")
            if not isinstance(semantic, Mapping):
                errors.append(f"stage {stage} lacks semantic_contract metadata")
            else:
                errors.extend(
                    f"stage {stage}: {error}"
                    for error in semantic_contract_errors(
                        semantic,
                        variant=variant,
                        stage=stage,
                        route=normalized_stage_route,
                        retained_stages=expected,
                    )
                )

    if not isinstance(route, list) or len(route) < 2:
        errors.append("planned_route must contain at least one hop")
    else:
        normalized_route = [int(node) for node in route]
        invalid = _invalid_hops(normalized_route, _browser_adjacency(metadata))
        if invalid:
            errors.append(f"planned_route contains invalid hops: {invalid}")
    return errors


def _validate_file_environment(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    root = manifest_path.parent
    variant = str(manifest.get("variant") or "")
    expected = expected_stages(variant)
    expected_cases = int(manifest.get("cases") or 0)
    generated_cases = int(manifest.get("generated_case_count") or 0)
    if expected_cases <= 0 or generated_cases != expected_cases:
        errors.append(
            f"single-pass generation must cover all cases: {generated_cases}/{expected_cases}"
        )

    planned_nodes = manifest.get("planned_stage_nodes")
    if not isinstance(planned_nodes, Mapping):
        errors.append("planned_stage_nodes is missing")
        planned_nodes = {}
    rows = _read_jsonl(root / "injection" / "injection_manifest.jsonl")
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row.get("case_id") or ""), []).append(row)
    if len(by_case) != expected_cases:
        errors.append(
            f"injection manifest must contain {expected_cases} cases, got {len(by_case)}"
        )

    pre_trees = sorted(root.glob("case_envs/*/env/env_pre_injection_tree.json"))
    if not pre_trees:
        errors.append("no File pre-injection tree is available for route validation")
        adjacency: Mapping[str, set[str]] = {}
    else:
        adjacency = {
            str(node): set(map(str, neighbors))
            for node, neighbors in load_file_graph(pre_trees[0]).adjacency.items()
        }

    for case_id, case_rows in sorted(by_case.items()):
        actual = [str(row.get("stage") or "") for row in case_rows]
        if len(case_rows) != len(expected) or set(actual) != set(expected):
            errors.append(
                f"{case_id}: generated stage set must be {list(expected)}, got {actual}"
            )
        stage_rows = {
            str(row.get("stage") or ""): row for row in case_rows
        }
        for stage in expected:
            row = stage_rows.get(stage)
            if row is None:
                continue
            stage = str(row.get("stage") or "")
            node = str(row.get("directory_logical_path") or "")
            if node != str(planned_nodes.get(stage) or ""):
                errors.append(
                    f"{case_id}:{stage} is at {node!r}, expected {planned_nodes.get(stage)!r}"
                )
            errors.extend(
                f"{case_id}:{stage}: {error}"
                for error in _generation_record_errors(row.get("generation"))
            )
            if not bool(row.get("checked_stage_text_generated")):
                errors.append(f"{case_id}:{stage} is not marked as checked text")
            route = row.get("route")
            if not isinstance(route, list) or len(route) < 2:
                errors.append(f"{case_id}:{stage} has no valid route metadata")
            else:
                normalized_stage_route = [str(node) for node in route]
                expected_index = expected.index(stage)
                planned_route = [str(node) for node in manifest.get("planned_route") or []]
                expected_goal = (
                    str(planned_nodes[expected[expected_index + 1]])
                    if expected_index + 1 < len(expected)
                    else (planned_route[-1] if planned_route else "")
                )
                if normalized_stage_route[0] != node:
                    errors.append(
                        f"{case_id}:{stage} route does not start at its stage node"
                    )
                if normalized_stage_route[-1] != expected_goal:
                    errors.append(
                        f"{case_id}:{stage} route does not end at the next retained stage/anchor"
                    )
                invalid = _invalid_hops(normalized_stage_route, adjacency)
                if invalid:
                    errors.append(
                        f"{case_id}:{stage} contains invalid route hops: {invalid}"
                    )
                semantic = row.get("semantic_contract")
                if not isinstance(semantic, Mapping):
                    errors.append(
                        f"{case_id}:{stage} lacks semantic_contract metadata"
                    )
                else:
                    errors.extend(
                        f"{case_id}:{stage}: {error}"
                        for error in semantic_contract_errors(
                            semantic,
                            variant=variant,
                            stage=stage,
                            route=normalized_stage_route,
                            retained_stages=expected,
                        )
                    )
    return errors


def validate_environment_contract(manifest_path: Path) -> list[str]:
    """Return all readiness violations; an empty list means formal-run ready."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        return [f"environment manifest is missing: {manifest_path}"]
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        return [f"environment manifest is not an object: {manifest_path}"]

    errors: list[str] = []
    variant = str(manifest.get("variant") or "")
    if variant not in FORMAL_VARIANTS:
        return [f"unknown formal variant: {variant!r}"]
    if tuple(manifest.get("retained_stages") or ()) != expected_stages(variant):
        errors.append(
            "retained_stages do not match the planned variant: "
            f"{manifest.get('retained_stages')!r}"
        )
    if tuple(manifest.get("moved_stages") or ()) != expected_moved_stages(variant):
        errors.append(
            "moved_stages do not match the planned variant: "
            f"{manifest.get('moved_stages')!r}"
        )
    if not bool(manifest.get("checked_stage_text_generated")):
        errors.append("checked stage text has not been generated")
    if bool(manifest.get("copied_stage_text_from_full", True)):
        errors.append("copied Full-condition text is still active")
    if not bool(manifest.get("route_specific_stage_text_ready")):
        errors.append("route-specific/standalone text is not ready")
    errors.extend(_generation_contract_errors(manifest))

    system = str(manifest.get("system") or "")
    if system == "Browser":
        errors.extend(_validate_browser_environment(manifest, manifest_path))
    elif system == "File":
        errors.extend(_validate_file_environment(manifest, manifest_path))
    else:
        errors.append(f"unknown environment system: {system!r}")
    return errors
