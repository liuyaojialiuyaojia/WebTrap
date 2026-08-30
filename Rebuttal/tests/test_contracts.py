from __future__ import annotations

import json
from pathlib import Path

from Rebuttal.experiments.contracts import (
    BROWSER_MAX_PARALLEL,
    FORMAL_VARIANTS,
    MODEL_ID,
    environment_fingerprint,
    runtime_contract_errors,
    validate_environment_contract,
)
from Rebuttal.experiments.semantic_prompts import (
    SEMANTIC_CHECKER_MODE,
    SINGLE_STAGE_SEMANTIC_POLICY_VERSION,
    stage_semantic_contract,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _browser_payload_only_environment(tmp_path: Path) -> Path:
    metadata_path = tmp_path / "static" / "page_metadata.json"
    pages = []
    for node in range(4):
        page = {
            "page_index": node,
            "click_targets": (
                [{"target_page": node + 1, "label": f"to-{node + 1}"}]
                if node < 3
                else []
            ),
            "injections": [],
        }
        pages.append(page)
    pages[2]["injections"] = [
        {
            "text": "opaque",
            "psaa": {
                "stage": "payload",
                "generation": {
                    "passed": True,
                    "attempts": 1,
                    "checker_used": False,
                    "checker_mode": SEMANTIC_CHECKER_MODE,
                },
                "checked_stage_text_generated": True,
                "stage_route": [2, 3],
                "semantic_contract": stage_semantic_contract(
                    variant="payload_only",
                    stage="payload",
                    route=[2, 3],
                    retained_stages=("payload",),
                ),
            },
        }
    ]
    _write_json(metadata_path, {"pages": pages})
    _write_json(tmp_path / "webarena_tasks" / "1000.json", {"task_id": 1000})
    _write_json(
        tmp_path / "webarena_tasks_attacker" / "1000.json",
        {"task_id": 1000},
    )
    _write_json(tmp_path / "attack_case.json", {"id": "opaque"})
    manifest_path = tmp_path / "environment_manifest.json"
    _write_json(
        manifest_path,
        {
            "system": "Browser",
            "suite": "gitlab",
            "variant": "payload_only",
            "retained_stages": ["payload"],
            "moved_stages": [],
            "planned_stage_nodes": {"payload": 2},
            "planned_route": [0, 1, 2, 3],
            "active_page_metadata": metadata_path.as_posix(),
            "copied_stage_text_from_full": False,
            "checked_stage_text_generated": True,
            "route_specific_stage_text_ready": True,
            "generation_contract": {
                "prompt_spec_path": "/tmp/spec.yaml",
                "prompt_spec_sha256": "abc",
                "prompt_spec_version": "psaa_v1",
                "attacker_model": "generator",
                "checker_model": "checker",
                "retry_num": 1,
                "checker_enabled": False,
                "checker_mode": SEMANTIC_CHECKER_MODE,
                "max_generation_attempts": 1,
                "semantic_policy_version": SINGLE_STAGE_SEMANTIC_POLICY_VERSION,
            },
        },
    )
    return manifest_path


def test_browser_environment_contract_accepts_complete_single_stage(
    tmp_path: Path,
) -> None:
    manifest_path = _browser_payload_only_environment(tmp_path)
    assert validate_environment_contract(manifest_path) == []


def test_browser_environment_contract_rejects_reddit_single_stage(
    tmp_path: Path,
) -> None:
    manifest_path = _browser_payload_only_environment(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suite"] = "reddit"
    _write_json(manifest_path, manifest)
    errors = validate_environment_contract(manifest_path)
    assert any("outside the planned scope" in error for error in errors)


def test_environment_fingerprint_changes_with_formal_input(tmp_path: Path) -> None:
    manifest_path = _browser_payload_only_environment(tmp_path)
    before = environment_fingerprint(manifest_path)
    task_path = tmp_path / "webarena_tasks" / "1000.json"
    _write_json(task_path, {"task_id": 1000, "changed": True})
    assert environment_fingerprint(manifest_path) != before


def test_runtime_contract_locks_browser_configuration() -> None:
    errors = runtime_contract_errors(
        system="Browser",
        model=MODEL_ID,
        check_model=MODEL_ID,
        temperature=1.0,
        defense_mode="default_attack",
        check_temperature=0.0,
        trials=3,
        seed=42,
        max_actions=20,
        max_observations=20,
        max_parallel=BROWSER_MAX_PARALLEL,
        check_max_tokens=4096,
    )
    assert errors == []
    assert runtime_contract_errors(
        system="Browser",
        model=MODEL_ID,
        check_model=MODEL_ID,
        temperature=0.0,
        defense_mode="default_attack",
        check_temperature=0.0,
        trials=3,
        seed=42,
        max_actions=20,
        max_observations=20,
        max_parallel=BROWSER_MAX_PARALLEL,
        check_max_tokens=4096,
    )


def test_runtime_contract_uses_path_normalized_variant_budgets() -> None:
    browser_common = {
        "system": "Browser",
        "model": MODEL_ID,
        "check_model": MODEL_ID,
        "temperature": 1.0,
        "defense_mode": "default_attack",
        "check_temperature": 0.0,
        "trials": 3,
        "seed": 42,
        "max_parallel": BROWSER_MAX_PARALLEL,
        "check_max_tokens": 4096,
    }
    assert runtime_contract_errors(
        **browser_common,
        variant="shift_s2",
        max_actions=22,
        max_observations=22,
    ) == []
    assert runtime_contract_errors(
        **browser_common,
        variant="shift_s2s3",
        max_actions=24,
        max_observations=24,
    ) == []
    assert runtime_contract_errors(
        **browser_common,
        variant="payload_only",
        max_actions=20,
        max_observations=20,
    ) == []

    file_common = {
        "system": "File",
        "model": MODEL_ID,
        "check_model": MODEL_ID,
        "temperature": 1.0,
        "defense_mode": "default_attack",
        "check_temperature": 0.0,
        "num_samples": 3,
        "agent_max_tokens": 512,
        "asr_max_tokens": 512,
    }
    assert runtime_contract_errors(
        **file_common,
        variant="shift_s3",
        max_steps=27,
    ) == []
    assert runtime_contract_errors(
        **file_common,
        variant="shift_s2s3",
        max_steps=29,
    ) == []
    assert runtime_contract_errors(
        **file_common,
        variant="lure_only",
        max_steps=25,
    ) == []


def test_formal_variant_set_contains_all_planned_conditions() -> None:
    assert set(FORMAL_VARIANTS) == {
        "shift_s2",
        "shift_s3",
        "shift_s2s3",
        "lure_only",
        "inertia_only",
        "payload_only",
    }
