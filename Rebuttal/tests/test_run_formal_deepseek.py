from __future__ import annotations

from Rebuttal.experiments.run_formal_deepseek import (
    _blocked_environment_result,
    _is_environment_ready,
    _runtime_contract_for_args,
    parse_args,
)


def test_not_ready_environment_is_blocked_before_model_run() -> None:
    row = {
        "variant": "shift_s2s3",
        "sample_id": "sample",
        "environment_root": "/tmp/sample",
        "ready_for_deepseek_v3_1_terminal": False,
        "model_run_status": "blocked_until_route_specific_injection_generation",
    }

    assert _is_environment_ready(row) is False
    result = _blocked_environment_result("Browser", row)
    assert result["status"] == "blocked_not_ready"
    assert result["reason"] == "blocked_until_route_specific_injection_generation"


def test_default_formal_run_includes_all_rebuttal_variants() -> None:
    args = parse_args([])
    assert set(args.browser_variants.split(",")) == {
        "shift_s2",
        "shift_s3",
        "shift_s2s3",
        "lure_only",
        "inertia_only",
        "payload_only",
    }
    assert set(args.file_variants.split(",")) == set(
        args.browser_variants.split(",")
    )


def test_runner_selects_path_normalized_budget_per_variant() -> None:
    args = parse_args([])
    assert _runtime_contract_for_args(
        "Browser", args, variant="shift_s2"
    )["passed"]
    assert _runtime_contract_for_args(
        "Browser", args, variant="shift_s2s3"
    )["passed"]
    assert _runtime_contract_for_args(
        "File", args, variant="shift_s3"
    )["passed"]
    assert _runtime_contract_for_args(
        "File", args, variant="shift_s2s3"
    )["passed"]
