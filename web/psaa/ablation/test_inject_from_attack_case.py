import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = Path(__file__).resolve().parent / "inject_from_attack_case.py"
SPEC = importlib.util.spec_from_file_location(
    "ablation_inject_from_attack_case_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
inject_from_attack_case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inject_from_attack_case)

from psaa.ablation import wo_inertia, wo_lure, wo_payload
from psaa.ablation import prompt_spec as ablation_prompt_spec


def test_validate_ablation_variant_allows_component_drop_variants() -> None:
    assert inject_from_attack_case._validate_ablation_variant("wo_lure") == "wo_lure"
    assert (
        inject_from_attack_case._validate_ablation_variant("wo_inertia")
        == "wo_inertia"
    )
    assert (
        inject_from_attack_case._validate_ablation_variant("wo_payload")
        == "wo_payload"
    )


def test_resolve_ablation_prompt_spec_path_defaults_to_base() -> None:
    resolved = ablation_prompt_spec.resolve_ablation_prompt_spec_path(None, None)

    assert resolved == REPO_ROOT / "psaa" / "psaa_v1.yaml"


def test_resolve_ablation_prompt_spec_path_supports_enhanced() -> None:
    resolved = ablation_prompt_spec.resolve_ablation_prompt_spec_path(None, "enhanced")

    assert resolved == REPO_ROOT / "psaa" / "psaa_v2.yaml"


def test_wo_lure_stage_plan_matches_component_drop() -> None:
    assert wo_lure.build_stage_plan(1) == [("payload", 0)]
    assert wo_lure.build_stage_plan(2) == [("inertia", 0), ("payload", 1)]
    assert wo_lure.build_stage_plan(8) == [("inertia", 0), ("payload", 7)]


def test_wo_inertia_stage_plan_matches_component_drop() -> None:
    assert wo_inertia.build_stage_plan(1) == [("payload", 0)]
    assert wo_inertia.build_stage_plan(2) == [("lure", 0), ("payload", 1)]
    assert wo_inertia.build_stage_plan(8) == [("lure", 0), ("payload", 7)]


def test_wo_payload_stage_plan_matches_component_drop() -> None:
    assert wo_payload.build_stage_plan(1) == [("inertia", 0)]
    assert wo_payload.build_stage_plan(2) == [("lure", 0), ("inertia", 1)]
    assert wo_payload.build_stage_plan(8) == [("lure", 0), ("inertia", 7)]
