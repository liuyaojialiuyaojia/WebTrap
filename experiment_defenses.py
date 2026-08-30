"""Stable imports for shared experiment defenses.

Some external evaluator modules use the top-level name ``utils`` for a plain
module. Loading the repository defense package under a private name avoids
making experiment runners depend on import order.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


_PACKAGE_NAME = "_repository_experiment_defenses"
_PACKAGE_ROOT = Path(__file__).resolve().parent / "utils" / "defenses"


def _load_package() -> ModuleType:
    existing = sys.modules.get(_PACKAGE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _PACKAGE_NAME,
        _PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(_PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load experiment defenses from {_PACKAGE_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_PACKAGE_NAME, None)
        raise
    return module


_defenses = _load_package()
_goal_reinforcement = importlib.import_module(
    f"{_PACKAGE_NAME}.goal_reinforcement"
)
_segmentation_removal = importlib.import_module(
    f"{_PACKAGE_NAME}.segmentation_removal"
)

ALLOWED_DEFENSE_MODES = _defenses.ALLOWED_DEFENSE_MODES
DEFENSE_MODE_DEFAULT = _defenses.DEFENSE_MODE_DEFAULT
DEFENSE_MODE_GOAL_REINFORCE_FAKECOM_T = (
    _defenses.DEFENSE_MODE_GOAL_REINFORCE_FAKECOM_T
)
DEFENSE_MODE_GOAL_REINFORCE_IGNORE = (
    _defenses.DEFENSE_MODE_GOAL_REINFORCE_IGNORE
)
DEFENSE_MODE_SEGMENT_REMOVE_DIRECT = (
    _defenses.DEFENSE_MODE_SEGMENT_REMOVE_DIRECT
)
DEFENSE_MODE_SEGMENT_REMOVE_GATED = (
    _defenses.DEFENSE_MODE_SEGMENT_REMOVE_GATED
)
DEFENSE_MODE_STEP_WISE = _defenses.DEFENSE_MODE_STEP_WISE

apply_step_wise_defense = _goal_reinforcement.apply_step_wise_defense
apply_system_prompt_defense = _goal_reinforcement.apply_system_prompt_defense
build_goal_reinforcement_messages = (
    _goal_reinforcement.build_goal_reinforcement_messages
)
is_goal_reinforcement_mode = (
    _goal_reinforcement.is_goal_reinforcement_mode
)
merge_goal_reinforcement_into_input = (
    _goal_reinforcement.merge_goal_reinforcement_into_input
)
sanitize_observation_text = _segmentation_removal.sanitize_observation_text

__all__ = [
    "ALLOWED_DEFENSE_MODES",
    "DEFENSE_MODE_DEFAULT",
    "DEFENSE_MODE_GOAL_REINFORCE_FAKECOM_T",
    "DEFENSE_MODE_GOAL_REINFORCE_IGNORE",
    "DEFENSE_MODE_SEGMENT_REMOVE_DIRECT",
    "DEFENSE_MODE_SEGMENT_REMOVE_GATED",
    "DEFENSE_MODE_STEP_WISE",
    "apply_step_wise_defense",
    "apply_system_prompt_defense",
    "build_goal_reinforcement_messages",
    "is_goal_reinforcement_mode",
    "merge_goal_reinforcement_into_input",
    "sanitize_observation_text",
]
