"""Helpers for loading A2Perf web primitives."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PRIMITIVES_PATH = (
    WORKSPACE_ROOT
    / "A2Perf"
    / "a2perf"
    / "domains"
    / "web_navigation"
    / "gwob"
    / "CoDE"
    / "web_primitives.py"
)
MODULE_NAME = "trap_site_stage1_primitives"

_PRIMITIVES_MODULE: ModuleType | None = None


def _patch_or_stub_gymnasium() -> None:
    if "gymnasium" in sys.modules:
        module = sys.modules["gymnasium"]
        envs = getattr(module, "envs", None)
        registration = getattr(envs, "registration", None)
        register = getattr(registration, "register", None)
        if callable(register):
            original_register = register

            def compat_register(*args, **kwargs):
                kwargs.pop("disable_env_checker", None)
                return original_register(*args, **kwargs)

            registration.register = compat_register  # type: ignore[attr-defined]
        return

    try:
        import gymnasium  # type: ignore

        envs = gymnasium.envs
        registration = envs.registration
        original_register = registration.register

        def compat_register(*args, **kwargs):
            kwargs.pop("disable_env_checker", None)
            return original_register(*args, **kwargs)

        registration.register = compat_register  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        stub = ModuleType("gymnasium")
        stub.envs = SimpleNamespace(registration=SimpleNamespace(register=lambda *a, **k: None))
        sys.modules["gymnasium"] = stub


def load_primitives_module() -> ModuleType:
    global _PRIMITIVES_MODULE
    if _PRIMITIVES_MODULE is not None:
        return _PRIMITIVES_MODULE

    _patch_or_stub_gymnasium()
    spec = importlib.util.spec_from_file_location(MODULE_NAME, PRIMITIVES_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate A2Perf primitive module at {PRIMITIVES_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _PRIMITIVES_MODULE = module
    return module


def get_concepts() -> List[str]:
    module = load_primitives_module()
    concepts = getattr(module, "CONCEPTS", None)
    if concepts is None:
        raise AttributeError("A2Perf primitives module does not expose CONCEPTS")
    return list(concepts)


def get_transition_names() -> List[str]:
    module = load_primitives_module()
    transitions = getattr(module, "TRANSITIONS2DESIGN", None)
    if transitions is None:
        raise AttributeError("A2Perf primitives module does not expose TRANSITIONS2DESIGN")
    return list(transitions.keys())
