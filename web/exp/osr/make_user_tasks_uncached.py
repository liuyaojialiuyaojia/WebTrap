#!/usr/bin/env python3
"""OSR-only wrapper around Stage 04 user task generation with cache disabled."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web.exp.osr.no_cache import wrap_llm_chat


def _load_shared_module():
    module_path = Path(__file__).resolve().parents[1] / "04_task_gen" / "make_user_tasks_from_leaves.py"
    spec = importlib.util.spec_from_file_location("web_osr_make_user_tasks_shared", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shared = _load_shared_module()


def install_patches() -> None:
    if shared.llm_chat is not None:
        shared.llm_chat = wrap_llm_chat(shared.llm_chat)


def main() -> None:
    install_patches()
    shared.main()


if __name__ == "__main__":
    main()
