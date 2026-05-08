#!/usr/bin/env python3
"""OSR-only exp1 wrapper that forces uncached user-question generation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.exp1 import build_cases_top20 as shared
from file.exp.exp1 import make_user_task_question as question_mod
from file.exp.osr.no_cache import wrap_generate


def install_patches() -> None:
    question_mod.generate = wrap_generate(question_mod.generate)


def main() -> None:
    install_patches()
    shared.main()


if __name__ == "__main__":
    main()
