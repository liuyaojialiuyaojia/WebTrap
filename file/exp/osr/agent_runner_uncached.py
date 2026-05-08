#!/usr/bin/env python3
"""OSR-only exp2 agent wrapper with cache disabled for all sample calls."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from file.exp.exp2 import agent_runner as shared
from file.exp.osr.no_cache import wrap_generate


def install_patches() -> None:
    shared.generate = wrap_generate(shared.generate)


def main() -> None:
    install_patches()
    shared.main()


if __name__ == "__main__":
    main()
