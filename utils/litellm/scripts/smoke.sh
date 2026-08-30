#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/scripts/libcompose.sh"
ensure_local_no_proxy
"$DIR/scripts/up.sh"
export PYTHONPATH="$DIR/python:${PYTHONPATH:-}"
python3 "$DIR/python/smoke_test.py"
