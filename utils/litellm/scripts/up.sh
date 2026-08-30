#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/scripts/libcompose.sh"
cd "$DIR"
cp -n .env.example .env >/dev/null 2>&1 || true
run_compose up -d
wait_for_litellm
