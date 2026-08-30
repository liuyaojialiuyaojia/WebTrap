#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/scripts/libcompose.sh"
cd "$DIR"
run_compose up -d proxy-bridge
run_compose up -d --force-recreate --no-deps litellm
wait_for_litellm
