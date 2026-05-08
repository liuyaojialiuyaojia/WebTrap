#!/usr/bin/env bash
set -euo pipefail

# One-click runner for file/find depth-range inertia experiments.
# Default task: state_judgment (outputs judgment_failure.csv under file/runs/...).
# Optional task: next_step (outputs action_inertia.csv under file/runs/...).
#
# Quick examples:
#   bash file/find/run_find.sh
#   TASK=next_step bash file/find/run_find.sh
#   DEPTH_MIN=2 DEPTH_MAX=8 bash file/find/run_find.sh
#   bash file/find/run_find.sh --tests-per-depth 30 --history-gap 3
#
# Note:
# - Model calls use the OpenAI-compatible client in utils/llm_client.py.
# - Set OPENAI_API_KEY and optionally OPENAI_BASE_URL/OPENAI_API_BASE first.
# - Values below are used as defaults.
# - Any CLI arguments appended to this script are forwarded to file/find/main.py
#   and can override these defaults.

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-next_step}" # state_judgment or next_step
MODEL="${MODEL:-gpt-4o-mini}"
TEMPERATURE="${TEMPERATURE:-1.0}"
HISTORY_GAP="${HISTORY_GAP:-2}"
DEPTH_MIN="${DEPTH_MIN:-1}"
DEPTH_MAX="${DEPTH_MAX:-10}"
SOURCE_TREE="${SOURCE_TREE:-file/create/file_tree/custom_d11_w2_r_1.json}"
FIXED_WIDTH="${FIXED_WIDTH:-2}"
TESTS_PER_DEPTH="${TESTS_PER_DEPTH:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" file/find/main.py \
  --task "${TASK}" \
  --model "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --history-gap "${HISTORY_GAP}" \
  --depth-min "${DEPTH_MIN}" \
  --depth-max "${DEPTH_MAX}" \
  --source-tree "${SOURCE_TREE}" \
  --fixed-width "${FIXED_WIDTH}" \
  --tests-per-depth "${TESTS_PER_DEPTH}" \
  "$@"
