#!/usr/bin/env bash
set -euo pipefail

# Run task generation + tool_call agent execution on an existing run directory.
#
# Usage:
#   bash file/exp/exp2/run_exp2.sh --run-dir file/runs/<run_id>

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-gpt-4o-mini}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-256}" # 0 = backend default
MAX_STEPS="${MAX_STEPS:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
USE_LLM_TASK_GEN="${USE_LLM_TASK_GEN:-0}"
DEFENSE_MODE="${DEFENSE_MODE:-default_attack}" # default_attack | system_prompt_defense | step_wise_prompt_defense | goal_reinforce_ignore | goal_reinforce_fakecom_t | segment_remove_gated | segment_remove_direct

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR=""
CASE_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --case-id)
      CASE_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "--run-dir is required" >&2
  exit 1
fi

TASK_CMD=(
  "${PYTHON_BIN}" file/exp/exp2/task_gen.py
  --run-dir "${RUN_DIR}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
)
if [[ "${USE_LLM_TASK_GEN}" == "1" ]]; then
  TASK_CMD+=(--use-llm)
fi
"${TASK_CMD[@]}"

RUN_CMD=(
  "${PYTHON_BIN}" file/exp/exp2/agent_runner.py
  --run-dir "${RUN_DIR}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --max-steps "${MAX_STEPS}"
  --num-samples "${NUM_SAMPLES}"
  --defense-mode "${DEFENSE_MODE}"
)
if [[ -n "${CASE_ID}" ]]; then
  RUN_CMD+=(--case-id "${CASE_ID}")
fi
"${RUN_CMD[@]}"
