#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-osr_attack_aligned_file}"
RUNS_ROOT="${RUNS_ROOT:-file/runs}"
RUN_DIR="${RUNS_ROOT%/}/${RUN_ID}"

SOURCE_TREE="${SOURCE_TREE:-file/create/file_tree/custom_d11_w2_r_1.json}"
TOP_K="${TOP_K:-20}"
DEPTH="${DEPTH:-10}"
WIDTH="${WIDTH:-2}"
SEED_BASE="${SEED_BASE:-42}"
ROOT_COORD="${ROOT_COORD:-}"
ANCHOR_LOGICAL_PATH="${ANCHOR_LOGICAL_PATH:-/2/2/2/2/2/2}"
DS_JSON="${DS_JSON:-file/exp/exp1/data/ds_top20.json}"
SECURITY_SPEC="${SECURITY_SPEC:-file/exp/exp1/data/security_spec.filetree_ds_top20.json}"
USER_QUESTION_MODEL="${USER_QUESTION_MODEL:-gpt-4o-mini}"
USER_QUESTION_CACHE_TTL="${USER_QUESTION_CACHE_TTL:-0}"
USER_QUESTION_MAX_TOKENS="${USER_QUESTION_MAX_TOKENS:-128}"
USER_QUESTION_TEMPERATURE="${USER_QUESTION_TEMPERATURE:-1.0}"
USER_QUESTION_BODY_MAX_INPUT_TOKENS="${USER_QUESTION_BODY_MAX_INPUT_TOKENS:-2048}"
OVERWRITE="${OVERWRITE:-0}"

MODEL="${MODEL:-gpt-4o-mini}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_STEPS="${MAX_STEPS:-25}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
DEFENSE_MODE="${DEFENSE_MODE:-default_attack}"
USE_LLM_TASK_GEN="${USE_LLM_TASK_GEN:-0}"
CASE_ID="${CASE_ID:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

START_TS="$(date +%s)"

echo "[osr] Stage 01: build clean file OSR cases (1 user task x 20 cases)"
echo "[osr] forcing uncached fresh sampling for OSR-only file runs"
EXP1_CMD=(
  "${PYTHON_BIN}" file/exp/osr/run_exp1_uncached.py
  --run-id "${RUN_ID}"
  --runs-root "${RUNS_ROOT}"
  --source-tree "${SOURCE_TREE}"
  --top-k "${TOP_K}"
  --depth "${DEPTH}"
  --width "${WIDTH}"
  --seed-base "${SEED_BASE}"
  --anchor-logical-path "${ANCHOR_LOGICAL_PATH}"
  --ds-json "${DS_JSON}"
  --security-spec "${SECURITY_SPEC}"
  --user-question-model "${USER_QUESTION_MODEL}"
  --user-question-cache-ttl "${USER_QUESTION_CACHE_TTL}"
  --user-question-max-tokens "${USER_QUESTION_MAX_TOKENS}"
  --user-question-temperature "${USER_QUESTION_TEMPERATURE}"
  --user-question-body-max-input-tokens "${USER_QUESTION_BODY_MAX_INPUT_TOKENS}"
)
if [[ -n "${ROOT_COORD}" ]]; then
  EXP1_CMD+=(--root-coord "${ROOT_COORD}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  EXP1_CMD+=(--overwrite)
fi
"${EXP1_CMD[@]}"

echo "[osr] Stage 02: generate clean task files"
TASK_GEN_CMD=(
  "${PYTHON_BIN}" file/exp/osr/task_gen_uncached.py
  --run-dir "${RUN_DIR}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
)
if [[ "${USE_LLM_TASK_GEN}" == "1" ]]; then
  TASK_GEN_CMD+=(--use-llm)
fi
"${TASK_GEN_CMD[@]}"

echo "[osr] Stage 03: inject aligned security-entry path hints into OSR tasks"
PREPARE_CMD=(
  "${PYTHON_BIN}" file/exp/osr/prepare_osr_tasks.py
  --run-dir "${RUN_DIR}"
)
if [[ -n "${CASE_ID}" ]]; then
  PREPARE_CMD+=(--case-id "${CASE_ID}")
fi
"${PREPARE_CMD[@]}"

echo "[osr] Stage 04: run agent"
AGENT_CMD=(
  "${PYTHON_BIN}" file/exp/osr/agent_runner_uncached.py
  --run-dir "${RUN_DIR}"
  --model "${MODEL}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --max-steps "${MAX_STEPS}"
  --num-samples "${NUM_SAMPLES}"
  --defense-mode "${DEFENSE_MODE}"
)
if [[ -n "${CASE_ID}" ]]; then
  AGENT_CMD+=(--case-id "${CASE_ID}")
fi
"${AGENT_CMD[@]}"

echo "[osr] Stage 05: evaluate user-task success with the standard SR rule"
EVAL_CMD=(
  "${PYTHON_BIN}" file/exp/exp3/eval_user_goal.py
  --run-dir "${RUN_DIR}"
  --include-samples
)
if [[ -n "${CASE_ID}" ]]; then
  EVAL_CMD+=(--case-id "${CASE_ID}")
fi
"${EVAL_CMD[@]}"

echo "[osr] Stage 06: summarize OSR metrics"
OSR_SUMMARY_CMD=(
  "${PYTHON_BIN}" file/exp/osr/evaluate_osr.py
  --run-dir "${RUN_DIR}"
)
if [[ -n "${CASE_ID}" ]]; then
  OSR_SUMMARY_CMD+=(--case-id "${CASE_ID}")
fi
"${OSR_SUMMARY_CMD[@]}"

END_TS="$(date +%s)"
TOTAL_SECONDS="$((END_TS - START_TS))"
printf '[osr] Total elapsed: %02d:%02d:%02d (%ds)\n' \
  "$((TOTAL_SECONDS / 3600))" \
  "$(((TOTAL_SECONDS % 3600) / 60))" \
  "$((TOTAL_SECONDS % 60))" \
  "${TOTAL_SECONDS}"

echo "[osr] Done: ${RUN_DIR}/eval/osr_metrics.json"
