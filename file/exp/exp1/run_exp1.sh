#!/usr/bin/env bash
set -euo pipefail

# Build exp1 single-environment tree-file assets:
#   1) extract one user subtree from full source tree
#   2) build one security subtree (contains target files for all cases)
#   3) merge into one pre-injection environment tree JSON under runs/<run_id>/env/
#   4) emit attack_cases.jsonl case-to-target logical-path mapping contract
#
# Examples:
#   bash file/exp/exp1/run_exp1.sh
#   RUN_ID=my_run TOP_K=20 bash file/exp/exp1/run_exp1.sh
#   RUN_ID=my_run ANCHOR_LOGICAL_PATH='/0/1' bash file/exp/exp1/run_exp1.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-}"
RUNS_ROOT="${RUNS_ROOT:-file/runs}"
SOURCE_TREE="${SOURCE_TREE:-file/create/file_tree/custom_d11_w2_r_1.json}"
TOP_K="${TOP_K:-20}"
DEPTH="${DEPTH:-6}"
WIDTH="${WIDTH:-2}"
SEED_BASE="${SEED_BASE:-42}"
ROOT_COORD="${ROOT_COORD:-}"

ANCHOR_LOGICAL_PATH="${ANCHOR_LOGICAL_PATH:-/2/2/2/2}"
DS_JSON="${DS_JSON:-file/exp/exp1/data/ds_top20.json}"
SECURITY_SPEC="${SECURITY_SPEC:-file/exp/exp1/data/security_spec.filetree_ds_top20.json}"

USER_QUESTION_MODEL="${USER_QUESTION_MODEL:-gpt-4o-mini}"
USER_QUESTION_CACHE_TTL="${USER_QUESTION_CACHE_TTL:-86400}"
USER_QUESTION_MAX_TOKENS="${USER_QUESTION_MAX_TOKENS:-128}"
USER_QUESTION_TEMPERATURE="${USER_QUESTION_TEMPERATURE:-1.0}"
USER_QUESTION_BODY_MAX_INPUT_TOKENS="${USER_QUESTION_BODY_MAX_INPUT_TOKENS:-2048}"
OVERWRITE="${OVERWRITE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}" file/exp/exp1/build_cases_top20.py
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

if [[ -n "${RUN_ID}" ]]; then
  CMD+=(--run-id "${RUN_ID}")
fi
if [[ -n "${ROOT_COORD}" ]]; then
  CMD+=(--root-coord "${ROOT_COORD}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

"${CMD[@]}"
