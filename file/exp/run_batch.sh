#!/usr/bin/env bash
set -euo pipefail

# Unified runner for file/exp pipeline:
#   exp1 (once) -> attack + exp2 (per case) -> exp3 (once, optional)
#
# Usage:
#   bash file/exp/run_batch.sh
#
# Common env vars:
#   RUN_ID:        run directory name under RUNS_ROOT (default auto-generated)
#   RUNS_ROOT:     default file/runs
#   PYTHON_BIN:    default python
#
# Pipeline config env vars (only this script is the "source of truth"):
#
# exp1 (environment build)
#   SOURCE_TREE:   source full tree JSON
#   TOP_K:         number of attack cases (default 20)
#   DEPTH/WIDTH:   extracted user subtree params
#   SEED_BASE:     seed base for exp1 case generation
#   ROOT_COORD:    optional subtree root coord "depth,index"
#   ANCHOR_LOGICAL_PATH: anchor index path like /0/1 (root is /)
#   DS_JSON:       dataset top-k config JSON
#   SECURITY_SPEC: security tree spec JSON
#   USER_QUESTION_MODEL: exp1 user-question LLM
#   USER_QUESTION_*:     exp1 user-question generation params
#   OVERWRITE:     0 | 1 (overwrite existing run_dir)
#
# attack (injection)
#   SETTING:       base | enhanced (baseline injection)
#   INJECTION_MODE:baseline | psaa | topicattack | combinedattack
#   PSAA_SPEC:     YAML spec path for psaa injection
#   PSAA_START_INDEX_PATH: psaa injection start dir (index path like /0/1, root is /)
#   TOPICATTACK_MODEL/TOPICATTACK_NUM_TURNS/TOPICATTACK_TEMPERATURE/TOPICATTACK_MAX_TOKENS:
#                  official TopicAttack agent construction params
#   TOPICATTACK_CACHE_TTL/TOPICATTACK_WITH_REMINDER/TOPICATTACK_SYSTEM_MESSAGE:
#                  TopicAttack runtime params
#   TOPICATTACK_INJECT_INDEX_PATH: topicattack injection dir (index path like /0/1, root is /)
#   COMBINEDATTACK_SEED: deterministic seed for Combined Attack sampling
#   COMBINEDATTACK_INJECT_INDEX_PATH: combinedattack injection dir (index path like /0/1, root is /)
#
# exp2 (task generation + agent run)
#   MODEL:         exp2 model (also used by exp2 task_gen when USE_LLM_TASK_GEN=1)
#   TEMPERATURE:   exp2 temperature (also used by exp2 task_gen when USE_LLM_TASK_GEN=1)
#   MAX_STEPS:     exp2 agent max steps
#   MAX_TOKENS:    exp2 agent max tokens (0 = backend default)
#   NUM_SAMPLES:   exp2 per-case sampling count (>=1, concurrent)
#   DEFENSE_MODE:  default_attack | system_prompt_defense | step_wise_prompt_defense | goal_reinforce_ignore | goal_reinforce_fakecom_t | segment_remove_gated | segment_remove_direct
#   USE_LLM_TASK_GEN: 0 | 1
#
# exp3 (evaluation)
#   EVAL:          0 | 1 (run exp3 at end; default 1)
#   ASR_FORMAT:    reasoning (enforced by the main file pipeline)
#   ASR_MODEL:     exp3 ASR-intermediate classifier model
#   ASR_TEMPERATURE: exp3 classifier temperature
#   ASR_MAX_TOKENS: exp3 classifier max tokens (0 = backend default)
#
# Case selection env vars:
#   CASES:         comma/space-separated case ids (e.g., "case_0001,case_0002")
#   CASE_START:    1-based start index in contract order (default 1)
#   CASE_END:      1-based end index in contract order (default last)
#
# Optional:
#   EVAL:          0 | 1 (run exp3 at end; default 1)
#   SKIP_DONE:     0 | 1 (skip cases with existing logs/trace_<case>.jsonl)

START_TS="$(date +%s)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-file/runs}"
RUN_ID="${RUN_ID:-}"

# exp1 defaults (must match the intended experiment config)
SOURCE_TREE="${SOURCE_TREE:-file/create/file_tree/custom_d11_w2_r_1.json}"
TOP_K="${TOP_K:-20}" # attack cases
DEPTH="${DEPTH:-10}"
WIDTH="${WIDTH:-2}"
SEED_BASE="${SEED_BASE:-42}"
ROOT_COORD="${ROOT_COORD:-}" # select root for user tree

ANCHOR_LOGICAL_PATH="${ANCHOR_LOGICAL_PATH:-/2/2/2/2/2/2}" # insertion anchor in the extracted user tree
DS_JSON="${DS_JSON:-file/exp/exp1/data/ds_top20.json}"
SECURITY_SPEC="${SECURITY_SPEC:-file/exp/exp1/data/security_spec.filetree_ds_top20.json}"

USER_QUESTION_MODEL="${USER_QUESTION_MODEL:-gpt-4o-mini}"
USER_QUESTION_CACHE_TTL="${USER_QUESTION_CACHE_TTL:-86400}"
USER_QUESTION_MAX_TOKENS="${USER_QUESTION_MAX_TOKENS:-128}"
USER_QUESTION_TEMPERATURE="${USER_QUESTION_TEMPERATURE:-1.0}"
USER_QUESTION_BODY_MAX_INPUT_TOKENS="${USER_QUESTION_BODY_MAX_INPUT_TOKENS:-2048}"
OVERWRITE="${OVERWRITE:-0}"

# attack defaults
INJECTION_MODE="${INJECTION_MODE:-psaa}" # baseline | psaa | topicattack | combinedattack
SETTING="${SETTING:-base}" # base | enhanced for baseline
PSAA_SPEC="${PSAA_SPEC:-file/exp/attack/psaa/file_psaa_v1.yaml}"
PSAA_START_INDEX_PATH="${PSAA_START_INDEX_PATH:-/}"

TOPICATTACK_MODEL="${TOPICATTACK_MODEL:-gpt-4o}"
TOPICATTACK_NUM_TURNS="${TOPICATTACK_NUM_TURNS:-5}"
TOPICATTACK_TEMPERATURE="${TOPICATTACK_TEMPERATURE:-0.0}"
TOPICATTACK_MAX_TOKENS="${TOPICATTACK_MAX_TOKENS:-2048}"
TOPICATTACK_CACHE_TTL="${TOPICATTACK_CACHE_TTL:-86400}"
TOPICATTACK_WITH_REMINDER="${TOPICATTACK_WITH_REMINDER:-true}"
TOPICATTACK_SYSTEM_MESSAGE="${TOPICATTACK_SYSTEM_MESSAGE:-}"
TOPICATTACK_INJECT_INDEX_PATH="${TOPICATTACK_INJECT_INDEX_PATH:-/}"
TOPICATTACK_USER_GOAL_STR="${TOPICATTACK_USER_GOAL_STR:-}"

COMBINEDATTACK_SEED="${COMBINEDATTACK_SEED:-42}"
COMBINEDATTACK_INJECT_INDEX_PATH="${COMBINEDATTACK_INJECT_INDEX_PATH:-/}"
COMBINEDATTACK_USER_GOAL_STR="${COMBINEDATTACK_USER_GOAL_STR:-}"

# exp2 defaults
MODEL="${MODEL:-gpt-4o-mini}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_STEPS="${MAX_STEPS:-25}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
DEFENSE_MODE="${DEFENSE_MODE:-default_attack}" # default_attack | system_prompt_defense | step_wise_prompt_defense | goal_reinforce_ignore | goal_reinforce_fakecom_t | segment_remove_gated | segment_remove_direct
USE_LLM_TASK_GEN="${USE_LLM_TASK_GEN:-0}"

CASES="${CASES:-}" # empty means run all cases
CASE_START="${CASE_START:-}"
CASE_END="${CASE_END:-}"

# exp3 defaults
EVAL="${EVAL:-1}"
ASR_FORMAT="${ASR_FORMAT:-reasoning}"
ASR_MODEL="${ASR_MODEL:-gpt-4o-mini}"
ASR_TEMPERATURE="${ASR_TEMPERATURE:-0.0}"
ASR_MAX_TOKENS="${ASR_MAX_TOKENS:-512}"

if [[ "${ASR_FORMAT}" != "reasoning" ]]; then
  echo "[run_batch] ERROR: ASR_FORMAT must be 'reasoning' in the main file pipeline." >&2
  echo "[run_batch] Refusing unsupported ASR_FORMAT='${ASR_FORMAT}'." >&2
  exit 2
fi

###
SKIP_DONE="${SKIP_DONE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_MODE_TAG="${INJECTION_MODE}"
  if [[ "${INJECTION_MODE}" == "baseline" ]]; then
    RUN_MODE_TAG="${SETTING}_${INJECTION_MODE}"
  fi
  RUN_ID="$(date +%Y%m%d_%H%M%S)_file_ds_top20_${RUN_MODE_TAG}"
fi

RUN_DIR="${RUNS_ROOT%/}/${RUN_ID}"

echo "[run_batch] run_id=${RUN_ID}"
echo "[run_batch] run_dir=${RUN_DIR}"
echo "[run_batch] injection_mode=${INJECTION_MODE} setting=${SETTING}"
if [[ "${INJECTION_MODE}" == "psaa" ]]; then
  echo "[run_batch] psaa: spec=${PSAA_SPEC} start_index_path=${PSAA_START_INDEX_PATH}"
elif [[ "${INJECTION_MODE}" == "topicattack" ]]; then
  echo "[run_batch] topicattack: model=${TOPICATTACK_MODEL} turns=${TOPICATTACK_NUM_TURNS} temp=${TOPICATTACK_TEMPERATURE} max_tokens=${TOPICATTACK_MAX_TOKENS} with_reminder=${TOPICATTACK_WITH_REMINDER} inject_index_path=${TOPICATTACK_INJECT_INDEX_PATH}"
elif [[ "${INJECTION_MODE}" == "combinedattack" ]]; then
  echo "[run_batch] combinedattack: seed=${COMBINEDATTACK_SEED} inject_index_path=${COMBINEDATTACK_INJECT_INDEX_PATH}"
fi
echo "[run_batch] exp1 subtree: source_tree=${SOURCE_TREE} depth=${DEPTH} width=${WIDTH} seed_base=${SEED_BASE}"
echo "[run_batch] exp1 security: spec=${SECURITY_SPEC} anchor=${ANCHOR_LOGICAL_PATH}"
echo "[run_batch] exp1 user question: model=${USER_QUESTION_MODEL} temp=${USER_QUESTION_TEMPERATURE}"
echo "[run_batch] exp2: model=${MODEL} temp=${TEMPERATURE} max_steps=${MAX_STEPS} max_tokens=${MAX_TOKENS} num_samples=${NUM_SAMPLES} defense=${DEFENSE_MODE} task_gen_llm=${USE_LLM_TASK_GEN}"
echo "[run_batch] exp3: eval=${EVAL} asr_format=${ASR_FORMAT} asr_model=${ASR_MODEL} asr_temp=${ASR_TEMPERATURE}"

echo "[run_batch] step 1/3: exp1 (build env + contract)"
RUN_ID="${RUN_ID}" RUNS_ROOT="${RUNS_ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
  SOURCE_TREE="${SOURCE_TREE}" TOP_K="${TOP_K}" DEPTH="${DEPTH}" WIDTH="${WIDTH}" \
  SEED_BASE="${SEED_BASE}" ROOT_COORD="${ROOT_COORD}" ANCHOR_LOGICAL_PATH="${ANCHOR_LOGICAL_PATH}" \
  DS_JSON="${DS_JSON}" SECURITY_SPEC="${SECURITY_SPEC}" \
  USER_QUESTION_MODEL="${USER_QUESTION_MODEL}" USER_QUESTION_CACHE_TTL="${USER_QUESTION_CACHE_TTL}" \
  USER_QUESTION_MAX_TOKENS="${USER_QUESTION_MAX_TOKENS}" USER_QUESTION_TEMPERATURE="${USER_QUESTION_TEMPERATURE}" \
  USER_QUESTION_BODY_MAX_INPUT_TOKENS="${USER_QUESTION_BODY_MAX_INPUT_TOKENS}" OVERWRITE="${OVERWRITE}" \
  bash file/exp/exp1/run_exp1.sh

CONTRACT_PATH="${RUN_DIR}/attack_cases.jsonl"
PRE_CONTRACT_PATH="${RUN_DIR}/attack_cases_pre_injection.jsonl"

if [[ ! -f "${CONTRACT_PATH}" ]]; then
  echo "[run_batch] ERROR: missing contract: ${CONTRACT_PATH}" >&2
  exit 1
fi

# Keep a stable copy of the original contract so exp3 can evaluate all cases.
if [[ ! -f "${PRE_CONTRACT_PATH}" ]]; then
  cp -f "${CONTRACT_PATH}" "${PRE_CONTRACT_PATH}"
fi

ALL_CASE_IDS="$(
  "${PYTHON_BIN}" - "${PRE_CONTRACT_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
case_ids = []
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    cid = str(row.get("case_id") or "").strip()
    if cid:
        case_ids.append(cid)
print(" ".join(case_ids))
PY
)"

if [[ -z "${ALL_CASE_IDS}" ]]; then
  echo "[run_batch] ERROR: no case_ids found in ${PRE_CONTRACT_PATH}" >&2
  exit 1
fi

SELECTED_CASE_IDS=()
if [[ -n "${CASES}" ]]; then
  normalized="${CASES//,/ }"
  read -r -a SELECTED_CASE_IDS <<<"${normalized}"
else
  read -r -a SELECTED_CASE_IDS <<<"${ALL_CASE_IDS}"
  start="${CASE_START:-1}"
  end="${CASE_END:-${#SELECTED_CASE_IDS[@]}}"
  if ! [[ "${start}" =~ ^[0-9]+$ && "${end}" =~ ^[0-9]+$ ]]; then
    echo "[run_batch] ERROR: CASE_START/CASE_END must be integers (got start=${start} end=${end})." >&2
    exit 1
  fi
  if (( start < 1 || end < start || end > ${#SELECTED_CASE_IDS[@]} )); then
    echo "[run_batch] ERROR: invalid case range start=${start} end=${end} total=${#SELECTED_CASE_IDS[@]}." >&2
    exit 1
  fi
  SELECTED_CASE_IDS=("${SELECTED_CASE_IDS[@]:$((start - 1)):$((end - start + 1))}")
fi

if (( ${#SELECTED_CASE_IDS[@]} == 0 )); then
  echo "[run_batch] ERROR: selected empty case set." >&2
  exit 1
fi

echo "[run_batch] step 2/3: per-case attack + exp2 (${#SELECTED_CASE_IDS[@]} case(s))"

for CASE_ID in "${SELECTED_CASE_IDS[@]}"; do
  if [[ -z "${CASE_ID}" ]]; then
    continue
  fi
  echo "[run_batch] --- ${CASE_ID} ---"

  TRACE_PATH="${RUN_DIR}/logs/trace_${CASE_ID}.jsonl"
  if [[ "${SKIP_DONE}" == "1" && -f "${TRACE_PATH}" ]]; then
    echo "[run_batch] skip: trace exists (${TRACE_PATH})"
    continue
  fi

  echo "[run_batch] attack (${INJECTION_MODE})"
  INJECTION_MODE="${INJECTION_MODE}" SETTING="${SETTING}" PSAA_SPEC="${PSAA_SPEC}" \
    PSAA_START_INDEX_PATH="${PSAA_START_INDEX_PATH}" \
    TOPICATTACK_MODEL="${TOPICATTACK_MODEL}" \
    TOPICATTACK_NUM_TURNS="${TOPICATTACK_NUM_TURNS}" \
    TOPICATTACK_TEMPERATURE="${TOPICATTACK_TEMPERATURE}" \
    TOPICATTACK_MAX_TOKENS="${TOPICATTACK_MAX_TOKENS}" \
    TOPICATTACK_CACHE_TTL="${TOPICATTACK_CACHE_TTL}" \
    TOPICATTACK_WITH_REMINDER="${TOPICATTACK_WITH_REMINDER}" \
    TOPICATTACK_SYSTEM_MESSAGE="${TOPICATTACK_SYSTEM_MESSAGE}" \
    TOPICATTACK_INJECT_INDEX_PATH="${TOPICATTACK_INJECT_INDEX_PATH}" \
    TOPICATTACK_USER_GOAL_STR="${TOPICATTACK_USER_GOAL_STR}" \
    COMBINEDATTACK_SEED="${COMBINEDATTACK_SEED}" \
    COMBINEDATTACK_INJECT_INDEX_PATH="${COMBINEDATTACK_INJECT_INDEX_PATH}" \
    COMBINEDATTACK_USER_GOAL_STR="${COMBINEDATTACK_USER_GOAL_STR}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash file/exp/attack/run_attack.sh --run-dir "${RUN_DIR}" --case-id "${CASE_ID}"

  echo "[run_batch] exp2"
  MODEL="${MODEL}" TEMPERATURE="${TEMPERATURE}" MAX_TOKENS="${MAX_TOKENS}" MAX_STEPS="${MAX_STEPS}" NUM_SAMPLES="${NUM_SAMPLES}" \
    DEFENSE_MODE="${DEFENSE_MODE}" USE_LLM_TASK_GEN="${USE_LLM_TASK_GEN}" PYTHON_BIN="${PYTHON_BIN}" \
    bash file/exp/exp2/run_exp2.sh --run-dir "${RUN_DIR}" --case-id "${CASE_ID}"
done

# Restore full contract for exp3 evaluation (attack injection overwrites attack_cases.jsonl per case).
cp -f "${PRE_CONTRACT_PATH}" "${CONTRACT_PATH}"

if [[ "${EVAL}" == "1" ]]; then
  echo "[run_batch] step 3/3: exp3 (evaluate)"
  PYTHON_BIN="${PYTHON_BIN}" \
    ASR_FORMAT="${ASR_FORMAT}" ASR_MODEL="${ASR_MODEL}" ASR_TEMPERATURE="${ASR_TEMPERATURE}" \
    ASR_MAX_TOKENS="${ASR_MAX_TOKENS}" \
    bash file/exp/exp3/run_exp3.sh --run-dir "${RUN_DIR}"
else
  echo "[run_batch] step 3/3: exp3 skipped (EVAL=${EVAL})"
fi

END_TS="$(date +%s)"
TOTAL_SECONDS="$((END_TS - START_TS))"
printf '[run_batch] Total elapsed: %02d:%02d:%02d (%ds)\n' \
  "$((TOTAL_SECONDS / 3600))" \
  "$(((TOTAL_SECONDS % 3600) / 60))" \
  "$((TOTAL_SECONDS % 60))" \
  "${TOTAL_SECONDS}"

echo "[run_batch] done: ${RUN_DIR}"
