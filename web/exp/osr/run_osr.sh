#!/usr/bin/env bash
set -euo pipefail

START_TS="$(date +%s)"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-web/runs}"
: "${EXPERIMENT_ID:=osr_attack_aligned_web}"
PORT="${PORT:-8000}"
START_URL="${START_URL:-http://127.0.0.1:${PORT}/index.html}"

SOURCE_ROOT="${SOURCE_ROOT:-web/exp/02_nested_builder/tree/d10_w2_s42}"
ROOT_PATH="${ROOT_PATH:-/}"
DEPTH="${DEPTH:-10}"
WIDTH="${WIDTH:-2}"
SELECTION_SEED="${SELECTION_SEED:-42}"

WASP_ENV="${WASP_ENV:-gitlab}" # gitlab | reddit
case "${WASP_ENV}" in
  gitlab)
    DEFAULT_SECURITY_SPEC="web/exp/02_5_security_microtree/security_spec.gitlab_wasp.json"
    ;;
  reddit)
    DEFAULT_SECURITY_SPEC="web/exp/02_5_security_microtree/security_spec.reddit_wasp.json"
    ;;
  *)
    echo "[osr] ERROR: unsupported WASP_ENV='${WASP_ENV}' (expected gitlab|reddit)" >&2
    exit 1
    ;;
esac

SECURITY_SPEC="${SECURITY_SPEC:-${DEFAULT_SECURITY_SPEC}}"
ANCHOR_BREADCRUMB="${ANCHOR_BREADCRUMB:-/1/1/1/1}"
USER_TASK_COUNT="${USER_TASK_COUNT:-2}"
ALIGN_CASE_COUNT="${ALIGN_CASE_COUNT:-}"
ATTACK_MICROTREE_IDS="${ATTACK_MICROTREE_IDS:-}"
ATTACK_MICROTREE_IDS_FILE="${ATTACK_MICROTREE_IDS_FILE:-}"

MODEL="${MODEL:-gpt-4o-mini}"
INTENT_MODEL="${INTENT_MODEL:-gpt-4o-mini}"
INTENT_CACHE_TTL="${INTENT_CACHE_TTL:-0}"
INTENT_TEMPERATURE="${INTENT_TEMPERATURE:-0.0}"
INTENT_MAX_TOKENS="${INTENT_MAX_TOKENS:-128}"
BODY_MAX_CHARS="${BODY_MAX_CHARS:-1200}"
TRIALS="${TRIALS:-3}"
MAX_ACTIONS="${MAX_ACTIONS:-20}"
MAX_OBS="${MAX_OBS:-20}"
INTRA_ATTACK_PARALLEL="${INTRA_ATTACK_PARALLEL:-${MAX_PARALLEL:-6}}"
AGENT_DEFENSE_MODE="${AGENT_DEFENSE_MODE:-${DEFENSE_MODE:-default_attack}}"
AGENT_TEMP="${AGENT_TEMP:-${AGENT_TEMPERATURE:-0.8}}"
AGENT_SEED="${AGENT_SEED:-}"

BASE_ROOT="${EXPERIMENT_ROOT%/}/${EXPERIMENT_ID}"
STATIC_DIR="${BASE_ROOT}/static"
BASE_USER_TASKS="${BASE_ROOT}/webarena_tasks"
OSR_TASKS="${BASE_ROOT}/webarena_tasks_osr"
LOG_DIR="${BASE_ROOT}/agent_logs_osr"
USER_UTILITY_OUT="${BASE_ROOT}/user_utility_osr.json"
METRICS_OUT="${BASE_ROOT}/metrics_osr.json"
PATH_STATS_OUT="${BASE_ROOT}/path_stats_osr.json"
SERVER_LOG="${BASE_ROOT}/http_${PORT}.log"

LOCAL_NO_PROXY_HOSTS="${LOCAL_NO_PROXY_HOSTS:-127.0.0.1,localhost}"
export NO_PROXY="${LOCAL_NO_PROXY_HOSTS}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${LOCAL_NO_PROXY_HOSTS}${no_proxy:+,${no_proxy}}"
if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL}}"
elif [[ -n "${OPENAI_API_BASE:-}" ]]; then
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[osr] ERROR: set OPENAI_API_KEY before running OSR." >&2
  exit 2
fi
export DATASET="${DATASET:-visualwebarena}"
export REDDIT="${REDDIT:-http://127.0.0.1:8001}"
export SHOPPING="${SHOPPING:-http://127.0.0.1:8002}"
export GITLAB="${GITLAB:-http://127.0.0.1:8003}"
export WIKIPEDIA="${WIKIPEDIA:-http://127.0.0.1:8004}"
export HOMEPAGE="${HOMEPAGE:-http://127.0.0.1:8005}"
export CLASSIFIEDS="${CLASSIFIEDS:-http://127.0.0.1:8006}"
export CLASSIFIEDS_RESET_TOKEN="${CLASSIFIEDS_RESET_TOKEN:-dummy}"
export REDDIT_RESET_URL="${REDDIT_RESET_URL:-http://127.0.0.1:8001/reset}"

source wasp/visualwebarena/venv/bin/activate

if [[ ! -f "${SECURITY_SPEC}" ]]; then
  echo "[osr] ERROR: SECURITY_SPEC not found: ${SECURITY_SPEC}" >&2
  exit 1
fi

SECURITY_SPEC_BASENAME="$(basename "${SECURITY_SPEC}")"
SPEC_ENV_HINT=""
case "${SECURITY_SPEC_BASENAME}" in
  *gitlab*)
    SPEC_ENV_HINT="gitlab"
    ;;
  *reddit*)
    SPEC_ENV_HINT="reddit"
    ;;
esac
if [[ -n "${SPEC_ENV_HINT}" && "${SPEC_ENV_HINT}" != "${WASP_ENV}" ]]; then
  echo "[osr] ERROR: SECURITY_SPEC (${SECURITY_SPEC_BASENAME}) conflicts with WASP_ENV=${WASP_ENV}" >&2
  exit 1
fi

echo "[osr] experiment=${BASE_ROOT}"
echo "[osr] wasp_env=${WASP_ENV} security_spec=${SECURITY_SPEC}"
echo "[osr] intent_model=${INTENT_MODEL} trials=${TRIALS} defense=${AGENT_DEFENSE_MODE}"
echo "[osr] forcing uncached fresh sampling for OSR-only web runs"

if [[ -n "${AGENT_SEED}" ]]; then
  echo "[osr] ERROR: AGENT_SEED is disabled for OSR fresh sampling. Remove AGENT_SEED to continue." >&2
  exit 2
fi

echo "[osr] Stage 02: extract_subtree"
"${PYTHON_BIN}" web/exp/02_nested_builder/extract_subtree.py \
  --source-root "${SOURCE_ROOT}" \
  --root-path "${ROOT_PATH}" \
  --depth "${DEPTH}" \
  --width "${WIDTH}" \
  --selection-seed "${SELECTION_SEED}" \
  --output-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --force

echo "[osr] Stage 02.5: compile_security_microtree"
"${PYTHON_BIN}" web/exp/02_5_security_microtree/compile_security_microtree.py \
  --spec "${SECURITY_SPEC}" \
  --anchor-breadcrumb "${ANCHOR_BREADCRUMB}" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}"

AVAILABLE_MICROTREE_COUNT="$("${PYTHON_BIN}" - "${BASE_ROOT}/security_microtree/security_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
microtrees = payload.get("microtrees") or []
print(len([row for row in microtrees if isinstance(row, dict)]))
PY
)"
TARGET_MICROTREE_COUNT=""
SELECTED_MICROTREE_COUNT=""
if [[ -n "${ATTACK_MICROTREE_IDS}" || -n "${ATTACK_MICROTREE_IDS_FILE}" ]]; then
  if [[ -n "${ALIGN_CASE_COUNT}" ]]; then
    TARGET_MICROTREE_COUNT="${ALIGN_CASE_COUNT}"
    SELECTED_MICROTREE_COUNT="${ALIGN_CASE_COUNT}"
  else
    SELECTED_MICROTREE_COUNT="$("${PYTHON_BIN}" - "${ATTACK_MICROTREE_IDS}" "${ATTACK_MICROTREE_IDS_FILE}" <<'PY'
import re
import sys
from pathlib import Path

raw_ids = str(sys.argv[1] or "").strip()
ids_file = str(sys.argv[2] or "").strip()
parts = [part.strip() for part in re.split(r"[\s,]+", raw_ids) if part.strip()]
if ids_file:
    file_text = Path(ids_file).read_text(encoding="utf-8")
    parts.extend(part.strip() for part in re.split(r"[\s,]+", file_text) if part.strip())
print(len(parts))
PY
)"
  fi
else
  TARGET_MICROTREE_COUNT="${ALIGN_CASE_COUNT:-${AVAILABLE_MICROTREE_COUNT}}"
  SELECTED_MICROTREE_COUNT="${TARGET_MICROTREE_COUNT}"
fi
echo "[osr] aligned samples=${USER_TASK_COUNT}*${SELECTED_MICROTREE_COUNT}"

echo "[osr] Stage 03: pack_static_site (merged tree)"
"${PYTHON_BIN}" web/exp/03_a2perf_render/pack_static_site.py \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --design "${BASE_ROOT}/security_microtree/website_designs.json" \
  --metadata "${BASE_ROOT}/security_microtree/page_metadata.json" \
  --transitions "${BASE_ROOT}/security_microtree/transitions.json"

rm -rf "${BASE_USER_TASKS}" "${OSR_TASKS}" "${LOG_DIR}"
rm -f "${USER_UTILITY_OUT}" "${METRICS_OUT}" "${PATH_STATS_OUT}"

echo "[osr] Serving ${STATIC_DIR} on ${START_URL} (log: ${SERVER_LOG})"
"${PYTHON_BIN}" -m http.server "${PORT}" --directory "${STATIC_DIR}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT
sleep 0.3
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[osr] ERROR: failed to start http.server (see ${SERVER_LOG})" >&2
  exit 1
fi

echo "[osr] Stage 04: make base user tasks"
"${PYTHON_BIN}" web/exp/osr/make_user_tasks_uncached.py \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --count "${USER_TASK_COUNT}" \
  --start-url "${START_URL}" \
  --intent-model "${INTENT_MODEL}" \
  --intent-cache-ttl "${INTENT_CACHE_TTL}" \
  --intent-temperature "${INTENT_TEMPERATURE}" \
  --intent-max-tokens "${INTENT_MAX_TOKENS}" \
  --body-max-chars "${BODY_MAX_CHARS}"

echo "[osr] Stage 04.5: build navigation-guided OSR tasks"
OSR_ALIGN_CMD=(
  "${PYTHON_BIN}" web/exp/osr/make_attack_aligned_osr_tasks.py
  --experiment-root "${EXPERIMENT_ROOT}"
  --experiment-id "${EXPERIMENT_ID}"
  --base-user-tasks "${BASE_USER_TASKS}"
  --security-manifest "${BASE_ROOT}/security_microtree/security_manifest.json"
  --page-metadata "${BASE_ROOT}/security_microtree/page_metadata.json"
  --output-dir "${OSR_TASKS}"
  --expected-base-task-count "${USER_TASK_COUNT}"
  --start-url "${START_URL}"
)
if [[ -n "${TARGET_MICROTREE_COUNT}" ]]; then
  OSR_ALIGN_CMD+=(--count "${TARGET_MICROTREE_COUNT}")
fi
if [[ -n "${ATTACK_MICROTREE_IDS}" ]]; then
  OSR_ALIGN_CMD+=(--microtree-ids "${ATTACK_MICROTREE_IDS}")
fi
if [[ -n "${ATTACK_MICROTREE_IDS_FILE}" ]]; then
  OSR_ALIGN_CMD+=(--microtree-ids-file "${ATTACK_MICROTREE_IDS_FILE}")
fi
"${OSR_ALIGN_CMD[@]}"

echo "[osr] Stage 05: run agent on navigation-guided OSR tasks"
RUN_AGENT_CMD=(
  "${PYTHON_BIN}" web/exp/osr/run_agent_uncached.py
  --experiment-root "${EXPERIMENT_ROOT}"
  --experiment-id "${EXPERIMENT_ID}"
  --tasks "${OSR_TASKS}"
  --result-dir "${LOG_DIR}"
  --model "${MODEL}"
  --dataset "${DATASET}"
  --defense-mode "${AGENT_DEFENSE_MODE}"
  --max-actions "${MAX_ACTIONS}"
  --max-observations "${MAX_OBS}"
  --trials "${TRIALS}"
  --max-parallel "${INTRA_ATTACK_PARALLEL}"
  --url-timeout 5
)
if [[ -n "${AGENT_TEMP}" ]]; then
  RUN_AGENT_CMD+=(--temperature "${AGENT_TEMP}")
fi
"${RUN_AGENT_CMD[@]}"

non_empty_traces=0
shopt -s nullglob
for trace_file in "${LOG_DIR%/}"/trace_*.jsonl; do
  if [[ -s "${trace_file}" ]]; then
    non_empty_traces=$((non_empty_traces + 1))
  fi
done
shopt -u nullglob
if [[ "${non_empty_traces}" -eq 0 ]]; then
  echo "[osr] ERROR: all trace logs are empty under ${LOG_DIR}." >&2
  echo "[osr] Hint: the configured OpenAI-compatible endpoint must support tool calls." >&2
  exit 1
fi

echo "[osr] Stage 06: extract breadcrumb paths"
"${PYTHON_BIN}" web/exp/05_evaluate/extract_paths.py \
  --logs "${LOG_DIR}" \
  --out "${PATH_STATS_OUT}" \
  --format gpt_web_tools \
  --page-metadata "${BASE_ROOT}/static/page_metadata.json" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}"

echo "[osr] Stage 07: evaluate final-page user success"
"${PYTHON_BIN}" web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py \
  --log-folder "${LOG_DIR}" \
  --task-folder "${OSR_TASKS}" \
  --page-metadata "${BASE_ROOT}/static/page_metadata.json" \
  --out "${USER_UTILITY_OUT}"

"${PYTHON_BIN}" web/exp/osr/summarize_osr_metrics.py \
  --task-manifest "${OSR_TASKS}/manifest.json" \
  --user-utility "${USER_UTILITY_OUT}" \
  --path-stats "${PATH_STATS_OUT}" \
  --out "${METRICS_OUT}"

END_TS="$(date +%s)"
TOTAL_SECONDS="$((END_TS - START_TS))"
printf '[osr] Total elapsed: %02d:%02d:%02d (%ds)\n' \
  "$((TOTAL_SECONDS / 3600))" \
  "$(((TOTAL_SECONDS % 3600) / 60))" \
  "$((TOTAL_SECONDS % 60))" \
  "${TOTAL_SECONDS}"

echo "[osr] Done: ${METRICS_OUT}"
