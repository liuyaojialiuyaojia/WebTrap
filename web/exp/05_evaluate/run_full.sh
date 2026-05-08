#!/usr/bin/env bash

set -euo pipefail

# Stage 05: run user tasks, extract paths, evaluate, and summarize.
#
# - User-only evaluation: `bash web/exp/05_evaluate/run_full.sh --user-only`
# - User + attacker evaluation: `bash web/exp/05_evaluate/run_full.sh --with-attacker`

LOCAL_NO_PROXY_HOSTS="${LOCAL_NO_PROXY_HOSTS:-127.0.0.1,localhost}"
export NO_PROXY="${LOCAL_NO_PROXY_HOSTS}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${LOCAL_NO_PROXY_HOSTS}${no_proxy:+,${no_proxy}}"

# User-only is the default; pass --with-attacker to evaluate attack tasks.
RUN_ATTACKER=0
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-}"
AGENT_SEED="${AGENT_SEED:-}"
AGENT_DEFENSE_MODE="${AGENT_DEFENSE_MODE:-default_attack}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-attacker)
      RUN_ATTACKER=1
      shift
      ;;
    --user-only)
      RUN_ATTACKER=0
      shift
      ;;
    --temperature)
      if [[ $# -lt 2 ]]; then
        echo "[stage05] ERROR: --temperature expects a value" >&2
        exit 2
      fi
      AGENT_TEMPERATURE="$2"
      shift 2
      ;;
    --seed)
      if [[ $# -lt 2 ]]; then
        echo "[stage05] ERROR: --seed expects a value" >&2
        exit 2
      fi
      AGENT_SEED="$2"
      shift 2
      ;;
    --defense-mode)
      if [[ $# -lt 2 ]]; then
        echo "[stage05] ERROR: --defense-mode expects a value" >&2
        exit 2
      fi
      AGENT_DEFENSE_MODE="$2"
      shift 2
      ;;
    *)
      echo "[stage05] Warning: unknown argument $1 ignored"
      shift
      ;;
  esac
done

case "${AGENT_DEFENSE_MODE}" in
  default_attack|system_prompt_defense|step_wise_prompt_defense|goal_reinforce_ignore|goal_reinforce_fakecom_t|segment_remove_gated|segment_remove_direct)
    ;;
  *)
    echo "[stage05] ERROR: invalid --defense-mode '${AGENT_DEFENSE_MODE}'." >&2
    echo "[stage05] Allowed values: default_attack | system_prompt_defense | step_wise_prompt_defense | goal_reinforce_ignore | goal_reinforce_fakecom_t | segment_remove_gated | segment_remove_direct" >&2
    exit 2
    ;;
esac

BASE_ROOT="${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}"
if [[ -n "${EXPERIMENT_ID:-}" ]]; then
  BASE_ROOT="${BASE_ROOT%/}/${EXPERIMENT_ID}"
fi
BASE_ROOT="${BASE_ROOT%/}"
LOG_DIR="${LOG_DIR:-${BASE_ROOT}/agent_logs_post_injection}"
USER_TASKS="${USER_TASKS:-${BASE_ROOT}/webarena_tasks}"
ATTACKER_TASKS="${ATTACKER_TASKS:-${BASE_ROOT}/webarena_tasks_attacker}"
PAGE_METADATA="${PAGE_METADATA:-${BASE_ROOT}/static/page_metadata.json}"
METRICS_OUT="${METRICS_OUT:-${BASE_ROOT}/metrics.json}"
MODEL="${MODEL:-gpt-4o-mini}"
CHECK_MODEL="${CHECK_MODEL:-gpt-4o-mini}"
DATASET="visualwebarena"
PATH_FORMAT="gpt_web_tools"
MAX_ACTIONS="${MAX_ACTIONS:-20}"
MAX_OBS="${MAX_OBS:-20}"
TRIALS="${TRIALS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_API_BASE="${OPENAI_API_BASE:-${OPENAI_BASE_URL}}"
elif [[ -n "${OPENAI_API_BASE:-}" ]]; then
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE}}"
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[stage05] ERROR: set OPENAI_API_KEY before running Stage 05." >&2
  exit 2
fi
export DATASET=${DATASET}

export REDDIT=${REDDIT:-http://127.0.0.1:8001}
export SHOPPING=${SHOPPING:-http://127.0.0.1:8002}
export GITLAB=${GITLAB:-http://127.0.0.1:8003}
export WIKIPEDIA=${WIKIPEDIA:-http://127.0.0.1:8004}
export HOMEPAGE=${HOMEPAGE:-http://127.0.0.1:8005}
export CLASSIFIEDS=${CLASSIFIEDS:-http://127.0.0.1:8006}
export CLASSIFIEDS_RESET_TOKEN=${CLASSIFIEDS_RESET_TOKEN:-dummy}
export REDDIT_RESET_URL=${REDDIT_RESET_URL:-http://127.0.0.1:8001/reset}

source wasp/visualwebarena/venv/bin/activate

echo "[stage05] 1/4 Running user tasks"
echo "[stage05] Defense mode: ${AGENT_DEFENSE_MODE}"
AGENT_ARGS=()
if [[ -n "${AGENT_TEMPERATURE}" ]]; then
  AGENT_ARGS+=(--temperature "${AGENT_TEMPERATURE}")
fi
if [[ -n "${AGENT_SEED}" ]]; then
  AGENT_ARGS+=(--seed "${AGENT_SEED}")
fi
AGENT_ARGS+=(--defense-mode "${AGENT_DEFENSE_MODE}")

python web/exp/05_evaluate/run_agent.py \
  --tasks "${USER_TASKS}" \
  --result-dir "${LOG_DIR}" \
  --model "${MODEL}" \
  "${AGENT_ARGS[@]}" \
  --dataset "${DATASET}" \
  --max-actions "${MAX_ACTIONS}" \
  --max-observations "${MAX_OBS}" \
  --trials "${TRIALS}" \
  --max-parallel "${MAX_PARALLEL}" \
  --url-timeout 5 \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"}

# Fail fast when model requests never produced valid traces (common when tool-calling is unsupported).
non_empty_traces=0
shopt -s nullglob
for trace_file in "${LOG_DIR%/}"/trace_*.jsonl; do
  if [[ -s "${trace_file}" ]]; then
    non_empty_traces=$((non_empty_traces + 1))
  fi
done
shopt -u nullglob
if [[ "${non_empty_traces}" -eq 0 ]]; then
  echo "[stage05] ERROR: all trace logs are empty under ${LOG_DIR}." >&2
  echo "[stage05] Hint: the configured OpenAI-compatible endpoint must support tool calls." >&2
  exit 1
fi

pick_canonical_trace() {
  local candidate
  for candidate in "$@"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

echo "[stage05] 2/4 Preparing canonical logs for path statistics and step-by-step evaluation"
CANON_LOG_DIR="${LOG_DIR%/}/canonical_logs"
rm -rf "${CANON_LOG_DIR}" || true
mkdir -p "${CANON_LOG_DIR}"
shopt -s nullglob
for task_json in "${USER_TASKS%/}"/*.json; do
  base="$(basename "${task_json}")"
  [[ "${base}" == "manifest.json" ]] && continue
  [[ "${base}" == "attack_case.json" ]] && continue
  stem="${base%.json}"
  [[ "${stem}" =~ ^[0-9]+$ ]] || continue
  tid="${stem}"
  picked=""
  candidates=()
  if [[ "${TRIALS}" -gt 1 ]]; then
    preferred="${LOG_DIR%/}/trace_${tid}_r01.jsonl"
    if [[ -f "${preferred}" ]]; then
      candidates+=("${preferred}")
    fi
    for candidate in "${LOG_DIR%/}/trace_${tid}_r"*.jsonl; do
      [[ -f "${candidate}" ]] || continue
      [[ "${candidate}" == "${preferred}" ]] && continue
      candidates+=("${candidate}")
    done
  fi
  if [[ -f "${LOG_DIR%/}/trace_${tid}.jsonl" ]]; then
    candidates+=("${LOG_DIR%/}/trace_${tid}.jsonl")
  fi
  if picked="$(pick_canonical_trace "${candidates[@]}")"; then
    cp -f "${picked}" "${CANON_LOG_DIR%/}/trace_${tid}.jsonl"
    if [[ ! -s "${picked}" ]]; then
      echo "[stage05] Warning: canonical trace for task ${tid} is empty: ${picked}" >&2
    fi
  else
    echo "[stage05] Warning: no trace found for task ${tid} under ${LOG_DIR}" >&2
  fi
done
shopt -u nullglob

echo "[stage05] 3/4 Extracting navigation paths"
python web/exp/05_evaluate/extract_paths.py \
  --logs "${CANON_LOG_DIR}" \
  --out "${LOG_DIR}/path_stats.json" \
  --format "${PATH_FORMAT}" \
  --page-metadata "${PAGE_METADATA}" \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"}

echo "[stage05] 4/4 Evaluating and summarizing"
rm -f /tmp/run_user_utility.json /tmp/run_attacker_utility.json /tmp/run_step_by_step_asr.json || true

python web/exp/05_evaluate/evaluator_step_by_step_reasoning.py \
  --log-folder "${CANON_LOG_DIR}" \
  --task-folder "${USER_TASKS}" \
  --model "${CHECK_MODEL}"

python web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py \
  --log-folder "${LOG_DIR}" \
  --task-folder "${USER_TASKS}" \
  --page-metadata "${PAGE_METADATA}" \
  --out /tmp/run_user_utility.json

if [[ "${RUN_ATTACKER}" == "1" ]]; then
  if [[ ! -d "${ATTACKER_TASKS}" ]]; then
    echo "[stage05] ERROR: attacker tasks dir not found: ${ATTACKER_TASKS}" >&2
    echo "[stage05] Hint: set ATTACKER_TASKS=/path/to/webarena_tasks_attacker or generate them in Stage 04." >&2
    exit 1
  fi
  python web/exp/05_evaluate/evaluator_leaf_marker_from_logs.py \
    --log-folder "${LOG_DIR}" \
    --task-folder "${ATTACKER_TASKS}" \
    --page-metadata "${PAGE_METADATA}" \
    --out /tmp/run_attacker_utility.json
else
  echo "[stage05] Skipping attacker evaluation (user-only mode)."
fi

python web/exp/05_evaluate/summarize_results.py \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"} \
  --out "${METRICS_OUT}"

echo "[stage05] Done: logs ${LOG_DIR} | metrics ${METRICS_OUT}"
