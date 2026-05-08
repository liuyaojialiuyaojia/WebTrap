#!/usr/bin/env bash

set -euo pipefail

PORT="${PORT:-8000}"
USE_SECURITY_MICROTREE="${USE_SECURITY_MICROTREE:-true}"
ATTACK_CASE="${ATTACK_CASE:-}"
COMBINEDATTACK_SEED="${COMBINEDATTACK_SEED:-42}"
INJECT_PAGE_PATH="${INJECT_PAGE_PATH:-/}"
COMBINEDATTACK_USER_GOAL_STR="${COMBINEDATTACK_USER_GOAL_STR:-}"
COMBINEDATTACK_USER_TASK_PATH="${COMBINEDATTACK_USER_TASK_PATH:-}"

SERVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve|--start-server)
      SERVE=1
      shift
      ;;
    *)
      echo "[combinedattack/run_injection] Warning: unknown arg $1 (ignored)" >&2
      shift
      ;;
  esac
done

BASE_ROOT="${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}"
if [[ -n "${EXPERIMENT_ID:-}" ]]; then
  BASE_ROOT="${BASE_ROOT%/}/${EXPERIMENT_ID}"
fi
BASE_ROOT="${BASE_ROOT%/}"

if [[ -z "${ATTACK_CASE}" ]]; then
  ATTACK_CASE="${BASE_ROOT}/attack_case.json"
fi
if [[ ! -f "${ATTACK_CASE}" ]]; then
  echo "[combinedattack/run_injection] ERROR: attack_case.json not found: ${ATTACK_CASE}" >&2
  exit 1
fi

USE_MICROTREE_FLAG=()
if [[ "${USE_SECURITY_MICROTREE}" == "false" ]]; then
  USE_MICROTREE_FLAG+=("--no-use-security-microtree")
fi

USER_TASK_FLAG=()
if [[ -n "${COMBINEDATTACK_USER_TASK_PATH}" ]]; then
  USER_TASK_FLAG=(--user-task-path "${COMBINEDATTACK_USER_TASK_PATH}")
fi

echo "[combinedattack/run_injection] Step 1/2: injecting from attack_case=${ATTACK_CASE}"
python web/baseline/combinedattack/inject_from_attack_case.py \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"} \
  --attack-case "${ATTACK_CASE}" \
  --seed "${COMBINEDATTACK_SEED}" \
  --inject-page-path "${INJECT_PAGE_PATH}" \
  --user-goal-str "${COMBINEDATTACK_USER_GOAL_STR}" \
  "${USER_TASK_FLAG[@]}" \
  "${USE_MICROTREE_FLAG[@]}"

echo "[combinedattack/run_injection] Step 2/2: packing injected static site"
python web/baseline/pack_injected_site.py \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"}

if [[ "${SERVE}" == "1" ]]; then
  echo "[combinedattack/run_injection] Starting static server on port ${PORT}"
  bash web/exp/03_a2perf_render/start_server.sh "${PORT}"
fi
