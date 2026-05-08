#!/usr/bin/env bash

set -euo pipefail

PORT="${PORT:-8000}"
USE_SECURITY_MICROTREE="${USE_SECURITY_MICROTREE:-true}"
ATTACK_CASE="${ATTACK_CASE:-}"
TOPICATTACK_MODEL="${TOPICATTACK_MODEL:-gpt-4o}"
TOPICATTACK_NUM_TURNS="${TOPICATTACK_NUM_TURNS:-5}"
TOPICATTACK_TEMPERATURE="${TOPICATTACK_TEMPERATURE:-0.0}"
TOPICATTACK_MAX_TOKENS="${TOPICATTACK_MAX_TOKENS:-2048}"
TOPICATTACK_CACHE_TTL="${TOPICATTACK_CACHE_TTL:-86400}"
TOPICATTACK_WITH_REMINDER="${TOPICATTACK_WITH_REMINDER:-true}"
TOPICATTACK_SYSTEM_MESSAGE="${TOPICATTACK_SYSTEM_MESSAGE:-}"
INJECT_PAGE_PATH="${INJECT_PAGE_PATH:-/}"
TOPICATTACK_USER_GOAL_STR="${TOPICATTACK_USER_GOAL_STR:-}"
TOPICATTACK_USER_TASK_PATH="${TOPICATTACK_USER_TASK_PATH:-}"

SERVE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --serve|--start-server)
      SERVE=1
      shift
      ;;
    *)
      echo "[topicattack/run_injection] Warning: unknown arg $1 (ignored)" >&2
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
  echo "[topicattack/run_injection] ERROR: attack_case.json not found: ${ATTACK_CASE}" >&2
  exit 1
fi

USE_MICROTREE_FLAG=()
if [[ "${USE_SECURITY_MICROTREE}" == "false" ]]; then
  USE_MICROTREE_FLAG+=("--no-use-security-microtree")
fi

WITH_REMINDER_FLAG=("--with-reminder")
if [[ "${TOPICATTACK_WITH_REMINDER}" == "false" ]]; then
  WITH_REMINDER_FLAG=("--no-with-reminder")
fi

USER_TASK_FLAG=()
if [[ -n "${TOPICATTACK_USER_TASK_PATH}" ]]; then
  USER_TASK_FLAG=(--user-task-path "${TOPICATTACK_USER_TASK_PATH}")
fi

echo "[topicattack/run_injection] Step 1/2: injecting from attack_case=${ATTACK_CASE}"
python web/baseline/topicattack/inject_from_attack_case.py \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"} \
  --attack-case "${ATTACK_CASE}" \
  --model "${TOPICATTACK_MODEL}" \
  --num-turns "${TOPICATTACK_NUM_TURNS}" \
  --temperature "${TOPICATTACK_TEMPERATURE}" \
  --max-tokens "${TOPICATTACK_MAX_TOKENS}" \
  --cache-ttl "${TOPICATTACK_CACHE_TTL}" \
  --system-message "${TOPICATTACK_SYSTEM_MESSAGE}" \
  --inject-page-path "${INJECT_PAGE_PATH}" \
  --user-goal-str "${TOPICATTACK_USER_GOAL_STR}" \
  "${USER_TASK_FLAG[@]}" \
  "${USE_MICROTREE_FLAG[@]}" \
  "${WITH_REMINDER_FLAG[@]}"

echo "[topicattack/run_injection] Step 2/2: packing injected static site"
python web/baseline/pack_injected_site.py \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"}

if [[ "${SERVE}" == "1" ]]; then
  echo "[topicattack/run_injection] Starting static server on port ${PORT}"
  bash web/exp/03_a2perf_render/start_server.sh "${PORT}"
fi
