#!/usr/bin/env bash
set -euo pipefail

# Unified attack stage runner under file/exp.
# Modes:
#   baseline    -> inject baseline payload into root readme
#   psaa        -> inject staged PSAA payload into scoped readmes
#   topicattack -> inject official TopicAttack agent transcript into one README
#   combinedattack -> inject official Combined Attack string into one README

PYTHON_BIN="${PYTHON_BIN:-python}"
INJECTION_MODE="${INJECTION_MODE:-psaa}"   # baseline | psaa | topicattack | combinedattack
SETTING="${SETTING:-enhanced}" # for baseline: base | enhanced
PSAA_SPEC="${PSAA_SPEC:-file/exp/attack/psaa/file_psaa_v1.yaml}" # for psaa: file_psaa_v1.yaml
PSAA_START_INDEX_PATH="${PSAA_START_INDEX_PATH:-}" # for psaa: index path like /0/1 (root is /)
TOPICATTACK_MODEL="${TOPICATTACK_MODEL:-gpt-4o}" # for topicattack
TOPICATTACK_NUM_TURNS="${TOPICATTACK_NUM_TURNS:-5}" # for topicattack
TOPICATTACK_TEMPERATURE="${TOPICATTACK_TEMPERATURE:-0.0}" # for topicattack
TOPICATTACK_MAX_TOKENS="${TOPICATTACK_MAX_TOKENS:-2048}" # for topicattack
TOPICATTACK_CACHE_TTL="${TOPICATTACK_CACHE_TTL:-86400}" # for topicattack
TOPICATTACK_WITH_REMINDER="${TOPICATTACK_WITH_REMINDER:-true}" # for topicattack
TOPICATTACK_SYSTEM_MESSAGE="${TOPICATTACK_SYSTEM_MESSAGE:-}" # for topicattack
TOPICATTACK_INJECT_INDEX_PATH="${TOPICATTACK_INJECT_INDEX_PATH:-/}" # for topicattack
TOPICATTACK_USER_GOAL_STR="${TOPICATTACK_USER_GOAL_STR:-}" # for topicattack
COMBINEDATTACK_SEED="${COMBINEDATTACK_SEED:-42}" # for combinedattack
COMBINEDATTACK_INJECT_INDEX_PATH="${COMBINEDATTACK_INJECT_INDEX_PATH:-/}" # for combinedattack
COMBINEDATTACK_USER_GOAL_STR="${COMBINEDATTACK_USER_GOAL_STR:-}" # for combinedattack

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
if [[ -z "${CASE_ID}" ]]; then
  echo "--case-id is required" >&2
  exit 1
fi

case "${INJECTION_MODE}" in
  baseline)
    CMD=(
      "${PYTHON_BIN}" file/exp/attack/baseline/inject.py
      --run-dir "${RUN_DIR}"
      --setting "${SETTING}"
      --case-id "${CASE_ID}"
    )
    "${CMD[@]}"
    ;;
  psaa)
    CMD=(
      "${PYTHON_BIN}" file/exp/attack/psaa/inject.py
      --run-dir "${RUN_DIR}"
      --spec "${PSAA_SPEC}"
      --case-id "${CASE_ID}"
    )
    if [[ -n "${PSAA_START_INDEX_PATH}" ]]; then
      CMD+=(--start-index-path "${PSAA_START_INDEX_PATH}")
    fi
    "${CMD[@]}"
    ;;
  topicattack)
    WITH_REMINDER_FLAG=("--with-reminder")
    if [[ "${TOPICATTACK_WITH_REMINDER}" == "false" ]]; then
      WITH_REMINDER_FLAG=("--no-with-reminder")
    fi
    CMD=(
      "${PYTHON_BIN}" file/exp/attack/topicattack/inject.py
      --run-dir "${RUN_DIR}"
      --case-id "${CASE_ID}"
      --model "${TOPICATTACK_MODEL}"
      --num-turns "${TOPICATTACK_NUM_TURNS}"
      --temperature "${TOPICATTACK_TEMPERATURE}"
      --max-tokens "${TOPICATTACK_MAX_TOKENS}"
      --cache-ttl "${TOPICATTACK_CACHE_TTL}"
      --system-message "${TOPICATTACK_SYSTEM_MESSAGE}"
      --inject-index-path "${TOPICATTACK_INJECT_INDEX_PATH}"
      --user-goal-str "${TOPICATTACK_USER_GOAL_STR}"
      "${WITH_REMINDER_FLAG[@]}"
    )
    "${CMD[@]}"
    ;;
  combinedattack)
    CMD=(
      "${PYTHON_BIN}" file/exp/attack/combinedattack/inject.py
      --run-dir "${RUN_DIR}"
      --case-id "${CASE_ID}"
      --seed "${COMBINEDATTACK_SEED}"
      --inject-index-path "${COMBINEDATTACK_INJECT_INDEX_PATH}"
      --user-goal-str "${COMBINEDATTACK_USER_GOAL_STR}"
    )
    "${CMD[@]}"
    ;;
  *)
    echo "Unsupported INJECTION_MODE: ${INJECTION_MODE}" >&2
    echo "Supported modes: baseline | psaa | topicattack | combinedattack" >&2
    exit 1
    ;;
esac
