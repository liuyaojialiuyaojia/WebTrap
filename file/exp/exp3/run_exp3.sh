#!/usr/bin/env bash
set -euo pipefail

# Run exp3 evaluation on an existing run directory.
#
# Usage:
#   bash file/exp/exp3/run_exp3.sh --run-dir file/runs/<run_id>

PYTHON_BIN="${PYTHON_BIN:-python}"
ASR_FORMAT="${ASR_FORMAT:-reasoning}"
ASR_MODEL="${ASR_MODEL:-gpt-4o-mini}"
ASR_TEMPERATURE="${ASR_TEMPERATURE:-0.0}"
ASR_MAX_TOKENS="${ASR_MAX_TOKENS:-512}" # 0 = backend default

if [[ "${ASR_FORMAT}" != "reasoning" ]]; then
  echo "[run_exp3] ERROR: ASR_FORMAT must be 'reasoning'." >&2
  echo "[run_exp3] Refusing unsupported ASR_FORMAT='${ASR_FORMAT}'." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
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

"${PYTHON_BIN}" file/exp/exp3/eval_user_goal.py --run-dir "${RUN_DIR}" --include-samples
"${PYTHON_BIN}" file/exp/exp3/eval_attack_goal.py --run-dir "${RUN_DIR}" --include-samples
"${PYTHON_BIN}" file/exp/exp3/eval_asr_intermediate.py \
  --run-dir "${RUN_DIR}" \
  --include-samples \
  --format "${ASR_FORMAT}" \
  --model "${ASR_MODEL}" \
  --temperature "${ASR_TEMPERATURE}" \
  --max-tokens "${ASR_MAX_TOKENS}"
"${PYTHON_BIN}" file/exp/exp3/summarize_metrics.py --run-dir "${RUN_DIR}"

# --include-samples \
