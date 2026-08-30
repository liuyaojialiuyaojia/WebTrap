#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-Rebuttal/runs/node_frequency_rerun}"
OUT_DIR="${OUT_DIR:-Rebuttal/results/node_frequency_rerun}"
MODEL="${MODEL:-deepseek-v3-1-terminus}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAMPLES="${SAMPLES:-16}"
BROWSER_PORT="${BROWSER_PORT:-8124}"
BROWSER_PARALLEL="${BROWSER_PARALLEL:-4}"
FILE_PARALLEL="${FILE_PARALLEL:-4}"
MAX_ACTIONS="${MAX_ACTIONS:-20}"
MAX_OBSERVATIONS="${MAX_OBSERVATIONS:-20}"
MAX_STEPS="${MAX_STEPS:-25}"
MAX_TOKENS="${MAX_TOKENS:-512}"

if [[ -d "${RUN_ROOT}/browser/logs" || -d "${RUN_ROOT}/file/logs" ]]; then
  echo "[cover-rerun] ERROR: run logs already exist under ${RUN_ROOT}." >&2
  echo "[cover-rerun] Use a new RUN_ROOT to preserve the completed evidence." >&2
  exit 2
fi

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:4000/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:4000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-yaojia-get-ccfa}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,192.168.11.218}"
export no_proxy="${no_proxy:-127.0.0.1,localhost,192.168.11.218}"
export DATASET="${DATASET:-visualwebarena}"
export REDDIT="${REDDIT:-http://127.0.0.1:8001}"
export SHOPPING="${SHOPPING:-http://127.0.0.1:8002}"
export GITLAB="${GITLAB:-http://127.0.0.1:8003}"
export WIKIPEDIA="${WIKIPEDIA:-http://127.0.0.1:8004}"
export HOMEPAGE="${HOMEPAGE:-http://127.0.0.1:8005}"
export CLASSIFIEDS="${CLASSIFIEDS:-http://127.0.0.1:8006}"
export CLASSIFIEDS_RESET_TOKEN="${CLASSIFIEDS_RESET_TOKEN:-dummy}"
export REDDIT_RESET_URL="${REDDIT_RESET_URL:-http://127.0.0.1:8001/reset}"
export PYTHONPATH=".:${PYTHONPATH:-}"

source wasp/visualwebarena/venv/bin/activate

echo "[cover-rerun] Stage 01: materialize two clean 16-target protocols"
"${PYTHON_BIN}" Rebuttal/experiments/prepare_node_frequency_runs.py \
  --run-root "${RUN_ROOT}" \
  --samples "${SAMPLES}" \
  --browser-start-url "http://127.0.0.1:${BROWSER_PORT}/index.html"

echo "[cover-rerun] Stage 02: render the pure Browser user tree"
"${PYTHON_BIN}" web/exp/03_a2perf_render/pack_static_site.py \
  --experiment-root "${RUN_ROOT}" \
  --experiment-id browser

mkdir -p "${RUN_ROOT}/browser/logs"
"${PYTHON_BIN}" -m http.server "${BROWSER_PORT}" \
  --directory "${RUN_ROOT}/browser/static" \
  >"${RUN_ROOT}/browser/http_${BROWSER_PORT}.log" 2>&1 &
BROWSER_SERVER_PID="$!"
trap 'kill "${BROWSER_SERVER_PID}" 2>/dev/null || true' EXIT
sleep 0.5
if ! kill -0 "${BROWSER_SERVER_PID}" 2>/dev/null; then
  echo "[cover-rerun] ERROR: Browser server failed to start." >&2
  exit 1
fi

echo "[cover-rerun] Stage 03: run 16 fresh Browser trajectories"
"${PYTHON_BIN}" web/exp/osr/run_agent_uncached.py \
  --experiment-root "${RUN_ROOT}" \
  --experiment-id browser \
  --tasks "${RUN_ROOT}/browser/tasks" \
  --result-dir "${RUN_ROOT}/browser/logs" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --defense-mode default_attack \
  --temperature "${TEMPERATURE}" \
  --max-actions "${MAX_ACTIONS}" \
  --max-observations "${MAX_OBSERVATIONS}" \
  --trials 1 \
  --max-parallel "${BROWSER_PARALLEL}" \
  --url-timeout 5

echo "[cover-rerun] Stage 04: run 16 fresh File trajectories"
"${PYTHON_BIN}" Rebuttal/experiments/run_clean_file_node_frequency.py \
  --run-dir "${RUN_ROOT}/file" \
  --model "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-steps "${MAX_STEPS}" \
  --max-parallel "${FILE_PARALLEL}" \
  --expected-cases "${SAMPLES}"

echo "[cover-rerun] Stage 05: aggregate the requested table"
"${PYTHON_BIN}" Rebuttal/experiments/analyze_node_frequency.py \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --expected-samples "${SAMPLES}"

echo "[cover-rerun] Done: ${OUT_DIR}/table.md"
