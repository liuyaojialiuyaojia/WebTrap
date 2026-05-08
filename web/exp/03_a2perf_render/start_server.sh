#!/usr/bin/env bash

set -euo pipefail

BASE_ROOT="${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}"
if [[ -n "${EXPERIMENT_ID:-}" ]]; then
  BASE_ROOT="${BASE_ROOT%/}/${EXPERIMENT_ID}"
fi
PORT="${1:-8000}"
STATIC_DIR="${STATIC_DIR:-${BASE_ROOT%/}/static}"

if [[ ! -d "${STATIC_DIR}" ]]; then
  echo "Static directory '${STATIC_DIR}' not found. Run pack_static_site.py first." >&2
  exit 1
fi

PIDFILE="${STATIC_DIR}/.http_server_${PORT}.pid"
stop_previous_if_ours() {
  if [[ -f "${PIDFILE}" ]]; then
    local old_pid
    old_pid="$(cat "${PIDFILE}" || true)"
    if [[ -n "${old_pid}" && -d "/proc/${old_pid}" ]]; then
      local cmdline
      cmdline="$(tr '\0' ' ' </proc/${old_pid}/cmdline || true)"
      if [[ "${cmdline}" == "python -m http.server ${PORT} --directory ${STATIC_DIR}" ]]; then
        if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -a -p "${old_pid}" >/dev/null 2>&1; then
          echo "Stopping previous server (PID ${old_pid})"
          kill "${old_pid}" || true
          for _ in {1..15}; do
            if ! kill -0 "${old_pid}" 2>/dev/null; then
              break
            fi
            sleep 0.2
          done
          if kill -0 "${old_pid}" 2>/dev/null; then
            echo "Previous server (PID ${old_pid}) is still running; not force-killing. Please check." >&2
            exit 1
          fi
        fi
      fi
    fi
    rm -f "${PIDFILE}" || true
  fi
}
stop_previous_if_ours

echo "Serving ${STATIC_DIR} at http://127.0.0.1:${PORT}/ (Ctrl+C to stop)"
python -m http.server "${PORT}" --directory "${STATIC_DIR}"
