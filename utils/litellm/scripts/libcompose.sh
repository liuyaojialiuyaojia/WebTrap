#!/usr/bin/env bash

ensure_local_no_proxy() {
  local local_hosts="localhost,127.0.0.1"
  export NO_PROXY="${local_hosts}${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="${local_hosts}${no_proxy:+,$no_proxy}"
}

run_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi

  echo "Neither 'docker compose' nor 'docker-compose' is available." >&2
  exit 1
}

wait_for_litellm() {
  local health_url="${LITELLM_HEALTH_URL:-http://127.0.0.1:4000/health/readiness}"
  local max_attempts="${LITELLM_READY_ATTEMPTS:-60}"
  local attempt

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if curl --noproxy localhost,127.0.0.1 \
      --fail --silent --show-error --max-time 2 \
      "$health_url" >/dev/null 2>&1; then
      echo "LiteLLM is ready at $health_url"
      return 0
    fi
    sleep 1
  done

  echo "LiteLLM did not become ready after ${max_attempts} attempts." >&2
  run_compose logs --tail=50 litellm >&2 || true
  return 1
}
