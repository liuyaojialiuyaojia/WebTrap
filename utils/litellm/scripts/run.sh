#!/usr/bin/env bash
# 用法：utils/litellm/scripts/run.sh python your_app.py
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/scripts/libcompose.sh"
cd "$DIR"
ensure_local_no_proxy

# 确保代理已启动
run_compose up -d litellm >/dev/null

# 注入运行时：让 Python 自动加载 utils/litellm/python/sitecustomize.py
export PYTHONPATH="$DIR/python:${PYTHONPATH:-}"
export LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://localhost:4000/v1}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-yaojia-get-ccfa}"

exec "$@"
