#!/usr/bin/env bash
set -euo pipefail
BASE=${LITELLM_BASE_URL:-http://localhost:4000/v1}
KEY=${LITELLM_MASTER_KEY:-sk-your-litellm-key}
curl "$BASE/chat/completions" \
 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
 -d '{"model":"ds-chat","messages":[{"role":"user","content":"你好"}],"cache":{"ttl":300,"namespace":"demo"}}'
