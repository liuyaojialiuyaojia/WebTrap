#!/usr/bin/env bash
set -euo pipefail

# Simple generator for user and attacker tasks under web/runs/trap_site_mvp/.
# - Picks k right-to-left leaves for user tasks
# - Generates aligned attacker tasks from security_manifest.json (requires MICROTREE_ID)

K_USERS=${K_USERS:-5}
MICROTREE_ID=${MICROTREE_ID:-}
ATTACK_EVAL_TYPE=${ATTACK_EVAL_TYPE:-exfil}
BASE_ROOT="${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}"
if [[ -n "${EXPERIMENT_ID:-}" ]]; then
  BASE_ROOT="${BASE_ROOT%/}/${EXPERIMENT_ID}"
fi
BASE_ROOT="${BASE_ROOT%/}"
USER_DIR="${USER_DIR:-${BASE_ROOT}/webarena_tasks}"
ATTACK_DIR="${ATTACK_DIR:-${BASE_ROOT}/webarena_tasks_attacker}"

# Clean previous outputs (web/runs/ is safe to delete/recreate per repo guidelines)
rm -rf "${USER_DIR}" "${ATTACK_DIR}"

python3 web/exp/04_task_gen/make_user_tasks_from_leaves.py \
  --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
  ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"} \
  --count "${K_USERS}" \
  --output-dir "${USER_DIR}" \
  --task-id-start 1000

if [[ -z "${MICROTREE_ID}" ]]; then
  echo "[generate_tasks] Skipping attacker tasks: MICROTREE_ID is not set." >&2
  echo "[generate_tasks] Hint: set MICROTREE_ID=<id from security_manifest.json> to generate aligned attacker tasks." >&2
else
  python3 web/exp/04_task_gen/make_security_attack_tasks.py \
    --experiment-root "${EXPERIMENT_ROOT:-web/runs/trap_site_mvp}" \
    ${EXPERIMENT_ID:+--experiment-id "${EXPERIMENT_ID}"} \
    --microtree-id "${MICROTREE_ID}" \
    --eval-type "${ATTACK_EVAL_TYPE}" \
    --user-tasks "${USER_DIR}" \
    --output-dir "${ATTACK_DIR}"
fi

USERS_GEN=$(find "${USER_DIR}" -maxdepth 1 -name '*.json' ! -name 'manifest.json' 2>/dev/null | wc -l | awk '{print $1}')
ATK_GEN=$(find "${ATTACK_DIR}" -maxdepth 1 -name '*.json' ! -name 'manifest.json' 2>/dev/null | wc -l | awk '{print $1}')
echo "Generated ${USERS_GEN} user tasks in ${USER_DIR}" >&2
echo "Generated ${ATK_GEN} attacker tasks in ${ATTACK_DIR}" >&2
