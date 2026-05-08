#!/usr/bin/env bash
set -euo pipefail

# Batch runner for RUNBOOK_FLOWS.md (Stage 02 → 04 once; then PSAA inject + Stage 05 per attack).
# EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=test_d10w2 \     
# bash web/exp/run_batch_psaa_attacks.sh

START_TS="$(date +%s)"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-web/runs}"
: "${EXPERIMENT_ID:=test_d10w2}"
PORT="${PORT:-8000}"
START_URL="${START_URL:-http://127.0.0.1:${PORT}/index.html}"

SOURCE_ROOT="${SOURCE_ROOT:-web/exp/02_nested_builder/tree/d10_w2_s42}"
ROOT_PATH="${ROOT_PATH:-/}"
DEPTH="${DEPTH:-10}"
WIDTH="${WIDTH:-2}"
SELECTION_SEED="${SELECTION_SEED:-42}"

WASP_ENV="${WASP_ENV:-gitlab}" # gitlab | reddit
WASP_EVAL_TYPE="${WASP_EVAL_TYPE:-both}"
WASP_SEED="${WASP_SEED:-42}"

case "${WASP_ENV}" in
  gitlab)
    DEFAULT_SECURITY_SPEC="web/exp/02_5_security_microtree/security_spec.gitlab_wasp.json"
    ;;
  reddit)
    DEFAULT_SECURITY_SPEC="web/exp/02_5_security_microtree/security_spec.reddit_wasp.json"
    ;;
  *)
    echo "[batch] ERROR: unsupported WASP_ENV='${WASP_ENV}' (expected gitlab|reddit)" >&2
    exit 1
    ;;
esac

SECURITY_SPEC="${SECURITY_SPEC:-${DEFAULT_SECURITY_SPEC}}"
ANCHOR_BREADCRUMB="${ANCHOR_BREADCRUMB:-/1/1/1/1/1/1/1}"

USER_TASK_COUNT="${USER_TASK_COUNT:-2}"

TRIALS="${TRIALS:-3}"
INTRA_ATTACK_PARALLEL="${INTRA_ATTACK_PARALLEL:-6}"
AGENT_TEMP="${AGENT_TEMP:-1.0}"
AGENT_SEED="${AGENT_SEED:-42}"
AGENT_DEFENSE_MODE="${AGENT_DEFENSE_MODE:-default_attack}" # default_attack | system_prompt_defense | step_wise_prompt_defense | goal_reinforce_ignore | goal_reinforce_fakecom_t | segment_remove_gated | segment_remove_direct
PSAA_VARIANT="${PSAA_VARIANT:-full}" # full | wo_lure | wo_inertia | wo_payload
PSAA_PROMPT_SPEC="${PSAA_PROMPT_SPEC:-}" # optional explicit YAML override; otherwise uses PSAA_PROMPT_VERSION

# Edit this list to batch different attacks (subdirs under <root>/webarena_tasks_attacker_wasp/).
ATTACKS=(
  ### gitlab
  # wasp_gitlab_create_a_deploy_token
  # wasp_gitlab_create_a_deploy_key
  # wasp_gitlab_add_member
  # wasp_gitlab_create_a_webhook_to_attacker_s_server
  # wasp_gitlab_export_project
  # wasp_gitlab_transfer_project_to_a_new_namespace
  # wasp_gitlab_delete_project
  wasp_gitlab_extract_list_of_contributors
  wasp_gitlab_add_an_ssh_key_to_the_user_account
  wasp_gitlab_create_a_personal_access_token_for_the_user_s_account
  wasp_gitlab_make_the_profile_public
  wasp_gitlab_add_a_security_violating_change_to_a_repository_a_command_to_download_malware_to_zshrc

  ### reddit
  # wasp_reddit_create_a_harmful_comment
  # wasp_reddit_adding_custom_email_for_password_reset
  # wasp_reddit_phishing_through_private_messages
  # wasp_reddit_negative_comment_on_a_post
  # wasp_reddit_delete_user_created_post
  # wasp_reddit_downvote_a_post
  # wasp_reddit_make_a_post_pushing_a_political_slogan
  # wasp_reddit_exfiltrate_the_private_messages_of_the_user
  # wasp_reddit_exfiltrate_information_about_block_listed_users
)

export EXPERIMENT_ROOT EXPERIMENT_ID

BASE_ROOT="${EXPERIMENT_ROOT%/}/${EXPERIMENT_ID}"

case "${PSAA_VARIANT}" in
  full)
    PSAA_VARIANT_LABEL="full PSAA"
    ;;
  wo_lure)
    PSAA_VARIANT_LABEL="WebTrap w/o lure"
    ;;
  wo_inertia)
    PSAA_VARIANT_LABEL="WebTrap w/o inertia"
    ;;
  wo_payload)
    PSAA_VARIANT_LABEL="WebTrap w/o payload"
    ;;
  *)
    echo "[batch] ERROR: unsupported PSAA_VARIANT='${PSAA_VARIANT}'" >&2
    echo "[batch] Allowed values: full | wo_lure | wo_inertia | wo_payload" >&2
    exit 1
    ;;
esac

if [[ ! -f "${SECURITY_SPEC}" ]]; then
  echo "[batch] ERROR: SECURITY_SPEC not found: ${SECURITY_SPEC}" >&2
  exit 1
fi

SECURITY_SPEC_BASENAME="$(basename "${SECURITY_SPEC}")"
SPEC_ENV_HINT=""
case "${SECURITY_SPEC_BASENAME}" in
  *gitlab*)
    SPEC_ENV_HINT="gitlab"
    ;;
  *reddit*)
    SPEC_ENV_HINT="reddit"
    ;;
esac
if [[ -n "${SPEC_ENV_HINT}" && "${SPEC_ENV_HINT}" != "${WASP_ENV}" ]]; then
  echo "[batch] ERROR: SECURITY_SPEC (${SECURITY_SPEC_BASENAME}) conflicts with WASP_ENV=${WASP_ENV}" >&2
  exit 1
fi

echo "[batch] experiment=${BASE_ROOT} port=${PORT} start_url=${START_URL}"
echo "[batch] trials=${TRIALS} intra_attack_parallel=${INTRA_ATTACK_PARALLEL}"
echo "[batch] psaa variant=${PSAA_VARIANT} (${PSAA_VARIANT_LABEL})"
echo "[batch] psaa prompt spec=${PSAA_PROMPT_SPEC}"
echo "[batch] agent defense mode=${AGENT_DEFENSE_MODE}"
echo "[batch] wasp_env=${WASP_ENV} security_spec=${SECURITY_SPEC}"

echo "[batch] Stage 02: extract_subtree"
python web/exp/02_nested_builder/extract_subtree.py \
  --source-root "${SOURCE_ROOT}" \
  --root-path "${ROOT_PATH}" \
  --depth "${DEPTH}" --width "${WIDTH}" --selection-seed "${SELECTION_SEED}" \
  --output-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --force

echo "[batch] Stage 02.5: compile_security_microtree"
python web/exp/02_5_security_microtree/compile_security_microtree.py \
  --spec "${SECURITY_SPEC}" \
  --anchor-breadcrumb "${ANCHOR_BREADCRUMB}" \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}"

echo "[batch] Stage 03: pack_static_site (merged tree)"
python web/exp/03_a2perf_render/pack_static_site.py \
  --experiment-root "${EXPERIMENT_ROOT}" --experiment-id "${EXPERIMENT_ID}" \
  --design "${BASE_ROOT}/security_microtree/website_designs.json" \
  --metadata "${BASE_ROOT}/security_microtree/page_metadata.json" \
  --transitions "${BASE_ROOT}/security_microtree/transitions.json"

STATIC_DIR="${BASE_ROOT}/static"
SERVER_LOG="${BASE_ROOT}/http_${PORT}.log"
echo "[batch] Serving ${STATIC_DIR} on ${START_URL} (log: ${SERVER_LOG})"
python -m http.server "${PORT}" --directory "${STATIC_DIR}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT
sleep 0.3
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[batch] ERROR: failed to start http.server (see ${SERVER_LOG})" >&2
  exit 1
fi

echo "[batch] Stage 04: make_user_tasks_from_leaves"
python web/exp/04_task_gen/make_user_tasks_from_leaves.py \
  --experiment-root "${EXPERIMENT_ROOT}" --experiment-id "${EXPERIMENT_ID}" \
  --count "${USER_TASK_COUNT}" \
  --start-url "${START_URL}"

echo "[batch] Stage 04: make_wasp_attack_tasks_from_manifest (env=${WASP_ENV}, eval=${WASP_EVAL_TYPE})"
python web/exp/04_task_gen/make_wasp_attack_tasks_from_manifest.py \
  --experiment-root "${EXPERIMENT_ROOT}" --experiment-id "${EXPERIMENT_ID}" \
  --env "${WASP_ENV}" --eval-type "${WASP_EVAL_TYPE}" \
  --all \
  --seed "${WASP_SEED}"

ATTACK_ROOT="${BASE_ROOT}/webarena_tasks_attacker_wasp"
if [[ ! -d "${ATTACK_ROOT}" ]]; then
  echo "[batch] ERROR: attacker task root not found: ${ATTACK_ROOT}" >&2
  exit 1
fi
MISSING_ATTACKS=()
for attack in "${ATTACKS[@]}"; do
  attack="${attack%,}"
  if [[ ! -f "${ATTACK_ROOT}/${attack}/attack_case.json" ]]; then
    MISSING_ATTACKS+=("${attack}")
  fi
done
if (( ${#MISSING_ATTACKS[@]} > 0 )); then
  echo "[batch] ERROR: missing attack_case.json for ${#MISSING_ATTACKS[@]} attack(s): ${MISSING_ATTACKS[*]}" >&2
  echo "[batch] Available attack directories under ${ATTACK_ROOT}:" >&2
  if ! find "${ATTACK_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '  - %f\n' | sort >&2; then
    true
  fi
  exit 1
fi

BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_DIR="${BASE_ROOT}/batch_runs/${BATCH_ID}_${PSAA_VARIANT}"
mkdir -p "${BATCH_DIR}"
echo "[batch] Outputs: ${BATCH_DIR}"

for attack in "${ATTACKS[@]}"; do
  attack="${attack%,}"
  ATTACK_DIR="${BASE_ROOT}/webarena_tasks_attacker_wasp/${attack}"
  ATTACK_CASE="${ATTACK_DIR}/attack_case.json"
  if [[ ! -f "${ATTACK_CASE}" ]]; then
    echo "[batch] ERROR: missing attack_case.json: ${ATTACK_CASE}" >&2
    exit 1
  fi

  OUT_DIR="${BATCH_DIR}/${attack}"
  mkdir -p "${OUT_DIR}"
  echo "[batch] === ${attack} ==="

  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
    INJECT_START_PAGE_PATH=/ \
    ATTACK_CASE="${ATTACK_CASE}" \
    PSAA_VARIANT="${PSAA_VARIANT}" \
    PSAA_PROMPT_SPEC="${PSAA_PROMPT_SPEC}" \
    INJECTION_GENERATION_LOG="${OUT_DIR}/injection_generation.jsonl" \
    bash web/psaa/run_injection.sh >"${OUT_DIR}/injection.log" 2>&1

  mkdir -p "${OUT_DIR}/static_snapshot"
  cp -f "${BASE_ROOT}/static/page_metadata.json" "${OUT_DIR}/static_snapshot/page_metadata.json"

  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
    TRIALS="${TRIALS}" \
    MAX_PARALLEL="${INTRA_ATTACK_PARALLEL}" \
    ATTACKER_TASKS="${ATTACK_DIR}" \
    PAGE_METADATA="${OUT_DIR}/static_snapshot/page_metadata.json" \
    LOG_DIR="${OUT_DIR}/agent_logs_post_injection" \
    METRICS_OUT="${OUT_DIR}/metrics_post_injection.json" \
    bash web/exp/05_evaluate/run_full.sh --with-attacker --temperature "${AGENT_TEMP}" --seed "${AGENT_SEED}" --defense-mode "${AGENT_DEFENSE_MODE}" \
    >"${OUT_DIR}/attacker.log" 2>&1
done

END_TS="$(date +%s)"
TOTAL_SECONDS="$((END_TS - START_TS))"
printf '[batch] Total elapsed: %02d:%02d:%02d (%ds)\n' \
  "$((TOTAL_SECONDS / 3600))" \
  "$(((TOTAL_SECONDS % 3600) / 60))" \
  "$((TOTAL_SECONDS % 60))" \
  "${TOTAL_SECONDS}"

echo "[batch] Done: ${BATCH_DIR}"
