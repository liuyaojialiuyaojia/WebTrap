#!/usr/bin/env bash
set -euo pipefail

# Batch runner for RUNBOOK_FLOWS.md (Stage 02 → 04 once; then baseline inject + Stage 05 per attack & target).
#
# Notes:
#   - This script batches WASP-derived microtrees under <root>/webarena_tasks_attacker_wasp/<microtree_id>/.
#   - Each run overwrites <root>/static; we snapshot page_metadata.json per run for evaluation.

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

WASP_ENV="${WASP_ENV:-gitlab}"
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
ANCHOR_BREADCRUMB="${ANCHOR_BREADCRUMB:-/1/1/1/1}"

USER_TASK_COUNT="${USER_TASK_COUNT:-2}"

TRIALS="${TRIALS:-3}"
INTRA_ATTACK_PARALLEL="${INTRA_ATTACK_PARALLEL:-6}"
AGENT_TEMP="${AGENT_TEMP:-0.8}"
AGENT_SEED="${AGENT_SEED:-42}"
AGENT_DEFENSE_MODE="${AGENT_DEFENSE_MODE:-default_attack}" # default_attack | system_prompt_defense | step_wise_prompt_defense | goal_reinforce_ignore | goal_reinforce_fakecom_t | segment_remove_gated | segment_remove_direct

# Baseline targets can be a mix of:
#   - WASP formats:
#       goal_hijacking_url_injection
#       goal_hijacking_plain_text
#       generic_url_injection
#       generic_plain_text
#   - Additional retained baseline families:
#       topicattack
#       combinedattack
BASELINE_FORMATS_STR="${BASELINE_FORMATS:-goal_hijacking_url_injection,goal_hijacking_plain_text,generic_url_injection,generic_plain_text,topicattack,combinedattack}" # goal_hijacking_url_injection, goal_hijacking_plain_text, generic_url_injection, generic_plain_text, topicattack, combinedattack
BASELINE_FORMATS_STR="${BASELINE_FORMATS_STR//,/ }"
read -r -a BASELINE_FORMATS <<<"${BASELINE_FORMATS_STR}"

INJECT_PAGE_PATH="${INJECT_PAGE_PATH:-/}"

TOPICATTACK_MODEL="${TOPICATTACK_MODEL:-gpt-4o}"
TOPICATTACK_NUM_TURNS="${TOPICATTACK_NUM_TURNS:-5}"
TOPICATTACK_TEMPERATURE="${TOPICATTACK_TEMPERATURE:-0.0}"
TOPICATTACK_MAX_TOKENS="${TOPICATTACK_MAX_TOKENS:-2048}"
TOPICATTACK_CACHE_TTL="${TOPICATTACK_CACHE_TTL:-86400}"
TOPICATTACK_WITH_REMINDER="${TOPICATTACK_WITH_REMINDER:-true}"
TOPICATTACK_USER_GOAL_STR="${TOPICATTACK_USER_GOAL_STR:-}"

COMBINEDATTACK_SEED="${COMBINEDATTACK_SEED:-42}"
COMBINEDATTACK_USER_GOAL_STR="${COMBINEDATTACK_USER_GOAL_STR:-}"

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
  # wasp_gitlab_extract_list_of_contributors
  # wasp_gitlab_add_an_ssh_key_to_the_user_account
  # wasp_gitlab_create_a_personal_access_token_for_the_user_s_account
  # wasp_gitlab_make_the_profile_public
  # wasp_gitlab_add_a_security_violating_change_to_a_repository_a_command_to_download_malware_to_zshrc

  ### reddit
  wasp_reddit_create_a_harmful_comment
  wasp_reddit_adding_custom_email_for_password_reset
  wasp_reddit_phishing_through_private_messages
  wasp_reddit_negative_comment_on_a_post
  wasp_reddit_delete_user_created_post
  wasp_reddit_downvote_a_post
  wasp_reddit_make_a_post_pushing_a_political_slogan
  wasp_reddit_exfiltrate_the_private_messages_of_the_user
  wasp_reddit_exfiltrate_information_about_block_listed_users
)

export EXPERIMENT_ROOT EXPERIMENT_ID
BASE_ROOT="${EXPERIMENT_ROOT%/}/${EXPERIMENT_ID}"

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
echo "[batch] trials=${TRIALS} attack_parallel=${INTRA_ATTACK_PARALLEL}"
echo "[batch] baseline targets: ${BASELINE_FORMATS[*]}"
echo "[batch] agent defense mode=${AGENT_DEFENSE_MODE}"
echo "[batch] wasp_env=${WASP_ENV} security_spec=${SECURITY_SPEC}"
echo "[batch] inject_page_path=${INJECT_PAGE_PATH}"
echo "[batch] topicattack_model=${TOPICATTACK_MODEL} topicattack_turns=${TOPICATTACK_NUM_TURNS} topicattack_with_reminder=${TOPICATTACK_WITH_REMINDER}"
echo "[batch] combinedattack_seed=${COMBINEDATTACK_SEED}"

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
BATCH_DIR="${BASE_ROOT}/batch_runs/${BATCH_ID}_baseline"
mkdir -p "${BATCH_DIR}"
echo "[batch] Outputs: ${BATCH_DIR}"

copy_single_task_file() {
  local src_file="$1"
  local dst_dir="$2"
  mkdir -p "${dst_dir}"
  cp -f "${src_file}" "${dst_dir}/$(basename "${src_file}")"
}

run_per_task_baseline() {
  local target="$1"
  local attack_case="$2"
  local out_dir="$3"
  local attack_dir
  local user_task
  local task_base
  local task_id
  local task_out
  local user_task_dir
  local attacker_task_dir
  local attacker_task_file

  attack_dir="$(dirname "${attack_case}")"
  mkdir -p "${out_dir}/per_task"

  for user_task in "${BASE_ROOT}/webarena_tasks"/*.json; do
    [[ -f "${user_task}" ]] || continue
    task_base="$(basename "${user_task}")"
    [[ "${task_base}" == "manifest.json" ]] && continue
    [[ "${task_base%.json}" =~ ^[0-9]+$ ]] || continue
    task_id="${task_base%.json}"
    attacker_task_file="${attack_dir}/${task_base}"
    if [[ ! -f "${attacker_task_file}" ]]; then
      echo "[batch] ERROR: missing attacker task for task_id=${task_id}: ${attacker_task_file}" >&2
      exit 1
    fi

    task_out="${out_dir}/per_task/${task_id}"
    mkdir -p "${task_out}"

    echo "[batch] ----- ${target} :: task ${task_id} -----"
    case "${target}" in
      goal_hijacking_url_injection|goal_hijacking_plain_text)
        EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
          ATTACK_CASE="${attack_case}" \
          BASELINE_INJECTION_FORMAT="${target}" \
          BASELINE_USER_TASK_PATH="${user_task}" \
          INJECT_PAGE_PATH="${INJECT_PAGE_PATH}" \
          bash web/baseline/run_injection.sh >"${task_out}/injection.log" 2>&1
        ;;
      topicattack)
        EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
          ATTACK_CASE="${attack_case}" \
          TOPICATTACK_MODEL="${TOPICATTACK_MODEL}" \
          TOPICATTACK_NUM_TURNS="${TOPICATTACK_NUM_TURNS}" \
          TOPICATTACK_TEMPERATURE="${TOPICATTACK_TEMPERATURE}" \
          TOPICATTACK_MAX_TOKENS="${TOPICATTACK_MAX_TOKENS}" \
          TOPICATTACK_CACHE_TTL="${TOPICATTACK_CACHE_TTL}" \
          TOPICATTACK_WITH_REMINDER="${TOPICATTACK_WITH_REMINDER}" \
          TOPICATTACK_USER_GOAL_STR="${TOPICATTACK_USER_GOAL_STR}" \
          TOPICATTACK_USER_TASK_PATH="${user_task}" \
          INJECT_PAGE_PATH="${INJECT_PAGE_PATH}" \
          bash web/baseline/topicattack/run_injection.sh >"${task_out}/injection.log" 2>&1
        ;;
      combinedattack)
        EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
          ATTACK_CASE="${attack_case}" \
          COMBINEDATTACK_SEED="${COMBINEDATTACK_SEED}" \
          COMBINEDATTACK_USER_GOAL_STR="${COMBINEDATTACK_USER_GOAL_STR}" \
          COMBINEDATTACK_USER_TASK_PATH="${user_task}" \
          INJECT_PAGE_PATH="${INJECT_PAGE_PATH}" \
          bash web/baseline/combinedattack/run_injection.sh >"${task_out}/injection.log" 2>&1
        ;;
      *)
        echo "[batch] ERROR: unsupported per-task baseline '${target}'" >&2
        exit 1
        ;;
    esac

    mkdir -p "${task_out}/static_snapshot"
    cp -f "${BASE_ROOT}/static/page_metadata.json" "${task_out}/static_snapshot/page_metadata.json"

    user_task_dir="${task_out}/user_tasks"
    attacker_task_dir="${task_out}/attacker_tasks"
    copy_single_task_file "${user_task}" "${user_task_dir}"
    copy_single_task_file "${attacker_task_file}" "${attacker_task_dir}"

    EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
      TRIALS="${TRIALS}" \
      MAX_PARALLEL="${INTRA_ATTACK_PARALLEL}" \
      USER_TASKS="${user_task_dir}" \
      ATTACKER_TASKS="${attacker_task_dir}" \
      PAGE_METADATA="${task_out}/static_snapshot/page_metadata.json" \
      LOG_DIR="${task_out}/agent_logs_post_injection" \
      METRICS_OUT="${task_out}/metrics_post_injection.json" \
      bash web/exp/05_evaluate/run_full.sh --with-attacker --temperature "${AGENT_TEMP}" --seed "${AGENT_SEED}" --defense-mode "${AGENT_DEFENSE_MODE}" \
      >"${task_out}/attacker.log" 2>&1
  done
}

run_injection_for_target() {
  local target="$1"
  local attack_case="$2"
  local out_dir="$3"

  case "${target}" in
    goal_hijacking_url_injection|goal_hijacking_plain_text|topicattack|combinedattack)
      run_per_task_baseline "${target}" "${attack_case}" "${out_dir}"
      ;;
    generic_url_injection|generic_plain_text)
      EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
        ATTACK_CASE="${attack_case}" \
        BASELINE_INJECTION_FORMAT="${target}" \
        INJECT_PAGE_PATH="${INJECT_PAGE_PATH}" \
        bash web/baseline/run_injection.sh >"${out_dir}/injection.log" 2>&1
      ;;
    *)
      echo "[batch] ERROR: unsupported baseline target '${target}'" >&2
      echo "[batch] Supported targets: goal_hijacking_url_injection | goal_hijacking_plain_text | generic_url_injection | generic_plain_text | topicattack | combinedattack" >&2
      exit 1
      ;;
  esac
}

for attack in "${ATTACKS[@]}"; do
  attack="${attack%,}"
  ATTACK_DIR="${BASE_ROOT}/webarena_tasks_attacker_wasp/${attack}"
  ATTACK_CASE="${ATTACK_DIR}/attack_case.json"
  if [[ ! -f "${ATTACK_CASE}" ]]; then
    echo "[batch] ERROR: missing attack_case.json: ${ATTACK_CASE}" >&2
    exit 1
  fi

  for fmt in "${BASELINE_FORMATS[@]}"; do
    fmt="${fmt%,}"
    OUT_DIR="${BATCH_DIR}/${attack}/${fmt}"
    mkdir -p "${OUT_DIR}"
    echo "[batch] === ${attack} :: ${fmt} ==="

    run_injection_for_target "${fmt}" "${ATTACK_CASE}" "${OUT_DIR}"

    case "${fmt}" in
      goal_hijacking_url_injection|goal_hijacking_plain_text|topicattack|combinedattack)
        continue
        ;;
    esac

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
done

END_TS="$(date +%s)"
TOTAL_SECONDS="$((END_TS - START_TS))"
printf '[batch] Total elapsed: %02d:%02d:%02d (%ds)\n' \
  "$((TOTAL_SECONDS / 3600))" \
  "$(((TOTAL_SECONDS % 3600) / 60))" \
  "$((TOTAL_SECONDS % 60))" \
  "${TOTAL_SECONDS}"

echo "[batch] Done: ${BATCH_DIR}"
