#!/bin/bash
# Shared interactive/Slurm runner. See README.md for smoke/live examples.
# RUN_MODE=live|smoke, TOOLKIT=baseline|kaleido,
# AUTO_ASSESS=true|false defaults to false (on-demand assessment).
# TARGET_CONTEXT=true|false (default false) pins target definitions for the actor.
# MODEL_BACKEND=vllm|openai (default vllm); openai uses OPENAI_API_KEY.
# MODEL, TP, PORT, VLLM_URL, TA3_ENDPOINT, ADM_PROFILE, TRAINING_SESSION, SCENARIO_ID,
# RUN_USERNAME, OUT_DIR, CONDA_SH, CONDA_ENV, KALEIDO_DEVICE are optional.
# VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_MODEL_LEN, VLLM_MAX_NUM_SEQS tune memory.
# REQUIRE_SCORES=true|false (default true) fails live runs that end without server scores.
# DRY_RUN=1 prints commands without activating Conda or contacting servers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

fail() { echo "Error: $*" >&2; exit 2; }

# Existing EXPERIMENT-based wrappers remain supported.
default_target_context=false
if [[ -n "${EXPERIMENT:-}" ]]; then
  case "${EXPERIMENT}" in
    phase2_langchain_agent_ow_live) toolkit=baseline ;;
    phase2_langchain_agent_ow_target_live)
      toolkit=baseline; default_target_context=true ;;
    phase2_langchain_agent_ow_kaleido_live) toolkit=kaleido ;;
    *) fail "Unsupported EXPERIMENT: ${EXPERIMENT}" ;;
  esac
else
  toolkit="${TOOLKIT:-baseline}"
fi
case "${toolkit}" in
  baseline) experiment=phase2_langchain_agent_ow_live; toolkit_key=; toolkit_label=Baseline ;;
  kaleido) experiment=phase2_langchain_agent_ow_kaleido_live; toolkit_key=kaleido; toolkit_label=Kaleido ;;
  *) fail "TOOLKIT must be baseline or kaleido" ;;
esac

assessment_mode="${ASSESSMENT_MODE:-}"
auto_assess="${AUTO_ASSESS:-false}"
target_context="${TARGET_CONTEXT:-${default_target_context}}"
requested_username="${RUN_USERNAME:-}"
session_type="${SESSION_TYPE:-}"
ta3_endpoint="${TA3_ENDPOINT:-}"
model_backend="${MODEL_BACKEND:-vllm}"
case "${model_backend}" in
  vllm)
    chat_model_config=vllm_qwen38_27b
    model="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
    base_url="${VLLM_URL:-}" ;;
  openai)
    chat_model_config=openai_api
    model="${MODEL:-${OPENAI_MODEL:-gpt-4.1}}"
    base_url=https://api.openai.com/v1 ;;
  *) fail "MODEL_BACKEND must be vllm or openai" ;;
esac
extra_args=()
# Consume managed Hydra overrides so naming reflects the actual settings and
# a trailing interface.username override cannot bypass the testrun- prefix.
for arg in "$@"; do
  key="${arg#+}"; key="${key#+}"
  case "${key}" in
    interface.username=*) requested_username="${key#*=}" ;;
    interface.session_type=*) session_type="${key#*=}" ;;
    interface.api_endpoint=*) ta3_endpoint="${key#*=}" ;;
    driver.chat_model.model=*) model="${key#*=}" ;;
    driver.chat_model.base_url=*) base_url="${key#*=}" ;;
    "driver.extra_tools.${toolkit_key}.assessment_mode="*) assessment_mode="${key#*=}" ;;
    "driver.extra_tools.${toolkit_key}.auto_assess="*) auto_assess="${key#*=}" ;;
    driver/target_context@driver.target_context=target) target_context=true ;;
    driver.target_context=null|'~driver.target_context') target_context=false ;;
    driver.target_context=*|driver/target_context@driver.target_context=*|'~driver/target_context@driver.target_context'*)
      fail "Select target context with TARGET_CONTEXT=true|false" ;;
    '~interface.username'*) fail "interface.username cannot be removed" ;;
    experiment=*) fail "Select the experiment with TOOLKIT or EXPERIMENT instead" ;;
    -m|--multirun) fail "Submit separate jobs for each assessment mode" ;;
    *) extra_args+=("${arg}") ;;
  esac
done

# SESSION_TYPE=test from older invocations implies smoke unless RUN_MODE is set.
run_mode="${RUN_MODE:-}"
if [[ -z "${run_mode}" ]]; then
  if [[ "${session_type}" == test ]]; then run_mode=smoke; else run_mode=live; fi
fi
case "${run_mode}" in
  smoke)
    session_type="${session_type:-test}"
    [[ "${session_type}" == test ]] || fail "Smoke runs require SESSION_TYPE=test"
    ta3_endpoint="${ta3_endpoint:-http://127.0.0.1:8081}"
    run_label=Smoke ;;
  live)
    session_type="${session_type:-eval}"
    [[ "${session_type}" == eval ]] || fail "Live runs require SESSION_TYPE=eval"
    ta3_endpoint="${ta3_endpoint:-https://darpaitm.caci.com}"
    run_label=Live ;;
  *) fail "RUN_MODE must be smoke or live" ;;
esac
require_scores="${REQUIRE_SCORES:-true}"
require_scores="${require_scores,,}"
case "${require_scores}" in
  true|false) ;;
  *) fail "REQUIRE_SCORES must be true or false" ;;
esac
if [[ "${toolkit}" == kaleido && -n "${assessment_mode}" ]]; then
  fail "ASSESSMENT_MODE applies only to compreg; Kaleido uses its original target-distance assessment"
fi
auto_assess="${auto_assess,,}"
case "${auto_assess}" in
  true) delivery_label=Auto ;;
  false) delivery_label=OnDemand ;;
  *) fail "AUTO_ASSESS must be true or false" ;;
esac
target_context="${target_context,,}"
case "${target_context}" in
  true|false) ;;
  *) fail "TARGET_CONTEXT must be true or false" ;;
esac

adm_profile="${ADM_PROFILE:-APRIL_OPENWORLD3}"
model_label="${model##*/}"
model_label="${model_label//[^a-zA-Z0-9_.-]/-}"
variant="${toolkit_label}"
if [[ -n "${toolkit_key}" ]]; then
  variant="${variant}-${delivery_label}"
fi
if [[ "${target_context}" == true ]]; then variant="${variant}-TargetAware"; fi
# The profile selects the scenario set, so runs of different profiles must not
# share a username or output directory.
variant="${variant}-${adm_profile}"
username="${requested_username:-ALIGN-ADM-Ph2-LangChainAgent-${variant}-${run_label}-${model_label}}"
username="testrun-${username#testrun-}"
[[ "${username}" =~ ^testrun-[a-zA-Z0-9_.-]+$ ]] || fail "RUN_USERNAME must contain only letters, digits, dots, underscores, or hyphens"

run_id="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
out_dir="${OUT_DIR:-outputs/ow_agent/${model}/${run_mode}/${variant}/${run_id}}"
port="${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
start_vllm=true
if [[ -n "${base_url}" ]]; then start_vllm=false; fi
base_url="${base_url:-http://127.0.0.1:${port}/v1}"

run_args=(
  "+experiment=phase2_feb_openworld/${experiment}"
  "driver/chat_model=${chat_model_config}"
  "driver.chat_model.model=${model}"
  "driver.chat_model.base_url=${base_url}"
  "interface.api_endpoint=${ta3_endpoint}"
  "interface.session_type=${session_type}"
  "interface.training_session=${TRAINING_SESSION:-null}"
  "interface.adm_profile=${adm_profile}"
  "hydra.run.dir=${out_dir}"
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  run_args+=("resume_from=${RESUME_FROM}")
fi
if [[ "${model_backend}" == vllm ]]; then
  run_args+=("driver.max_messages_in_context=12")
fi
if [[ "${run_mode}" == smoke ]]; then
  run_args+=("driver.max_actions_per_scenario=${SMOKE_MAX_ACTIONS:-10}")
fi
if [[ -n "${SCENARIO_ID:-}" ]]; then
  run_args+=("interface.scenario_ids=[${SCENARIO_ID}]")
fi
if [[ -n "${toolkit_key}" ]]; then
  run_args+=("driver.extra_tools.${toolkit_key}.auto_assess=${auto_assess}")
fi
if [[ "${target_context}" == true ]]; then
  run_args+=("+driver/target_context@driver.target_context=target")
else
  run_args+=("driver.target_context=null")
fi
if [[ "${toolkit}" == kaleido ]]; then
  # vLLM claims most GPU memory when started in this job.
  if [[ "${start_vllm}" == true ]]; then default_device=cpu; else default_device=cuda; fi
  run_args+=("+driver.extra_tools.kaleido.kaleido_instance.device=${KALEIDO_DEVICE:-${default_device}}")
fi
run_args+=("${extra_args[@]}" "interface.username=${username}")

# Alignment scores come from the TA3 server at the end of each live eval
# scenario (docs/agent_alignment.md); test sessions never score. After the
# run, report every scenario's score or the reason it is missing.
report_args=("${out_dir}")
if [[ "${run_mode}" == live && "${require_scores}" == true ]]; then
  report_args+=(--require)
fi

vllm_args=(
  serve "${model}"
  --language-model-only
  --tensor-parallel-size "${TP:-1}"
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --max-model-len "${VLLM_MAX_MODEL_LEN:-65536}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-8}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
  --port "${port}"
)

echo "==> ${run_mode}: ${variant} as ${username} against ${ta3_endpoint} -> ${out_dir}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  if [[ "${start_vllm}" == true ]]; then
    printf '%q ' env VLLM_USE_FLASHINFER_SAMPLER=0 vllm "${vllm_args[@]}"
    printf '\n'
  fi
  printf '%q ' run_align_system "${run_args[@]}"
  printf '\n'
  printf '%q ' python -m align_system.cli.report_agent_scores "${report_args[@]}"
  printf '\n'
  exit 0
fi

if ! command -v run_align_system >/dev/null 2>&1; then
  conda_sh="${CONDA_SH:-/home/local/KHQ/vincenzo.dimatteo/miniconda3/etc/profile.d/conda.sh}"
  [[ -f "${conda_sh}" ]] || fail "Set CONDA_SH or activate the align environment first"
  source "${conda_sh}"
  conda activate "${CONDA_ENV:-align}"
fi

if [[ "${model_backend}" == openai ]]; then
  [[ -n "${OPENAI_API_KEY:-}" ]] || fail "Export OPENAI_API_KEY before submitting the job"
elif [[ "${start_vllm}" == false ]]; then
  curl -sf "${base_url%/v1}/health" >/dev/null || fail "No vLLM at ${base_url}"
else
  mkdir -p "${out_dir}"
  vllm_log="${out_dir}/vllm.log"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  echo "==> starting vLLM on port ${port} (log: ${vllm_log})"
  vllm "${vllm_args[@]}" \
    > >(tee "${vllm_log}") 2>&1 &
  vllm_pid=$!
  cleanup() {
    kill "${vllm_pid}" 2>/dev/null || true
    wait "${vllm_pid}" 2>/dev/null || true
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for ((attempt=0; attempt<360; attempt++)); do
    curl -sf "${base_url%/v1}/health" >/dev/null 2>&1 && break
    kill -0 "${vllm_pid}" 2>/dev/null || fail "vLLM exited; see ${vllm_log}"
    sleep 10
  done
  curl -sf "${base_url%/v1}/health" >/dev/null || fail "vLLM did not come up in time"
fi
run_status=0
run_align_system "${run_args[@]}" || run_status=$?

# Summarize the run and report its scores even after a failed run, so the
# Slurm log always says which scenarios were scored.
if ! python -m align_system.cli.summarize_agent_alignment "${out_dir}/input_output.json" \
    > "${out_dir}/alignment_summary.json"; then
  rm -f "${out_dir}/alignment_summary.json"
  echo "Warning: could not summarize ${out_dir}/input_output.json" >&2
fi
report_status=0
python -m align_system.cli.report_agent_scores "${report_args[@]}" || report_status=$?
if [[ "${run_status}" != 0 ]]; then exit "${run_status}"; fi
exit "${report_status}"
