#!/bin/bash
set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
export HF_HOME="/data/shared/models/huggingface"
export TRANSFORMERS_CACHE="/data/shared/models/huggingface"
export HUGGINGFACE_HUB_CACHE="/data/shared/models/huggingface"
export HF_DATASETS_CACHE="/data/shared/datasets/huggingface"
export OUTLINES_CACHE_DIR="/data/shared/outlines_cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/align_system/configs/experiment"
ALL_ATTRS=(AF MF PS SS)
TARGETS_PER_ATTR=8

# ── Parse flags ──────────────────────────────────────────────────────────────
FLAG_EXP_GROUP=""
FLAG_PIPELINE=""
FLAG_ATTRS=""
FLAG_MODE=""
FLAG_OUTDIR=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options (all optional — prompts interactively if omitted):"
    echo "  -e, --experiment GROUP   Experiment group directory name"
    echo "  -p, --pipeline   NAME    Pipeline config name (without .yaml)"
    echo "  -a, --attrs      LIST    Comma-separated attributes (e.g. AF,MF,PS,SS)"
    echo "  -m, --mode       MODE    Run mode: sequential | parallel | dry-run"
    echo "  -o, --outdir     DIR     Output base directory (default: phase2_feb2026_results_local)"
    echo "  -h, --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                          # fully interactive"
    echo "  $0 -e claude_sonnet -p baseline -a AF -m dry-run"
    echo "  $0 -e claude_sonnet -p baseline -a AF,MF,PS,SS -m parallel"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment) FLAG_EXP_GROUP="$2"; shift 2 ;;
        -p|--pipeline)   FLAG_PIPELINE="$2";  shift 2 ;;
        -a|--attrs)      FLAG_ATTRS="$2";     shift 2 ;;
        -m|--mode)       FLAG_MODE="$2";      shift 2 ;;
        -o|--outdir)     FLAG_OUTDIR="$2";    shift 2 ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1" >&2; echo "Use --help for usage." >&2; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
prompt_selection() {
    local prompt_msg="$1"
    shift
    local options=("$@")

    echo ""
    echo "${prompt_msg}"
    for i in "${!options[@]}"; do
        printf "  %d) %s\n" $((i + 1)) "${options[$i]}"
    done

    while true; do
        read -rp "Enter number [1-${#options[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
            SELECTED="${options[$((choice - 1))]}"
            return 0
        fi
        echo "Invalid selection. Try again."
    done
}

prompt_multi_selection() {
    local prompt_msg="$1"
    shift
    local options=("$@")

    echo ""
    echo "${prompt_msg}"
    printf "  0) ALL\n"
    for i in "${!options[@]}"; do
        printf "  %d) %s\n" $((i + 1)) "${options[$i]}"
    done

    while true; do
        read -rp "Enter numbers separated by spaces (e.g. '1 3' or '0' for all): " -a choices
        MULTI_SELECTED=()

        local valid=true
        for c in "${choices[@]}"; do
            if ! [[ "$c" =~ ^[0-9]+$ ]] || (( c < 0 || c > ${#options[@]} )); then
                valid=false
                break
            fi
            if (( c == 0 )); then
                MULTI_SELECTED=("${options[@]}")
                return 0
            fi
            MULTI_SELECTED+=("${options[$((c - 1))]}")
        done

        # Deduplicate selections
        if $valid && (( ${#MULTI_SELECTED[@]} > 0 )); then
            local -A seen=()
            local unique=()
            for item in "${MULTI_SELECTED[@]}"; do
                if [[ -z "${seen[$item]+x}" ]]; then
                    seen[$item]=1
                    unique+=("$item")
                fi
            done
            MULTI_SELECTED=("${unique[@]}")
            return 0
        fi
        echo "Invalid selection. Try again."
    done
}

# ── 1. Select experiment group ───────────────────────────────────────────────
if [[ -n "$FLAG_EXP_GROUP" ]]; then
    if [[ ! -d "${CONFIG_DIR}/${FLAG_EXP_GROUP}" ]]; then
        echo "Error: Experiment group '${FLAG_EXP_GROUP}' not found in ${CONFIG_DIR}" >&2
        exit 1
    fi
    exp_group="$FLAG_EXP_GROUP"
else
    mapfile -t exp_groups < <(find "$CONFIG_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    if (( ${#exp_groups[@]} == 0 )); then
        echo "Error: No experiment groups found in ${CONFIG_DIR}" >&2
        exit 1
    fi
    prompt_selection "Select experiment group:" "${exp_groups[@]}"
    exp_group="$SELECTED"
fi

# ── 2. Select pipeline config ────────────────────────────────────────────────
if [[ -n "$FLAG_PIPELINE" ]]; then
    if [[ ! -f "${CONFIG_DIR}/${exp_group}/${FLAG_PIPELINE}.yaml" ]]; then
        echo "Error: Pipeline '${FLAG_PIPELINE}.yaml' not found in ${CONFIG_DIR}/${exp_group}/" >&2
        exit 1
    fi
    pipeline="$FLAG_PIPELINE"
else
    mapfile -t pipelines < <(find "${CONFIG_DIR}/${exp_group}" -maxdepth 1 -name '*.yaml' -printf '%f\n' | sed 's/\.yaml$//' | sort)
    if (( ${#pipelines[@]} == 0 )); then
        echo "Error: No pipeline configs found in ${CONFIG_DIR}/${exp_group}/" >&2
        exit 1
    fi
    prompt_selection "Select pipeline config:" "${pipelines[@]}"
    pipeline="$SELECTED"
fi

# ── 3. Select attributes ────────────────────────────────────────────────────
if [[ -n "$FLAG_ATTRS" ]]; then
    IFS=',' read -ra selected_attrs <<< "$FLAG_ATTRS"
    # Validate each attr
    for a in "${selected_attrs[@]}"; do
        local_valid=false
        for valid_attr in "${ALL_ATTRS[@]}"; do
            [[ "$a" == "$valid_attr" ]] && local_valid=true && break
        done
        if ! $local_valid; then
            echo "Error: Invalid attribute '${a}'. Valid: ${ALL_ATTRS[*]}" >&2
            exit 1
        fi
    done
else
    prompt_multi_selection "Select attributes to evaluate:" "${ALL_ATTRS[@]}"
    selected_attrs=("${MULTI_SELECTED[@]}")
fi

# ── 4. Select run mode ──────────────────────────────────────────────────────
if [[ -n "$FLAG_MODE" ]]; then
    case "$FLAG_MODE" in
        sequential|parallel|dry-run) RUN_MODE="$FLAG_MODE" ;;
        *) echo "Error: Invalid mode '${FLAG_MODE}'. Valid: sequential, parallel, dry-run" >&2; exit 1 ;;
    esac
else
    RUN_MODES=("sequential" "parallel (tmux grid)" "dry run")
    prompt_selection "Select run mode:" "${RUN_MODES[@]}"
    case "$SELECTED" in
        "sequential")            RUN_MODE="sequential" ;;
        "parallel (tmux grid)")  RUN_MODE="parallel"   ;;
        "dry run")               RUN_MODE="dry-run"    ;;
    esac
fi

# ── 5. Confirm and run ──────────────────────────────────────────────────────
output_basedir="${FLAG_OUTDIR:-phase2_feb2026_results_local}"
date_now="$(date +"%Y-%m-%d__%H-%M-%S")"

total_runs=$(( ${#selected_attrs[@]} * TARGETS_PER_ATTR ))

# Build the equivalent flag-based command
flag_cmd="./run_eval.sh -e ${exp_group} -p ${pipeline} -a $(IFS=,; echo "${selected_attrs[*]}") -m ${RUN_MODE}"
[[ "$output_basedir" != "phase2_feb2026_results_local" ]] && flag_cmd+=" -o ${output_basedir}"

echo ""
echo "═══════════════════════════════════════════"
echo "  Experiment:  ${exp_group}/${pipeline}"
echo "  Attributes:  ${selected_attrs[*]}"
echo "  Targets:     1..${TARGETS_PER_ATTR} per attribute"
echo "  Total runs:  ${total_runs}"
echo "  Output dir:  ${output_basedir}/${pipeline}/${date_now}/"
echo "  Mode:        ${RUN_MODE}"
echo "═══════════════════════════════════════════"
echo ""
echo "  Equivalent command:"
echo "  ${flag_cmd}"
echo ""

if [[ "$RUN_MODE" != "dry-run" ]]; then
    read -rp "Proceed? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""

# ── Build command list ────────────────────────────────────────────────────────
commands=()
labels=()
for attr in "${selected_attrs[@]}"; do
    scenario_ids="Feb2026-${attr}0-train"
    for i in {1..9}; do
        scenario_ids="${scenario_ids},Feb2026-${attr}${i}-train"
    done

    for t in $(seq 1 "$TARGETS_PER_ATTR"); do
        target="Feb2026-${attr}-${t}"
        cmd="uv run run_align_system \
\"+experiment=${exp_group}/${pipeline}\" \
\"interface.scenario_ids=[${scenario_ids}]\" \
\"+alignment_target=feb2026/${target}\" \
\"hydra.run.dir=${output_basedir}/${pipeline}/${date_now}/${target}\""
        commands+=("$cmd")
        labels+=("$target")
    done
done

# ── Dry run ───────────────────────────────────────────────────────────────────
if [[ "$RUN_MODE" == "dry-run" ]]; then
    echo "[dry-run] Would run ${#commands[@]} commands:"
    echo ""
    for i in "${!commands[@]}"; do
        echo "[${labels[$i]}] ${commands[$i]}"
    done
    exit 0
fi

# ── Parallel tmux execution ──────────────────────────────────────────────────
if [[ "$RUN_MODE" == "parallel" ]]; then
    # Export env vars so tmux panes inherit them
    env_exports="export HF_HOME='${HF_HOME}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}' HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}' HF_DATASETS_CACHE='${HF_DATASETS_CACHE}' OUTLINES_CACHE_DIR='${OUTLINES_CACHE_DIR}'"
    # Propagate API keys if set
    [[ -n "${OPENAI_API_KEY:-}" ]]    && env_exports+=" OPENAI_API_KEY='${OPENAI_API_KEY}'"
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && env_exports+=" ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY}'"

    session_name="eval-${pipeline}-${date_now}"

    if [[ -n "${TMUX:-}" ]]; then
        current_session="$(tmux display-message -p '#S')"
    else
        # Create a detached session with a dummy window (replaced below)
        tmux new-session -d -s "$session_name" -n "_init"
    fi

    first_window=true
    # One window per attribute, panes for each target within
    for attr in "${selected_attrs[@]}"; do
        window_name="${attr}"

        # Collect commands for this attr
        attr_cmds=()
        attr_labels=()
        for i in "${!labels[@]}"; do
            if [[ "${labels[$i]}" == Feb2026-${attr}-* ]]; then
                attr_cmds+=("${commands[$i]}")
                attr_labels+=("${labels[$i]}")
            fi
        done

        if [[ -n "${TMUX:-}" ]]; then
            tmux new-window -t "$current_session" -n "$window_name"
            target_window="${current_session}:${window_name}"
        else
            if $first_window; then
                # Rename the dummy init window
                tmux rename-window -t "${session_name}:_init" "$window_name"
                first_window=false
            else
                tmux new-window -t "$session_name" -n "$window_name"
            fi
            target_window="${session_name}:${window_name}"
        fi

        # First target goes into the initial pane
        tmux send-keys -t "$target_window" "cd ${SCRIPT_DIR} && ${env_exports} && ${attr_cmds[0]}; echo -e '\\n── ${attr_labels[0]} DONE (exit \$?) ──'; exec bash" Enter

        # Remaining targets each get a new pane
        for j in $(seq 1 $(( ${#attr_cmds[@]} - 1 ))); do
            tmux split-window -t "$target_window" -h
            tmux select-layout -t "$target_window" tiled
            tmux send-keys -t "$target_window" "cd ${SCRIPT_DIR} && ${env_exports} && ${attr_cmds[$j]}; echo -e '\\n── ${attr_labels[$j]} DONE (exit \$?) ──'; exec bash" Enter
        done

        tmux select-layout -t "$target_window" tiled
    done

    if [[ -n "${TMUX:-}" ]]; then
        # Switch to the first new window
        tmux select-window -t "${current_session}:${selected_attrs[0]}"
        echo "Launched ${#selected_attrs[@]} windows (${selected_attrs[*]}) with ${TARGETS_PER_ATTR} panes each"
    else
        echo "Launched ${#selected_attrs[@]} windows (${selected_attrs[*]}) with ${TARGETS_PER_ATTR} panes each"
        echo "Attach with: tmux attach -t ${session_name}"
    fi
    exit 0
fi

# ── Sequential execution ─────────────────────────────────────────────────────
set -x
for i in "${!commands[@]}"; do
    eval "${commands[$i]}"
done
