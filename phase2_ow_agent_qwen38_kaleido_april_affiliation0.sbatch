#!/bin/bash
#SBATCH --account=itm
#SBATCH --nodelist=itm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=ow_april_affiliation0
#SBATCH --output=slurm-%x-%j.out

# One April desert episode, with affiliation 0.0 for both actor and scoring.
# Starts a private TA3 process using the shared installation and local TA1.
# Shared server configuration and the service on port 8081 are not modified.
# DRY_RUN=1 prints the agent command without starting services.
set -euo pipefail
cd "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
project_dir="$PWD"
[[ $# == 0 ]] || { echo "This launcher takes no Hydra overrides; configure model/toolkit settings through environment variables." >&2; exit 2; }

export TOOLKIT=kaleido EXPERIMENT=phase2_langchain_agent_ow_kaleido_live
export RUN_MODE=live SESSION_TYPE=eval TRAINING_SESSION=null
export ADM_PROFILE=APRIL_AFFILIATION_0 SCENARIO_ID=April2026-OW_desert3
export AUTO_ASSESS="${AUTO_ASSESS:-false}" TARGET_CONTEXT=true
export OUT_DIR="${OUT_DIR:-outputs/ow_agent/Qwen/Qwen3.8-27B-FP8/live/April-Desert-Affiliation0/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    export TA3_ENDPOINT=http://127.0.0.1:18081
    echo "==> Private TA3: April desert, affiliation 0.0 only (runtime port allocated automatically)"
    exec bash ./phase2_ow_agent_qwen38.sh
fi

server_dir="${TA3_SERVER_DIR:-/data/shared/open-world-servers/itm-evaluation-server}"
server_python="${server_dir}/.venv/bin/python"
[[ -x "$server_python" ]] || { echo "Missing TA3 Python: $server_python" >&2; exit 2; }
mkdir -p "$OUT_DIR"
export OUT_DIR="$(cd "$OUT_DIR" && pwd)"
private_dir="${OUT_DIR}/ta3"
mkdir -p "$private_dir"
config_path="${private_dir}/config.ini"

# Copy settings into a job-owned file. Restrict scenario-to-target mappings,
# including any extra groups the shared installation may gain in the future.
server_port="$("$server_python" - "$server_dir" "$private_dir" <<'PY'
import configparser
from pathlib import Path
import socket
import sys

server, private = map(Path, sys.argv[1:])
config = configparser.ConfigParser()
with (server / "swagger_server/config.ini").open() as source:
    config.read_file(source)
profile = dict(config["APRIL_OPENWORLD3"])
for key in profile:
    if key.startswith("adept_eval_") and key.endswith("_scenarios"):
        profile[key] = ""
profile.update(
    scenario_directory=str(server / "swagger_server/itm/data/april2026/scenarios"),
    adept_eval_filenames="april2026-desert-openworld3_redacted.yaml",
    adept_train_filenames="",
    adept_eval_k2_scenarios="April2026-OW_desert3",
    adept_k2_alignment_targets="ADEPT-June2025-affiliation-0.0",
    save_history_to_s3="False",
    history_directory=str(private / "history"),
)
config["APRIL_AFFILIATION_0"] = profile
config["DEFAULT"]["save_history_to_s3"] = "False"
with (private / "config.ini").open("w") as output:
    config.write(output)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
export TA3_ENDPOINT="http://127.0.0.1:${server_port}"
server_pid=
runner_pid=
cleanup() {
    if [[ -n "$runner_pid" ]]; then
        kill "$runner_pid" 2>/dev/null || true
        wait "$runner_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
    cd "$server_dir"
    export PYTHONDONTWRITEBYTECODE=1
    exec "$server_python" -m swagger_server \
        -f "$config_path" -c "$ADM_PROFILE" -p "$server_port"
) > "${private_dir}/server.log" 2>&1 &
server_pid=$!
echo "==> Private TA3 at $TA3_ENDPOINT; log: ${private_dir}/server.log"
ready=false
for ((attempt=0; attempt<60; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -40 "${private_dir}/server.log" >&2
        exit 1
    fi
    if curl --max-time 2 -fsS "${TA3_ENDPOINT}/openapi.json" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { echo "TA3 startup timed out; see ${private_dir}/server.log" >&2; exit 1; }

bash "${project_dir}/phase2_ow_agent_qwen38.sh" &
runner_pid=$!
status=0
wait "$runner_pid" || status=$?
runner_pid=
exit "$status"
