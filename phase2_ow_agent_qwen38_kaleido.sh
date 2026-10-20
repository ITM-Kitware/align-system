#!/bin/bash
# Interactive kaleido launcher; all shared runner settings apply.
# RUN_MODE=smoke|live AUTO_ASSESS=true|false (default true).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export TOOLKIT=kaleido
export EXPERIMENT=phase2_langchain_agent_ow_kaleido_live
exec bash ./phase2_ow_agent_qwen38.sh "$@"
