#!/bin/bash
# Interactive compreg launcher; all shared runner settings apply.
# RUN_MODE=smoke|live ASSESSMENT_MODE=evidence|ranked AUTO_ASSESS=true|false.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export TOOLKIT=compreg
export EXPERIMENT=phase2_langchain_agent_ow_comparative_regression_live
exec bash ./phase2_ow_agent_qwen38.sh "$@"
