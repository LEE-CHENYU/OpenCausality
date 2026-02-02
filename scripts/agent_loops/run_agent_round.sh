#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-/Users/lichenyu/econometric-research}"
export RUN_ONCE="1"

bash "$SCRIPT_DIR/datascout_loop.sh"
bash "$SCRIPT_DIR/modelsmith_loop.sh"
bash "$SCRIPT_DIR/estimator_loop.sh"
bash "$SCRIPT_DIR/judge_loop.sh"
