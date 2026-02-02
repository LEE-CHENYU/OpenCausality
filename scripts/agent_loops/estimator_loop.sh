#!/usr/bin/env bash
set -euo pipefail

export ROOT="${ROOT:-/Users/lichenyu/econometric-research}"
export AGENT_NAME="estimator"

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agent_loop.sh"
