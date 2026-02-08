#!/usr/bin/env bash
# Codex CLI loop for econometric DAG estimation pipeline
# Adapted from codex_24h_loop.sh for causal inference automation
#
# Usage:
#   ./codex_estimation_loop.sh              # Run with defaults (8 hours)
#   DURATION_HOURS=24 ./codex_estimation_loop.sh  # Run for 24 hours
#   ./codex_estimation_loop.sh --once       # Single iteration only

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT="${ROOT:-/Users/lichenyu/econometric-research}"
LOG="$ROOT/outputs/codex_loop.log"
PIDFILE="$ROOT/outputs/codex_loop.pid"
STOPFILE="$ROOT/outputs/codex_loop.stop"

# Timing
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"          # 5 min between iterations
DURATION_HOURS="${DURATION_HOURS:-8}"          # Default 8 hours
DURATION_SECONDS=$((DURATION_HOURS * 3600))

# Files
OBJECTIVE_FILE="$ROOT/scripts/codex_loop/codex_objective.txt"
RESUME_FILE="$ROOT/scripts/codex_loop/codex_resume.md"
DAG_PATH="$ROOT/config/agentic/dags/kspi_k2_full.yaml"
REPORT_PATH="$ROOT/outputs/agentic/KSPI_K2_REAL_ESTIMATION_REPORT.md"

# Provider and model settings
PROVIDER="${PROVIDER:-codex}"  # codex | claude
SANDBOX_MODE="${SANDBOX_MODE:-danger-full-access}"
MODEL="${MODEL:-gpt-5.3-codex}"

# =============================================================================
# DEFAULT OBJECTIVE
# =============================================================================

OBJECTIVE_DEFAULT="Econometric DAG Estimation Pipeline:
1. Run pre-estimation validation on the DAG specification
2. Execute estimation for any edges that need re-estimation
3. Run post-estimation validation on EdgeCards
4. Check report consistency (report values match EdgeCards)
5. Fix any issues found automatically
6. Update the report if estimates changed
7. Commit changes with clear message

Priority: Fix validation errors > Fix warnings > Improve estimates

Key files:
- DAG: config/agentic/dags/kspi_k2_full.yaml
- Estimation: scripts/run_real_estimation.py
- Validation: shared/agentic/validation.py
- Report checker: shared/agentic/report_checker.py
- Output: outputs/agentic/KSPI_K2_REAL_ESTIMATION_REPORT.md"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

cleanup() {
  log "Cleaning up..."
  rm -f "$PIDFILE"
}
trap cleanup EXIT

# =============================================================================
# SINGLE RUN MODE
# =============================================================================

if [ "${1:-}" = "--once" ]; then
  log "=== Single iteration mode ==="

  objective="$OBJECTIVE_DEFAULT"
  if [ -f "$OBJECTIVE_FILE" ]; then
    objective="$(cat "$OBJECTIVE_FILE")"
  fi

  PROMPT_TEXT="You are improving the econometric research repository.

OBJECTIVE:
$objective

CONTEXT:
$(cat "$RESUME_FILE" 2>/dev/null || echo 'No previous context.')

INSTRUCTIONS:
1. Read the resume file for context on previous work
2. Run validation: python -c \"from shared.agentic.validation import run_full_validation; print(run_full_validation('$DAG_PATH').to_markdown())\"
3. If issues found, fix them
4. If estimation needed, run: python scripts/run_real_estimation.py
5. Check report consistency
6. Update $RESUME_FILE with: current focus, changes made, issues found, next steps
7. If code changed and tests pass, commit with message: codex(econometric): <summary>

Be thorough but focused. Prefer fixing validation errors over adding features."

  cd "$ROOT"
  if [ "$PROVIDER" = "claude" ]; then
    claude -p "$PROMPT_TEXT" \
      --allowedTools "Bash(git:*),Bash(python:*),Read,Write,Edit,Glob,Grep"
  else
    codex exec -s "$SANDBOX_MODE" -m "$MODEL" "$PROMPT_TEXT"
  fi

  exit 0
fi

# =============================================================================
# MAIN LOOP
# =============================================================================

mkdir -p "$(dirname "$LOG")"
mkdir -p "$(dirname "$PIDFILE")"

start_ts=$(date +%s)
end_ts=$((start_ts + DURATION_SECONDS))
echo "$$" > "$PIDFILE"

log "=== Codex estimation loop started ==="
log "Root: $ROOT"
log "Duration: ${DURATION_HOURS}h (${DURATION_SECONDS}s)"
log "Sleep: ${SLEEP_SECONDS}s between iterations"
log "Model: $MODEL"
log "PID: $$"

# Initialize resume file if missing
if [ ! -f "$RESUME_FILE" ]; then
  cat > "$RESUME_FILE" << 'EOF'
# Codex Estimation Loop Resume

## Current Focus
- Initial run: validate DAG and run estimation pipeline

## Previous Changes
- None yet (first iteration)

## Issues Found
- TBD

## Next Steps
1. Run pre-estimation validation
2. Execute estimation pipeline
3. Check report consistency

## Risks / Blockers
- None identified
EOF
  log "Created initial resume file"
fi

# Initialize objective file if missing
if [ ! -f "$OBJECTIVE_FILE" ]; then
  echo "$OBJECTIVE_DEFAULT" > "$OBJECTIVE_FILE"
  log "Created initial objective file"
fi

iter=0
while [ "$(date +%s)" -lt "$end_ts" ]; do
  iter=$((iter + 1))

  # Check for stop signal
  if [ -f "$STOPFILE" ]; then
    log "Stop file detected, exiting gracefully"
    rm -f "$STOPFILE"
    break
  fi

  log "=== Iteration $iter started ==="

  # Read current objective
  objective="$(cat "$OBJECTIVE_FILE" 2>/dev/null || echo "$OBJECTIVE_DEFAULT")"

  # Read resume context
  resume_context="$(cat "$RESUME_FILE" 2>/dev/null || echo 'No previous context.')"

  # Build prompt
  ITER_PROMPT="You are improving the econometric research repository at $ROOT.

ITERATION: $iter
OBJECTIVE:
$objective

PREVIOUS CONTEXT:
$resume_context

VALIDATION COMMANDS:
- Pre-estimation: python -c \"from shared.agentic.validation import DAGValidator; v=DAGValidator.from_yaml('$DAG_PATH'); print(v.validate_pre_estimation().to_markdown())\"
- Full pipeline: python scripts/run_real_estimation.py
- Report check: python -c \"from shared.agentic.report_checker import check_report_consistency; from pathlib import Path; print('Run after loading EdgeCards')\"

RULES:
1. Run validation FIRST before making changes
2. Fix errors before warnings
3. Keep changes minimal and focused
4. Update $RESUME_FILE before finishing with:
   - Current focus
   - Changes made this iteration
   - Validation results
   - Next steps
5. If code changed and tests pass, commit: codex(iter $iter): <summary>
6. If no changes needed, say 'No changes required' and explain why

OUTPUT: Brief summary of what was done and validation status."

  # Run with selected provider
  cd "$ROOT"
  if [ "$PROVIDER" = "claude" ]; then
    claude -p "$ITER_PROMPT" \
      --allowedTools "Bash(git:*),Bash(python:*),Read,Write,Edit,Glob,Grep" \
      >> "$LOG" 2>&1 || log "Claude execution had non-zero exit"
  else
    codex exec -s "$SANDBOX_MODE" -m "$MODEL" "$ITER_PROMPT" \
      >> "$LOG" 2>&1 || log "Codex execution had non-zero exit"
  fi

  # Log git status
  log "Git status after iteration $iter:"
  git -C "$ROOT" status -sb >> "$LOG" 2>&1 || true

  log "=== Iteration $iter completed ==="

  # Sleep before next iteration
  if [ "$(date +%s)" -lt "$end_ts" ]; then
    log "Sleeping ${SLEEP_SECONDS}s before next iteration..."
    sleep "$SLEEP_SECONDS"
  fi
done

log "=== Codex estimation loop ended ==="
log "Total iterations: $iter"
