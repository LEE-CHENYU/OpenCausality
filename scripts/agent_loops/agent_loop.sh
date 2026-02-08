#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/lichenyu/econometric-research}"
AGENT_NAME="${AGENT_NAME:-datascout}"
CONFIG_DIR="${CONFIG_DIR:-$ROOT/config/agentic}"
PROMPT_FILE="${PROMPT_FILE:-$CONFIG_DIR/prompts/${AGENT_NAME}.txt}"
OBJECTIVE_FILE="${OBJECTIVE_FILE:-$CONFIG_DIR/objectives/${AGENT_NAME}.txt}"
EVAL_FILE="${EVAL_FILE:-$CONFIG_DIR/eval/${AGENT_NAME}.md}"

LOG_DIR="${LOG_DIR:-$ROOT/outputs/agentic/logs}"
RUN_DIR="${RUN_DIR:-$ROOT/outputs/agentic/${AGENT_NAME}}"
LOG="${LOG:-$LOG_DIR/${AGENT_NAME}_loop.log}"
PIDFILE="${PIDFILE:-$RUN_DIR/${AGENT_NAME}_loop.pid}"
STOPFILE="${STOPFILE:-$RUN_DIR/${AGENT_NAME}_loop.stop}"
RESUME_FILE="${RESUME_FILE:-$RUN_DIR/resume.md}"

SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
DURATION_SECONDS="${DURATION_SECONDS:-86400}"
SANDBOX_MODE="${SANDBOX_MODE:-danger-full-access}"
RUN_ONCE="${RUN_ONCE:-0}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
PROVIDER="${PROVIDER:-codex}"  # codex | claude

OBJECTIVE_DEFAULT="Maintain backward compatibility while producing agent artifacts for the DAG workflow."

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "$$" > "$PIDFILE"

start_ts=$(date +%s)
end_ts=$((start_ts + DURATION_SECONDS))

{
  echo "=== ${AGENT_NAME} loop started: $(date) ==="
  echo "root: $ROOT"
  echo "sleep: ${SLEEP_SECONDS}s"
  echo "duration: ${DURATION_SECONDS}s"
} >> "$LOG"

iter=0
while [ "$(date +%s)" -lt "$end_ts" ]; do
  iter=$((iter + 1))
  if [ -f "$STOPFILE" ]; then
    echo "Stopfile detected at $(date); exiting." >> "$LOG"
    break
  fi

  echo "=== Iteration $iter @ $(date) ===" >> "$LOG"

  objective="$OBJECTIVE_DEFAULT"
  if [ -n "${OBJECTIVE:-}" ]; then
    objective="$OBJECTIVE"
  elif [ -f "$OBJECTIVE_FILE" ]; then
    objective=$(cat "$OBJECTIVE_FILE" | tr -d '\r')
  fi

  prompt_path="$RUN_DIR/${AGENT_NAME}_prompt.txt"
  {
    if [ -f "$PROMPT_FILE" ]; then
      cat "$PROMPT_FILE"
    else
      echo "You are ${AGENT_NAME}."
    fi
    echo ""
    echo "Objective:"
    echo "$objective"
    if [ -f "$EVAL_FILE" ]; then
      echo ""
      echo "Evaluation criteria:"
      cat "$EVAL_FILE"
    fi
    echo ""
    echo "Runtime paths:"
    echo "- Repo root: $ROOT"
    echo "- Inbox: $ROOT/outputs/agentic/inbox"
    echo "- Outbox: $ROOT/outputs/agentic/outbox"
    echo "- Queue: $ROOT/outputs/agentic/queue"
    echo "- Ledger: $ROOT/outputs/agentic/ledger"
    echo "- Resume file: $RESUME_FILE"
  } > "$prompt_path"

  (
    cd "$ROOT"
    if [ "$PROVIDER" = "claude" ]; then
      claude -p "$(cat "$prompt_path")" \
        --allowedTools "Bash(git:*),Bash(python:*),Read,Write,Edit,Glob,Grep" \
        2>&1
    else
      codex exec --sandbox "$SANDBOX_MODE" --full-auto "$(cat "$prompt_path")"
    fi
  ) >> "$LOG" 2>&1

  {
    echo "--- git status ---"
    cd "$ROOT" && git status -sb
    echo "--- git diff --stat ---"
    cd "$ROOT" && git diff --stat
    echo "--- end iteration ---"
  } >> "$LOG" 2>&1

  if [ "$RUN_ONCE" = "1" ]; then
    break
  fi

  sleep "$SLEEP_SECONDS"
done

echo "=== ${AGENT_NAME} loop ended: $(date) ===" >> "$LOG"
