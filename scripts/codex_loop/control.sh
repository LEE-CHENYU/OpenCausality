#!/usr/bin/env bash
# Control script for codex estimation loop
#
# Usage:
#   ./control.sh start      # Start the loop (background)
#   ./control.sh stop       # Stop gracefully
#   ./control.sh status     # Check if running
#   ./control.sh once       # Run single iteration
#   ./control.sh tail       # Follow the log
#   ./control.sh log        # Show recent log

set -euo pipefail

ROOT="/Users/lichenyu/econometric-research"
LOOP_SCRIPT="$ROOT/scripts/codex_loop/codex_estimation_loop.sh"
PIDFILE="$ROOT/outputs/codex_loop.pid"
STOPFILE="$ROOT/outputs/codex_loop.stop"
LOG="$ROOT/outputs/codex_loop.log"

case "${1:-help}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Loop already running (PID: $(cat "$PIDFILE"))"
      exit 1
    fi
    # Pass through PROVIDER env var if set
    export PROVIDER="${PROVIDER:-codex}"
    echo "Starting estimation loop (provider: $PROVIDER)..."
    nohup "$LOOP_SCRIPT" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "Started with PID: $!"
    echo "Log: $LOG"
    ;;

  stop)
    if [ -f "$PIDFILE" ]; then
      echo "Sending stop signal..."
      touch "$STOPFILE"
      echo "Loop will stop after current iteration"
      echo "To force stop: kill $(cat "$PIDFILE")"
    else
      echo "No PID file found"
    fi
    ;;

  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Running (PID: $(cat "$PIDFILE"))"
      echo "Log tail:"
      tail -5 "$LOG" 2>/dev/null || echo "(no log yet)"
    else
      echo "Not running"
      rm -f "$PIDFILE" 2>/dev/null || true
    fi
    ;;

  once)
    echo "Running single iteration..."
    "$LOOP_SCRIPT" --once
    ;;

  tail)
    echo "Following log (Ctrl+C to stop)..."
    tail -f "$LOG"
    ;;

  log)
    echo "=== Recent log (last 50 lines) ==="
    tail -50 "$LOG" 2>/dev/null || echo "(no log yet)"
    ;;

  *)
    echo "Codex Estimation Loop Control"
    echo ""
    echo "Usage: $0 {start|stop|status|once|tail|log}"
    echo ""
    echo "Commands:"
    echo "  start   Start the loop in background"
    echo "  stop    Stop gracefully after current iteration"
    echo "  status  Check if loop is running"
    echo "  once    Run single iteration (foreground)"
    echo "  tail    Follow the log file"
    echo "  log     Show recent log entries"
    echo ""
    echo "Configuration (environment variables):"
    echo "  DURATION_HOURS=8    How long to run (default: 8)"
    echo "  SLEEP_SECONDS=300   Seconds between iterations (default: 300)"
    echo "  MODEL=gpt-5.2-codex Which model to use"
    ;;
esac
