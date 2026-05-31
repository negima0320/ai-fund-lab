#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

PROFILE="${AI_FUND_PROFILE:-rookie_dealer_02_v2_1}"
PROVIDER="${AI_FUND_PROVIDER:-jquants}"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOG_FILE="$ROOT_DIR/logs/paper_run.log"

START_DATE="${AI_FUND_START_DATE:-$(date +%F)}"
END_DATE="${AI_FUND_END_DATE:-$(date +%F)}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

run_step() {
  log "start: $*"
  if "$@" >> "$LOG_FILE" 2>&1; then
    log "done: $*"
  else
    status=$?
    log "failed: $* exit_code=$status"
    exit "$status"
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  log "failed: python not found or not executable: $PYTHON"
  exit 127
fi

log "daily paper run started profile=$PROFILE provider=$PROVIDER start_date=$START_DATE end_date=$END_DATE"

run_step "$PYTHON" src/main.py --mode preflight --provider "$PROVIDER" --profile "$PROFILE"
run_step "$PYTHON" src/main.py --mode full-paper-run --provider "$PROVIDER" --profile "$PROFILE" --start-date "$START_DATE" --end-date "$END_DATE"
run_step "$PYTHON" src/main.py --mode analyze --profile "$PROFILE"

log "daily paper run completed"
