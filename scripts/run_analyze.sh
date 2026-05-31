#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

PROFILE="${AI_FUND_PROFILE:-rookie_dealer_02_v2_1}"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOG_FILE="$ROOT_DIR/logs/paper_run.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

if [[ ! -x "$PYTHON" ]]; then
  log "failed: python not found or not executable: $PYTHON"
  exit 127
fi

log "analyze started profile=$PROFILE"

if "$PYTHON" src/main.py --mode analyze --profile "$PROFILE" >> "$LOG_FILE" 2>&1; then
  log "analyze completed"
else
  status=$?
  log "analyze failed exit_code=$status"
  exit "$status"
fi
