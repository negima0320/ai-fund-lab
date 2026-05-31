#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

PROFILE="${AI_FUND_PROFILE:-rookie_dealer_02_v2_1}"
PROVIDER="${AI_FUND_PROVIDER:-jquants}"
TARGET_DATE="${AI_FUND_TARGET_DATE:-$(date +%F)}"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOG_FILE="$ROOT_DIR/logs/evening_selection.log"

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

log "evening selection started profile=$PROFILE provider=$PROVIDER date=$TARGET_DATE"

run_step "$PYTHON" src/main.py --mode preflight --provider "$PROVIDER" --profile "$PROFILE"
run_step "$PYTHON" src/main.py --mode fetch-prices --provider "$PROVIDER" --profile "$PROFILE" --date "$TARGET_DATE"
run_step "$PYTHON" src/main.py --mode calculate-indicators --provider "$PROVIDER" --profile "$PROFILE" --date "$TARGET_DATE"
run_step "$PYTHON" src/main.py --mode screen --provider "$PROVIDER" --profile "$PROFILE" --date "$TARGET_DATE"
run_step "$PYTHON" src/main.py --mode score --provider "$PROVIDER" --profile "$PROFILE" --date "$TARGET_DATE"
run_step "$PYTHON" src/main.py --mode preview-orders --provider "$PROVIDER" --profile "$PROFILE" --date "$TARGET_DATE"
run_step "$PYTHON" src/main.py --mode analyze --profile "$PROFILE"

log "evening selection completed. next business day order candidates saved under reports/$PROFILE/order_previews"
