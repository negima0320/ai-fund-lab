#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

PROFILE="${AI_FUND_PROFILE:-rookie_dealer_02_v2_1}"
PROVIDER="${AI_FUND_PROVIDER:-jquants}"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOG_FILE="$ROOT_DIR/logs/demo_auto_order.log"

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

log "demo auto order started profile=$PROFILE"

"$PYTHON" -c 'import sys, yaml; from pathlib import Path; sys.path.insert(0, "src"); from profile_loader import load_profile; profile=load_profile(sys.argv[1]); schedule=yaml.safe_load(Path("config/operation_schedule.yaml").read_text()); broker=profile.get("broker",{}).get("provider"); env=profile.get("tachibana",{}).get("environment"); policy=schedule.get("execution_policy",{}); safety=schedule.get("safety",{}); assert env == "demo", "env must be demo"; assert broker == "tachibana_demo", "broker must be tachibana_demo"; assert broker != "tachibana_live", "live broker is forbidden"; assert policy.get("auto_order_enabled") is True, "auto_order_enabled must be true"; assert policy.get("broker") == "tachibana_demo", "schedule broker must be tachibana_demo"; assert safety.get("forbid_live_auto_order") is True, "forbid_live_auto_order must be true"' "$PROFILE" >> "$LOG_FILE" 2>&1 || {
  status=$?
  log "failed: demo safety gate exit_code=$status"
  exit "$status"
}

run_step "$PYTHON" src/main.py --mode preflight --provider "$PROVIDER" --profile "$PROFILE"
run_step "$PYTHON" src/main.py --mode account-snapshot --profile "$PROFILE"
run_step "$PYTHON" src/main.py --mode demo-auto-order --profile "$PROFILE"

log "demo auto order completed. result appended to logs/demo_orders.log"
