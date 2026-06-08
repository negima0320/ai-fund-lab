#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase8d_pm_ai_v2_calibration_audit import Phase8DPMCalibrationAudit  # noqa: E402


def main() -> int:
    audit = Phase8DPMCalibrationAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    verdict = report.get("verdict", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"calibration_rule_recommended={verdict.get('calibration_rule_recommended')}")
    print(f"pm_ai_v2_calibration_feasible={verdict.get('pm_ai_v2_calibration_feasible')}")
    print(f"ready_for_phase8e_backtest={verdict.get('ready_for_phase8e_backtest')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

