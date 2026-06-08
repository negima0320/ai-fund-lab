#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.phase8f_pm_ai_label_redesign_audit import Phase8FPMAILabelRedesignAudit  # noqa: E402


def main() -> int:
    audit = Phase8FPMAILabelRedesignAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    verdict = report["verdict"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"current_label_is_correct={verdict.get('current_label_is_correct')}")
    print(f"pm_ai_v2_problem_is_label={verdict.get('pm_ai_v2_problem_is_label')}")
    print(f"recommended_label_design={verdict.get('recommended_label_design')}")
    print(f"ready_for_phase8g_retraining={verdict.get('ready_for_phase8g_retraining')}")
    print(f"next_phase_recommended={verdict.get('next_phase_recommended')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
