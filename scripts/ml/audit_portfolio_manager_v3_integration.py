#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_v3_integration_audit import PMAIV3CandidateIntegrationAudit  # noqa: E402


def main() -> None:
    audit = PMAIV3CandidateIntegrationAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    verdict = report["verdict"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"prediction_columns={','.join(report['prediction_columns'])}")
    print(f"best_mapping={verdict.get('best_mapping_name')}")
    print(f"best_mapping_pm130_count={verdict.get('best_mapping_pm130_count')}")
    print(f"best_mapping_pm130_actual_downside_mean={verdict.get('best_mapping_pm130_actual_downside_mean')}")
    print(f"overall_actual_downside_mean={verdict.get('overall_actual_downside_mean')}")
    print(f"leakage_risk={report['leakage_guard'].get('leakage_risk')}")
    print(f"phase9f_backtest_worth_testing={verdict.get('phase9f_backtest_worth_testing')}")


if __name__ == "__main__":
    main()

