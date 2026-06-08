#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.phase8g_pm_ai_ranking_audit import Phase8GPMAIRankingAudit  # noqa: E402


def main() -> int:
    audit = Phase8GPMAIRankingAudit(ROOT)
    report = audit.build_report()
    paths = audit.save_report(report)
    verdict = report["verdict"]
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"pm_problem_type_recommended={report['problem_definition'].get('pm_problem_type_recommended')}")
    print(f"ranking_problem_supported={verdict.get('ranking_problem_supported')}")
    print(f"relative_allocation_worth_testing={verdict.get('relative_allocation_worth_testing')}")
    print(f"best_next_approach={verdict.get('best_next_approach')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
