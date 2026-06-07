#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase3k_candidate_ranking import PortfolioManagerPhase3KCandidateRankingAudit


def main() -> int:
    audit = PortfolioManagerPhase3KCandidateRankingAudit(root=ROOT)
    result = audit.build()
    paths = audit.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"profile={result['profile']}")
    print(f"selected_sort_columns={result['ranking_basis']['selected_sort_columns']}")
    for row in result["candidate_path_classification"]:
        print(f"classification={row['classification']} count={row['count']}")
    for row in result["fallback_decision_flags"]:
        print(f"flag={row['flag']} value={row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
