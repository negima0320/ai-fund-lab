#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase4g_exit_delay_candidate_hold import (  # noqa: E402
    PortfolioManagerPhase4GExitDelayCandidateHoldAudit,
)


def main() -> int:
    audit = PortfolioManagerPhase4GExitDelayCandidateHoldAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    delay = result.get("exit_delay_1d_audit", {}).get("summary", {})
    judgement = result.get("clean_v280_candidate_judgement", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(
        "summary="
        f"exit_delay_profit_delta={delay.get('profit_delta')} "
        f"best_rule={judgement.get('best_rule')} "
        f"clean_v280_worth_implementing={judgement.get('clean_v280_worth_implementing')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

