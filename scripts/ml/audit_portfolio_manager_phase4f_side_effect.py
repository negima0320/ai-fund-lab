#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.portfolio_manager_phase4f_side_effect_audit import PortfolioManagerPhase4FSideEffectAudit  # noqa: E402


def main() -> int:
    audit = PortfolioManagerPhase4FSideEffectAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    daily = result.get("daily_path_divergence", {})
    judgement = result.get("side_effect_judgement", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(
        "summary="
        f"first_divergence_date={daily.get('first_divergence_date')} "
        f"type={daily.get('divergence_type')} "
        f"minimum_hold_directly_effective={judgement.get('minimum_hold_directly_effective')} "
        f"v279_safe_to_adopt={judgement.get('v279_safe_to_adopt')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

