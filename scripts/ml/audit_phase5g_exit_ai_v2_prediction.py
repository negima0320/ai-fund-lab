#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase5g_exit_ai_v2_prediction_audit import Phase5GExitAIV2PredictionAudit  # noqa: E402


def main() -> int:
    audit = Phase5GExitAIV2PredictionAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"prediction_coverage={result['prediction_summary']['coverage_rate']}")
    print(f"leakage_risk={result['leakage_integrity_audit']['leakage_risk']}")
    print(f"recommended_next_phase={result['recommended_next_phase']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
