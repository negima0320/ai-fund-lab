#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase5d_exit_ai_v2_training_design import Phase5DExitAIV2TrainingDesignAudit  # noqa: E402


def main() -> int:
    audit = Phase5DExitAIV2TrainingDesignAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"recommended_task={result['recommended_task']}")
    print(f"recommended_feature_set={result['recommended_feature_set']}")
    print(f"leakage_risk={result['leakage_audit']['leakage_risk']}")
    print(f"recommended_next_phase={result['recommended_next_phase']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
