#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase7c_pm_ai_api_only_dataset_design import Phase7CPMAIAPIOnlyDatasetDesignAudit  # noqa: E402


def main() -> int:
    audit = Phase7CPMAIAPIOnlyDatasetDesignAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    verdict = result.get("final_judgement", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"api_only_pm_dataset_feasible={verdict.get('api_only_pm_dataset_feasible')}")
    print(f"recommended_retraining_plan={verdict.get('recommended_retraining_plan')}")
    print(f"ready_for_phase7d={verdict.get('ready_for_phase7d')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

