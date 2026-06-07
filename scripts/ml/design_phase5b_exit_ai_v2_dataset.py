#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.phase5b_exit_ai_v2_dataset_design import Phase5BExitAIV2DatasetDesignAudit  # noqa: E402


def main() -> int:
    audit = Phase5BExitAIV2DatasetDesignAudit(ROOT)
    result = audit.build_report()
    paths = audit.save_report(result)
    feasibility = result.get("sample_generation_feasibility", {})
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"recommended_next_phase={result.get('recommended_next_phase')}")
    print(f"candidate_rows={feasibility.get('candidate_rows_before_label_horizon_filtering', 0)}")
    print(f"rows_after_20d_label_available={feasibility.get('rows_after_20d_label_available', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
