#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_v3_trainer import PMAIV3TrainerPrototype  # noqa: E402


def main() -> None:
    trainer = PMAIV3TrainerPrototype(ROOT)
    report = trainer.run()
    paths = trainer.save_report(report)
    split_rows = {row["split"]: row["rows"] for row in report["split"]["rows"]}
    print(f"model_dir={paths.model_dir}")
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"feature_count_after_drops={report['feature_plan']['feature_count_after_drops']}")
    print(f"dropped_features={','.join(report['feature_plan']['dropped_features'])}")
    print(f"train_rows={split_rows.get('train')}")
    print(f"validation_rows={split_rows.get('validation')}")
    print(f"test_rows={split_rows.get('test')}")
    print(f"leakage_risk={report['leakage_checklist']['leakage_risk']}")
    print(f"phase9e_worth_testing={report['verdict']['phase9e_integration_audit_worth_testing']}")


if __name__ == "__main__":
    main()

