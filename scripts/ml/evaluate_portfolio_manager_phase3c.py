#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ml.portfolio_manager_phase3c import PortfolioManagerPhase3CLightBacktest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Portfolio Manager AI Phase 3-C lightweight sizing rules.")
    parser.add_argument("--dataset", default="data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet")
    parser.add_argument("--model-dir", default="models/ml/portfolio_manager/current_v2_73_phase3b_clean")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    args = parser.parse_args()

    evaluator = PortfolioManagerPhase3CLightBacktest(
        root=ROOT,
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        start_date=args.start,
        end_date=args.end,
    )
    result = evaluator.build()
    paths = evaluator.save(result)
    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"trades_csv={paths.trades_csv}")
    for row in result["summary"]:
        print(
            f"{row['rule']}: net_profit={row.get('net_profit')} "
            f"pf={row.get('profit_factor')} dd={row.get('max_drawdown')} "
            f"delta={row.get('profit_delta_vs_baseline')} "
            f"monthly_win_rate={row.get('monthly_win_rate')} "
            f"67400={row.get('focus_67400_contribution')}"
        )
    for item in result["recommendation"]:
        print(f"recommendation={item}")


if __name__ == "__main__":
    main()
