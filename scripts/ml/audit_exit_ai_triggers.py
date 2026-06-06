#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.exit_ai_trigger_audit import ExitAITriggerAudit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Exit AI triggered trades from existing backtest logs.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--period-key", default="2023-01-01_to_2026-05-31")
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--exit-dataset", default=None)
    args = parser.parse_args()

    audit = ExitAITriggerAudit(
        root=args.root,
        period_key=args.period_key,
        comparison_json=args.comparison_json,
        exit_dataset_path=args.exit_dataset,
    )
    result = audit.build()
    paths = audit.save(result)
    summary = result["v2_68_trigger_summary"]
    improvement = result["v2_68_improvement_summary"]
    march = result["march_2026_analysis"]["summary"]
    dd = result["drawdown_analysis"]["summary"]

    print(f"markdown={paths.markdown}")
    print(f"json={paths.json}")
    print(f"trigger_trades_csv={paths.trigger_trades_csv}")
    print(f"trade_delta_csv={paths.trade_delta_csv}")
    print(f"v2_68_trigger_count={summary['trigger_count']}")
    print(f"v2_68_improvement_total={improvement['improvement_total']:.2f}")
    print(f"v2_68_worsening_total={improvement['worsening_total']:.2f}")
    print(f"v2_68_net_effect={improvement['net_effect']:.2f}")
    print(f"march_2026_delta={march['profit_delta']:.2f}")
    print(f"march_2026_triggered_delta={march['triggered_delta']:.2f}")
    print(f"drawdown_improvement={dd['drawdown_improvement']}")


if __name__ == "__main__":
    main()
