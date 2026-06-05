#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.realistic_portfolio import MLRealisticPortfolioSimulator, RealisticPortfolioConfig
from ml.realistic_portfolio_bottleneck import MLRealisticPortfolioBottleneckAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bottlenecks in realistic ML portfolios.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--ranking", default="risk_adjusted_return", choices=["risk_adjusted_return", "expected_return_10d"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--position-size", type=float, default=100_000)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--exit-rule", default="close_10d", choices=["close_5d", "close_10d", "close_20d"])
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--min-turnover-value", type=float, default=50_000_000)
    parser.add_argument("--predictions-root", default="data/ml/walk_forward_predictions")
    parser.add_argument("--features-root", default="data/ml/features")
    parser.add_argument("--labels-root", default="data/ml/labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = MLRealisticPortfolioSimulator(
        predictions_root=args.predictions_root,
        features_root=args.features_root,
        labels_root=args.labels_root,
    )
    analyzer = MLRealisticPortfolioBottleneckAnalyzer(simulator=simulator)
    config = RealisticPortfolioConfig(
        ranking=args.ranking,
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        position_size=args.position_size,
        max_positions=args.max_positions,
        exit_rule=args.exit_rule,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        min_turnover_value=args.min_turnover_value,
    )
    result = analyzer.analyze(args.start, args.end, config)
    md_path = analyzer.save_report(result)
    json_path = analyzer.save_json(result)
    csv_path = analyzer.save_candidates_csv(result)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    print(f"saved bought_vs_rejected csv to {csv_path}")
    baseline = result["baseline_summary"]
    print(
        "baseline "
        f"config={baseline.get('config_id')} total_return={baseline.get('total_return')} "
        f"pf={baseline.get('profit_factor')} dd={baseline.get('max_drawdown')} "
        f"trades={baseline.get('total_trades')}"
    )
    best = result.get("best_grid") or {}
    print(
        "best_grid "
        f"config={best.get('config_id')} total_return={best.get('total_return')} "
        f"pf={best.get('profit_factor')} dd={best.get('max_drawdown')} "
        f"trades={best.get('total_trades')}"
    )


if __name__ == "__main__":
    main()
