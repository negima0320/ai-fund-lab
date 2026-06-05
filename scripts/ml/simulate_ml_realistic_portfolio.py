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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate realistic report-only ML portfolios.")
    parser.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--ranking", default="expected_return_10d", choices=["expected_return_10d", "expected_max_return_20d", "ml_score"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--initial-cash", type=float, default=1_000_000)
    parser.add_argument("--position-size", type=float, default=100_000)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--exit-rule", default="close_20d", choices=["close_10d", "close_20d"])
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.001)
    parser.add_argument("--min-turnover-value", type=float, default=50_000_000)
    parser.add_argument("--grid", action="store_true", help="Run the predefined grid over ranking, max positions, exit rule, and liquidity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = MLRealisticPortfolioSimulator()
    if args.grid:
        result = simulator.simulate_grid(
            args.start,
            args.end,
            top_n=args.top_n,
            initial_cash=args.initial_cash,
            position_size=args.position_size,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )
    else:
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
        result = simulator.simulate_one(args.start, args.end, config)

    md_path = simulator.save_report(result)
    json_path = simulator.save_json(result)
    csv_path = simulator.save_trades_csv(result)
    print(f"saved markdown report to {md_path}")
    print(f"saved json report to {json_path}")
    print(f"saved realistic trades csv to {csv_path}")
    for row in result["summary"]:
        print(
            "config="
            f"{row['config_id']} final_assets={row['final_assets']} "
            f"total_return={row['total_return']} total_profit={row['total_profit']} "
            f"trades={row['total_trades']} win_rate={row['win_rate']} "
            f"profit_factor={row['profit_factor']} max_drawdown={row['max_drawdown']} "
            f"liq_reject={row['rejected_by_liquidity']}"
        )


if __name__ == "__main__":
    main()
