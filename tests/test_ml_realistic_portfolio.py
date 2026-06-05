from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml.data_loader import JQuantsDataLoader
from ml.realistic_portfolio import MLRealisticPortfolioSimulator, RealisticPortfolioConfig
from ml.realistic_portfolio_bottleneck import MLRealisticPortfolioBottleneckAnalyzer


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _prediction(date: str, code: str, score: float) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "expected_return_10d": score,
        "expected_max_return_20d": score,
        "ml_score": score,
    }


def _label(date: str, code: str) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "future_10d_return": 0.10,
        "future_max_return_20d": 0.10,
    }


def _feature(date: str, code: str, turnover_value: float) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "turnover_value": turnover_value,
        "volume": 1000,
        "close": 100.0,
    }


def _price(date: pd.Timestamp, code: str, close: float = 100.0) -> dict:
    return {
        "Date": date.strftime("%Y-%m-%d"),
        "Code": code,
        "O": 100.0,
        "H": max(101.0, close),
        "L": 99.0,
        "C": close,
        "Vo": 1000,
        "Va": 100000000,
    }


def _simulator(tmp_path: Path) -> MLRealisticPortfolioSimulator:
    return MLRealisticPortfolioSimulator(
        predictions_root=tmp_path / "data" / "ml" / "predictions",
        labels_root=tmp_path / "data" / "ml" / "labels",
        features_root=tmp_path / "data" / "ml" / "features",
        report_root=tmp_path / "reports" / "ml",
        data_loader=JQuantsDataLoader(tmp_path / "jquants"),
    )


def _write_daily_inputs(simulator: MLRealisticPortfolioSimulator, date: str, rows: list[tuple[str, float, float]]) -> None:
    _write_parquet(
        simulator.predictions_root / f"predictions_{date}.parquet",
        [_prediction(date, code, score) for code, score, _ in rows],
    )
    _write_parquet(
        simulator.labels_root / f"labels_{date}.parquet",
        [_label(date, code) for code, _, _ in rows],
    )
    _write_parquet(
        simulator.features_root / f"features_{date}.parquet",
        [_feature(date, code, turnover) for code, _, turnover in rows],
    )


def _write_price_cache(tmp_path: Path, codes: list[str]) -> None:
    for offset, date in enumerate(pd.date_range("2026-01-01", periods=20, freq="D")):
        rows = []
        for code in codes:
            close = 110.0 if code == "1001" and offset >= 10 else 100.0
            rows.append(_price(date, code, close=close))
        _write_json(tmp_path / "jquants" / "prices" / f"{date:%Y-%m-%d}.json", rows)


def test_realistic_portfolio_applies_position_duplicate_and_liquidity_constraints(tmp_path) -> None:
    simulator = _simulator(tmp_path)
    _write_daily_inputs(
        simulator,
        "2026-01-01",
        [
            ("1001", 0.9, 100_000_000),
            ("1002", 0.8, 100_000_000),
            ("1003", 0.7, 1_000_000),
        ],
    )
    _write_daily_inputs(simulator, "2026-01-02", [("1001", 0.9, 100_000_000)])
    _write_price_cache(tmp_path, ["1001", "1002", "1003"])
    config = RealisticPortfolioConfig(
        ranking="expected_return_10d",
        top_n=3,
        initial_cash=300_000,
        position_size=100_000,
        max_positions=1,
        exit_rule="close_10d",
        fee_rate=0.0,
        slippage_rate=0.0,
        min_turnover_value=50_000_000,
    )

    result = simulator.simulate_one("2026-01-01", "2026-01-02", config)
    summary = result["summary"][0]

    assert summary["total_trades"] == 1
    assert summary["final_assets"] == pytest.approx(310_000)
    assert summary["total_profit"] == pytest.approx(10_000)
    assert summary["rejected_by_max_positions"] == 1
    assert summary["rejected_by_duplicate"] == 1
    assert summary["rejected_by_liquidity"] == 1
    assert result["trades"][0]["code"] == "1001"


def test_realistic_portfolio_counts_cash_rejections_and_saves_outputs(tmp_path) -> None:
    simulator = _simulator(tmp_path)
    _write_daily_inputs(simulator, "2026-01-01", [("1001", 0.9, 100_000_000)])
    _write_price_cache(tmp_path, ["1001"])
    config = RealisticPortfolioConfig(
        ranking="expected_return_10d",
        top_n=1,
        initial_cash=100_000,
        position_size=100_000,
        max_positions=5,
        exit_rule="close_10d",
        fee_rate=0.001,
        slippage_rate=0.001,
        min_turnover_value=50_000_000,
    )

    result = simulator.simulate_one("2026-01-01", "2026-01-01", config)
    summary = result["summary"][0]
    md_path = simulator.save_report(result)
    json_path = simulator.save_json(result)
    csv_path = simulator.save_trades_csv(result)

    assert summary["total_trades"] == 0
    assert summary["rejected_by_cash"] == 1
    assert md_path.exists()
    assert json_path.exists()
    assert csv_path.exists()
    assert "ML Realistic Portfolio Simulation" in md_path.read_text(encoding="utf-8")


def test_realistic_portfolio_supports_close_5d_exit(tmp_path) -> None:
    simulator = _simulator(tmp_path)
    _write_daily_inputs(simulator, "2026-01-01", [("1001", 0.9, 100_000_000)])
    _write_price_cache(tmp_path, ["1001"])
    config = RealisticPortfolioConfig(
        ranking="expected_return_10d",
        top_n=1,
        initial_cash=300_000,
        position_size=100_000,
        max_positions=5,
        exit_rule="close_5d",
        fee_rate=0.0,
        slippage_rate=0.0,
        min_turnover_value=50_000_000,
    )

    result = simulator.simulate_one("2026-01-01", "2026-01-01", config)

    assert result["summary"][0]["total_trades"] == 1
    assert result["trades"][0]["holding_days"] == 5


def test_bottleneck_analyzer_reports_bought_vs_rejected(tmp_path) -> None:
    simulator = _simulator(tmp_path)
    _write_daily_inputs(
        simulator,
        "2026-01-01",
        [
            ("1001", 0.9, 100_000_000),
            ("1002", 0.8, 1_000_000),
        ],
    )
    _write_price_cache(tmp_path, ["1001", "1002"])
    config = RealisticPortfolioConfig(
        ranking="expected_return_10d",
        top_n=2,
        initial_cash=300_000,
        position_size=100_000,
        max_positions=5,
        exit_rule="close_10d",
        fee_rate=0.0,
        slippage_rate=0.0,
        min_turnover_value=50_000_000,
    )
    analyzer = MLRealisticPortfolioBottleneckAnalyzer(simulator=simulator, report_root=tmp_path / "reports" / "ml")

    result = analyzer.analyze("2026-01-01", "2026-01-01", config)
    md_path = analyzer.save_report(result)
    csv_path = analyzer.save_candidates_csv(result)

    reasons = {(row["status"], row["reason"]) for row in result["candidate_rows"]}
    assert ("bought", "bought") in reasons
    assert ("rejected", "liquidity") in reasons
    assert result["monthly_rejections"][0]["rejected_by_liquidity"] == 1
    assert md_path.exists()
    assert csv_path.exists()
