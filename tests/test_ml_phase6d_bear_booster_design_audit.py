from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.phase6d_bear_booster_design_audit import Phase6DBearBoosterDesignAudit


PROFILE = "rookie_dealer_02_v2_78_pm_aware_order_fallback_w025"
PERIOD = "2023-01-01_to_2026-05-31"


def _write_fixture(root: Path) -> None:
    log_dir = root / "logs" / "backtests" / PROFILE / PERIOD
    log_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2023-01-04", periods=100)
    pd.DataFrame(
        [
            {
                "action": "SELL",
                "code": "11110",
                "sector_name": "Tech",
                "entry_date": dates[70].strftime("%Y-%m-%d"),
                "exit_date": dates[73].strftime("%Y-%m-%d"),
                "net_profit": 30_000,
                "net_profit_rate": 0.12,
                "holding_days": 3,
                "entry_price": 200,
                "shares": 100,
                "pm_score": 0.5,
                "pm_multiplier": 1.30,
                "volume_ratio": 2.8,
            },
            {
                "action": "SELL",
                "code": "22220",
                "sector_name": "Retail",
                "entry_date": dates[72].strftime("%Y-%m-%d"),
                "exit_date": dates[76].strftime("%Y-%m-%d"),
                "net_profit": 12_000,
                "net_profit_rate": 0.06,
                "holding_days": 4,
                "entry_price": 100,
                "shares": 100,
                "pm_score": 0.2,
                "pm_multiplier": 1.15,
                "volume_ratio": 2.2,
            },
            {
                "action": "SELL",
                "code": "33330",
                "sector_name": "Retail",
                "entry_date": dates[74].strftime("%Y-%m-%d"),
                "exit_date": dates[79].strftime("%Y-%m-%d"),
                "net_profit": 8_000,
                "net_profit_rate": 0.05,
                "holding_days": 5,
                "entry_price": 100,
                "shares": 100,
                "pm_score": -0.2,
                "pm_multiplier": 0.80,
                "volume_ratio": 2.4,
            },
            {
                "action": "SELL",
                "code": "44440",
                "sector_name": "Bank",
                "entry_date": dates[75].strftime("%Y-%m-%d"),
                "exit_date": dates[77].strftime("%Y-%m-%d"),
                "net_profit": -5_000,
                "net_profit_rate": -0.03,
                "holding_days": 2,
                "entry_price": 100,
                "shares": 100,
                "pm_score": 0.1,
                "pm_multiplier": 1.00,
                "volume_ratio": 1.1,
            },
        ]
    ).to_csv(log_dir / "trades.csv", index=False)
    pd.DataFrame(
        [
            {"code": "11110", "entry_date": dates[70].strftime("%Y-%m-%d"), "final_amount": 20_000},
            {"code": "22220", "entry_date": dates[72].strftime("%Y-%m-%d"), "final_amount": 10_000},
            {"code": "33330", "entry_date": dates[74].strftime("%Y-%m-%d"), "final_amount": 10_000},
            {"code": "44440", "entry_date": dates[75].strftime("%Y-%m-%d"), "final_amount": 10_000},
        ]
    ).to_csv(log_dir / "purchase_audit.csv", index=False)

    topix_dir = root / "data" / "cache" / "jquants" / "topix_prices"
    topix_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, day in enumerate(pd.bdate_range("2022-10-03", periods=180)):
        close = 1000 + idx * 2 if idx < 105 else 1210 - (idx - 105) * 8
        records.append({"date": day.strftime("%Y-%m-%d"), "open": close, "high": close, "low": close, "close": close})
    (topix_dir / "2022-10-03_to_2023-06-09.json").write_text(json.dumps({"records": records}), encoding="utf-8")

    listed_dir = root / "data" / "cache" / "jquants" / "listed_info"
    listed_dir.mkdir(parents=True, exist_ok=True)
    listed = [
        {"Date": "2026-05-29", "Code": "11110", "S33Nm": "Tech", "ScaleCat": "TOPIX Large70", "MktNm": "Prime"},
        {"Date": "2026-05-29", "Code": "22220", "S33Nm": "Retail", "ScaleCat": "TOPIX Small 1", "MktNm": "Prime"},
        {"Date": "2026-05-29", "Code": "33330", "S33Nm": "Retail", "ScaleCat": "TOPIX Small 2", "MktNm": "Prime"},
        {"Date": "2026-05-29", "Code": "44440", "S33Nm": "Bank", "ScaleCat": "TOPIX Mid400", "MktNm": "Prime"},
    ]
    (listed_dir / "2026-05-29.json").write_text(json.dumps({"records": listed}), encoding="utf-8")


def test_phase6d_builds_bear_booster_design_audit(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    result = Phase6DBearBoosterDesignAudit(tmp_path).build_report()

    assert result["metadata"]["audit_only"] is True
    assert result["metadata"]["full_backtest_executed"] is False
    assert result["coverage"]["bear_trades"] >= 4
    assert len(result["booster_rule_design_audit"]) == 7
    rules = {row["rule"]: row for row in result["booster_rule_design_audit"]}
    assert rules["Rule_B_Bear_PM_gte_1_15_buy_amount_plus_50pct"]["profit_delta"] > 0
    assert result["pm_080_protection_audit"]["summary"]["profit"] > 0
    assert result["dd_risk_audit"]
    assert "ready_for_phase6e" in result["verdict"]


def test_phase6d_saves_reports(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    audit = Phase6DBearBoosterDesignAudit(tmp_path)

    paths = audit.save_report(audit.build_report())

    assert paths.markdown.exists()
    assert paths.json.exists()
    assert "Phase 6-D" in paths.markdown.read_text(encoding="utf-8")
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    assert payload["sources"]["approximation_note"].startswith("Profit impact")
