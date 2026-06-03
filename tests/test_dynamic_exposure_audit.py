from __future__ import annotations

import csv
import json
from pathlib import Path

from feature_analysis import _affordable_fallback_buy_audit, _affordable_fallback_buy_audit_lines, _dynamic_exposure_audit, _dynamic_exposure_audit_lines


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_dynamic_exposure_audit_outputs_json_and_markdown(tmp_path: Path) -> None:
    profile_id = "test_profile"
    start_date = "2026-01-05"
    end_date = "2026-01-06"
    log_dir = tmp_path / "logs" / "backtests" / profile_id / f"{start_date}_to_{end_date}"
    _write_csv(
        log_dir / "summary.csv",
        [
            {
                "date": "2026-01-05",
                "cash": 600000,
                "positions_value": 400000,
                "total_assets": 1000000,
                "open_positions_count": 1,
            },
            {
                "date": "2026-01-06",
                "cash": 950000,
                "positions_value": 50000,
                "total_assets": 1000000,
                "open_positions_count": 1,
            },
        ],
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "market_context_2026-01-03.json").write_text(
        json.dumps({"advance_ratio": 0.75, "average_change_rate": 0.01, "market_regime": "risk_on"}),
        encoding="utf-8",
    )
    (processed / "market_context_2026-01-05.json").write_text(
        json.dumps({"advance_ratio": 0.20, "average_change_rate": -0.01, "market_regime": "risk_off"}),
        encoding="utf-8",
    )
    (processed / "market_context_2026-01-06.json").write_text(
        json.dumps({"advance_ratio": 0.80, "average_change_rate": 0.02, "market_regime": "risk_on"}),
        encoding="utf-8",
    )
    config = {
        "capital_utilization_policy": {"target_exposure": 0.9},
        "dynamic_exposure": {
            "enabled": True,
            "target_exposure_by_regime": {
                "strong_bull": 0.95,
                "bull": 0.90,
                "range": 0.90,
                "bear": 0.90,
                "strong_bear": 0.0,
            },
        },
    }
    backtest_summary = {
        "all_trades": [
            {"action": "BUY", "signal_date": "2026-01-05", "entry_date": "2026-01-06", "amount": 100000},
            {"action": "SKIP_BUY", "signal_date": "2026-01-06", "entry_date": "2026-01-07", "skipped_reason": "selected_but_not_affordable"},
        ]
    }

    audit = _dynamic_exposure_audit(tmp_path, profile_id, start_date, end_date, backtest_summary, config)
    lines = _dynamic_exposure_audit_lines(audit)

    assert audit["enabled"] is True
    assert audit["target_exposure_by_regime"]["range"] == 0.90
    assert audit["target_exposure_by_regime"]["bear"] == 0.90
    assert audit["target_exposure_by_regime"]["strong_bear"] == 0.0
    assert audit["regime_day_count"]["strong_bull"] == 1
    assert audit["regime_day_count"]["strong_bear"] == 1
    assert audit["selected_but_not_affordable_by_regime"]["strong_bear"] == 1
    assert audit["same_day_market_context_used_count"] == 0
    assert audit["future_data_leak_guard_status"] == "OK"
    assert audit["regime_source_date_sample"][0]["source_date"] == "2026-01-03"
    assert audit["regime_source_date_sample"][-1]["source_date"] == "2026-01-05"
    assert audit["dynamic_exposure_trigger_count"] == 2
    assert any("target_exposure_by_regime" in line for line in lines)


def test_affordable_fallback_buy_audit_outputs_counts_and_samples() -> None:
    audit = _affordable_fallback_buy_audit(
        {
            "all_trades": [
                {
                    "action": "SKIP_BUY",
                    "signal_date": "2026-01-05",
                    "code": "1001",
                    "skipped_reason": "selected_but_not_affordable",
                    "affordable_fallback_attempted": True,
                    "affordable_fallback_replaced_by_code": "1002",
                },
                {
                    "action": "BUY",
                    "signal_date": "2026-01-05",
                    "code": "1002",
                    "name": "Fallback",
                    "market_section": "TSEStandard",
                    "amount": 300000,
                    "score": 50,
                    "allocation_limit": 500000,
                    "affordable_fallback_buy_selected": True,
                    "affordable_fallback_original_code": "1001",
                    "affordable_fallback_round_lot_amount": 300000,
                    "affordable_fallback_reason": "selected_but_not_affordable",
                },
            ]
        },
        {"selection": {"min_score": 45}, "affordable_fallback_buy": {"enabled": True}},
    )
    lines = _affordable_fallback_buy_audit_lines(audit)

    assert audit["enabled"] is True
    assert audit["affordable_fallback_candidate_count"] == 1
    assert audit["fallback_attempt_count"] == 1
    assert audit["fallback_buy_trade_count"] == 1
    assert audit["fallback_selected_by_market"]["Standard"] == 1
    assert audit["fallback_rejected_reason_counts"]["no_affordable_candidate"] == 0
    assert audit["selected_but_not_affordable_count"] == 1
    assert audit["selected_but_not_affordable_replaced_count"] == 1
    assert audit["selected_but_not_affordable_after_fallback_count"] == 0
    assert audit["fallback_average_total_score"] == 50
    assert audit["fallback_total_buy_amount"] == 300000
    assert audit["fallback_samples"][0]["fallback_code"] == "1002"
    assert any("fallback_buy_trade_count" in line for line in lines)


def test_affordable_fallback_buy_audit_counts_surplus_fallback_buy() -> None:
    audit = _affordable_fallback_buy_audit(
        {
            "all_trades": [
                {
                    "action": "BUY",
                    "signal_date": "2026-01-05",
                    "code": "2001",
                    "name": "Surplus",
                    "market_section": "TSEStandard",
                    "amount": 200000,
                    "score": 50,
                    "rank": 2,
                    "allocation_limit": 300000,
                    "affordable_fallback_buy_selected": True,
                    "affordable_fallback_original_code": "",
                    "affordable_fallback_round_lot_amount": 200000,
                    "affordable_fallback_reason": "surplus_available_cash",
                    "affordable_fallback_candidate_count": 1,
                },
            ]
        },
        {
            "selection": {"min_score": 45},
            "affordable_fallback_buy": {"enabled": True, "surplus_after_selection": True},
        },
    )

    assert audit["candidate_count"] == 1
    assert audit["selected_count"] == 1
    assert audit["fallback_buy_trade_count"] == 1
    assert audit["selected_by_market"]["Standard"] == 1
    assert audit["fallback_label_only_count"] == 0
    assert audit["fallback_samples"][0]["fallback_code"] == "2001"


def test_affordable_fallback_buy_audit_rejections_exclude_selected_successes() -> None:
    audit = _affordable_fallback_buy_audit(
        {
            "all_trades": [
                {
                    "action": "BUY",
                    "signal_date": "2026-01-05",
                    "code": "2001",
                    "market_section": "Prime",
                    "amount": 200000,
                    "score": 50,
                    "affordable_fallback_buy_selected": True,
                    "affordable_fallback_round_lot_amount": 200000,
                    "affordable_fallback_reason": "surplus_available_cash",
                    "affordable_fallback_candidate_count": 1,
                },
                {
                    "action": "BUY",
                    "signal_date": "2026-01-06",
                    "code": "2002",
                    "market_section": "Prime",
                    "amount": 180000,
                    "score": 48,
                    "affordable_fallback_buy_selected": True,
                    "affordable_fallback_round_lot_amount": 180000,
                    "affordable_fallback_reason": "surplus_available_cash",
                    "affordable_fallback_candidate_count": 1,
                },
                {
                    "action": "SKIP_BUY",
                    "signal_date": "2026-01-07",
                    "code": "1001",
                    "affordable_fallback_attempted": True,
                    "affordable_fallback_no_candidate": True,
                },
            ]
        },
        {
            "selection": {"min_score": 45},
            "affordable_fallback_buy": {"enabled": True, "surplus_after_selection": True},
        },
    )

    assert audit["fallback_selected_count"] == 2
    assert audit["rejected_total_count"] == 1
    assert audit["fallback_attempt_count"] == 3
    assert audit["fallback_attempt_count"] == audit["fallback_selected_count"] + audit["rejected_total_count"]
    assert audit["fallback_rejected_reason_counts"]["no_affordable_candidate"] == 1


def test_affordable_fallback_buy_audit_does_not_count_label_only_buy() -> None:
    audit = _affordable_fallback_buy_audit(
        {
            "all_trades": [
                {
                    "action": "BUY",
                    "signal_date": "2026-01-05",
                    "entry_date": "2026-01-06",
                    "code": "1002",
                    "name": "Normally Bought",
                    "amount": 300000,
                    "affordable_fallback_buy_selected": True,
                    "affordable_fallback_original_code": "1001",
                    "affordable_fallback_reason": "selected_but_not_affordable",
                }
            ]
        },
        {"selection": {"min_score": 45}, "affordable_fallback_buy": {"enabled": True}},
    )

    assert audit["fallback_buy_trade_count"] == 0
    assert audit["fallback_selected_count"] == 0
    assert audit["fallback_label_only_count"] == 1


def test_affordable_fallback_buy_audit_outputs_quality_rejection_counts() -> None:
    audit = _affordable_fallback_buy_audit(
        {
            "all_trades": [
                {
                    "action": "SKIP_BUY",
                    "signal_date": "2026-01-05",
                    "code": "1001",
                    "affordable_fallback_attempted": True,
                    "affordable_fallback_no_candidate": True,
                    "fallback_score_below_min_count": 2,
                    "fallback_rank_out_of_range_count": 1,
                }
            ]
        },
        {"selection": {"min_score": 45}, "affordable_fallback_buy": {"enabled": True, "min_total_score": 55}},
    )
    lines = _affordable_fallback_buy_audit_lines(audit)

    assert audit["fallback_score_below_min_count"] == 2
    assert audit["fallback_rank_out_of_range_count"] == 1
    assert any("fallback_score_below_min_count" in line for line in lines)
    assert any("fallback_rank_out_of_range_count" in line for line in lines)
