from __future__ import annotations

from main import build_order_preview, render_order_preview_markdown


def test_order_preview_builds_manual_approval_daily_paper_report(config_copy: dict) -> None:
    config_copy["broker"]["provider"] = "tachibana_live"
    config_copy["broker"]["live_trading_enabled"] = True
    config_copy["safety"]["allow_live_trading"] = True
    state = {
        "cash": 800_000,
        "total_assets": 1_020_000,
        "max_drawdown": -0.01,
        "daily_profit_rate": 0.0,
        "positions": [
            {
                "code": "1001",
                "name": "Held Winner",
                "shares": 100,
                "entry_price": 1_000,
                "holding_days": 2,
                "sector_name": "機械",
            }
        ],
    }
    scored_candidates = [
        {
            "code": "1001",
            "name": "Held Winner",
            "close": 1_070,
            "selected": False,
            "total_score": 80,
            "confidence": 0.9,
            "sector_name": "機械",
        },
        {
            "code": "2001",
            "name": "Buy Candidate",
            "close": 1_000,
            "selected": True,
            "total_score": 76,
            "confidence": 0.9,
            "selection_reason": "score and volume breakout",
            "sector_name": "情報通信",
            "market_regime": "risk_on",
            "candlestick_signals": ["volume_confirmed_breakout"],
        },
        {
            "code": "3001",
            "name": "Rejected Candidate",
            "close": 900,
            "selected": False,
            "total_score": 61,
            "confidence": 0.8,
            "rejected_reason": "score below threshold",
            "sector_name": "小売業",
            "market_regime": "neutral",
        },
    ]

    preview = build_order_preview(scored_candidates, state, config_copy, "2026-03-06")
    markdown = render_order_preview_markdown(preview)

    assert preview["mode"] == "MANUAL_APPROVAL_PREVIEW"
    assert preview["broker_provider"] == "paper"
    assert preview["broker_candidates"] == ["paper"]
    assert preview["order_submission_enabled"] is False
    assert preview["manual_approval_required"] is True
    assert preview["manual_approval_flow"]["status"] == "PENDING_MANUAL_APPROVAL"
    assert preview["summary"]["live_trading_enabled"] is False
    assert preview["summary"]["configured_broker_provider"] == "tachibana_live"
    assert preview["summary"]["configured_live_trading_enabled"] is True
    assert preview["sell_candidates"][0]["sell_reason"] == "利確"
    assert preview["buy_candidates"][0]["buy_reason"] == "score and volume breakout"
    assert preview["positions"][0]["unrealized_pnl"] == 7000
    assert preview["summary"]["unrealized_pnl"] == 7000
    assert preview["preview_orders"][0]["order_status"] == "PREVIEW"
    assert preview["preview_orders"][0]["approval_status"] == "PENDING_MANUAL_APPROVAL"
    assert all(item["broker_provider"] == "paper" for item in preview["preview_orders"])
    assert all(item["live_trading"] is False for item in preview["preview_orders"])
    assert any(item["reason"] == "score below threshold" for item in preview["skipped"])
    assert preview["risk_check_summary"]["checked_count"] == 2
    assert preview["risk_check_summary"]["rejected_count"] == 0

    assert "# Daily Paper Report" in markdown
    assert "## Manual Approval Flow" in markdown
    assert "## preview_orders" in markdown
    assert "## 今日の買い候補" in markdown
    assert "買い理由=score and volume breakout" in markdown
    assert "## 今日の売り候補" in markdown
    assert "売り理由=利確" in markdown
    assert "## 保有中ポジション" in markdown
    assert "含み損益=+7,000" in markdown
    assert "## 除外理由" in markdown
    assert "score below threshold" in markdown
    assert "## リスクチェック結果" in markdown
    assert "order_submission_enabled: false" in markdown
