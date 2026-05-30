from __future__ import annotations

from pathlib import Path

from article import generate_note_article


def test_note_article_contains_required_sections(config_copy: dict, tmp_path: Path) -> None:
    summary = {
        "day_number": 1,
        "date": "2026-03-06",
        "total_assets": 1_000_000,
        "daily_profit": 0,
        "day_change_pct": 0,
        "cumulative_profit": 0,
        "gross_cumulative_profit": 0,
        "estimated_tax_total": 0,
        "net_cumulative_profit": 0,
        "total_commission": 0,
        "win_rate": None,
        "max_drawdown": 0,
    }
    paper_trade_log = {
        "day_number": 1,
        "orders": [],
        "skipped_buys": [],
        "closed_trades": [],
        "positions": [],
        "safety_events": [],
    }
    article = generate_note_article(summary, paper_trade_log, config_copy, repo_root=tmp_path)
    assert "# AIファンド1号 Day 1" in article
    assert "## 今日の開発内容" in article
    assert "## ネギマコメント" in article
    assert "## 免責事項" in article
    assert "本記事は投資助言を目的としたものではありません。" in article
