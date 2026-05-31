from __future__ import annotations

import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_operations_files_exist() -> None:
    assert (ROOT / "scripts" / "run_daily_paper.sh").exists()
    assert (ROOT / "scripts" / "run_analyze.sh").exists()
    assert (ROOT / "scripts" / "run_evening_selection.sh").exists()
    assert (ROOT / "scripts" / "run_demo_auto_order.sh").exists()
    assert (ROOT / "docs" / "operations.md").exists()
    assert (ROOT / "docs" / "launchd" / "com.negima.ai-fund-lab.paper-run.plist").exists()
    assert (ROOT / "config" / "operation_schedule.yaml").exists()


def test_operations_readme_section_exists() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## 運用方法" in readme
    assert "[docs/operations.md](docs/operations.md)" in readme
    assert "scripts/run_daily_paper.sh" in readme
    assert "config/operation_schedule.yaml" in readme


def test_operation_scripts_are_executable_and_safe() -> None:
    daily = ROOT / "scripts" / "run_daily_paper.sh"
    analyze = ROOT / "scripts" / "run_analyze.sh"
    evening = ROOT / "scripts" / "run_evening_selection.sh"
    demo = ROOT / "scripts" / "run_demo_auto_order.sh"
    daily_text = daily.read_text(encoding="utf-8")
    analyze_text = analyze.read_text(encoding="utf-8")
    evening_text = evening.read_text(encoding="utf-8")
    demo_text = demo.read_text(encoding="utf-8")

    assert os.access(daily, os.X_OK)
    assert os.access(analyze, os.X_OK)
    assert os.access(evening, os.X_OK)
    assert os.access(demo, os.X_OK)
    assert "set -euo pipefail" in daily_text
    assert ".venv/bin/python" in daily_text
    assert "--mode preflight" in daily_text
    assert "--mode full-paper-run" in daily_text
    assert "--mode analyze" in daily_text
    assert "tachibana_live" not in daily_text
    assert "set -euo pipefail" in analyze_text
    assert "--mode analyze" in analyze_text
    assert "--mode preview-orders" in evening_text
    assert "--mode demo-auto-order" in demo_text
    assert "broker must be tachibana_demo" in demo_text
    assert "live broker is forbidden" in demo_text


def test_operations_doc_includes_safety_and_scheduling_guidance() -> None:
    operations = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
    plist = (ROOT / "docs" / "launchd" / "com.negima.ai-fund-lab.paper-run.plist").read_text(encoding="utf-8")

    for text in [
        "## 基本運用フロー",
        "## cron運用例",
        "## launchd運用例",
        "## ログ",
        "## 安全運用ルール",
        "## よく使うコマンド",
        "paper brokerのみ",
        "tachibana_live",
        "秘密情報",
        "16:30",
        "09:00〜09:30",
    ]:
        assert text in operations

    for text in [
        "WorkingDirectory",
        "ProgramArguments",
        "StandardOutPath",
        "StandardErrorPath",
        "StartCalendarInterval",
        "Weekday",
        "Hour",
        "Minute",
    ]:
        assert text in plist


def test_operation_schedule_safety_defaults() -> None:
    schedule = yaml.safe_load((ROOT / "config" / "operation_schedule.yaml").read_text(encoding="utf-8"))

    assert schedule["market"] == "tokyo"
    assert schedule["timezone"] == "Asia/Tokyo"
    assert schedule["execution_policy"]["auto_order_enabled"] is True
    assert schedule["execution_policy"]["execution_mode"] == "auto_demo"
    assert schedule["execution_policy"]["broker"] == "tachibana_demo"
    assert schedule["execution_policy"]["require_preflight"] is True
    assert schedule["execution_policy"]["require_cash_check"] is True
    assert schedule["execution_policy"]["require_position_check"] is True
    assert schedule["safety"]["require_manual_approval"] is True
    assert schedule["safety"]["forbid_live_auto_order"] is True
