from __future__ import annotations

import os
from pathlib import Path

import yaml
import main as main_module


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


def test_operation_simulation_reads_schedule_and_estimates_api() -> None:
    simulation = main_module.build_operation_simulation("rookie_dealer_02_v2_1", 1)

    assert simulation["dry_run"] is True
    assert simulation["operation_days"]
    assert simulation["api_usage"]["daily_total"] > 0
    assert simulation["orders"]["actual_orders"] == 0
    assert simulation["launchd_validation"]["status"] in {"OK", "WARN", "SKIP"}
    assert "reason" in simulation["launchd_validation"]
    assert "checked" in simulation["launchd_validation"]


def test_operation_simulation_skips_launchd_validation_in_ci(monkeypatch) -> None:
    monkeypatch.setenv("CI", "true")

    simulation = main_module.build_operation_simulation("rookie_dealer_02_v2_1", 1)

    assert simulation["launchd_validation"]["status"] == "SKIP"
    assert simulation["launchd_validation"]["reason"] == "launchd validation skipped in CI"
    assert simulation["launchd_validation"]["checked"] == 0


def test_operation_simulation_does_not_call_external_execution(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(main_module, "run_backtest", lambda *args, **kwargs: called.append("backtest"))
    monkeypatch.setattr(main_module, "run_demo_auto_order", lambda *args, **kwargs: called.append("order"))

    main_module.build_operation_simulation("rookie_dealer_02_v2_1", 7)

    assert called == []


def test_launchd_validation_detects_missing_script(tmp_path, monkeypatch) -> None:
    launchd_dir = tmp_path / "docs" / "launchd"
    launchd_dir.mkdir(parents=True)
    plist = launchd_dir / "bad.plist"
    plist.write_text(
        """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
<key>Label</key><string>bad</string>
<key>WorkingDirectory</key><string>{root}</string>
<key>ProgramArguments</key><array><string>{root}/scripts/missing.sh</string></array>
</dict></plist>
""".format(root=tmp_path),
        encoding="utf-8",
    )
    monkeypatch.setattr(main_module, "ROOT", tmp_path)

    result = main_module.validate_launchd_files()

    assert result["status"] == "ERROR"
    assert "missing script" in result["checks"][0]["message"]
