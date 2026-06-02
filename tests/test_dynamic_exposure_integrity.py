from __future__ import annotations

import json

import main as main_module


def test_dynamic_exposure_source_date_before_signal_is_not_future_leak() -> None:
    row = {
        "date": "2026-01-06",
        "dynamic_exposure_source_date": "2026-01-05",
        "dynamic_exposure_same_day_context_used": False,
    }

    assert main_module._score_row_has_future_data(row) is False


def test_dynamic_exposure_same_day_source_is_future_leak() -> None:
    row = {
        "date": "2026-01-06",
        "dynamic_exposure_source_date": "2026-01-06",
        "dynamic_exposure_same_day_context_used": True,
    }

    assert main_module._score_row_has_future_data(row) is True


def test_previous_market_context_read_is_not_stale_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    monkeypatch.setattr(main_module, "BACKTEST_MODE_ACTIVE", True)
    monkeypatch.setattr(main_module, "BACKTEST_JSON_READ_PERIOD", ("2026-01-06", "2026-03-06"))
    monkeypatch.setattr(main_module, "BACKTEST_JSON_READ_AUDIT", {"file_count": 0, "out_of_range_count": 0, "out_of_range_sample": []})
    path = tmp_path / "data" / "processed" / "market_context_2026-01-05.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"advance_ratio": 0.2, "average_change_rate": -0.01}), encoding="utf-8")

    main_module.read_json(path)

    audit = main_module.BACKTEST_JSON_READ_AUDIT
    assert audit["file_count"] == 1
    assert audit["out_of_range_count"] == 0


def test_effective_dynamic_exposure_context_reads_only_latest_previous_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(main_module, "ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for day in ["2026-01-02", "2026-01-05", "2026-01-06"]:
        (processed / f"market_context_{day}.json").write_text(
            json.dumps({"date": day, "advance_ratio": 0.75 if day == "2026-01-05" else 0.2, "average_change_rate": 0.01}),
            encoding="utf-8",
        )
    calls: list[str] = []
    original_read_json = main_module.read_json

    def tracking_read_json(path):
        calls.append(path.name)
        return original_read_json(path)

    monkeypatch.setattr(main_module, "read_json", tracking_read_json)

    resolved = main_module.load_effective_dynamic_exposure_context("2026-01-06", "jquants")

    assert resolved["source_date"] == "2026-01-05"
    assert resolved["regime"] == "strong_bull"
    assert calls == ["market_context_2026-01-05.json"]
