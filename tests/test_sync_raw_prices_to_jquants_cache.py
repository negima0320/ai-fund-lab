from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ml" / "sync_raw_prices_to_jquants_cache.py"
    spec = importlib.util.spec_from_file_location("sync_raw_prices_to_jquants_cache", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_raw(path: Path, prices: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prices": prices}), encoding="utf-8")


def test_sync_one_converts_raw_prices_to_cache_records(tmp_path) -> None:
    module = _load_module()
    raw_path = tmp_path / "data" / "raw" / "prices_2026-05-20.json"
    cache_path = tmp_path / "data" / "cache" / "jquants" / "prices" / "2026-05-20.json"
    _write_raw(
        raw_path,
        [
            {
                "code": 1001,
                "date": "2026-05-20",
                "open": "100",
                "high": "105",
                "low": "99",
                "close": "104",
                "volume": "1000",
                "turnover_value": "104000",
            }
        ],
    )

    result = module.sync_one(raw_path, cache_path, "2026-05-20")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))

    assert result["written"] is True
    assert result["records"] == 1
    assert payload == {
        "records": [
            {
                "date": "2026-05-20",
                "code": "1001",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 1000.0,
                "turnover_value": 104000.0,
            }
        ]
    }


def test_sync_one_does_not_overwrite_existing_cache_by_default(tmp_path) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw" / "prices_2026-05-20.json"
    cache_path = tmp_path / "cache" / "2026-05-20.json"
    _write_raw(raw_path, [{"code": "1001", "date": "2026-05-20", "close": 100}])
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"code":"old"}]}', encoding="utf-8")

    result = module.sync_one(raw_path, cache_path, "2026-05-20")

    assert result["written"] is False
    assert result["skipped"] is True
    assert json.loads(cache_path.read_text(encoding="utf-8"))["records"][0]["code"] == "old"
    assert "already exists" in result["warnings"][0]


def test_sync_one_overwrites_when_requested(tmp_path) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw" / "prices_2026-05-20.json"
    cache_path = tmp_path / "cache" / "2026-05-20.json"
    _write_raw(raw_path, [{"code": "1001", "date": "2026-05-20", "close": 100}])
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text('{"records":[{"code":"old"}]}', encoding="utf-8")

    result = module.sync_one(raw_path, cache_path, "2026-05-20", overwrite=True)

    assert result["written"] is True
    assert json.loads(cache_path.read_text(encoding="utf-8"))["records"][0]["code"] == "1001"


def test_sync_one_dry_run_does_not_write(tmp_path) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw" / "prices_2026-05-20.json"
    cache_path = tmp_path / "cache" / "2026-05-20.json"
    _write_raw(raw_path, [{"code": "1001", "date": "2026-05-20", "close": 100}])

    result = module.sync_one(raw_path, cache_path, "2026-05-20", dry_run=True)

    assert result["written"] is False
    assert result["skipped"] is False
    assert result["records"] == 1
    assert not cache_path.exists()


def test_sync_one_empty_prices_array_warns_but_does_not_crash(tmp_path) -> None:
    module = _load_module()
    raw_path = tmp_path / "raw" / "prices_2026-05-20.json"
    cache_path = tmp_path / "cache" / "2026-05-20.json"
    _write_raw(raw_path, [])

    result = module.sync_one(raw_path, cache_path, "2026-05-20")

    assert result["written"] is True
    assert result["records"] == 0
    assert "empty or invalid" in result["warnings"][0]
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"records": []}


def test_sync_dates_supports_small_date_range(tmp_path) -> None:
    module = _load_module()
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    _write_raw(raw_root / "prices_2026-05-20.json", [{"code": "1001", "date": "2026-05-20", "close": 100}])
    _write_raw(raw_root / "prices_2026-05-21.json", [{"code": "1001", "date": "2026-05-21", "close": 101}])

    results = module.sync_dates(["2026-05-20", "2026-05-21"], raw_root=raw_root, cache_root=cache_root, dry_run=True)

    assert [item["records"] for item in results] == [1, 1]
    assert not (cache_root / "2026-05-20.json").exists()
