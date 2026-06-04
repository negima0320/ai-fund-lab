from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.dataset_builder import DatasetBuilder


def _builder(tmp_path: Path) -> DatasetBuilder:
    return DatasetBuilder(
        feature_root=tmp_path / "data" / "ml" / "features",
        label_root=tmp_path / "data" / "ml" / "labels",
        dataset_root=tmp_path / "data" / "ml" / "datasets",
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")


def _feature(date: str, code: str, close: float = 100.0, volume: float = 1000.0) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "close": close,
        "volume": volume,
        "turnover_value": close * volume,
        "return_1d": 0.01,
    }


def _label(date: str, code: str, future_5d: float = 0.05, future_10d: float = 0.10) -> dict:
    return {
        "date": pd.Timestamp(date),
        "code": code,
        "entry_price": 101.0,
        "future_5d_return": future_5d,
        "future_10d_return": future_10d,
        "upside_10d": True,
        "bad_entry_10d": False,
    }


def _install_parquet_reader(monkeypatch, frames: dict[Path, pd.DataFrame]) -> None:
    def fake_read_parquet(path: Path) -> pd.DataFrame:
        return frames[Path(path)].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


def test_build_dataset_inner_joins_features_and_labels_by_date_code(monkeypatch, tmp_path) -> None:
    builder = _builder(tmp_path)
    feature_path = builder.feature_root / "features_2026-05-01.parquet"
    label_path = builder.label_root / "labels_2026-05-01.parquet"
    _touch(feature_path)
    _touch(label_path)
    _install_parquet_reader(
        monkeypatch,
        {
            feature_path: pd.DataFrame([_feature("2026-05-01", "1001"), _feature("2026-05-01", "1002")]),
            label_path: pd.DataFrame([_label("2026-05-01", "1001")]),
        },
    )

    df = builder.build_dataset("2026-05-01", "2026-05-01")

    assert df["code"].tolist() == ["1001"]
    assert df.loc[0, "future_10d_return"] == 0.10


def test_build_dataset_filters_invalid_rows(monkeypatch, tmp_path) -> None:
    builder = _builder(tmp_path)
    feature_path = builder.feature_root / "features_2026-05-01.parquet"
    label_path = builder.label_root / "labels_2026-05-01.parquet"
    _touch(feature_path)
    _touch(label_path)
    _install_parquet_reader(
        monkeypatch,
        {
            feature_path: pd.DataFrame(
                [
                    _feature("2026-05-01", "keep"),
                    _feature("2026-05-01", "zero_volume", volume=0),
                    _feature("2026-05-01", "missing_close", close=float("nan")),
                    _feature("2026-05-01", "extreme"),
                ]
            ),
            label_path: pd.DataFrame(
                [
                    _label("2026-05-01", "keep"),
                    _label("2026-05-01", "zero_volume"),
                    _label("2026-05-01", "missing_close"),
                    _label("2026-05-01", "extreme", future_10d=1.5),
                    _label("2026-05-01", "missing_label", future_10d=float("nan")),
                ]
            ),
        },
    )

    df = builder.build_dataset("2026-05-01", "2026-05-01")

    assert df["code"].tolist() == ["keep"]


def test_split_by_time_uses_ordered_date_boundaries(tmp_path) -> None:
    builder = _builder(tmp_path)
    df = pd.DataFrame(
        [
            {**_feature("2026-05-25", "test"), **_label("2026-05-25", "test")},
            {**_feature("2026-05-01", "train"), **_label("2026-05-01", "train")},
            {**_feature("2026-05-20", "valid"), **_label("2026-05-20", "valid")},
        ]
    )

    train, valid, test = builder.split_by_time(df, "2026-05-15", "2026-05-24")

    assert train["code"].tolist() == ["train"]
    assert valid["code"].tolist() == ["valid"]
    assert test["code"].tolist() == ["test"]


def test_save_dataset_writes_parquet_path(monkeypatch, tmp_path) -> None:
    builder = _builder(tmp_path)
    df = pd.DataFrame([_feature("2026-05-01", "1001")])
    calls = {}

    def fake_to_parquet(_self: pd.DataFrame, path: Path, index: bool = False) -> None:
        calls["path"] = path
        calls["index"] = index
        path.write_text("parquet", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    path = builder.save_dataset(df, "train")

    assert path == tmp_path / "data" / "ml" / "datasets" / "train.parquet"
    assert calls == {"path": path, "index": False}
    assert path.exists()
