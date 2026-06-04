from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.config import (
    DATASET_REQUIRED_COLUMNS,
    MAX_ABS_LABEL_RETURN,
    ML_DATASETS_ROOT,
    ML_FEATURES_ROOT,
    ML_LABELS_ROOT,
    REQUIRED_LABEL_COLUMNS,
)


class DatasetBuilder:
    """Build LightGBM-ready datasets from saved feature and label parquet files."""

    def __init__(
        self,
        feature_root: str | Path = ML_FEATURES_ROOT,
        label_root: str | Path = ML_LABELS_ROOT,
        dataset_root: str | Path = ML_DATASETS_ROOT,
        max_abs_label_return: float = MAX_ABS_LABEL_RETURN,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.label_root = Path(label_root)
        self.dataset_root = Path(dataset_root)
        self.max_abs_label_return = max_abs_label_return

    def build_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        frames = []
        for date_text in self._date_texts(start_date, end_date):
            feature_path = self.feature_root / f"features_{date_text}.parquet"
            label_path = self.label_root / f"labels_{date_text}.parquet"
            if not feature_path.exists() or not label_path.exists():
                continue
            features = self._read_frame(feature_path)
            labels = self._read_frame(label_path)
            if features.empty or labels.empty:
                continue
            frames.append(self._join_daily_frames(features, labels))

        if not frames:
            return pd.DataFrame()
        dataset = pd.concat(frames, ignore_index=True)
        dataset = self._filter_dataset(dataset)
        return dataset.sort_values(["date", "code"]).reset_index(drop=True)

    def split_by_time(self, df: pd.DataFrame, train_end: str, valid_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df.copy(), df.copy(), df.copy()
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values(["date", "code"]).reset_index(drop=True)
        train_cutoff = pd.Timestamp(train_end)
        valid_cutoff = pd.Timestamp(valid_end)
        train = data[data["date"] <= train_cutoff].copy()
        valid = data[(data["date"] > train_cutoff) & (data["date"] <= valid_cutoff)].copy()
        test = data[data["date"] > valid_cutoff].copy()
        return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)

    def save_dataset(self, df: pd.DataFrame, name: str) -> Path:
        filename = name if name.endswith(".parquet") else f"{name}.parquet"
        path = self.dataset_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def _join_daily_frames(self, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        left = self._normalize_keys(features)
        right = self._normalize_keys(labels)
        label_columns = ["date", "code", "entry_price", *REQUIRED_LABEL_COLUMNS]
        if any(column not in right.columns for column in label_columns):
            return pd.DataFrame()
        available_label_columns = [column for column in label_columns if column in right.columns]
        return left.merge(right[available_label_columns], on=["date", "code"], how="inner")

    def _filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or any(column not in df.columns for column in DATASET_REQUIRED_COLUMNS):
            return pd.DataFrame(columns=df.columns)
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if "code" in data.columns:
            data["code"] = data["code"].astype("string")
        for column in ["close", "entry_price", "future_5d_return", "future_10d_return", "volume"]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        if "upside_10d" in data.columns:
            data["upside_10d"] = data["upside_10d"].astype("boolean")
        if "bad_entry_10d" in data.columns:
            data["bad_entry_10d"] = data["bad_entry_10d"].astype("boolean")

        required = [column for column in DATASET_REQUIRED_COLUMNS if column in data.columns]
        data = data.dropna(subset=required)
        if "volume" in data.columns:
            data = data[data["volume"] > 0]
        data = data[data["future_5d_return"].abs() <= self.max_abs_label_return]
        data = data[data["future_10d_return"].abs() <= self.max_abs_label_return]
        return data

    def _normalize_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        return data

    def _read_frame(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _date_texts(self, start_date: str, end_date: str) -> list[str]:
        return [day.strftime("%Y-%m-%d") for day in pd.date_range(start=start_date, end=end_date, freq="D")]
