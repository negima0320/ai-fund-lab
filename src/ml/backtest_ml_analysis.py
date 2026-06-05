from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.config import ML_PREDICTIONS_ROOT, ML_REPORTS_ROOT, ROOT


ML_PREDICTION_COLUMNS = [
    "expected_return_5d",
    "expected_return_10d",
    "upside_probability_10d",
    "bad_entry_probability_10d",
    "expected_max_return_10d",
    "expected_max_return_20d",
    "swing_success_probability_20d",
    "entry_risk_label",
    "ml_score",
]


class BacktestMLAnalyzer:
    """Join existing backtest trades with ML predictions for reporting only."""

    def __init__(
        self,
        root: str | Path = ROOT,
        predictions_root: str | Path = ML_PREDICTIONS_ROOT,
        report_root: str | Path = ML_REPORTS_ROOT,
    ) -> None:
        self.root = Path(root)
        self.predictions_root = Path(predictions_root)
        self.report_root = Path(report_root)

    def analyze_profile(self, profile: str, start_date: str, end_date: str, top_n: int = 10) -> dict[str, Any]:
        trades_path = self.find_trades_csv(profile, start_date, end_date)
        return self.analyze_trades_csv(trades_path, start_date, end_date, profile=profile, top_n=top_n)

    def analyze_trades_csv(
        self,
        trades_path: str | Path,
        start_date: str,
        end_date: str,
        profile: str | None = None,
        top_n: int = 10,
    ) -> dict[str, Any]:
        trades_path = Path(trades_path)
        trades = self.load_trades(trades_path)
        trades = self.filter_closed_trades(trades, start_date, end_date)
        predictions, warnings = self.load_predictions(start_date, end_date)
        joined = self.join_predictions(trades, predictions)
        analysis = self.evaluate_joined(joined, top_n=top_n)
        analysis["win_loss_analysis"] = self.evaluate_win_loss(joined, top_n=top_n)
        analysis["ml_trade_details_csv"] = self._trade_csv_rows(joined)
        analysis.update(
            {
                "profile": profile,
                "period": {"start_date": start_date, "end_date": end_date},
                "source": {
                    "trades_csv": self._relative_path(trades_path),
                    "join_key": "signal_date + code, fallback entry_date + code",
                    "note": "report-only ML join; trading logic is unchanged",
                },
                "warnings": warnings,
            }
        )
        return analysis

    def evaluate_win_loss(self, joined: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
        data = self._normalize_metrics(joined)
        matched = data[data["ml_prediction_joined"]].copy() if "ml_prediction_joined" in data.columns else data.head(0)
        if matched.empty:
            return {
                "ml_average_by_result": [],
                "top_profit_trades": [],
                "bottom_profit_trades": [],
                "risk_label_win_loss_cross": [],
                "danger_win_loss_difference": [],
                "watch_distribution": [],
                "watch_trade_details": [],
            }

        matched["win_loss"] = matched["is_win"].map({True: "win", False: "loss"}).fillna("unknown")
        return {
            "ml_average_by_result": self._ml_average_by_result(matched),
            "top_profit_trades": self._ranked_trade_details(matched, top_n=top_n, ascending=False),
            "bottom_profit_trades": self._ranked_trade_details(matched, top_n=top_n, ascending=True),
            "risk_label_win_loss_cross": self._multi_group_performance(matched, ["entry_risk_label", "win_loss"]),
            "danger_win_loss_difference": self._danger_win_loss_difference(matched),
            "watch_distribution": self._watch_distribution(matched),
            "watch_trade_details": self._watch_trade_details(matched),
        }

    def find_trades_csv(self, profile: str, start_date: str, end_date: str) -> Path:
        profile_root = self.root / "logs" / "backtests" / profile
        exact = profile_root / f"{start_date}_to_{end_date}" / "trades.csv"
        if exact.exists():
            return exact
        if not profile_root.exists():
            raise FileNotFoundError(f"backtest profile directory not found: {profile_root}")

        candidates = []
        target_start = pd.Timestamp(start_date)
        target_end = pd.Timestamp(end_date)
        for path in profile_root.glob("*_to_*/trades.csv"):
            period = path.parent.name.split("_to_")
            if len(period) != 2:
                continue
            try:
                period_start = pd.Timestamp(period[0])
                period_end = pd.Timestamp(period[1])
            except ValueError:
                continue
            if period_start <= target_start and target_end <= period_end:
                candidates.append((period_end - period_start, path))
        if candidates:
            return sorted(candidates, key=lambda item: item[0])[0][1]
        raise FileNotFoundError(f"trades.csv not found for {profile} covering {start_date} to {end_date}")

    def load_trades(self, path: str | Path) -> pd.DataFrame:
        rows = self._read_csv_rows(Path(path))
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date", "code", *ML_PREDICTION_COLUMNS])
        df["date"] = self._join_date(df)
        df["code"] = df["code"].astype("string")
        return df

    def filter_closed_trades(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        mask = (data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))
        if "action" in data.columns:
            mask &= data["action"].astype(str).str.upper().eq("SELL")
        return data[mask].reset_index(drop=True)

    def load_predictions(self, start_date: str, end_date: str) -> tuple[pd.DataFrame, list[str]]:
        frames = []
        warnings = []
        for day in pd.date_range(start=start_date, end=end_date, freq="D"):
            date_text = day.strftime("%Y-%m-%d")
            path = self.predictions_root / f"predictions_{date_text}.parquet"
            if not path.exists():
                warnings.append(f"prediction missing for {date_text}: {self._relative_path(path)}")
                continue
            frame = pd.read_parquet(path)
            if frame.empty:
                warnings.append(f"prediction empty for {date_text}: {self._relative_path(path)}")
                continue
            frames.append(self._normalize_prediction_frame(frame))
        if not frames:
            return pd.DataFrame(columns=["date", "code", *ML_PREDICTION_COLUMNS]), warnings
        return pd.concat(frames, ignore_index=True), warnings

    def join_predictions(self, trades: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        left = trades.copy()
        if left.empty:
            for column in ML_PREDICTION_COLUMNS:
                left[column] = pd.NA
            left["ml_prediction_joined"] = False
            return left
        left["date"] = pd.to_datetime(left["date"], errors="coerce")
        left["code"] = left["code"].astype("string")
        right = self._normalize_prediction_frame(predictions)
        joined = left.merge(right[["date", "code", *ML_PREDICTION_COLUMNS]], on=["date", "code"], how="left")
        joined["ml_prediction_joined"] = joined["ml_score"].notna()
        return joined

    def evaluate_joined(self, joined: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
        data = self._normalize_metrics(joined)
        matched = data[data["ml_prediction_joined"]].copy() if "ml_prediction_joined" in data.columns else data.head(0)
        return {
            "join_summary": {
                "trade_rows": int(len(data)),
                "joined_count": int(len(matched)),
                "missing_count": int(len(data) - len(matched)),
                "join_rate": self._ratio(len(matched), len(data)),
            },
            "overall_performance": self._performance(data),
            "joined_performance": self._performance(matched),
            "risk_label_performance": self._group_performance(matched, "entry_risk_label"),
            "bad_entry_probability_bands": self._bad_probability_band_performance(matched),
            "ml_score_top_bottom": self._score_top_bottom(matched, top_n=top_n),
            "risk_label_ml_score_bands": self._multi_group_performance(matched, ["entry_risk_label", "ml_score_band"]),
            "bad_probability_expected_return_matrix": self._multi_group_performance(
                matched,
                ["bad_entry_probability_band", "expected_return_10d_band"],
            ),
            "danger_expected_return_comparison": self._danger_group_performance(matched, "expected_return_10d_band"),
            "danger_upside_probability_comparison": self._danger_group_performance(matched, "upside_probability_10d_band"),
            "virtual_filter_simulation": self._virtual_filter_simulation(matched),
            "virtual_position_sizing_simulation": self._virtual_position_sizing_simulation(matched),
            "trade_details": self._trade_details(matched),
        }

    def save_report(self, analysis: dict[str, Any]) -> Path:
        profile = analysis.get("profile") or "direct_trades"
        period = analysis["period"]
        path = self.report_root / f"backtest_ml_join_{profile}_{period['start_date']}_to_{period['end_date']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_markdown(analysis), encoding="utf-8")
        return path

    def save_json(self, analysis: dict[str, Any]) -> Path:
        profile = analysis.get("profile") or "direct_trades"
        period = analysis["period"]
        path = self.report_root / f"backtest_ml_join_{profile}_{period['start_date']}_to_{period['end_date']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_win_loss_report(self, analysis: dict[str, Any]) -> Path:
        profile = analysis.get("profile") or "direct_trades"
        period = analysis["period"]
        path = self.report_root / (
            f"backtest_ml_win_loss_analysis_{profile}_{period['start_date']}_to_{period['end_date']}.md"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.format_win_loss_markdown(analysis), encoding="utf-8")
        return path

    def save_win_loss_json(self, analysis: dict[str, Any]) -> Path:
        profile = analysis.get("profile") or "direct_trades"
        period = analysis["period"]
        path = self.report_root / (
            f"backtest_ml_win_loss_analysis_{profile}_{period['start_date']}_to_{period['end_date']}.json"
        )
        payload = {
            "profile": analysis.get("profile"),
            "period": analysis["period"],
            "source": analysis["source"],
            "join_summary": analysis["join_summary"],
            "win_loss_analysis": analysis.get("win_loss_analysis", {}),
            "warnings": analysis.get("warnings", []),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_ml_trades_csv(self, analysis: dict[str, Any]) -> Path:
        profile = analysis.get("profile") or "direct_trades"
        period = analysis["period"]
        path = self.report_root / f"backtest_ml_trades_{profile}_{period['start_date']}_to_{period['end_date']}.csv"
        rows = analysis.get("ml_trade_details_csv", [])
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def format_markdown(self, analysis: dict[str, Any]) -> str:
        source = analysis["source"]
        period = analysis["period"]
        join = analysis["join_summary"]
        lines = [
            "# Backtest ML Join Analysis",
            "",
            f"- profile: {analysis.get('profile') or ''}",
            f"- period: {period['start_date']} to {period['end_date']}",
            f"- trades_csv: {source['trades_csv']}",
            f"- join_key: {source['join_key']}",
            f"- note: {source['note']}",
            "",
            "## Join Summary",
            "",
            f"- trade_rows: {join['trade_rows']}",
            f"- joined_count: {join['joined_count']}",
            f"- missing_count: {join['missing_count']}",
            f"- join_rate: {self._fmt(join['join_rate'])}",
            "",
            "## Existing Trade Performance By ML Risk Label",
            "",
            self._table(analysis["risk_label_performance"], ["entry_risk_label", *self._performance_columns()]),
            "",
            "## Existing Trade Performance By Bad Entry Probability",
            "",
            self._table(analysis["bad_entry_probability_bands"], ["band", *self._performance_columns()]),
            "",
            "## ML Score Top / Bottom",
            "",
            self._table(analysis["ml_score_top_bottom"], ["bucket", *self._performance_columns(), "ml_score_mean"]),
            "",
            "## Risk Label x ML Score Band",
            "",
            self._table(analysis.get("risk_label_ml_score_bands", []), ["entry_risk_label", "ml_score_band", *self._performance_columns(), "ml_score_mean"]),
            "",
            "## Bad Entry Probability x Expected Return",
            "",
            self._table(
                analysis.get("bad_probability_expected_return_matrix", []),
                ["bad_entry_probability_band", "expected_return_10d_band", *self._performance_columns(), "expected_return_10d_mean", "bad_entry_probability_10d_mean"],
            ),
            "",
            "## Danger By Expected Return",
            "",
            self._table(analysis.get("danger_expected_return_comparison", []), ["expected_return_10d_band", *self._performance_columns(), "expected_return_10d_mean", "bad_entry_probability_10d_mean", "ml_score_mean"]),
            "",
            "## Danger By Upside Probability",
            "",
            self._table(analysis.get("danger_upside_probability_comparison", []), ["upside_probability_10d_band", *self._performance_columns(), "upside_probability_10d_mean", "bad_entry_probability_10d_mean", "ml_score_mean"]),
            "",
            "## Virtual ML Filter Simulation",
            "",
            self._table(
                analysis.get("virtual_filter_simulation", []),
                [
                    "filter_id",
                    "condition",
                    "original_trade_count",
                    "kept_trade_count",
                    "removed_trade_count",
                    "original_net_profit_total",
                    "kept_net_profit_total",
                    "removed_net_profit_total",
                    "original_win_rate",
                    "kept_win_rate",
                    "removed_win_rate",
                    "profit_delta",
                ],
            ),
            "",
            "## Virtual ML Position Sizing Simulation",
            "",
            self._table(
                analysis.get("virtual_position_sizing_simulation", []),
                [
                    "sizing_id",
                    "rule",
                    "original_net_profit_total",
                    "adjusted_net_profit_total",
                    "profit_delta",
                    "original_gross_profit_total",
                    "adjusted_gross_profit_total",
                    "average_position_multiplier",
                    "max_position_multiplier",
                    "min_position_multiplier",
                    "weighted_win_rate",
                    "trade_count",
                ],
            ),
            "",
            "## Trade Details With ML Predictions",
            "",
            self._table(
                analysis.get("trade_details", []),
                [
                    "signal_date",
                    "code",
                    "net_profit",
                    "net_profit_rate",
                    "expected_return_10d",
                    "expected_max_return_10d",
                    "expected_max_return_20d",
                    "upside_probability_10d",
                    "bad_entry_probability_10d",
                    "swing_success_probability_20d",
                    "entry_risk_label",
                    "ml_score",
                ],
            ),
            "",
            "## Warnings",
            "",
            "\n".join(f"- {warning}" for warning in analysis.get("warnings", [])) if analysis.get("warnings") else "_None._",
            "",
        ]
        return "\n".join(lines)

    def format_win_loss_markdown(self, analysis: dict[str, Any]) -> str:
        source = analysis["source"]
        period = analysis["period"]
        join = analysis["join_summary"]
        win_loss = analysis.get("win_loss_analysis", {})
        lines = [
            "# Backtest ML Win/Loss Analysis",
            "",
            f"- profile: {analysis.get('profile') or ''}",
            f"- period: {period['start_date']} to {period['end_date']}",
            f"- trades_csv: {source['trades_csv']}",
            f"- note: {source['note']}",
            f"- joined_count: {join['joined_count']}",
            f"- missing_count: {join['missing_count']}",
            "",
            "## Win vs Loss ML Averages",
            "",
            self._table(
                win_loss.get("ml_average_by_result", []),
                [
                    "win_loss",
                    "count",
                    "expected_return_5d_mean",
                    "expected_return_10d_mean",
                    "expected_max_return_10d_mean",
                    "expected_max_return_20d_mean",
                    "upside_probability_10d_mean",
                    "bad_entry_probability_10d_mean",
                    "swing_success_probability_20d_mean",
                    "ml_score_mean",
                    "net_profit_total",
                    "net_profit_rate_mean",
                ],
            ),
            "",
            "## Top Profit Trades",
            "",
            self._table(win_loss.get("top_profit_trades", []), self._win_loss_detail_columns()),
            "",
            "## Bottom Profit Trades",
            "",
            self._table(win_loss.get("bottom_profit_trades", []), self._win_loss_detail_columns()),
            "",
            "## Entry Risk Label x Win/Loss",
            "",
            self._table(
                win_loss.get("risk_label_win_loss_cross", []),
                ["entry_risk_label", "win_loss", *self._performance_columns(), "ml_score_mean", "expected_return_10d_mean", "upside_probability_10d_mean", "bad_entry_probability_10d_mean"],
            ),
            "",
            "## Danger Win/Loss Difference",
            "",
            self._table(
                win_loss.get("danger_win_loss_difference", []),
                [
                    "bucket",
                    "count",
                    "expected_return_10d_mean",
                    "expected_max_return_10d_mean",
                    "expected_max_return_20d_mean",
                    "upside_probability_10d_mean",
                    "bad_entry_probability_10d_mean",
                    "swing_success_probability_20d_mean",
                    "ml_score_mean",
                    "net_profit_rate_mean",
                ],
            ),
            "",
            "## Watch Distribution",
            "",
            self._table(
                win_loss.get("watch_distribution", []),
                [
                    "bucket",
                    "count",
                    "expected_return_10d_mean",
                    "expected_return_10d_min",
                    "expected_return_10d_max",
                    "upside_probability_10d_mean",
                    "upside_probability_10d_min",
                    "upside_probability_10d_max",
                    "bad_entry_probability_10d_mean",
                    "bad_entry_probability_10d_min",
                    "bad_entry_probability_10d_max",
                    "ml_score_mean",
                    "ml_score_min",
                    "ml_score_max",
                    "net_profit_total",
                    "win_rate",
                ],
            ),
            "",
            "## Watch Trade Details",
            "",
            self._table(win_loss.get("watch_trade_details", []), self._win_loss_detail_columns()),
            "",
            "## Warnings",
            "",
            "\n".join(f"- {warning}" for warning in analysis.get("warnings", [])) if analysis.get("warnings") else "_None._",
            "",
        ]
        return "\n".join(lines)

    def _normalize_prediction_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if data.empty:
            return pd.DataFrame(columns=["date", "code", *ML_PREDICTION_COLUMNS])
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["code"] = data["code"].astype("string")
        for column in ML_PREDICTION_COLUMNS:
            if column not in data.columns:
                data[column] = pd.NA
        return data[["date", "code", *ML_PREDICTION_COLUMNS]]

    def _join_date(self, df: pd.DataFrame) -> pd.Series:
        if "date" in df.columns and df["date"].astype(str).str.len().gt(0).any():
            return pd.to_datetime(df["date"], errors="coerce")
        if "signal_date" in df.columns and df["signal_date"].astype(str).str.len().gt(0).any():
            return pd.to_datetime(df["signal_date"], errors="coerce")
        if "entry_date" in df.columns:
            return pd.to_datetime(df["entry_date"], errors="coerce")
        return pd.Series(pd.NaT, index=df.index)

    def _normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for column in [
            "gross_profit",
            "gross_profit_rate",
            "net_profit",
            "net_profit_rate",
            "ml_score",
            "expected_return_5d",
            "expected_return_10d",
            "expected_max_return_10d",
            "expected_max_return_20d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "swing_success_probability_20d",
        ]:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        if "gross_profit" in data.columns:
            data["is_win"] = data["gross_profit"] > 0
        else:
            data["is_win"] = pd.NA
        data["ml_score_band"] = self._band_column(data, "ml_score", [("< 0", None, 0), ("0 to 5", 0, 5), ("5 to 10", 5, 10), (">= 10", 10, None)])
        data["expected_return_10d_band"] = self._band_column(
            data,
            "expected_return_10d",
            [("< 0", None, 0), ("0 to 0.03", 0, 0.03), ("0.03 to 0.10", 0.03, 0.10), (">= 0.10", 0.10, None)],
        )
        data["bad_entry_probability_band"] = self._band_column(
            data,
            "bad_entry_probability_10d",
            [("0 to 0.25", 0, 0.25), ("0.25 to 0.40", 0.25, 0.40), ("0.40 to 0.70", 0.40, 0.70), (">= 0.70", 0.70, None)],
        )
        data["upside_probability_10d_band"] = self._band_column(
            data,
            "upside_probability_10d",
            [("< 0.40", None, 0.40), ("0.40 to 0.60", 0.40, 0.60), ("0.60 to 0.80", 0.60, 0.80), (">= 0.80", 0.80, None)],
        )
        return data

    def _group_performance(self, df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if df.empty or column not in df.columns:
            return []
        rows = []
        for key, group in df.groupby(column, dropna=False):
            item = self._performance(group)
            item[column] = str(key)
            rows.append(item)
        return rows

    def _virtual_position_sizing_simulation(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        sizing_rules = [
            ("A", "safe=1.0, watch=1.0, danger=0.5", self._risk_label_multiplier(df, safe=1.0, watch=1.0, danger=0.5)),
            ("B", "safe=1.2, watch=1.0, danger=0.5", self._risk_label_multiplier(df, safe=1.2, watch=1.0, danger=0.5)),
            ("C", "safe=1.2, watch=1.0, danger=0.7", self._risk_label_multiplier(df, safe=1.2, watch=1.0, danger=0.7)),
            ("D", "bad_entry_probability bands: <0.25=1.2, 0.25-0.40=1.0, 0.40-0.70=0.7, >=0.70=0.4", self._bad_probability_multiplier(df)),
            ("E", "safe=1.2, watch=1.0, danger upside>=0.80=1.0, other danger=0.5", self._danger_upside_multiplier(df)),
        ]
        return [
            self._position_sizing_result(df, sizing_id, rule, multipliers)
            for sizing_id, rule, multipliers in sizing_rules
        ]

    def _risk_label_multiplier(self, df: pd.DataFrame, safe: float, watch: float, danger: float) -> pd.Series:
        labels = df["entry_risk_label"].astype(str)
        multipliers = pd.Series(1.0, index=df.index)
        multipliers = multipliers.mask(labels.eq("safe"), safe)
        multipliers = multipliers.mask(labels.eq("watch"), watch)
        multipliers = multipliers.mask(labels.eq("danger"), danger)
        return multipliers

    def _bad_probability_multiplier(self, df: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(df["bad_entry_probability_10d"], errors="coerce")
        multipliers = pd.Series(1.0, index=df.index)
        multipliers = multipliers.mask(values < 0.25, 1.2)
        multipliers = multipliers.mask((values >= 0.25) & (values < 0.40), 1.0)
        multipliers = multipliers.mask((values >= 0.40) & (values < 0.70), 0.7)
        multipliers = multipliers.mask(values >= 0.70, 0.4)
        return multipliers

    def _danger_upside_multiplier(self, df: pd.DataFrame) -> pd.Series:
        labels = df["entry_risk_label"].astype(str)
        upside = pd.to_numeric(df["upside_probability_10d"], errors="coerce")
        multipliers = pd.Series(1.0, index=df.index)
        multipliers = multipliers.mask(labels.eq("safe"), 1.2)
        multipliers = multipliers.mask(labels.eq("watch"), 1.0)
        multipliers = multipliers.mask(labels.eq("danger"), 0.5)
        multipliers = multipliers.mask(labels.eq("danger") & (upside >= 0.80), 1.0)
        return multipliers

    def _position_sizing_result(self, df: pd.DataFrame, sizing_id: str, rule: str, multipliers: pd.Series) -> dict[str, Any]:
        gross_profit = pd.to_numeric(df.get("gross_profit"), errors="coerce")
        net_profit = pd.to_numeric(df.get("net_profit"), errors="coerce")
        adjusted_gross = gross_profit * multipliers
        adjusted_net = net_profit * multipliers
        original_net_total = self._sum(df, "net_profit")
        original_gross_total = self._sum(df, "gross_profit")
        return {
            "sizing_id": sizing_id,
            "rule": rule,
            "original_net_profit_total": original_net_total,
            "adjusted_net_profit_total": self._series_sum(adjusted_net),
            "profit_delta": self._profit_delta(self._series_sum(adjusted_net), original_net_total),
            "original_gross_profit_total": original_gross_total,
            "adjusted_gross_profit_total": self._series_sum(adjusted_gross),
            "average_position_multiplier": self._series_mean(multipliers),
            "max_position_multiplier": self._series_max(multipliers),
            "min_position_multiplier": self._series_min(multipliers),
            "weighted_win_rate": self._weighted_win_rate(df, multipliers),
            "trade_count": int(len(df)),
        }

    def _multi_group_performance(self, df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
        if df.empty or any(column not in df.columns for column in columns):
            return []
        rows = []
        grouped = df.groupby(columns, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            item = self._performance(group)
            item.update({column: str(key) for column, key in zip(columns, keys)})
            item.update(self._ml_means(group))
            rows.append(item)
        return rows

    def _danger_group_performance(self, df: pd.DataFrame, column: str) -> list[dict[str, Any]]:
        if df.empty or "entry_risk_label" not in df.columns:
            return []
        danger = df[df["entry_risk_label"].astype(str).eq("danger")]
        return self._multi_group_performance(danger, [column])

    def _bad_probability_band_performance(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "bad_entry_probability_10d" not in df.columns:
            return []
        rows = []
        for label, lower, upper in [("0.0-0.25", 0.0, 0.25), ("0.25-0.40", 0.25, 0.40), ("0.40-1.0", 0.40, 1.0)]:
            if upper == 1.0:
                group = df[(df["bad_entry_probability_10d"] >= lower) & (df["bad_entry_probability_10d"] <= upper)]
            else:
                group = df[(df["bad_entry_probability_10d"] >= lower) & (df["bad_entry_probability_10d"] < upper)]
            item = self._performance(group)
            item["band"] = label
            rows.append(item)
        return rows

    def _score_top_bottom(self, df: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
        if df.empty or "ml_score" not in df.columns:
            return []
        sorted_df = df.sort_values("ml_score", ascending=False)
        rows = []
        for bucket, group in [("top", sorted_df.head(top_n)), ("bottom", sorted_df.tail(top_n))]:
            item = self._performance(group)
            item["bucket"] = f"{bucket}_{top_n}"
            item.update(self._ml_means(group))
            rows.append(item)
        return rows

    def _trade_details(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        columns = [
            "signal_date",
            "code",
            "net_profit",
            "net_profit_rate",
            "expected_return_10d",
            "expected_max_return_10d",
            "expected_max_return_20d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "swing_success_probability_20d",
            "entry_risk_label",
            "ml_score",
        ]
        rows = []
        available = [column for column in columns if column in df.columns]
        for row in df.sort_values(["date", "code"])[available].to_dict("records"):
            rows.append({key: self._json_value(value) for key, value in row.items()})
        return rows

    def _trade_csv_rows(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        columns = [
            "signal_date",
            "entry_date",
            "exit_date",
            "code",
            "name",
            "sector_name",
            "gross_profit",
            "gross_profit_rate",
            "net_profit",
            "net_profit_rate",
            "result",
            "exit_reason",
            "expected_return_5d",
            "expected_return_10d",
            "expected_max_return_10d",
            "expected_max_return_20d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "swing_success_probability_20d",
            "entry_risk_label",
            "ml_score",
            "ml_prediction_joined",
        ]
        data = self._normalize_metrics(df)
        available = [column for column in columns if column in data.columns]
        return [
            {key: self._json_value(value) for key, value in row.items()}
            for row in data.sort_values(["date", "code"])[available].to_dict("records")
        ]

    def _ml_average_by_result(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        rows = []
        for label in ["win", "loss"]:
            group = df[df["win_loss"].eq(label)]
            item = {
                "win_loss": label,
                "count": int(len(group)),
                "expected_return_5d_mean": self._mean(group, "expected_return_5d"),
                "expected_return_10d_mean": self._mean(group, "expected_return_10d"),
                "expected_max_return_10d_mean": self._mean(group, "expected_max_return_10d"),
                "expected_max_return_20d_mean": self._mean(group, "expected_max_return_20d"),
                "upside_probability_10d_mean": self._mean(group, "upside_probability_10d"),
                "bad_entry_probability_10d_mean": self._mean(group, "bad_entry_probability_10d"),
                "swing_success_probability_20d_mean": self._mean(group, "swing_success_probability_20d"),
                "ml_score_mean": self._mean(group, "ml_score"),
                "net_profit_total": self._sum(group, "net_profit"),
                "net_profit_rate_mean": self._mean(group, "net_profit_rate"),
            }
            rows.append(item)
        return rows

    def _ranked_trade_details(self, df: pd.DataFrame, top_n: int, ascending: bool) -> list[dict[str, Any]]:
        if df.empty or "net_profit" not in df.columns:
            return []
        ranked = df.sort_values("net_profit", ascending=ascending).head(top_n)
        return self._win_loss_trade_details(ranked)

    def _watch_trade_details(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "entry_risk_label" not in df.columns:
            return []
        watch = df[df["entry_risk_label"].astype(str).eq("watch")]
        return self._win_loss_trade_details(watch.sort_values("net_profit", ascending=True))

    def _win_loss_trade_details(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        columns = self._win_loss_detail_columns()
        available = [column for column in columns if column in df.columns]
        rows = []
        for row in df[available].to_dict("records"):
            rows.append({key: self._json_value(value) for key, value in row.items()})
        return rows

    def _win_loss_detail_columns(self) -> list[str]:
        return [
            "signal_date",
            "code",
            "net_profit",
            "net_profit_rate",
            "expected_return_5d",
            "expected_return_10d",
            "expected_max_return_10d",
            "expected_max_return_20d",
            "upside_probability_10d",
            "bad_entry_probability_10d",
            "swing_success_probability_20d",
            "entry_risk_label",
            "ml_score",
        ]

    def _danger_win_loss_difference(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "entry_risk_label" not in df.columns:
            return []
        danger = df[df["entry_risk_label"].astype(str).eq("danger")]
        win = danger[danger["win_loss"].eq("win")]
        loss = danger[danger["win_loss"].eq("loss")]
        rows = []
        for bucket, group in [("win", win), ("loss", loss)]:
            rows.append(self._win_loss_mean_row(bucket, group))
        rows.append(
            {
                "bucket": "win_minus_loss",
                "count": int(len(win) - len(loss)),
                "expected_return_10d_mean": self._diff_mean(win, loss, "expected_return_10d"),
                "expected_max_return_10d_mean": self._diff_mean(win, loss, "expected_max_return_10d"),
                "expected_max_return_20d_mean": self._diff_mean(win, loss, "expected_max_return_20d"),
                "upside_probability_10d_mean": self._diff_mean(win, loss, "upside_probability_10d"),
                "bad_entry_probability_10d_mean": self._diff_mean(win, loss, "bad_entry_probability_10d"),
                "swing_success_probability_20d_mean": self._diff_mean(win, loss, "swing_success_probability_20d"),
                "ml_score_mean": self._diff_mean(win, loss, "ml_score"),
                "net_profit_rate_mean": self._diff_mean(win, loss, "net_profit_rate"),
            }
        )
        return rows

    def _win_loss_mean_row(self, bucket: str, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "bucket": bucket,
            "count": int(len(df)),
            "expected_return_10d_mean": self._mean(df, "expected_return_10d"),
            "expected_max_return_10d_mean": self._mean(df, "expected_max_return_10d"),
            "expected_max_return_20d_mean": self._mean(df, "expected_max_return_20d"),
            "upside_probability_10d_mean": self._mean(df, "upside_probability_10d"),
            "bad_entry_probability_10d_mean": self._mean(df, "bad_entry_probability_10d"),
            "swing_success_probability_20d_mean": self._mean(df, "swing_success_probability_20d"),
            "ml_score_mean": self._mean(df, "ml_score"),
            "net_profit_rate_mean": self._mean(df, "net_profit_rate"),
        }

    def _watch_distribution(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty or "entry_risk_label" not in df.columns:
            return []
        watch = df[df["entry_risk_label"].astype(str).eq("watch")]
        if watch.empty:
            return []
        rows = []
        for bucket, group in [("all", watch), ("win", watch[watch["win_loss"].eq("win")]), ("loss", watch[watch["win_loss"].eq("loss")])]:
            rows.append(self._distribution_row(bucket, group))
        return rows

    def _distribution_row(self, bucket: str, df: pd.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {"bucket": bucket, "count": int(len(df))}
        for column in ["expected_return_10d", "upside_probability_10d", "bad_entry_probability_10d", "ml_score"]:
            row[f"{column}_mean"] = self._mean(df, column)
            row[f"{column}_min"] = self._series_stat(df, column, "min")
            row[f"{column}_max"] = self._series_stat(df, column, "max")
        row["net_profit_total"] = self._sum(df, "net_profit")
        row["win_rate"] = self._mean(df, "is_win")
        return row

    def _virtual_filter_simulation(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        filters = [
            ("A", 'entry_risk_label == "danger"', df["entry_risk_label"].astype(str).eq("danger")),
            ("B", "bad_entry_probability_10d >= 0.70", df["bad_entry_probability_10d"] >= 0.70),
            ("C", 'entry_risk_label == "danger" and ml_score < 0', df["entry_risk_label"].astype(str).eq("danger") & (df["ml_score"] < 0)),
            (
                "D",
                'entry_risk_label == "danger" and upside_probability_10d < 0.80',
                df["entry_risk_label"].astype(str).eq("danger") & (df["upside_probability_10d"] < 0.80),
            ),
            (
                "E",
                "bad_entry_probability_10d >= 0.40 and expected_return_10d < 0.03",
                (df["bad_entry_probability_10d"] >= 0.40) & (df["expected_return_10d"] < 0.03),
            ),
            (
                "F",
                "bad_entry_probability_10d >= 0.40 and ml_score < 5",
                (df["bad_entry_probability_10d"] >= 0.40) & (df["ml_score"] < 5),
            ),
        ]
        original = self._performance(df)
        rows = []
        for filter_id, condition, remove_mask in filters:
            remove_mask = remove_mask.fillna(False)
            kept = df[~remove_mask]
            removed = df[remove_mask]
            kept_perf = self._performance(kept)
            removed_perf = self._performance(removed)
            rows.append(
                {
                    "filter_id": filter_id,
                    "condition": condition,
                    "original_trade_count": original["count"],
                    "kept_trade_count": kept_perf["count"],
                    "removed_trade_count": removed_perf["count"],
                    "original_net_profit_total": original["net_profit_total"],
                    "kept_net_profit_total": kept_perf["net_profit_total"],
                    "removed_net_profit_total": removed_perf["net_profit_total"],
                    "original_win_rate": original["win_rate"],
                    "kept_win_rate": kept_perf["win_rate"],
                    "removed_win_rate": removed_perf["win_rate"],
                    "profit_delta": self._profit_delta(kept_perf["net_profit_total"], original["net_profit_total"]),
                }
            )
        return rows

    def _performance(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "count": int(len(df)),
            "win_rate": self._mean(df, "is_win"),
            "gross_profit_total": self._sum(df, "gross_profit"),
            "net_profit_total": self._sum(df, "net_profit"),
            "gross_profit_rate_mean": self._mean(df, "gross_profit_rate"),
            "net_profit_rate_mean": self._mean(df, "net_profit_rate"),
        }

    def _performance_columns(self) -> list[str]:
        return ["count", "win_rate", "gross_profit_total", "net_profit_total", "gross_profit_rate_mean", "net_profit_rate_mean"]

    def _ml_means(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "ml_score_mean": self._mean(df, "ml_score"),
            "expected_return_10d_mean": self._mean(df, "expected_return_10d"),
            "upside_probability_10d_mean": self._mean(df, "upside_probability_10d"),
            "bad_entry_probability_10d_mean": self._mean(df, "bad_entry_probability_10d"),
            "expected_max_return_10d_mean": self._mean(df, "expected_max_return_10d"),
            "expected_max_return_20d_mean": self._mean(df, "expected_max_return_20d"),
            "swing_success_probability_20d_mean": self._mean(df, "swing_success_probability_20d"),
        }

    def _band_column(self, df: pd.DataFrame, column: str, bands: list[tuple[str, float | None, float | None]]) -> pd.Series:
        if column not in df.columns:
            return pd.Series(pd.NA, index=df.index, dtype="string")
        values = pd.to_numeric(df[column], errors="coerce")
        output = pd.Series(pd.NA, index=df.index, dtype="string")
        for label, lower, upper in bands:
            mask = values.notna()
            if lower is not None:
                mask &= values >= lower
            if upper is not None:
                mask &= values < upper
            output = output.mask(mask, label)
        return output

    def _read_csv_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    def _sum(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        value = df[column].sum()
        return None if pd.isna(value) else float(value)

    def _mean(self, df: pd.DataFrame, column: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        value = df[column].mean()
        return None if pd.isna(value) else float(value)

    def _ratio(self, numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def _profit_delta(self, kept: Any, original: Any) -> float | None:
        if kept is None or original is None:
            return None
        return float(kept) - float(original)

    def _diff_mean(self, left: pd.DataFrame, right: pd.DataFrame, column: str) -> float | None:
        left_mean = self._mean(left, column)
        right_mean = self._mean(right, column)
        if left_mean is None or right_mean is None:
            return None
        return left_mean - right_mean

    def _series_stat(self, df: pd.DataFrame, column: str, stat: str) -> float | None:
        if df.empty or column not in df.columns:
            return None
        series = pd.to_numeric(df[column], errors="coerce")
        value = series.min() if stat == "min" else series.max()
        return None if pd.isna(value) else float(value)

    def _series_sum(self, series: pd.Series) -> float | None:
        value = series.sum()
        return None if pd.isna(value) else float(value)

    def _series_mean(self, series: pd.Series) -> float | None:
        value = series.mean()
        return None if pd.isna(value) else float(value)

    def _series_max(self, series: pd.Series) -> float | None:
        value = series.max()
        return None if pd.isna(value) else float(value)

    def _series_min(self, series: pd.Series) -> float | None:
        value = series.min()
        return None if pd.isna(value) else float(value)

    def _weighted_win_rate(self, df: pd.DataFrame, multipliers: pd.Series) -> float | None:
        if df.empty:
            return None
        total_weight = multipliers.sum()
        if total_weight == 0:
            return None
        wins = df["is_win"].fillna(False).astype(bool)
        return float(multipliers[wins].sum() / total_weight)

    def _table(self, rows: list[dict[str, Any]], columns: list[str]) -> str:
        if not rows:
            return "_No rows._"
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = ["| " + " | ".join(self._fmt(row.get(column)) for column in columns) + " |" for row in rows]
        return "\n".join([header, separator, *body])

    def _fmt(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    def _json_value(self, value: Any) -> Any:
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)
