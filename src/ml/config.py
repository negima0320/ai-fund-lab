from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

JQUANTS_CACHE_ROOT = ROOT / "data" / "cache" / "jquants"
ML_DATA_ROOT = ROOT / "data" / "ml"
ML_FEATURES_ROOT = ML_DATA_ROOT / "features"
ML_LABELS_ROOT = ML_DATA_ROOT / "labels"
ML_DATASETS_ROOT = ML_DATA_ROOT / "datasets"
ML_PREDICTIONS_ROOT = ML_DATA_ROOT / "predictions"
ML_MODELS_ROOT = ROOT / "models" / "ml"
ML_MODEL_ARCHIVE_ROOT = ML_MODELS_ROOT / "archive"
ML_MODEL_CURRENT_ROOT = ML_MODELS_ROOT / "current"
ML_REPORTS_ROOT = ROOT / "reports" / "ml"

FEATURE_HISTORY_DAYS = 180

RETURN_WINDOWS = [1, 3, 5, 10, 20]
MOVING_AVERAGE_WINDOWS = [5, 10, 25, 75]
VOLUME_RATIO_WINDOWS = [5, 20]
TURNOVER_RATIO_WINDOWS = [5, 20]

FEATURE_COLUMNS = [
    "date",
    "code",
    "close",
    "volume",
    "turnover_value",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma5_gap",
    "ma10_gap",
    "ma25_gap",
    "ma75_gap",
    "ma5_slope",
    "ma25_slope",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "turnover_ratio_5d",
    "turnover_ratio_20d",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_position",
    "gap_up_ratio",
    "daily_range_ratio",
    "EPS",
    "BPS",
    "EqAR",
    "Sales_growth",
    "OP_growth",
    "NP_growth",
    "FEPS_growth",
    "FSales_growth",
    "FOP_growth",
    "PayoutRatioAnn",
    "days_to_earnings",
    "days_after_earnings",
    "is_near_earnings",
    "market",
    "sector_name",
    "scale_category",
    "margin_category",
    "credit_category",
    "topix_return_5d",
    "topix_return_10d",
    "topix_return_20d",
    "relative_return_5d",
    "relative_return_10d",
    "relative_return_20d",
]

TECHNICAL_FEATURE_COLUMNS = [
    "close",
    "volume",
    "turnover_value",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma5_gap",
    "ma10_gap",
    "ma25_gap",
    "ma75_gap",
    "ma5_slope",
    "ma25_slope",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "turnover_ratio_5d",
    "turnover_ratio_20d",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_position",
    "gap_up_ratio",
    "daily_range_ratio",
]

FINANCIAL_FEATURE_COLUMNS = [
    "EPS",
    "BPS",
    "EqAR",
    "Sales_growth",
    "OP_growth",
    "NP_growth",
    "FEPS_growth",
    "FSales_growth",
    "FOP_growth",
    "PayoutRatioAnn",
]

FINANCIAL_NUMERIC_COLUMNS = [
    "Sales",
    "OP",
    "NP",
    "EPS",
    "BPS",
    "EqAR",
    "FEPS",
    "FSales",
    "FOP",
    "PayoutRatioAnn",
]

EARNINGS_FEATURE_COLUMNS = [
    "days_to_earnings",
    "days_after_earnings",
    "is_near_earnings",
]

LISTED_INFO_FEATURE_COLUMNS = [
    "market",
    "sector_name",
    "scale_category",
    "margin_category",
    "credit_category",
]

TOPIX_FEATURE_COLUMNS = [
    "topix_return_5d",
    "topix_return_10d",
    "topix_return_20d",
    "relative_return_5d",
    "relative_return_10d",
    "relative_return_20d",
]

LABEL_LOOKAHEAD_DAYS = 20
LABEL_HORIZONS = [5, 10]
UPSIDE_THRESHOLD = 0.05
BAD_ENTRY_THRESHOLD = -0.05
SWING_SUCCESS_THRESHOLD = 0.10

LABEL_COLUMNS = [
    "date",
    "code",
    "entry_price",
    "future_5d_return",
    "future_10d_return",
    "upside_10d",
    "bad_entry_10d",
    "future_max_return_10d",
    "future_max_return_20d",
    "future_swing_success_20d",
]

REQUIRED_LABEL_COLUMNS = [
    "future_5d_return",
    "future_10d_return",
    "upside_10d",
    "bad_entry_10d",
]

OPTIONAL_LABEL_COLUMNS = [
    "future_max_return_10d",
    "future_max_return_20d",
    "future_swing_success_20d",
]

DATASET_REQUIRED_COLUMNS = [
    "date",
    "code",
    "close",
    "entry_price",
    "future_5d_return",
    "future_10d_return",
    "upside_10d",
    "bad_entry_10d",
]

MAX_ABS_LABEL_RETURN = 1.0

MODEL_TARGETS = {
    "future_5d_return_regression": {"target": "future_5d_return", "task": "regression"},
    "future_10d_return_regression": {"target": "future_10d_return", "task": "regression"},
    "upside_10d_classification": {"target": "upside_10d", "task": "classification"},
    "bad_entry_10d_classification": {"target": "bad_entry_10d", "task": "classification"},
    "future_max_return_10d_regression": {"target": "future_max_return_10d", "task": "regression"},
    "future_max_return_20d_regression": {"target": "future_max_return_20d", "task": "regression"},
    "future_swing_success_20d_classification": {"target": "future_swing_success_20d", "task": "classification"},
}

MODEL_EXCLUDE_COLUMNS = [
    "date",
    "code",
    "entry_price",
    "future_5d_return",
    "future_10d_return",
    "upside_10d",
    "bad_entry_10d",
    "future_max_return_10d",
    "future_max_return_20d",
    "future_swing_success_20d",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "market",
    "sector_name",
    "scale_category",
    "margin_category",
    "credit_category",
]

LIGHTGBM_REGRESSION_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "verbose": -1,
}

LIGHTGBM_CLASSIFICATION_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "verbose": -1,
}

MODEL_FILENAMES = {
    "future_5d_return_regression": "future_5d_return_regression.joblib",
    "future_10d_return_regression": "future_10d_return_regression.joblib",
    "upside_10d_classification": "upside_10d_classification.joblib",
    "bad_entry_10d_classification": "bad_entry_10d_classification.joblib",
    "future_max_return_10d_regression": "future_max_return_10d_regression.joblib",
    "future_max_return_20d_regression": "future_max_return_20d_regression.joblib",
    "future_swing_success_20d_classification": "future_swing_success_20d_classification.joblib",
}

PREDICTION_COLUMNS = [
    "date",
    "code",
    "market",
    "sector_name",
    "scale_category",
    "margin_category",
    "credit_category",
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

JQUANTS_CACHE_DIRS = {
    "prices": "prices",
    "listed_info": "listed_info",
    "topix_prices": "topix_prices",
    "earnings_calendar": "earnings_calendar",
    "trading_calendar": "trading_calendar",
    "investor_types": "investor_types",
    "financial_statements": "financial_statements",
}

PRICE_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover_value",
]

ADJUSTED_PRICE_COLUMNS = [
    "adjusted_open",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "adjusted_volume",
]

TRADING_CALENDAR_COLUMNS = [
    "date",
    "holiday_division",
    "is_business_day",
]

INVESTOR_TYPE_DATE_COLUMNS = [
    "PubDate",
    "StDate",
    "EnDate",
    "date",
]

INVESTOR_TYPE_NUMERIC_COLUMNS = [
    "FrgnBal",
    "IndBal",
    "BrkBal",
    "PropBal",
    "InvTrBal",
    "TrstBnkBal",
]
