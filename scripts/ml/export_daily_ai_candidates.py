#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml.config import ML_MODEL_CURRENT_ROOT
from ml.daily_candidates import DailyAICandidateExporter, ENRICHED_V2_REQUIRED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export report-only daily AI stock candidates.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--min-turnover-value", type=float, default=50_000_000)
    parser.add_argument("--max-bad-entry-probability", type=float, default=None)
    parser.add_argument("--model-root", default=str(ML_MODEL_CURRENT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = DailyAICandidateExporter()
    candidates = exporter.build_candidates(
        args.date,
        top_n=args.top_n,
        min_turnover_value=args.min_turnover_value,
        max_bad_entry_probability=args.max_bad_entry_probability,
    )
    csv_path = exporter.save_csv(candidates, args.date)
    md_path = exporter.save_markdown(candidates, args.date)
    print(f"saved csv to {csv_path}")
    print(f"saved markdown to {md_path}")
    print(f"rows={len(candidates)} columns={len(candidates.columns)}")
    warning = _current_model_profile_warning(Path(args.model_root))
    if warning:
        print(f"warning={warning}")
    if candidates.empty:
        print("warning=no candidates matched the ranking/filter conditions")
    else:
        for row in candidates.head(args.top_n).to_dict("records"):
            print(
                f"rank={row['rank']} code={row['code']} name={row.get('name', '')} "
                f"risk_adjusted_score={row.get('risk_adjusted_score')} "
                f"expected_return_10d={row.get('expected_return_10d')} "
                f"expected_max_return_20d={row.get('expected_max_return_20d')} "
                f"swing_success_probability_20d={row.get('swing_success_probability_20d')} "
                f"bad_entry_probability_10d={row.get('bad_entry_probability_10d')} "
                f"turnover_value={row.get('turnover_value')}"
            )


def _current_model_profile_warning(model_root: Path) -> str | None:
    feature_path = model_root / "feature_columns.json"
    try:
        feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "current ML model profile could not be checked; expected enriched_v2 for daily AI candidates"
    feature_set = {str(column) for column in feature_columns}
    missing = sorted(ENRICHED_V2_REQUIRED_FEATURES - feature_set)
    if missing:
        return (
            "current ML model does not look like enriched_v2; "
            f"missing enriched features: {', '.join(missing)}"
        )
    return None


if __name__ == "__main__":
    main()
