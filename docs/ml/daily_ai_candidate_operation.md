# Daily AI Candidate Operation

This document describes the report-only daily AI candidate mode.

## Purpose

The daily AI candidate report is a human-review list generated after the ML daily pipeline.
It does not place orders, modify existing trading logic, or connect to broker execution.

## Current Operating Profile

- model_profile: `enriched_v2`
- ranking: `risk_adjusted_return`
- score: `expected_return_10d - 0.5 * bad_entry_probability_10d`
- top_n: `10`
- liquidity filter: `turnover_value >= 50,000,000`
- assumed exit rule: `close_20d`
- suggested position size: `200,000` JPY
- assumed max positions: `5`

The report includes:

- `risk_adjusted_score`
- `expected_return_10d`
- `expected_max_return_20d`
- `swing_success_probability_20d`
- `bad_entry_probability_10d`
- `turnover_value`
- `suggested_position_size`
- `assumed_exit_rule`
- `model_profile`

## Validation Memo

The current report header references the enriched v2 validation:

- 5-year walk-forward enriched v2 `risk_adjusted_return`: PF `1.7225`, DD `-13.64%`
- realistic portfolio main condition: total_return `+20.57%`, PF `1.4376`, DD `-7.33%`

These are research results for context, not a trading instruction.

## Listed Info Handling

`listed_info` is not used as a historical training feature unless an as-of-safe snapshot exists.
The available `/equities/master` cache is a current snapshot and can leak future information if applied to past dates.
For that reason, `market`, `sector_name`, `scale_category`, `margin_category`, and `credit_category` are not forced into historical learning data.

Candidate reports may display names or sectors when a safe or display-only cache is available, but this display metadata is separate from the training feature decision.

## Commands

Run the full daily pipeline and export candidates:

```bash
python3 scripts/ml/daily_pipeline.py --date 2026-05-15
```

Export candidates from existing features and predictions:

```bash
python3 scripts/ml/export_daily_ai_candidates.py --date 2026-05-15
```

Optional filters:

```bash
python3 scripts/ml/export_daily_ai_candidates.py \
  --date 2026-05-15 \
  --top-n 10 \
  --min-turnover-value 50000000 \
  --max-bad-entry-probability 0.70
```

## Safety

- No order placement.
- No existing backtest rerun.
- No J-Quants API refetch.
- No OpenAI API call.
- `models/ml/current` is not switched by the candidate exporter.
- If `models/ml/current/feature_columns.json` does not look like an enriched v2 model, the CLI/pipeline emits a warning.
