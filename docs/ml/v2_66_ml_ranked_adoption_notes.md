# v2_66 ML Ranked Adoption Notes

This note summarizes why `rookie_dealer_02_v2_66_ml_ranked` is the next main candidate profile, and what must be monitored before any production use.

## Scope

- Evaluation period: `2023-01-01` to `2026-05-31`
- Baseline: `rookie_dealer_02_v2_65`
- Candidate: `rookie_dealer_02_v2_66_ml_ranked`
- AI-only reference: `rookie_dealer_02_v2_67_ml_standalone`
- Prediction source for the backtest: `data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet`
- No live trading, broker execution, OpenAI API call, or J-Quants refetch was used in this evaluation.

## Result Summary

| profile | final_assets | net_profit | win_rate | profit_factor | max_drawdown | trades |
|---|---:|---:|---:|---:|---:|---:|
| `rookie_dealer_02_v2_65` | 1,606,699.70 | 491,656.21 | 41.09% | 1.3016 | -27.81% | 421 |
| `rookie_dealer_02_v2_66_ml_ranked` | 2,121,672.30 | 904,163.62 | 41.23% | 1.4808 | -23.58% | 439 |
| `rookie_dealer_02_v2_67_ml_standalone` | 1,280,860.00 | 241,094.94 | 53.95% | 1.6305 | -12.37% | 215 |

`v2_66_ml_ranked` keeps the existing `v2_65` candidate extraction and trading rules, then joins ML predictions and prioritizes candidates by:

```text
risk_adjusted_score = expected_return_10d - 0.5 * bad_entry_probability_10d
```

Missing ML predictions are not hard-excluded in the implementation, but in the 2023-01 to 2026-05 diagnostic run the ML join success rate was 100%.

## Why v2_66 Is the Main Candidate

`v2_66_ml_ranked` is the strongest candidate because it improves the existing strategy without replacing its core logic.

- It outperformed `v2_65` in every evaluated calendar year.
- It improved `25` months and worsened `14` months versus `v2_65`.
- It increased net profit from `491,656.21` JPY to `904,163.62` JPY.
- It improved profit factor from `1.3016` to `1.4808`.
- It reduced max drawdown from `-27.81%` to `-23.58%`.
- It preserved the same basic trade style as `v2_65`: average holding days remained about `3.7`.
- It avoids the bigger operational shift of an AI-only profile.

`v2_67_ml_standalone` is useful as a risk-controlled AI-only benchmark, but it is not the main replacement candidate yet. It has higher win rate and lower drawdown, but lower total profit and a different holding profile.

## Known Weaknesses

### Large Single-Code Contribution

`67400` contributed `+415,843.88` JPY in `v2_66_ml_ranked`.

This is a meaningful concentration risk. The profile still looks better than `v2_65`, but monthly and code-level attribution must be monitored so one large winner does not hide structural weakness.

### Weak Months

The largest months where `v2_66` underperformed `v2_65` were:

| month | v2_66 net_profit | v2_65 net_profit | difference |
|---|---:|---:|---:|
| 2026-04 | -35,927.90 | 51,885.16 | -87,813.06 |
| 2025-09 | -44,416.09 | -14,375.23 | -30,040.86 |
| 2023-01 | -58,544.17 | -30,494.09 | -28,050.08 |
| 2025-10 | -72,398.87 | -52,440.86 | -19,958.01 |
| 2025-07 | 48,010.03 | 66,374.01 | -18,363.98 |

The `2025-09` to `2025-10` weakness should be reviewed as a contiguous regime, not only as independent monthly noise.

### Risk Adjusted Score Band Behavior

The diagnostic result does not show a simple monotonic relationship between `risk_adjusted_score` bands and realized profit inside the `v2_66` trade set.

Notably:

- `< -0.30` was highly profitable in this sample.
- `-0.30 to -0.20` was negative.
- `-0.20 to -0.10` was positive.
- `-0.10 to 0` was negative.

This means the ML score is useful for ranking within the existing candidate universe, but should not yet be interpreted as an absolute trade-quality probability.

## Prediction Source Policy

For historical backtests, use walk-forward predictions:

```text
data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet
```

This prevents current-model leakage into past dates.

For daily operation, use the current enriched model only for the current target date after the daily data pipeline has produced features:

```text
models/ml/current_enriched_v2/
```

The two sources must not be confused:

- Walk-forward predictions are for historical validation.
- `current_enriched_v2` is for daily forward candidate generation.
- Backtests should not silently regenerate past predictions using `models/ml/current_enriched_v2`.

## Monitoring Items

Track these at least monthly:

- Monthly profit factor
- Monthly max drawdown
- Monthly net profit versus `v2_65`
- Monthly win rate
- Code-level contribution and concentration
- Top winner contribution share
- Worst losing codes
- ML join success rate
- Prediction missing count
- Risk-adjusted-score band performance
- Number of trades affected by missing prediction fallback

If ML join success drops below 100%, the affected days and codes should be listed before trusting the run.

## Daily Operation Prerequisites

To use the `v2_66` idea in daily operation, the following data flow must be complete before candidate ranking:

1. J-Quants cache update
   - prices
   - TOPIX prices
   - earnings calendar when available
   - financial statements when available

2. Feature generation
   - `FeatureBuilder` must generate the target date features.
   - Future prices must not be used.

3. Prediction generation
   - The daily predictor must use the intended model directory.
   - For current operation this is expected to be `models/ml/current_enriched_v2/`.

4. Candidate review
   - Daily AI candidates remain human-review output.
   - Candidate reports are not trading instructions.

5. Backtest or live integration caution
   - Do not connect to live order placement until a separate safety review is complete.
   - Do not delete `v2_65`; keep it as the control profile.
   - Do not auto-switch model directories inside backtests.
   - Confirm prediction source in every report.

## Adoption Position

`rookie_dealer_02_v2_66_ml_ranked` is suitable as the next main profile candidate for further paper/backtest validation.

It should not yet be treated as an unattended live trading profile. The remaining risks are concentration, weak-regime behavior, prediction pipeline dependency, and the possibility that risk-adjusted score bands behave differently out of sample.
