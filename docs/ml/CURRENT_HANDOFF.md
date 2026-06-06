# Current ML Handoff

Last updated: `2026-06-07 05:43 JST`

This document is the short handoff for continuing the AI / ML work in a fresh
chat. It intentionally summarizes only the current state, key constraints, and
next useful actions. For the full history, see
`docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`.

## Current State

The current strongest research profile is:

```text
rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing
```

The fallback / reference baseline is:

```text
rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue
```

v2_75 is derived from v2_73 and adds Portfolio Manager AI sizing. It changes
planned buy amount using:

```text
pm_score = high_conviction_proba - avoid_proba
```

Multiplier rule:

| condition | multiplier |
|---|---:|
| `pm_score >= 0.40` | `1.30` |
| `pm_score >= 0.20` | `1.15` |
| `pm_score >= 0.00` | `1.00` |
| `pm_score >= -0.20` | `0.80` |
| otherwise | `0.60` |

The multiplier is applied before existing constraints:

- cash
- `daily_buy_limit = 900000`
- round lot
- scaled buy
- max positions
- Exit AI

## Latest Commit and Working Tree

Latest committed code:

```text
44b01c4 Add PM phase3d detail audit
```

Recent relevant commits:

```text
44b01c4 Add PM phase3d detail audit
486f896 Document ML portfolio manager progress
493324c Add portfolio manager AI sizing backtest profile
```

Current uncommitted documentation updates at the time this handoff was written:

- `docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`
- `docs/ml/README.md`
- `docs/ml/v2_73_adoption_notes.md`
- `docs/ml/CURRENT_HANDOFF.md`

## Important Artifacts

Models:

```text
models/ml/current_enriched_v2/
models/ml/exit/current_v2_66/
models/ml/portfolio_manager/current_v2_73_phase3b_clean/
```

Historical walk-forward predictions:

```text
data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet
```

Portfolio Manager clean dataset:

```text
data/ml/portfolio_manager/portfolio_manager_dataset_v2_73_clean_2023-01_to_2026-05.parquet
```

Backtest logs:

```text
logs/backtests/rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue/2023-01-01_to_2026-05-31/
logs/backtests/rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing/2023-01-01_to_2026-05-31/
```

Key reports:

```text
reports/ml/portfolio_manager_phase3d_full_backtest_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.json
```

Generated `data/ml`, `models/ml`, `reports/ml`, and `logs/backtests` artifacts
are intentionally not committed when ignored.

## Latest Results

Period:

```text
2023-01-01 to 2026-05-31
```

Comparison:

| profile | net_profit | PF | DD | win_rate | trades |
|---|---:|---:|---:|---:|---:|
| v2_73 baseline | `1,169,366` | `1.5683` | `-19.13%` | `43.01%` | `481` |
| v2_75 | `2,679,727` | `2.2054` | `-9.17%` | `48.53%` | `546` |

v2_75 detail audit:

- promotion checks: `7 / 8` passed
- PM fields are now carried into `trades.csv`
- `all_trades` BUY rows with PM values: `549 / 549`
- `trades.csv` SELL rows with PM values: `546 / 546`
- `purchase_audit.csv` PM match rate: `100%`
- `pm_status`: `ok` for all closed trades

67400 dependency:

- v2_75 67400 profit: `539,696`
- 67400 contribution to v2_75 net_profit: `20.14%`
- v2_75 excluding 67400 profit: `1,573,282`
- v2_75 excluding 67400 PF: `1.5670`
- v2_75 excluding 67400 DD: `-10.64%`
- v2_75 still beats v2_73 net profit after excluding 67400.

PM multiplier performance:

| multiplier | trades | net_profit | PF | win_rate |
|---:|---:|---:|---:|---:|
| `0.60` | `110` | `-136,272` | `0.7795` | `28.18%` |
| `0.80` | `152` | `244,615` | `1.3123` | `46.05%` |
| `1.00` | `145` | `512,770` | `1.5592` | `44.83%` |
| `1.15` | `40` | `363,848` | `3.6735` | `65.00%` |
| `1.30` | `99` | `1,128,016` | `4.3629` | `72.73%` |

Interpretation:

- `1.30` and `1.15` are strongly positive.
- `0.60` is clearly weak.
- PM score is directionally useful, especially by PF, win rate, and return on
  buy amount.

## Hard Constraints

Do not do these unless explicitly asked:

- Do not call OpenAI API.
- Do not refetch J-Quants API data.
- Do not regenerate historical predictions with `models/ml/current`.
- Do not use `selected_count_in_day`.
- Do not use actual/backtest/audit/decision/skip/exit/cash/final/result/profit
  columns as Portfolio Manager AI features.
- Do not destructively modify existing profiles such as v2_73.
- Do not connect to live order placement.
- Do not change Exit AI or `daily_buy_limit` unless the task explicitly says so.

Historical backtests should use:

```text
data/ml/walk_forward_predictions/
```

not current-model regenerated predictions.

## Useful Commands

Run v2_75 full backtest:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 src/main.py \
  --mode backtest \
  --provider jquants \
  --profile rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing \
  --start-date 2023-01-01 \
  --end-date 2026-05-31 \
  --skip-price-fetch \
  --quiet \
  --summary-only \
  --no-daily-logs
```

Run v2_75 detail audit:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_portfolio_manager_phase3d_detail.py
```

Relevant tests:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest -q \
  tests/test_ml_portfolio_manager_dataset.py \
  tests/test_ml_portfolio_manager_trainer.py \
  tests/test_ml_portfolio_manager_data_lineage.py \
  tests/test_ml_portfolio_manager_phase3c.py \
  tests/test_ml_portfolio_manager_phase3d.py \
  tests/test_ml_portfolio_manager_phase3d_detail_audit.py
```

Latest known test result:

```text
20 passed, 1 warning
```

## Next Good Tasks

Recommended next experiments:

1. Test a v2_75-derived profile where `pm_multiplier=0.60` becomes skip instead
   of reduced buy.
2. Test multiplier caps, such as removing the `1.30` boost.
3. Audit adverse periods only, especially weak months like `2025-03` and
   `2024-04`.
4. Check whether `daily_buy_limit_scaled_below_round_lot` keeps increasing in
   live-like daily runs.
5. Create v2_75 adoption notes if it survives the next robustness checks.

Keep v2_73 as the fallback baseline until v2_75 survives those additional
robustness checks.

## Documentation Map

Use these documents:

- `docs/ml/README.md`: ML documentation index
- `docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`: full recent history
- `docs/ml/v2_73_adoption_notes.md`: why v2_73 became the prior baseline
- `docs/ml/daily_ai_candidate_operation.md`: human-review daily AI candidates

