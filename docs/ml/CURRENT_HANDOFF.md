# Current ML Handoff

Last updated: `2026-06-07`

This document is the short handoff for continuing the AI / ML work in a fresh
chat. It intentionally summarizes only the current state, key constraints, and
next useful actions. For the full history, see
`docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`.

## Current State

The current strongest balanced research profile is:

```text
rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030
```

Important reference profiles are:

```text
rookie_dealer_02_v2_76_pm_ai_low_score_skip
rookie_dealer_02_v2_75_pm_ai_high_minus_avoid_sizing
rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue
```

Profile lineage:

- v2_73: ML-ranked + Exit AI + scaled buy continue; prior baseline.
- v2_75: adds Portfolio Manager AI sizing.
- v2_76: derives from v2_75 and skips very low PM score trades.
- v2_77 cap 0.30: derives from v2_76 and adds per-code exposure cap `0.30`.

Portfolio Manager AI score:

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

v2_76 additionally skips the lowest PM-score trades. v2_77 keeps that behavior
and adds a per-code exposure cap to address v2_76's drawdown concentration.

## Latest Commit and Working Tree

Latest committed code before this handoff update:

```text
ea139c5 Add portfolio manager AI audits and v2_77 variants
```

Recent relevant commits:

```text
ea139c5 Add portfolio manager AI audits and v2_77 variants
f48cf9c Add PM phase3e low score skip profile
e46914b Document current ML handoff
44b01c4 Add PM phase3d detail audit
```

Current documentation updates in progress:

- `docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`
- `docs/ml/README.md`
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
logs/backtests/rookie_dealer_02_v2_76_pm_ai_low_score_skip/2023-01-01_to_2026-05-31/
logs/backtests/rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030/2023-01-01_to_2026-05-31/
```

Key reports:

```text
reports/ml/portfolio_manager_phase3d_full_backtest_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3d_detail_audit_2023-01_to_2026-05.json
reports/ml/portfolio_manager_phase3f_drawdown_audit_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3g_per_code_cap_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3h_capital_utilization_2023-01_to_2026-05.md
reports/ml/portfolio_manager_phase3i_candidate_pool_expansion_2023-01_to_2026-05.md
```

Generated `data/ml`, `models/ml`, `reports/ml`, and `logs/backtests` artifacts
are intentionally not committed when ignored.

## Latest Results

Period:

```text
2023-01-01 to 2026-05-31
```

Latest comparison:

| profile | net_profit | PF | DD | win_rate | trades |
|---|---:|---:|---:|---:|---:|
| v2_73 baseline | `1,169,366` | `1.5683` | `-19.13%` | `43.01%` | `481` |
| v2_75 | `2,679,727` | `2.2054` | `-9.17%` | `48.53%` | `546` |
| v2_76 | `3,812,364` | `2.5720` | `-19.38%` | `55.05%` | `507` |
| v2_77 cap 0.20 | `1,343,154` | `2.2436` | `-7.58%` | `52.95%` | `477` |
| v2_77 cap 0.30 | `2,914,686` | `2.5430` | `-7.54%` | `53.15%` | `511` |

Interpretation:

- v2_76 has the highest profit/PF/win rate, but DD is too large.
- Phase 3-F found v2_76 DD was mainly a specific-code exposure issue.
- v2_77 cap 0.30 is the current best balance: profit above v2_75, PF near
  v2_76, and DD below v2_75/v2_76.
- v2_75 remains the simpler PM-sizing reference.

v2_77 cap 0.30 capital utilization:

| metric | value |
|---|---:|
| average capital utilization | `50.995%` |
| median capital utilization | `53.121%` |
| days below 50% | `355` |
| cash idle days | `63` |
| average holding count | about `2.44` |

Low-utilization dominant reasons:

| reason | days |
|---|---:|
| `no_candidates` | `114` |
| `exit_only_day` | `67` |
| `candidates_all_low_pm_skipped` | `35` |

Candidate pool expansion result:

| variant | max_selected | net_profit | PF | DD | avg utilization | no_candidates |
|---|---:|---:|---:|---:|---:|---:|
| current | `10` | `2,914,686` | `2.5430` | `-7.54%` | `50.995%` | `114` |
| pool x2 | `20` | `2,520,271` | `2.3186` | `-8.04%` | `51.010%` | `114` |
| pool x3 | `30` | `2,520,271` | `2.3186` | `-8.04%` | `51.010%` | `114` |

Decision:

- Do not adopt candidate pool expansion for now.
- It did not reduce `no_candidates`, barely improved utilization, and worsened
  profit/PF/DD.

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

Run v2_77 cap 0.30 full backtest:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 src/main.py \
  --mode backtest \
  --provider jquants \
  --profile rookie_dealer_02_v2_77_pm_ai_low_score_skip_per_code_cap_030 \
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

Run latest capital utilization and candidate-pool audits:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_portfolio_manager_phase3h_capital_utilization.py
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 scripts/ml/audit_portfolio_manager_phase3i_candidate_pool.py
```

Relevant tests:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest -q \
  tests/test_ml_portfolio_manager_dataset.py \
  tests/test_ml_portfolio_manager_trainer.py \
  tests/test_ml_portfolio_manager_data_lineage.py \
  tests/test_ml_portfolio_manager_phase3c.py \
  tests/test_ml_portfolio_manager_phase3d.py \
  tests/test_ml_portfolio_manager_phase3d_detail_audit.py \
  tests/test_ml_portfolio_manager_phase3e.py \
  tests/test_ml_portfolio_manager_phase3f_drawdown_audit.py \
  tests/test_ml_portfolio_manager_phase3g_per_code_cap.py \
  tests/test_ml_portfolio_manager_phase3h_capital_utilization.py \
  tests/test_ml_portfolio_manager_phase3i_candidate_pool.py
```

Latest known test result:

```text
34 passed, 1 warning
```

## Next Good Tasks

Recommended next experiments:

1. Keep v2_77 cap 0.30 as the current balanced research candidate.
2. Do not continue candidate-pool expansion unless the upstream candidate
   shortage definition changes.
3. Try utilization-improvement paths that do not dilute candidate quality:
   - low-score skip threshold tuning
   - replacement candidate filling after cap / affordability blocks
   - total-assets-linked `daily_buy_limit`
   - fallback quality improvement
4. Keep v2_75 and v2_73 as fallback references.
5. Continue monitoring top-code contribution and DD-period concentration.

Do not promote v2_76 directly without an exposure guard because its DD is too
large.

## Documentation Map

Use these documents:

- `docs/ml/README.md`: ML documentation index
- `docs/ml/ML_Phase_25_to_Portfolio_Manager_AI_Summary.md`: full recent history
- `docs/ml/v2_73_adoption_notes.md`: why v2_73 became the prior baseline
- `docs/ml/daily_ai_candidate_operation.md`: human-review daily AI candidates
