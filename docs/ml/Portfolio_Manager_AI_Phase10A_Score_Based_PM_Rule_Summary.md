# Portfolio Manager AI Phase 10-A Score-Based PM Rule Summary

作成日: 2026-06-09

## 目的

Phase 9では、PM AI v3 candidateもPM disabled equal-weight baselineも、現行チャンピオン `rookie_dealer_02_v2_82_cap38` を置換できなかった。

Phase 10-Aでは、新規学習やPM model推論を使わず、Stock Selection AIが予測時点で出しているスコアだけを使うルールベースPMを検証した。

使用featureは以下4つに限定した。

- `risk_adjusted_score`
- `expected_return`
- `stock_selection_rank_score`
- `candidate_strength`

バックテスト結果、売買結果、損益、cash、portfolio、position、selected/bought/affordable、exit/skip、final_assets、current PM multiplier模倣はfeatureとして使用していない。

## 追加Profile

- `rookie_dealer_02_v2_96_score_based_pm_rule_a`
- `rookie_dealer_02_v2_96b_score_based_pm_rule_b`
- `rookie_dealer_02_v2_96c_score_based_pm_rule_c`

alias:

- `rookie_dealer_02_v2_96`
- `rookie_dealer_02_v2.96`
- `rookie_dealer_02_v2_96b`
- `rookie_dealer_02_v2_96c`

## Rule Definitions

Rule A:

- 同日候補内 `risk_adjusted_score` percentile
- top 10%: PM1.30
- top 25%: PM1.15
- middle: PM1.00
- bottom 25%: PM0.80
- bottom 10%: PM0.60

Rule B:

- 同日候補内 `stock_selection_rank_score` percentile
- multiplier thresholdはRule Aと同じ

Rule C:

```text
composite_score =
0.35 * risk_adjusted_score_percentile
+ 0.30 * expected_return_percentile
+ 0.25 * stock_selection_rank_score_percentile
+ 0.10 * candidate_strength_percentile
```

PM multiplier:

- `composite_score >= 0.90`: PM1.30
- `composite_score >= 0.75`: PM1.15
- `composite_score >= 0.35`: PM1.00
- `composite_score >= 0.10`: PM0.80
- otherwise: PM0.60

## Backtest Result

期間: 2023-01-01 to 2026-05-31

v2_82は参考値であり、主比較はPM disabled baseline v2_95に対して行う。

| profile | net_profit | PF | DD | win_rate | monthly_win_rate | trades | final_assets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v2_82_reference` | 3,777,545 | 2.7309 | -6.54% | 55.11% | 78.05% | 502 | 5,720,597 |
| `v2_95_pm_disabled` | 801,879 | 1.4499 | -18.13% | 44.05% | 53.66% | 515 | 2,008,511 |
| `v2_96_rule_a` | 685,289 | 1.3971 | -10.62% | 42.23% | 58.54% | 523 | 1,855,998 |
| `v2_96b_rule_b` | 706,051 | 1.4018 | -13.62% | 44.67% | 46.34% | 499 | 1,888,052 |
| `v2_96c_rule_c` | 923,134 | 1.4837 | -11.72% | 45.00% | 58.54% | 522 | 2,160,679 |

## PM Disabled Baseline Comparison

| profile | net_profit_delta | PF_delta | DD_delta | candidate_gate |
| --- | ---: | ---: | ---: | --- |
| `v2_96_rule_a` | -116,590 | -0.0528 | +7.51pp | fail |
| `v2_96b_rule_b` | -95,828 | -0.0481 | +4.51pp | fail |
| `v2_96c_rule_c` | +121,255 | +0.0338 | +6.41pp | pass |

Rule Cだけが、v2_95 PM disabledに対してprofit/PF/DDの最低条件を満たした。

## PM Multiplier Quality

Rule CのPM別品質:

| PM | buy_count | trade_count | profit | PF | win_rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.30 | 218 | 217 | 434,260 | 1.4248 | 46.54% |
| 1.15 | 44 | 46 | -64,023 | 0.7632 | 26.09% |
| 1.00 | 231 | 233 | 11,136 | 1.0108 | 46.78% |
| 0.80 | 17 | 17 | -32,717 | 0.4222 | 29.41% |
| 0.60 | 8 | 9 | 87,895 | 7.9824 | 77.78% |

解釈:

- Rule Cはprofile全体ではv2_95を上回った。
- ただしPM0.60が非常に良く、PM1.15/0.80が悪いなど、倍率bucketの単調性は崩れている。
- このまま採用ではなく、Phase 10-Bでthreshold/bucket設計を再監査すべき。

## Adoption Decision

Phase 10-Aの採用判定:

- best profile: `v2_96c_rule_c`
- candidate gate: pass
- strong gate: fail
- recommendation: `candidate_for_review`
- next phase: Phase 10-B detailed robustness audit

強い採用候補条件は満たしていない。

- PF >= 2.0: fail
- max drawdown within -10%: fail (`-11.72%`)
- net_profit >= 2,500,000: fail
- win_rate >= 50%: fail
- monthly_win_rate >= 70%: fail

## Leakage Checklist

| item | result |
| --- | --- |
| allowed feature count | 4 |
| forbidden feature count | 0 |
| disallowed feature count | 0 |
| PM AI model used | false |
| PM AI v3 model used | false |
| current PM multiplier used | false |
| backtest results used as features | false |
| cash/portfolio used for score | false |
| leakage risk | low |

## Phase 9 Cleanup

削除前にdry-run manifestを出し、以下18パスを削除した。

- `data/ml/portfolio_manager_v3`
- `models/ml/portfolio_manager_v3`
- `logs/backtests/rookie_dealer_02_v2_93_pm_ai_v3_candidate`
- `logs/backtests/rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative`
- `logs/backtests/rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130`
- `logs/backtests/rookie_dealer_02_v2_94_pm_ai_v3_e139_candidate`
- `logs/backtests/rookie_dealer_02_v2_94b_pm_ai_v3_e140_candidate`
- `logs/backtests/rookie_dealer_02_v2_94c_pm_ai_v3_e120_candidate`
- `logs/backtests/rookie_dealer_02_v2_94d_pm_ai_v3_rank_score_candidate`
- `logs/backtests/rookie_dealer_02_v2_94e_pm_ai_v3_rank_downside_blend_candidate`
- `config/profiles/rookie_dealer_02_v2_93_pm_ai_v3_candidate.yaml`
- `config/profiles/rookie_dealer_02_v2_93b_pm_ai_v3_candidate_conservative.yaml`
- `config/profiles/rookie_dealer_02_v2_93c_pm_ai_v3_candidate_half_pm130.yaml`
- `config/profiles/rookie_dealer_02_v2_94_pm_ai_v3_e139_candidate.yaml`
- `config/profiles/rookie_dealer_02_v2_94b_pm_ai_v3_e140_candidate.yaml`
- `config/profiles/rookie_dealer_02_v2_94c_pm_ai_v3_e120_candidate.yaml`
- `config/profiles/rookie_dealer_02_v2_94d_pm_ai_v3_rank_score_candidate.yaml`
- `config/profiles/rookie_dealer_02_v2_94e_pm_ai_v3_rank_downside_blend_candidate.yaml`

容量:

- before: 500,684,097 bytes
- after: 0 bytes
- saved: 500,684,097 bytes

残したもの:

- Phase 9 reports
- Phase 9 audit scripts/tests
- v2_95 PM disabled baseline profile/logs
- v2_82 logs
- current PM AI
- current Exit AI
- Stock Selection AI

## Verification

```bash
PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest -q tests/test_ml_phase10a_score_based_pm_rule.py
```

Result:

- `5 passed in 0.19s`
