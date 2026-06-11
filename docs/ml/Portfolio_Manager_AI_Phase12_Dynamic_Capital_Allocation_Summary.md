# Portfolio Manager AI Phase 12 Dynamic Capital Allocation Summary

作成日: 2026-06-11

## Phase 12の目的

Phase 11では以下が確認された。

- Valuation Engineはcandidate qualityを改善する。
- Opportunity ExitはDD抑制に効く。
- Downside modelは大きな下落候補を検出できる。
- ただしstrict OOSでは、Opportunity単独ではdownsideを拾い、Downside単独ではopportunityを削りすぎる。

Phase 12では、以下を使って購入額を動的に決める。

```text
Opportunity
+
Downside
+
Confidence
```

目的は、候補を単純に入れ替えることではなく、Opportunityが高くDownsideが低い候補を大きく買い、Opportunityが高くてもDownsideが高い候補を小さく買うCapital Allocation Engineを研究することである。

## Phase 12-A Implementation Status

実装済み:

- `src/ml/phase12a_dynamic_capital_allocation.py`
- `scripts/ml/run_phase12a_dynamic_capital_allocation.py`
- `tests/test_ml_phase12a_dynamic_capital_allocation.py`

生成report / artifact:

- `reports/ml/phase12a_dynamic_capital_allocation_2025.md`
- `reports/ml/phase12a_dynamic_capital_allocation_2025.json`
- `data/ml/valuation_engine/phase12a_dynamic_capital_allocation_2025.parquet`

Scope:

- 2025年のみ
- Phase 11-A datasetを使用
- Phase 11-B3 research modelをread-onlyで使用
- allocation quality auditのみ
- strategy backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

Leakage:

| item | value |
| --- | --- |
| future_columns_used_as_features | `[]` |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

## Allocation Rules

比較対象:

- `baseline_equal_weight_top5`
- `opportunity_only_top5`
- `downside_safe_top5`
- `score_a_weighted`: `opportunity_proba * (1 - downside_bad_proba)`
- `score_b_weighted`: `opportunity_proba - downside_bad_proba`
- `score_c_weighted`: `opportunity_rank_percentile * (1 - downside_bad_proba)`
- `score_d_weighted`: `opportunity_proba * (1 - downside_rank_percentile)`
- `score_e_weighted`: `0.7 * opportunity_rank_percentile + 0.3 * (1 - downside_rank_percentile)`

Weighted rules use same-day percentile weights:

| percentile | weight |
| --- | ---: |
| `>= p95` | `1.00` |
| `>= p90` | `0.70` |
| `>= p80` | `0.40` |
| `>= p70` | `0.20` |
| otherwise | `0.00` |

## Phase 12-A Result

| rule | weighted_top_decile_rate | weighted_downside_bad_rate | weighted_opportunity_value | weighted_max_drawdown |
| --- | ---: | ---: | ---: | ---: |
| `baseline_equal_weight_top5` | `0.0885` | `0.1358` | `0.0185` | `-0.0514` |
| `downside_safe_top5` | `0.0436` | `0.0412` | `0.0103` | `-0.0241` |
| `opportunity_only_top5` | `0.2400` | `0.3794` | `0.0269` | `-0.1042` |
| `score_a_weighted` | `0.1537` | `0.2260` | `0.0282` | `-0.0724` |
| `score_b_weighted` | `0.0814` | `0.1209` | `0.0209` | `-0.0483` |
| `score_c_weighted` | `0.1374` | `0.1933` | `0.0267` | `-0.0655` |
| `score_d_weighted` | `0.0780` | `0.1166` | `0.0203` | `-0.0481` |
| `score_e_weighted` | `0.1349` | `0.1958` | `0.0267` | `-0.0659` |

## Interpretation

- Opportunity onlyはtop-decile rate `24.00%` を維持するが、downside bad rate `37.94%` が高すぎる。
- Downside safe onlyはdownside bad rate `4.12%` まで下げるが、top-decile rate `4.36%` まで機会を削る。
- `score_a_weighted` はdownside bad rateを `22.60%` まで下げるが、top-decile rateは `15.37%` で最低ライン `20%` 未達。
- `score_c_weighted` / `score_e_weighted` はopportunity valueを比較的残しつつdownsideを `20%` 前後に抑えるが、top-decile rateが不足。
- p70以上を広く拾うweight設計では候補が薄まり、Opportunity濃度を維持できない。

## Decision

Phase 12-Aの最低ライン:

```text
weighted_opportunity_top_decile_rate >= 0.20
weighted_downside_bad_rate <= 0.25
```

この条件を満たすruleはなかった。

判断:

- `ready_for_phase12b`: `false`
- `recommended_next_phase`: `Phase12-A2 allocation score refinement`

Phase 12-A2では、strategy backtestへ進まず、まず以下を軽量auditする。

- p80 / p90 / p95中心のより狭いweight threshold
- score_aとscore_cの中間weight
- opportunity minimum thresholdとdownside maximum thresholdの組み合わせ
- top5固定ではなくtopN候補に対するweight配分の再設計

## Phase 12-A2 Implementation Status

実装済み:

- `src/ml/phase12a2_allocation_score_refinement.py`
- `scripts/ml/run_phase12a2_allocation_score_refinement.py`
- `tests/test_ml_phase12a2_allocation_score_refinement.py`

生成report:

- `reports/ml/phase12a2_allocation_score_refinement_2025.md`
- `reports/ml/phase12a2_allocation_score_refinement_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- allocation score refinementのみ
- strategy backtestなし
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

Candidate universe:

- `opportunity_top5`
- `opportunity_top10`
- `opportunity_top20`
- `opportunity_p95`
- `opportunity_p90`

Penalty:

- `penalty_none`
- `penalty_soft`
- `penalty_medium`
- `penalty_rank_soft`
- `penalty_rank_medium`

## Phase 12-A2 Result

主要結果:

| rule | avg candidates/day | weighted_top_decile_rate | weighted_downside_bad_rate | weighted_opportunity_value | budget_usage_proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `opportunity_top5__penalty_none` | `5.0` | `0.2400` | `0.3794` | `0.0269` | `1.0000` |
| `opportunity_top5__penalty_rank_soft` | `5.0` | `0.2517` | `0.2970` | `0.0409` | `0.3261` |
| `opportunity_top5__penalty_rank_medium` | `5.0` | `0.2454` | `0.2664` | `0.0453` | `0.1907` |
| `opportunity_top10__penalty_rank_medium` | `10.0` | `0.2122` | `0.2721` | `0.0413` | `0.3916` |
| `opportunity_top20__penalty_rank_medium` | `20.0` | `0.1890` | `0.2673` | `0.0351` | `0.7194` |
| `opportunity_p95__penalty_rank_soft` | `94.7` | `0.1773` | `0.2452` | `0.0363` | `1.0000` |
| `opportunity_p90__penalty_medium` | `188.6` | `0.1724` | `0.2478` | `0.0309` | `1.0000` |

Minimum target:

```text
weighted_opportunity_top_decile_rate >= 0.20
weighted_downside_bad_rate <= 0.25
```

この条件を満たすruleはなかった。

Best rule:

```text
opportunity_top5__penalty_rank_medium
```

理由:

- top-decile rate `0.2454` でOpportunity濃度を維持した。
- downside bad rateを `0.3794` から `0.2664` まで下げた。
- average candidates/dayは `5.0` で過剰に広がっていない。
- weighted opportunity valueは `0.0453` と高い。

ただし、downside bad rate `0.2664` は最低ライン `0.25` に届かない。

Interpretation:

- A2の方向性は正しい。Opportunity top5を維持し、Downsideをrank penaltyで減額すると、Opportunityを残したままdownsideを大きく削れる。
- ただし、現行の`penalty_rank_medium`ではdownsideがまだ少し高い。
- `opportunity_p95` / `opportunity_p90` は一部downside targetを満たしそうだが、候補数が日次約95〜189件まで広がり、今回の目的から外れる。
- Phase 12-Bへ進む前に、top5/top10を中心にrank penaltyをもう少しだけ強めるA3が必要。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_as_features | `[]` |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| strategy_backtest_executed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `ready_for_phase12b`: `false`
- `recommended_next_phase`: `Phase12-A3 allocation refinement`

Phase 12-A3では、strategy backtestへ進まず、以下を軽量auditする。

- `opportunity_top5`固定
- stronger rank penalty
- hybrid rank/proba penalty
- top-decile rate `>= 0.20` を維持しながら downside_bad_rate `<= 0.25` を達成できるか

## Phase 12-A3 Implementation Status

実装済み:

- `src/ml/phase12a3_top5_penalty_refinement.py`
- `scripts/ml/run_phase12a3_top5_penalty_refinement.py`
- `tests/test_ml_phase12a3_top5_penalty_refinement.py`

生成report:

- `reports/ml/phase12a3_top5_penalty_refinement_2025.md`
- `reports/ml/phase12a3_top5_penalty_refinement_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- `opportunity_top5`固定
- downside penalty refinementのみ
- strategy backtestなし
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-A3 Result

比較対象:

| rule | weighted_top_decile_rate | weighted_downside_bad_rate | weighted_opportunity_value | average_weight | minimum_target |
| --- | ---: | ---: | ---: | ---: | --- |
| `A2_baseline_penalty_rank_medium` | `0.2454` | `0.2664` | `0.0453` | `0.1907` | false |
| `A3_1_rank_medium_plus` | `0.2386` | `0.2447` | `0.0445` | `0.1194` | true |
| `A3_2_rank_medium_stronger_tail` | `0.2466` | `0.2221` | `0.0523` | `0.1460` | true |
| `A3_3_rank_medium_floor_zero` | `0.2614` | `0.1432` | `0.0683` | `0.1168` | true |
| `A3_4_hybrid_rank_and_proba` | `0.2444` | `0.2621` | `0.0458` | `0.1882` | false |
| `A3_5_hybrid_rank_and_proba_strict` | `0.2461` | `0.2556` | `0.0470` | `0.1845` | false |

Minimum target:

```text
weighted_opportunity_top_decile_rate >= 0.20
weighted_downside_bad_rate <= 0.25
```

Minimum targetを満たしたrule:

- `A3_1_rank_medium_plus`
- `A3_2_rank_medium_stronger_tail`
- `A3_3_rank_medium_floor_zero`

Ideal target:

```text
weighted_opportunity_top_decile_rate >= 0.24
weighted_downside_bad_rate <= 0.20
```

Ideal targetを満たしたrule:

- `A3_3_rank_medium_floor_zero`

Best rule:

```text
A3_3_rank_medium_floor_zero
```

理由:

- weighted top-decile rate `0.2614` でOpportunity濃度を維持した。
- weighted downside bad rate `0.1432` まで下げた。
- weighted opportunity value `0.0683` が比較対象内で最も高い。
- average weight `0.1168` で、警告ライン `0.10` は下回らなかった。

Interpretation:

- A3により、Opportunity top5固定のままDownside penaltyだけで最低ラインを達成した。
- `A3_3_rank_medium_floor_zero` はtail downside rankにweight `0` を与えるため、budget usage proxyは低くなる可能性がある。
- これはBUY品質監査であり、strategy backtestではない。Phase 12-Bでは、実際の約定・単元株・cash・exitとの接続時に、低weight候補をどう扱うかを限定検証する必要がある。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_as_features | `[]` |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| strategy_backtest_executed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `ready_for_phase12b`: `true`
- `recommended_next_phase`: `Phase12-B limited allocation strategy check`

Phase 12-Bでは、フルバックテストではなく、2025年限定で`A3_3_rank_medium_floor_zero`を実売買ロジックへ最小接続する。

## Phase 12-B Implementation Status

実装済み:

- `src/ml/phase12b_limited_allocation_strategy_check.py`
- `scripts/ml/run_phase12b_limited_allocation_strategy_check.py`
- `tests/test_ml_phase12b_limited_allocation_strategy_check.py`

生成report:

- `reports/ml/phase12b_limited_allocation_strategy_check_2025.md`
- `reports/ml/phase12b_limited_allocation_strategy_check_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- 少数strategy比較のみ
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

Conditions:

```text
initial_cash = 1,000,000
daily_buy_budget = 900,000
max_positions = 5
holding_days = 20
round_lot = 100
cost_rate = 0.2% one-way
stop_loss = -8%
opportunity_exit = enabled for S2/S3
```

## Phase 12-B Result

Strategy results:

| strategy | net_profit | PF | DD | trades | utilization | cost_paid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `S0_baseline_equal_allocation` | `66,840` | `1.2506` | `-13.56%` | `60` | `0.8646` | `43,070` |
| `S1_opportunity_top5_equal` | `-58,049` | `0.9116` | `-24.26%` | `63` | `0.9007` | `46,049` |
| `S2_opportunity_top5_E4` | `7,724` | `1.0094` | `-23.86%` | `149` | `0.8771` | `114,536` |
| `S3a_dynamic_raw_weight` | `39,770` | `1.5971` | `-2.66%` | `56` | `0.1007` | `16,730` |
| `S3b_dynamic_normalized_weight` | `135,752` | `1.2712` | `-19.16%` | `75` | `0.7506` | `97,248` |

BUY quality:

| strategy | top_decile_rate | downside_bad_rate | opportunity_value | average_allocation_weight |
| --- | ---: | ---: | ---: | ---: |
| `S0_baseline_equal_allocation` | `0.1500` | `0.1333` | `0.0274` | `1.0000` |
| `S1_opportunity_top5_equal` | `0.2381` | `0.3333` | `0.0313` | `1.0000` |
| `S2_opportunity_top5_E4` | `0.2282` | `0.3423` | `0.0378` | `1.0000` |
| `S3a_dynamic_raw_weight` | `0.2321` | `0.1786` | `0.0616` | `0.5411` |
| `S3b_dynamic_normalized_weight` | `0.2533` | `0.1867` | `0.0650` | `0.4453` |

Interpretation:

- Dynamic allocation clearly improved BUY quality versus S2: downside_bad_rate fell from `0.3423` to `0.1786` / `0.1867`, and opportunity_value rose from `0.0378` to `0.0616` / `0.0650`.
- `S3a_dynamic_raw_weight` improved PF and DD versus S2, with PF `1.5971` and DD `-2.66%`, but capital utilization was only `0.1007` and net profit did not beat baseline.
- `S3b_dynamic_normalized_weight` produced the highest net profit `135,752`, but PF `1.2712` and DD `-19.16%` failed the Phase 12-B thresholds.
- Raw weighting is too conservative; normalized weighting over-concentrates risk. The next useful step is an execution adjustment between raw and normalized.

Leakage:

| item | value |
| --- | --- |
| future_columns_used_as_features | `[]` |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `best_strategy`: `S3a_dynamic_raw_weight`
- `dynamic_allocation_improved_vs_s2`: `true`
- `dynamic_allocation_improved_vs_baseline`: `false`
- `ready_for_phase12c`: `false`
- `recommended_next_phase`: `Phase12-B2 allocation execution adjustment`

Phase 12-B2では、strategy universeやmodelを増やさず、以下のような実行配分だけを少数比較する。

- raw weightとnormalized weightの中間
- minimum budget usage floor
- max allocation cap per candidate
- weight 0銘柄の扱いは維持
