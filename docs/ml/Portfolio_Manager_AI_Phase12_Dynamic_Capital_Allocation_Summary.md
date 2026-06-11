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

## Phase 12-B2 Implementation Status

実装済み:

- `src/ml/phase12b2_allocation_execution_adjustment.py`
- `scripts/ml/run_phase12b2_allocation_execution_adjustment.py`
- `tests/test_ml_phase12b2_allocation_execution_adjustment.py`

生成report:

- `reports/ml/phase12b2_allocation_execution_adjustment_2025.md`
- `reports/ml/phase12b2_allocation_execution_adjustment_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- execution adjustmentのみ
- 少数variantのみ
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-B2 Result

Strategy results:

| strategy | net_profit | PF | DD | capital_utilization | avg_daily_budget_usage |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S0_baseline_equal_allocation` | `66,840` | `1.2506` | `-13.56%` | `0.8646` | `0.0721` |
| `S2_opportunity_top5_E4` | `7,724` | `1.0094` | `-23.86%` | `0.8771` | `0.1924` |
| `S3a_dynamic_raw_weight` | `39,770` | `1.5971` | `-2.66%` | `0.1007` | `0.0280` |
| `S3b_dynamic_normalized_weight` | `135,752` | `1.2712` | `-19.16%` | `0.7506` | `0.1629` |
| `S4_partial_normalized_50` | `-4,321` | `0.9874` | `-15.65%` | `0.5169` | `0.1160` |
| `S5_partial_normalized_30` | `2,434` | `1.0094` | `-11.75%` | `0.3910` | `0.0784` |
| `S6_min_usage_guard_30pct` | `-127,207` | `0.6708` | `-18.12%` | `0.4485` | `0.0928` |
| `S7_min_usage_guard_50pct` | `-57,603` | `0.8556` | `-19.96%` | `0.6230` | `0.1172` |
| `S8_capped_normalized` | `-34,065` | `0.9120` | `-17.16%` | `0.5922` | `0.1162` |
| `S9_capped_normalized_tighter` | `-23,504` | `0.9258` | `-13.02%` | `0.4257` | `0.0940` |

BUY quality highlights:

| strategy | top_decile_rate | downside_bad_rate | opportunity_value | avg_target_buy_amount |
| --- | ---: | ---: | ---: | ---: |
| `S3a_dynamic_raw_weight` | `0.2321` | `0.1786` | `0.0616` | `93,237` |
| `S3b_dynamic_normalized_weight` | `0.2533` | `0.1867` | `0.0650` | `377,341` |
| `S5_partial_normalized_30` | `0.2169` | `0.2048` | `0.0393` | `173,247` |
| `S8_capped_normalized` | `0.1954` | `0.2414` | `0.0298` | `239,912` |

Minimum target:

```text
net_profit > 0
PF >= 1.5
DD >= -12%
capital_utilization >= 0.20
```

この条件を満たすstrategyはなかった。

Interpretation:

- `S3a_dynamic_raw_weight` はPF/DDが良いが、capital utilization `0.1007` で不足。
- `S3b_dynamic_normalized_weight` は利益最大だが、PF `1.2712` とDD `-19.16%` が不足。
- `S5_partial_normalized_30` はDD `-11.75%` とutilization `0.3910` は最低ラインに近いが、PF `1.0094` で失格。
- min usage guard / capped normalizedは、資金利用率を上げてもPFが悪化し、今回の形では有効ではなかった。

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

- `strategies_meeting_minimum_target`: none
- `strategies_meeting_ideal_target`: none
- `ready_for_phase12c`: `false`
- `recommended_next_phase`: `Phase12-B3 execution adjustment or Phase12-A4 risk score refinement`

Phase 12-B2では、単純なraw/normalized中間やusage floorでは解決しなかった。次は以下のどちらかを検討する。

- B3: entry churn / Opportunity Exitの組み合わせを調整して、Dynamic Allocationの高品質BUYを利益化できるか確認する。
- A4: Downside penaltyだけでなく、Exit耐性や短期反転リスクを含むrisk scoreを再設計する。

## Phase 12-B3 Implementation Status

実装済み:

- `src/ml/phase12b3_exit_hold_audit.py`
- `scripts/ml/run_phase12b3_exit_hold_audit.py`
- `tests/test_ml_phase12b3_exit_hold_audit.py`

生成report:

- `reports/ml/phase12b3_exit_hold_audit_2025.md`
- `reports/ml/phase12b3_exit_hold_audit_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- exit / hold decision auditのみ
- 対象strategyは`S3a_dynamic_raw_weight`と比較用`S2_opportunity_top5_E4`
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は監査用contextのみ

## Phase 12-B3 Result

Recommendation:

| item | value |
| --- | --- |
| `main_exit_problem` | `late_exit` |
| `early_exit_detected` | `true` |
| `late_exit_detected` | `true` |
| `opportunity_exit_effective` | `false` |
| `recommended_exit_improvement` | `Prototype trailing exit or faster decay guard before stop-loss.` |
| `recommended_next_phase` | `Phase12-B4 trailing_exit_prototype` |

Trade summary:

| strategy | trades | avg_realized_return | win_rate | main exit reasons |
| --- | ---: | ---: | ---: | --- |
| `S2_opportunity_top5_E4` | `149` | `0.0040` | `0.4497` | opportunity_proba_drop `99`, stop_loss `24`, time_exit_20d `20` |
| `S3a_dynamic_raw_weight` | `56` | `0.0044` | `0.5000` | opportunity_proba_drop `46`, stop_loss `5`, time_exit_20d `2` |

Early Exit:

| strategy | avg_post_exit_20d_return | p90_post_exit_20d_return | count_post_exit_10pct_plus |
| --- | ---: | ---: | ---: |
| `S2_opportunity_top5_E4` | `0.0799` | `0.1809` | `46` |
| `S3a_dynamic_raw_weight` | `0.0653` | `0.1657` | `13` |

Late Exit:

| strategy | late_exit_trades | avg_realized_return | avg_max_profit_before_exit | avg_profit_decay_before_exit |
| --- | ---: | ---: | ---: | ---: |
| `S2_opportunity_top5_E4` | `24` | `-0.1078` | `0.0154` | `0.1232` |
| `S3a_dynamic_raw_weight` | `5` | `-0.1228` | `0.0476` | `0.1703` |

Opportunity Exit Quality:

| strategy | opportunity_exit_count | avg_realized_return | avg_post_exit_20d_return | p90_post_exit_20d_return | effective |
| --- | ---: | ---: | ---: | ---: | --- |
| `S2_opportunity_top5_E4` | `100` | `0.0199` | `0.0768` | `0.1685` | false |
| `S3a_dynamic_raw_weight` | `47` | `0.0134` | `0.0577` | `0.1501` | false |

Hold Candidate:

| strategy | time_exit_trades | avg_extra_return_after_20d | p90_extra_return_after_20d |
| --- | ---: | ---: | ---: |
| `S2_opportunity_top5_E4` | `20` | `0.0805` | `0.1286` |
| `S3a_dynamic_raw_weight` | `2` | `0.0722` | `0.0723` |

Interpretation:

- S3aはBUY品質は良いが、exit後20営業日で平均`+6.53%`の伸びを残しており、早売りも存在する。
- stop_loss系のS3a tradeは平均`-12.28%`で終わる前に平均`+4.76%`の含み益があり、平均profit decayは`17.03%`と大きい。
- Opportunity ExitはS3aで47件発生し、exit後20営業日に平均`+5.77%`残しているため、現閾値は有効とは言い切れない。
- ただし主問題としては、含み益からstop_lossまで落とすLate Exitの損失が大きい。次はtrailing exit / faster decay guardを限定prototypeするのが自然。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_audit | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `recommended_next_phase`: `Phase12-B4 trailing_exit_prototype`
- フルバックテストではなく、まず2025年限定でS3aにtrailing exit / profit decay guardを少数追加する。

## Phase 12-B4 Implementation Status

実装済み:

- `src/ml/phase12b4_trailing_exit_prototype.py`
- `scripts/ml/run_phase12b4_trailing_exit_prototype.py`
- `tests/test_ml_phase12b4_trailing_exit_prototype.py`

生成report:

- `reports/ml/phase12b4_trailing_exit_prototype_2025.md`
- `reports/ml/phase12b4_trailing_exit_prototype_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- S3a Dynamic Raw WeightのBUY/allocationは固定
- exit variantのみ少数比較
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-B4 Result

Variant results:

| variant | net_profit | PF | DD | capital_utilization | avg_holding_days | exit reasons |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `T0_current_opportunity_plus_stop` | `39,770` | `1.5971` | `-2.66%` | `0.1007` | `6.20` | opportunity_exit `47`, stop_loss `5` |
| `T1_stop_loss_only` | `14,489` | `1.1587` | `-6.83%` | `0.1856` | `17.49` | time_exit_20d `24`, stop_loss `8` |
| `T2_trailing_5pct` | `-7,503` | `0.9111` | `-5.52%` | `0.1662` | `13.66` | trailing_exit `23`, stop_loss `5` |
| `T3_trailing_8pct` | `9,149` | `1.1017` | `-5.50%` | `0.1806` | `16.84` | trailing_exit `6`, time_exit_20d `20` |
| `T4_trailing_10pct` | `15,387` | `1.1702` | `-6.74%` | `0.1854` | `17.46` | time_exit_20d `24`, stop_loss `7` |
| `T5_opportunity_plus_trailing_8pct` | `43,962` | `1.7044` | `-2.50%` | `0.0998` | `6.09` | opportunity_exit `47`, stop_loss `3`, trailing_exit `2` |

Hold quality:

| variant | avg_profit_capture | avg_max_profit_before_exit | profit_capture_ratio |
| --- | ---: | ---: | ---: |
| `T0_current_opportunity_plus_stop` | `0.0044` | `0.0386` | `-0.4070` |
| `T1_stop_loss_only` | `0.0105` | `0.0741` | `-0.6280` |
| `T3_trailing_8pct` | `0.0081` | `0.0738` | `-0.7800` |
| `T5_opportunity_plus_trailing_8pct` | `0.0066` | `0.0386` | `-0.3633` |

Interpretation:

- Opportunity Exitを無効化すると保有期間と利用率は上がるが、PFが`1.1587`まで落ち、利益も伸びない。
- Trailing単独は5%/8%/10%いずれもT0を上回らない。
- `T5_opportunity_plus_trailing_8pct`はT0よりnet profit、PF、DDを少し改善したが、capital utilizationは`0.0998`で改善せず、最低ライン未達。
- Trailing Exitはstop_loss件数を減らす補助にはなるが、Opportunity Exitの早さが支配的で、保有期間延長にはつながらなかった。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `best_variant`: `T5_opportunity_plus_trailing_8pct`
- `trailing_exit_improved_vs_opportunity_exit`: `false`
- `ready_for_phase12c`: `false`
- `recommended_next_phase`: `Phase12-B5 exit threshold recalibration`

Phase 12-B4の結果から、利益を伸ばすにはTrailing単独ではなく、Opportunity Exitの発火条件を遅らせる/緩める検証が必要。

## Phase 12-B5 Implementation Status

実装済み:

- `src/ml/phase12b5_exit_threshold_recalibration.py`
- `scripts/ml/run_phase12b5_exit_threshold_recalibration.py`
- `tests/test_ml_phase12b5_exit_threshold_recalibration.py`

生成report:

- `reports/ml/phase12b5_exit_threshold_recalibration_2025.md`
- `reports/ml/phase12b5_exit_threshold_recalibration_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- S3a Dynamic Raw WeightのBUY/allocationは固定
- stop_loss `-8%` は固定
- Opportunity Exit条件だけを少数比較
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-B5 Result

Variant results:

| variant | net_profit | PF | DD | capital_utilization | avg_holding_days | opportunity_exit | stop_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B5_0_baseline` | `39,770` | `1.5971` | `-2.66%` | `0.1007` | `6.20` | `47` | `5` |
| `B5_1_rank_floor_lower` | `39,770` | `1.5971` | `-2.66%` | `0.1007` | `6.20` | `47` | `5` |
| `B5_2_proba_drop_larger` | `71,922` | `2.1827` | `-3.24%` | `0.1613` | `13.34` | `19` | `4` |
| `B5_3_both_relaxed` | `71,450` | `2.0482` | `-4.02%` | `0.1755` | `14.15` | `17` | `5` |
| `B5_4_confirmation_2days` | `29,107` | `1.3509` | `-3.41%` | `0.1466` | `10.85` | `29` | `5` |
| `B5_5_confirmation_3days` | `7,287` | `1.0853` | `-4.71%` | `0.1594` | `14.18` | `13` | `7` |
| `B5_6_profit_protected_exit` | `15,868` | `1.2099` | `-3.75%` | `0.1310` | `8.71` | `30` | `8` |
| `B5_7_loss_only_hard_exit` | `9,747` | `1.1021` | `-6.70%` | `0.1773` | `17.11` | `5` | `7` |

Hold / Exit quality:

| variant | avg_profit_capture | avg_max_profit_before_exit | avg_post_exit_20d_return |
| --- | ---: | ---: | ---: |
| `B5_0_baseline` | `0.0044` | `0.0386` | `0.0653` |
| `B5_2_proba_drop_larger` | `0.0275` | `0.0724` | `0.0782` |
| `B5_3_both_relaxed` | `0.0280` | `0.0749` | `0.0821` |

Minimum target:

```text
PF >= 1.5
DD >= -12%
net_profit > 0
capital_utilization > B5-0 baseline
average_holding_days > B5-0 baseline
```

Minimum targetを満たしたvariant:

- `B5_2_proba_drop_larger`
- `B5_3_both_relaxed`

Best variant:

```text
B5_2_proba_drop_larger
```

理由:

- net_profitが`39,770`から`71,922`へ改善。
- PFが`1.5971`から`2.1827`へ改善。
- DDは`-3.24%`で十分低い。
- capital utilizationが`0.1007`から`0.1613`へ改善。
- average holding daysが`6.20`から`13.34`へ伸びた。
- opportunity_exit_countが`47`から`19`へ減り、早売りを抑えられた。

Interpretation:

- rank floorを`0.50`から`0.30`へ下げても結果は変わらなかった。exit発火の主因はrank floorではなくproba drop threshold。
- proba drop thresholdを`0.15`から`0.30`へ広げると、保有期間・利用率・PF・利益が同時に改善した。
- 2日/3日confirmationは保有期間を伸ばすがPFが落ちた。
- profit protected / loss only hard exitは、Exitを遅らせすぎて利益化に失敗した。
- B5により、Opportunity Exitは「少し弱くなったら即売り」ではなく「大きく崩れたら売り」に寄せる方が良い可能性が高い。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| backtest_columns_used_as_features | `[]` |
| trade_result_columns_used_as_features | `[]` |
| cash_or_portfolio_columns_used_as_model_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `best_variant`: `B5_2_proba_drop_larger`
- `variants_meeting_minimum_target`: `B5_2_proba_drop_larger`, `B5_3_both_relaxed`
- `ready_for_phase12c`: `true`
- `recommended_next_phase`: `Phase12-C dynamic allocation + recalibrated exit`

Phase 12-Cでは、フルバックテストではなく、2025年限定で以下を統合評価する。

```text
A3_3 Dynamic Allocation
+
B5_2 recalibrated Opportunity Exit
```

## Phase 12-C Implementation Status

実装済み:

- `src/ml/phase12c_dynamic_allocation_recalibrated_exit.py`
- `scripts/ml/run_phase12c_dynamic_allocation_recalibrated_exit.py`
- `tests/test_ml_phase12c_dynamic_allocation_recalibrated_exit.py`

生成report:

- `reports/ml/phase12c_dynamic_allocation_recalibrated_exit_2025.md`
- `reports/ml/phase12c_dynamic_allocation_recalibrated_exit_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- B5_2 recalibrated Opportunity Exitを固定
- Dynamic Allocation実行方式を少数比較
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-C Result

Strategy results:

| strategy | net_profit | PF | DD | capital_utilization | avg_holding_days | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `C0_baseline_equal_allocation` | `138,402` | `1.6472` | `-9.78%` | `0.9138` | `19.09` | `64` |
| `C1_dynamic_raw_B5_2_exit` | `71,922` | `2.1827` | `-3.24%` | `0.1613` | `13.34` | `41` |
| `C2_dynamic_normalized_B5_2_exit` | `306,382` | `2.0680` | `-18.88%` | `0.9076` | `14.64` | `44` |
| `C3_partial_normalized_30_B5_2_exit` | `113,140` | `1.4931` | `-12.36%` | `0.5041` | `14.17` | `54` |
| `C4_partial_normalized_50_B5_2_exit` | `74,741` | `1.2070` | `-19.15%` | `0.6343` | `15.23` | `53` |

Minimum target:

```text
PF >= 1.8
DD >= -10%
net_profit > 0
capital_utilization >= 0.20
```

Minimum targetを満たしたstrategy:

```text
none
```

Best strategy by score:

```text
C2_dynamic_normalized_B5_2_exit
```

ただし、C2は利益とPFは高いがDDが`-18.88%`まで悪化し、最低ラインを満たさない。

Interpretation:

- B5_2 Exitを統合しても、raw dynamic allocationはPF/DDが良い一方で利用率が`16.13%`に留まる。
- normalized allocationは利益と利用率を大きく上げるが、DDが`-18.88%`まで悪化する。
- partial normalized 30/50はrawとnormalizedの中間を狙ったが、PFまたはDDが最低ライン未達。
- 利用率だけを引き上げるとDDが崩れる構造はPhase 12-B/B2と同じで、Exit改善だけでは解消しなかった。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `best_strategy`: `C2_dynamic_normalized_B5_2_exit`
- `capital_utilization_improved`: `false`
- `pf_improved`: `true`
- `dd_improved`: `false`
- `ready_for_phase13`: `false`
- `recommended_next_phase`: `Phase12-C2 allocation utilization refinement`

Phase 12-Cでは、Dynamic Allocation + Recalibrated Exitの統合による決定的改善は確認できなかった。Phase 13へ進まず、次は利用率を上げてもDDを壊さない実行制約を検討する。

## Phase 12-C2 Implementation Status

実装済み:

- `src/ml/phase12c2_utilization_without_dd_explosion.py`
- `scripts/ml/run_phase12c2_utilization_without_dd_explosion.py`
- `tests/test_ml_phase12c2_utilization_without_dd_explosion.py`

生成report:

- `reports/ml/phase12c2_utilization_without_dd_explosion_2025.md`
- `reports/ml/phase12c2_utilization_without_dd_explosion_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- C2 normalized + B5_2 ExitをbaseにDD attributionを監査
- normalized系の少数variantのみ比較
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-C2 Result

Base DD attribution:

| item | value |
| --- | ---: |
| `main_dd_cause` | `single_name_concentration` |
| `max_concurrent_positions` | `5` |
| `largest_position_weight_mean` | `0.5049` |
| `largest_position_weight_p90` | `0.7585` |
| `largest_position_weight_max` | `0.8011` |
| `top2_weight_mean` | `0.7565` |
| `top3_weight_mean` | `0.8230` |
| `largest_within_invested_weight_mean` | `0.6035` |
| `loss_contribution_pct_top20` | `1.0000` |

Downside exposure:

| bucket | trades | avg_buy_amount | avg_realized_return | loss_rate | avg_normalized_weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| `downside_proba_lt_0.40` | `44` | `315,016` | `0.0226` | `0.4545` | `0.3991` |

Interpretation:

- C2のDD悪化は、`downside_bad_proba >= 0.40`銘柄へ大きく張ったことが主因ではない。
- 実際のtradeはすべて`downside_bad_proba < 0.40` bucketに入り、downside model上は極端に危険な候補ではなかった。
- 一方でnormalized allocationが候補数の少ない日やweight分布の偏った日に1銘柄へ大きく寄せ、largest position weightが最大`80.11%`まで上がった。
- top2平均weightも`75.65%`で、DDは「高downside銘柄」より「normalizedによる単一/少数銘柄集中」が主因。

Variant results:

| variant | net_profit | PF | DD | utilization | avg_holding_days |
| --- | ---: | ---: | ---: | ---: | ---: |
| `C2_base_dynamic_normalized_B5_2_exit` | `306,382` | `2.0680` | `-18.88%` | `0.9076` | `14.64` |
| `C2a_normalized_cap_20pct` | `132,958` | `1.4752` | `-17.99%` | `0.5799` | `15.50` |
| `C2b_normalized_cap_15pct` | `103,392` | `1.4541` | `-14.99%` | `0.4242` | `14.76` |
| `C2c_normalized_downside_penalty_squared` | `315,227` | `2.1172` | `-18.26%` | `0.9157` | `14.89` |
| `C2d_normalized_top_weight_cap_30pct` | `64,548` | `1.2234` | `-16.19%` | `0.5064` | `14.47` |
| `C2e_normalized_cash_reserve_80pct` | `118,114` | `1.3973` | `-21.08%` | `0.6152` | `14.16` |

Minimum target:

```text
PF >= 1.8
DD >= -12%
utilization >= 0.50
```

Minimum targetを満たしたvariant:

```text
none
```

Best variant:

```text
C2c_normalized_downside_penalty_squared
```

ただし、C2cはnet profit/PF/utilizationを改善したものの、DDは`-18.26%`で最低ライン未達。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `main_dd_cause`: `single_name_concentration`
- `best_variant`: `C2c_normalized_downside_penalty_squared`
- `ready_for_phase13`: `false`
- `recommended_next_phase`: `Phase12-C3 DD guard refinement`

Phase 12-C2では、利用率を維持したままDDを`-12%`以内へ抑える候補は見つからなかった。次はdownside penaltyではなく、1銘柄集中を直接抑えるposition concentration guard / rebalancing capを検証する。

## Phase 12-C3 Implementation Status

実装済み:

- `src/ml/phase12c3_position_concentration_guard.py`
- `scripts/ml/run_phase12c3_position_concentration_guard.py`
- `tests/test_ml_phase12c3_position_concentration_guard.py`

生成report:

- `reports/ml/phase12c3_position_concentration_guard_2025.md`
- `reports/ml/phase12c3_position_concentration_guard_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- C2c normalized downside-squared allocation + B5_2 Exitをbaseに固定
- position concentration guardを少数比較
- cap余り資金は原則再配分しない
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-C3 Result

Variant results:

| variant | net_profit | PF | DD | utilization | largest max | top2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `C3_0_baseline_downside_squared` | `315,227` | `2.1172` | `-18.26%` | `0.9157` | `0.8011` | `0.7556` |
| `C3_1_per_name_cap_40pct` | `106,084` | `1.2357` | `-26.94%` | `0.7213` | `0.4951` | `0.6446` |
| `C3_2_per_name_cap_30pct` | `53,101` | `1.1459` | `-19.96%` | `0.6817` | `0.3470` | `0.4920` |
| `C3_3_per_name_cap_25pct` | `91,652` | `1.2799` | `-19.56%` | `0.6216` | `0.2978` | `0.4179` |
| `C3_4_top2_cap_60pct` | `247,206` | `1.6596` | `-23.96%` | `0.8345` | `0.6466` | `0.6948` |
| `C3_5_per_name_30pct_and_top2_60pct` | `53,101` | `1.1459` | `-19.96%` | `0.6817` | `0.3470` | `0.4920` |
| `C3_6_concentration_scaled` | `16,113` | `1.0456` | `-20.40%` | `0.6083` | `0.4644` | `0.4760` |

Minimum target:

```text
PF >= 1.8
DD >= -12%
utilization >= 0.50
net_profit > 0
```

Minimum targetを満たしたvariant:

```text
none
```

Concentration reduction:

- baseline largest position max: `80.11%`
- best concentration variant: `C3_3_per_name_cap_25pct`
- best largest position max: `29.78%`
- baseline top2 mean: `75.56%`
- best top2 mean: `41.79%`

Interpretation:

- per-name capは集中度を大きく下げられる。
- しかし、cap余りを再配分しない単純capではPFが大きく落ち、DDも`-12%`以内へ改善しなかった。
- `C3_1_per_name_cap_40pct`は集中を減らしたにもかかわらずDDが`-26.94%`まで悪化し、capだけではDD guardとして不十分。
- `C3_4_top2_cap_60pct`は利益を比較的残したが、DDは`-23.96%`で悪化。
- concentration_scaledは利用率を維持しつつ張り過ぎを抑える狙いだったが、利益/PFが崩れた。
- Phase 12-C3では「normalizedの集中がDD原因」という診断は維持されたが、「単純な集中cap」で解ける問題ではないことが分かった。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `best_variant`: `C3_0_baseline_downside_squared`
- `variants_meeting_minimum_target`: none
- `ready_for_phase13`: `false`
- `recommended_next_phase`: `Phase12-C4 concentration guard refinement`

Phase 12-C3では、1銘柄集中を抑えることでDDを改善できるとは確認できなかった。次は単純capではなく、cap後の候補補充、同時保有の相関/同日entry制限、またはnormalized自体を日次ではなくportfolio-levelで制御する方向を検討する。

## Phase 12-C4 Implementation Status

実装済み:

- `src/ml/phase12c4_concentration_guard_refinement.py`
- `scripts/ml/run_phase12c4_concentration_guard_refinement.py`
- `tests/test_ml_phase12c4_concentration_guard_refinement.py`

生成report:

- `reports/ml/phase12c4_concentration_guard_refinement_2025.md`
- `reports/ml/phase12c4_concentration_guard_refinement_2025.json`

Scope:

- Phase 12-A artifactを使用
- 2025年のみ
- C2c normalized downside-squared allocation + B5_2 Exitをbaseに固定
- cap後redistribution、score gap dynamic cap、staged buy proxyを少数比較
- full backtestなし
- 既存model上書きなし
- profile追加/変更なし
- historical prediction再生成なし
- future系は評価指標のみ

## Phase 12-C4 Result

Variant results:

| variant | net_profit | PF | DD | utilization | largest max | top2 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `C4_0_baseline_downside_squared` | `315,227` | `2.1172` | `-18.26%` | `0.9157` | `0.8011` | `0.7556` |
| `C4_1_cap_40_redistribute` | `68,939` | `1.1527` | `-26.94%` | `0.7228` | `0.4951` | `0.6490` |
| `C4_2_cap_35_redistribute` | `-114,807` | `0.7319` | `-28.02%` | `0.6429` | `0.4173` | `0.6191` |
| `C4_3_cap_30_redistribute` | `4,564` | `1.0115` | `-24.82%` | `0.6608` | `0.3518` | `0.5113` |
| `C4_4_dynamic_cap_by_score_gap` | `292,483` | `1.7602` | `-25.78%` | `0.8163` | `0.5576` | `0.6828` |
| `C4_5_staged_buy_half_first` | `60,236` | `1.1699` | `-19.62%` | `0.5905` | `0.4644` | `0.4838` |
| `C4_6_staged_buy_70pct_first` | `265,290` | `1.7452` | `-20.47%` | `0.7630` | `0.6386` | `0.5751` |

Minimum target:

```text
PF >= 1.8
DD >= -12%
utilization >= 0.50
net_profit > 0
```

Minimum targetを満たしたvariant:

```text
none
```

Effect summary:

- cap redistributionは集中を下げたが、利益/PF/DDが大きく悪化した。
- dynamic cap by score gapは利益を`292,483`まで残したが、PF `1.7602`、DD `-25.78%`で最低ライン未達。
- staged buy 70% proxyは利用率`0.7630`と利益`265,290`を残したが、PF `1.7452`、DD `-20.47%`で未達。
- staged buy 50% proxyは集中を下げたが、利益/PFが大きく低下した。

Interpretation:

- C4でも、集中を落とすだけではDDは改善しなかった。
- 良い集中と悪い集中を日次allocation時点で完全には区別できていない。
- ただし`C4_4` / `C4_6`は利益を比較的残せたため、単純capよりは「強い候補への露出を残す」方向性がある。
- DDは個別position capより、portfolio-levelのリスク状態、同時被弾、含み損拡大時の新規買い停止/縮小などで制御する方が自然。

Leakage:

| item | value |
| --- | --- |
| future_columns_used_only_for_evaluation | `future_return_20d`, `future_max_return_20d`, `future_max_drawdown_20d`, `opportunity_value_20d`, `opportunity_top_decile_20d`, `downside_bad_20d` |
| future_columns_used_as_features | `[]` |
| existing_model_overwritten | `false` |
| profile_changed | `false` |
| full_backtest_executed | `false` |
| leakage_risk | `low` |
| blocking_issues | `0` |

Decision:

- `best_variant`: `C4_0_baseline_downside_squared`
- `variants_meeting_minimum_target`: none
- `ready_for_phase13`: `false`
- `recommended_next_phase`: `Phase12-C5 portfolio-level risk gate`

Phase 12-C4では、集中を落としつつ良い露出を残すvariantは見つからなかった。次はposition単位のcapではなく、portfolio-levelのDD/risk gateを2025年限定で検証する。
