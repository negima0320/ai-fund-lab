# v2_73 暫定本命 Profile Adoption Notes

このメモは `rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue` を、今後の検証・運用準備における暫定本命profileとして扱う理由と、監視すべきリスクを整理するものです。

## Scope

- 対象profile: `rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue`
- 評価期間: `2023-01-01` to `2026-05-31`
- 主な参照レポート:
  - `reports/ml/ml_exit_ai_backtest_comparison_2023-01_to_2026-05.md`
  - `reports/ml/scaled_buy_backtest_comparison_2023-01_to_2026-05.md`
  - `reports/ml/capital_allocation_phase5_v2_73_comparison_2023-01_to_2026-05.md`
  - `reports/ml/capital_allocation_phase7_affordable_fallback_2023-01_to_2026-05.md`
  - `reports/ml/capital_allocation_phase8_fallback_filter_2023-01_to_2026-05.md`
  - `reports/ml/position_sizing_phase2_soft_rules_2023-01_to_2026-05.md`
- Historical prediction source: `data/ml/walk_forward_predictions/predictions_YYYY-MM-DD.parquet`
- この整理では売買ロジック、profile、バックテスト結果は変更していない。

## Improvement History

| profile | 主な変更 | net_profit | PF | DD | trades |
|---|---|---:|---:|---:|---:|
| `v2_65` | MLなし既存基準 | 491,656 | 1.3016 | -27.81% | 422 |
| `v2_66_ml_ranked` | 既存候補をML `risk_adjusted_score` で順位付け | 904,164 | 1.4808 | -23.58% | 441 |
| `v2_68_ml_ranked_exit_ai_050` | v2_66 + Exit AI threshold 0.50 | 930,204 | 1.4798 | -15.30% | 453 |
| `v2_71_ml_ranked_exit_ai_050_scaled_buy` | v2_68 + daily buy limit超過時の縮小買付 | 1,502,024 | 1.7053 | -16.14% | 486 |
| `v2_73_ml_ranked_exit_ai_050_scaled_buy_continue` | v2_71 + 監査可能な継続候補処理 | 1,502,024 | 1.7053 | -16.14% | 486 |

v2_73 は、v2_71 と同等の成績を維持しながら、購入監査ログを残せる形に整理されたprofileである。利益だけでなく、PF/DD/監査性のバランスが最も良い。

## Why v2_73 Is the Tentative Main Profile

### AI Ranking

v2_66 で導入したML rankingは、既存候補抽出を壊さずに候補順位を改善した。

```text
risk_adjusted_score = expected_return_10d - 0.5 * bad_entry_probability_10d
```

v2_65からv2_66への改善は大きく、net_profitは `491,656` から `904,164` に増加し、PFも `1.3016` から `1.4808` に改善した。

### Exit AI

v2_68では `avoid_loss_5d_probability >= 0.50` を売り側に加えた。利益改善幅は大きくないが、DD改善に寄与した。

- v2_66 DD: `-23.58%`
- v2_68 DD: `-15.30%`

Exit AI単体には大勝ち月を削る副作用もあったため、単独で過信せず、資金配分・買付制御とセットで扱う必要がある。

### Scaled Buy

v2_71で、日次買付上限を超える注文を丸ごとREJECTせず、上限内に収まる最大100株単位へ縮小する `scaled buy` を追加した。

これにより、v2_68から大きく改善した。

- v2_68 net_profit: `930,204`
- v2_71 net_profit: `1,502,024`
- v2_68 PF: `1.4798`
- v2_71 PF: `1.7053`
- v2_68 DD: `-15.30%`
- v2_71 DD: `-16.14%`

DDはわずかに悪化したが、利益とPFの改善が大きい。

### Purchase Audit

v2_73は、v2_71相当の強さを維持しつつ `purchase_audit.csv` によって候補ごとの判断を追える。

監査上重要な列:

- `signal_date`
- `code`
- `candidate_rank`
- `score_rank`
- `risk_adjusted_score`
- `expected_return_10d`
- `bad_entry_probability_10d`
- `planned_shares`
- `planned_amount`
- `scaled_shares`
- `scaled_amount`
- `final_shares`
- `final_amount`
- `decision`
- `skip_reason`
- `scale_reason`

この監査性があるため、今後の資金配分AIやfallback改善の基準profileとして扱いやすい。

### PF/DD Balance

v2_74 は利益だけなら v2_73を上回るが、fallbackが増えすぎてPF/DDが悪化した。

| profile | net_profit | PF | DD | capital_utilization | average_holding_count |
|---|---:|---:|---:|---:|---:|
| `v2_73` | 1,502,024 | 1.7053 | -16.14% | 47.12% | 1.57 |
| `v2_74` | 1,743,282 | 1.5617 | -22.38% | 66.27% | 3.10 |

v2_73は利益、PF、DD、資金利用率、保有数のバランスが良い。暫定本命としては、v2_74より安定性を優先する。

## Rejected or Deferred Candidates

### v2_74: 利益は高いがPF/DD悪化

v2_74 は affordability-aware fallback により net_profit は `1,743,282` まで伸びたが、fallback単体PFは薄く、DDも悪化した。

Phase 8のfallback絞り込みでは:

- `max_fallback_buys_per_day = 1` は net_profit `1,619,798` まで改善したが、PFは `1.5809` に留まった。
- `expected_return_10d >= 0.02 and bad_entry <= 0.70` はDDを `-13.97%` まで改善したが、net_profitは `888,893` まで低下した。
- `risk_adjusted_score >= 0.05` はfallbackがほぼ消え、実質v2_73と同等になった。

fallbackは今後改善余地があるが、現時点ではv2_73を置き換えるほどではない。

### v2_72: 保守的すぎ

v2_72 はAI前提のrank別 30/20/10% 配分や候補継続を導入したが、保守的すぎて資金利用率が低下し、利益/PFが悪化した。

- v2_72 net_profit: `617,631`
- v2_72 PF: `1.3845`
- v2_72 DD: `-13.40%`

DDは良いが、利益を削りすぎる。

### Position Sizing: 現状はDD悪化または利益削りすぎ

Position Sizing Phase 1/2 では、事後的にnet_profitへ倍率を掛ける検証を行った。

v2_73の主な結果:

| rule | profit_delta | PF | DD |
|---|---:|---:|---:|
| `bad_entry_defensive_soft` | +53,050 | 1.3528 | -30.95% |
| `bad_entry_defensive_very_soft` | +25,967 | 1.3555 | -30.06% |
| `expected_return_soft` | -94,215 | 1.3962 | -22.93% |
| `combined_soft` | -98,928 | 1.3941 | -22.93% |

利益改善するルールはDDが悪化し、PF/DD改善するルールは利益を削りすぎる。現時点では本格バックテストへ進める強い根拠はない。

## Monitoring Items

v2_73を暫定本命として扱う場合、最低限以下を継続監視する。

### Monthly Stability

- 月次PF
- 月次DD
- 月次net_profit
- 月次win_rate
- losing months
- worst month / best month

### Concentration

- `67400` のような上位寄与銘柄
- top1 trade contribution
- top3 trade contribution
- code別profit
- sector別profit

v2_73は大きな勝ちトレードへの依存がある。利益総額だけではなく、上位寄与を除いた成績も定期的に確認する。

### Scaled Buy

- scaled buy発動件数
- scaled buy発動損益
- scaled buy発動銘柄
- scaled buy発動月
- scaled buy excluding top winner の成績

scaled buyはv2_73の重要な改善要素だが、集中リスクもある。

### Cash / Affordability

- cash不足skip
- daily buy limit skip
- max_positions skip
- duplicate holding skip
- 買えなかった上位候補の事後成績

v2_73はv2_74ほどfallbackを使わないため、買い逃しの監視が重要。

### Purchase Audit

- `purchase_audit.csv` の出力有無
- `decision` 別件数
- `skip_reason` 別件数
- ML join成功率
- prediction missing件数
- `risk_adjusted_score` 帯別成績

purchase auditはv2_73を本命候補にする大きな理由なので、ログ欠損は運用上の異常として扱う。

## Future Improvement Candidates

### Fallback Quality Improvement

v2_74で利益は伸びたが、fallback単体PFが薄かった。今後は無条件fallbackではなく、以下のような質制御が必要。

- fallback candidate のML条件強化
- fallback後の月次DD監視
- fallback候補のsector concentration制限
- fallbackを市場環境別にON/OFF

### Portfolio Manager AI

Exit AIや買いAI単体ではなく、以下を統合して判断するportfolio manager AIが候補。

- cash
- open positions
- max positions
- daily buy limit
- candidate score
- current drawdown
- sector concentration
- existing holdingsのExit AI probability

### Position Sizing AI

単純倍率ルールはまだ弱い。次に試すなら、固定倍率ではなく制約付きの購入額推定が良い。

- 最大1銘柄投資額を制限
- 低bad_entry銘柄だけ小幅増額
- 高expected_returnでも67400依存を増やしすぎない制約
- monthly DD budgetを持つ

### Tachibana Test Environment

将来的に立花証券テスト環境へ接続する場合は、v2_73を直接発注に使う前に以下が必要。

- demo/test broker only
- live broker禁止設定の再確認
- order size dry-run
- daily buy limit dry-run
- purchase_auditと注文ログの突合
- 自動発注OFFでの日次候補確認

## Adoption Position

`rookie_dealer_02_v2_73_ml_ranked_exit_ai_050_scaled_buy_continue` は、現時点の暫定本命profileとして扱う。

理由:

- v2_65から大きく改善したAI rankingを維持している。
- Exit AIによりDD改善要素を持つ。
- scaled buyにより買付REJECTを救済し、利益/PFを大きく改善した。
- v2_71相当の成績を維持しながら、purchase_auditで監査可能。
- v2_74ほど攻めすぎず、PF/DDバランスが良い。
- Position Sizingの単純倍率案より安定している。

ただし、これは本番売買profileではない。現段階では、今後のpaper/backtest検証・日次候補確認・テスト環境接続準備の基準profileとして採用する。

## Remaining Risks

- 大型勝ちトレードへの依存
- scaled buyの集中リスク
- Exit AIが大勝ち月を削る可能性
- fallbackなしによる買い逃し
- ML prediction pipeline依存
- walk-forward predictionとcurrent model predictionの混同リスク
- purchase_audit欠損時に判断根拠が追えないリスク

これらを監視しながら、v2_73を暫定本命として次の検証を進める。
