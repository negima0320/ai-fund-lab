# AI Fund Lab

AI Fund Lab は、日本株の短期売買ルールを **研究・検証・PaperBroker運用** するためのローカル実験環境です。投資助言や実売買の推奨を目的にしたものではありません。通常の売買実行先は `PaperBroker` で、live自動売買は安全ロックにより禁止・未運用です。

このREADMEは入口です。詳細仕様は `docs/` に分けています。古いドキュメントの文言ではなく、現在の正は `src/`、`config/profiles/*.yaml`、`config/profile_registry.yaml`、CLIの出力、生成済みレポート項目です。

## 現在の状態

| 領域 | 状態 |
| --- | --- |
| J-Quants | `free` / `light` planを設定で切替。価格、銘柄マスター、取引カレンダー、TOPIX、投資部門別、決算予定、財務サマリはplanに応じて取得・fallbackします |
| PaperBroker | 実装済み。バックテスト、日次paper run、資金制約、100株単位、fallback buy監査に使います |
| Tachibana | Read-only broker実装。口座・残高・注文・約定参照IFはありますが、実API発注は無効化されています |
| KabuStation | 現在は後退候補・スタブ扱い。注文実装はありません |
| Profile実験 | `config/profile_registry.yaml` で baseline / experiment / deprecated を管理します |
| レポート | `reports/<profile_id>/backtests/feature_analysis.*` と `reports/experiments/.../experiment_summary.*` が主要監査出力です |

## セットアップ

秘密情報は `.env` で管理し、READMEやdocsには書きません。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/main.py --mode init-db --profile rookie_dealer_02_v2_1
.venv/bin/python src/main.py --mode validate-config
```

CLI全体は次で確認できます。

```bash
.venv/bin/python src/main.py --mode help
```

## よく使うコマンド

```bash
# profile一覧と詳細
.venv/bin/python src/main.py --mode list-profiles
.venv/bin/python src/main.py --mode profile-info --profile rookie_dealer_02_v2_38

# 設定検証
.venv/bin/python src/main.py --mode validate-config
.venv/bin/python src/main.py --mode validate-config --profile rookie_dealer_02_v2_38 --strict

# 単体backtest
.venv/bin/python src/main.py --mode backtest --profile rookie_dealer_02_v2_26 --start-date YYYY-MM-DD --end-date YYYY-MM-DD --skip-price-fetch --fast-analysis --strict-integrity

# 実験比較
.venv/bin/python src/main.py --mode run-experiments \
  --base-profile rookie_dealer_02_v2_26 \
  --profiles rookie_dealer_02_v2_26 rookie_dealer_02_v2_38 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --skip-price-fetch \
  --fast-analysis \
  --summary-only \
  --strict-integrity

# 既存成果物からfeature_analysisだけ再生成
.venv/bin/python src/feature_analysis.py --profile rookie_dealer_02_v2_38
PYTHONPATH=. .venv/bin/python -m src.feature_analysis --profile rookie_dealer_02_v2_38

# 不要profile成果物の削除候補確認
.venv/bin/python src/main.py --mode cleanup-retired-profiles --verbose
```

重い5年backtestや `run-experiments` は、キャッシュと空き容量を確認してから手動で実行してください。

## Profileの見方

profile本体は `config/profiles/<profile_id>.yaml`、実験メタデータは `config/profile_registry.yaml` です。

- `role: baseline`: 比較基準。例: `rookie_dealer_02_v2_1`
- `role: experiment`: `compare_to` で指定したbaseに対する実験候補
- `role: deprecated`: 保管用。通常の一括実験対象から外します
- `required_plan`: `free` / `light` の推奨J-Quants plan
- `features`: データ取得・特徴量生成の有効化
- `enabled_features`: `list-profiles` / `profile-info` / experiment summaryに表示される識別用機能名
- `profile_id` / `profile_name` / `config_version`: DB、logs、reports、scored candidatesに保存されます

未指定時のprofile loader defaultは `rookie_dealer_01` ですが、現在の実験系の中心baselineは `rookie_dealer_02_v2_1`、資金利用率改善系の暫定baseは `rookie_dealer_02_v2_26` です。

## スコア計算の要点

現在の銘柄スコアは `src/scoring.py` の `score_real_candidates()` が正です。旧資料にあった「100点満点、ニュース30、財務20」の固定式は現在の実装では使っていません。

現在の実式は概念的に次です。

```text
total_score =
  technical_score
  + relative_strength_score
  + investor_context_score
  + market_context_score
  + winner_loser_rule_score
  + penalty_score
```

`technical_score` は最大50点です。`relative_strength_score`、`investor_context_score`、`affordability_filter`、`winner_loser_rule_adjustment` などはprofile設定で有効化された場合だけ実質的に効きます。`market_context_score` は現在0固定です。財務情報は監査・future candidate扱いで、通常の `total_score` には加算されません。

詳しくは [docs/scoring_spec.md](docs/scoring_spec.md) と [docs/score_formula_audit.md](docs/score_formula_audit.md) を参照してください。

## 選定・買付の要点

選定はスコア順に行われ、`selection.min_score`、市場別min score override、confidence、RSI/出来高/決算/投資部門別/market filter、最大選定数を通過した候補だけが `selected=true` になります。

買付はPaperBroker上で、現金、保有数、100株単位、`allocation_limit`、`target_exposure`、`min_cash_buffer`、`max_position_value_rate` を見て行われます。`affordable_fallback_buy` が有効なprofileでは、通常選定後に余剰現金で買える高順位候補を追加選定できます。fallback由来の取引は `selection_source` / `affordable_fallback_buy_selected` で通常選定と区別され、integrity auditでは選定済みとして扱います。

市場区分は `market_filter.allowed_sections` で制御します。`allow_unknown_market: false` の場合、Unknown / None / 空文字の市場区分は除外です。Standard/Growthを使う実験では、Candidate Universe Audit、Screening Audit、Standard Funnel Audit、Trade Market Auditでどの段階まで到達したかを確認します。

売却は損切り、利確、最大保有期間、market/risk exitが基本です。`conditional_hold_extension` が有効なprofileでは、最大保有期間到達時だけ含み益、relative strength、MA25上昇などを確認して保有延長を検証できます。詳細は [docs/trading-rules.md](docs/trading-rules.md) と [docs/rookie-dealer-decision-flow.md](docs/rookie-dealer-decision-flow.md) を参照してください。

## 主要出力

| 出力 | 内容 |
| --- | --- |
| `logs/backtests/<profile>/<START>_to_<END>/backtest_summary.json` | backtestの機械可読summary |
| `logs/backtests/<profile>/<START>_to_<END>/summary.csv` | 日次資産推移 |
| `logs/backtests/<profile>/<START>_to_<END>/trades.csv` | 売買イベント |
| `reports/<profile>/backtests/analysis_latest.*` | profile単体の分析 |
| `reports/<profile>/backtests/feature_analysis.*` | integrity、score、market、capital、monthlyなどの詳細監査 |
| `reports/<profile>/backtests/selection_quality.*` | 選定品質分析 |
| `reports/experiments/<START>_to_<END>/<base>/experiment_summary.*` | profile比較、verdict、practical_effect、market別集計 |

`final_assets` は最終総資産です。資産検算は `Compounding / Capital Flow Audit` の `asset_reconciliation` を見てください。`net_cumulative_profit` はレポート上の累積損益指標であり、最終資産の検算では `initial_capital + realized_profit_total + unrealized_profit_total + external_cash_flow_total` を優先します。

## ログとキャッシュ

主な階層は次の通りです。

- `data/raw/`: raw snapshot。`prices_YYYY-MM-DD.json`、`listed_stocks_jquants.json` など
- `data/cache/jquants/`: J-Quants API response cache
- `data/processed/common/`: profile互換のprocessed common cache
- `data/processed/<profile_id>/`: profile runtime I/O path
- `logs/`: 日次ログ、backtestログ、portfolio/trades/safety/reflections
- `reports/`: 人間確認用レポート、実験比較、記事
- `storage/`: SQLite DBと `STOP_TRADING`

詳細は [docs/log-design.md](docs/log-design.md) と [docs/operations.md](docs/operations.md) を参照してください。

## ドキュメント

- [docs/docs_audit_report.md](docs/docs_audit_report.md): 今回の棚卸し結果と修正方針
- [docs/scoring_spec.md](docs/scoring_spec.md): スコア計算仕様
- [docs/score_formula_audit.md](docs/score_formula_audit.md): Score Formula Auditの読み方
- [docs/rookie-dealer-decision-flow.md](docs/rookie-dealer-decision-flow.md): 候補生成から買付までの流れ
- [docs/operations.md](docs/operations.md): 運用・検証手順
- [docs/jquants_plan_matrix.md](docs/jquants_plan_matrix.md): J-Quants planとfallback
- [docs/log-design.md](docs/log-design.md): ログ・キャッシュ階層
- [docs/openai-optional.md](docs/openai-optional.md): OpenAI任意利用
- [docs/tachibana-plan.md](docs/tachibana-plan.md): Tachibana brokerの現状
- [docs/kabu-station-plan.md](docs/kabu-station-plan.md): KabuStationの現状
