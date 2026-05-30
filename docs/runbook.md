# AI Fund Lab Runbook

このRunbookは、AI Fund Labを初回セットアップから日次運用、バックテスト、分析、note記事公開管理まで通しで確認するための手順です。

現時点では実売買は行いません。Brokerは `paper` を使用し、立花証券 e支店 API、kabuステーションAPIへの実接続・実発注は行いません。

## 1. 初回セットアップ

Python 3.12以上で環境を作成します。`pandas-ta` の現行配布はPython 3.12系で解決できるため、ローカル環境もCIと同じPython版に合わせてください。

```bash
python -m venv .venv
source .venv/bin/activate
```

依存パッケージをインストールします。

```bash
pip install -r requirements.txt
```

`.env.example` を参考に `.env` を作成します。

```bash
cp .env.example .env
```

`.env` に `JQUANTS_API_KEY` を設定します。実際のAPIキーはGit管理しません。

```text
JQUANTS_API_KEY=
```

Tachibana系の環境変数は、現時点では未設定でも構いません。立花証券 e支店 APIは未実装で、実発注もしません。

```text
TACHIBANA_USER_ID=
TACHIBANA_PASSWORD=
TACHIBANA_SECOND_PASSWORD=
TACHIBANA_PRIVATE_KEY_PATH=
TACHIBANA_PUBLIC_KEY_ID=
```

`OPENAI_API_KEY` は任意です。未設定の場合は、ルールベースのコメント生成へフォールバックします。

```text
OPENAI_API_KEY=
```

## 2. 事前チェック

```bash
python src/main.py --mode preflight
```

確認すること:

- config が OK
- DB が OK、または未初期化の場合は案内が出る
- J-Quants設定が OK または WARN
- OpenAI未設定なら `rule_based` fallback になる
- broker が `paper`
- live trading が disabled
- Tachibana / kabuステーションは未接続

preflight結果はprofile別に以下にも保存されます。

- `reports/<profile_id>/backtests/preflight_latest.md`
- `reports/<profile_id>/backtests/preflight_latest.json`

## 3. DB初期化

```bash
python src/main.py --mode init-db
```

SQLite DBを初期化します。DBは運用データの正本ですが、Git管理しません。

デフォルト保存先:

```text
storage/ai_fund_lab.sqlite3
```

## 4. J-Quants疎通確認

```bash
python src/main.py --mode healthcheck --provider jquants
```

確認すること:

- `JQUANTS_API_KEY` が設定されている
- J-Quantsの軽量エンドポイントへ接続できる
- 上場銘柄一覧の件数が表示される

APIキーの値は表示・保存されません。

## 5. 東証プライム銘柄一覧取得

```bash
python src/main.py --mode list-stocks --provider jquants
```

J-Quantsから上場銘柄一覧を取得し、東証プライム銘柄だけに絞り込みます。

保存先:

```text
data/raw/prime_stocks_jquants.json
```

確認すること:

- 全取得件数
- 東証プライム件数
- 保存先

## 6. 日次運用を1日分実行

```bash
python src/main.py --mode run-daily --provider jquants --date YYYY-MM-DD
```

例:

```bash
python src/main.py --mode run-daily --provider jquants --date 2026-03-06
```

処理内容:

1. 価格データ取得
2. 指標計算
3. スクリーニング
4. スコアリング
5. 仮想売買
6. AI振り返り生成
7. 日報Markdown生成
8. note記事Markdown生成
9. `summary.csv` / `trades.csv` 更新
10. グラフ画像更新

出力確認:

- candidates が生成されている
- scoring が生成されている
- trade log が生成されている
- portfolio summary が生成されている
- report が生成されている
- article draft が生成されている
- SQLite DBに保存されている

代表的な出力先:

```text
data/processed/candidates_YYYY-MM-DD.json
data/processed/scored_candidates_YYYY-MM-DD.json
logs/trades/trades_YYYY-MM-DD.json
logs/portfolio/portfolio_YYYY-MM-DD.json
reports/day_YYYY-MM-DD.md
articles/drafts/day_YYYY-MM-DD.md
```

## 7. 注文プレビュー

```bash
python src/main.py --mode preview-orders --provider jquants --date YYYY-MM-DD
```

確認すること:

- 実発注しない
- Broker は `PaperBroker`
- Safetyチェックの通過/拒否理由が表示される
- 買付候補、売却候補、見送り理由が確認できる

保存先:

```text
reports/order_previews/order_preview_YYYY-MM-DD.md
reports/order_previews/order_preview_YYYY-MM-DD.json
```

このモードでは `broker.place_buy_order` / `broker.place_sell_order` は呼びません。

## 8. バックテスト

```bash
python src/main.py --mode backtest --provider jquants --profile rookie_dealer_01 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

例:

```bash
python src/main.py --mode backtest --provider jquants --profile rookie_dealer_01 --start-date 2026-03-02 --end-date 2026-06-05
python src/main.py --mode analyze --profile rookie_dealer_01
```

確認すること:

- `initial_capital`
- `final_assets`
- `gross_cumulative_profit`
- `net_cumulative_profit`
- `net_cumulative_profit_rate`
- `total_trades`
- `win_rate`
- `max_drawdown`
- `profit_factor`
- `take_profit_count`
- `stop_loss_count`
- `max_holding_exit_count`
- `no_trade_days`
- `selected_count_total`

バックテストは通常運用の `logs/portfolio/state.json` を壊さず、バックテスト専用の状態で実行します。

OpenAI / ChatGPT APIを使わず、ルールベースのみで90日バックテストする場合は、`config/profiles/rookie_dealer_01.yaml` を以下の状態にします。

- `ai_decision.enabled: false`
- `ai_commentary.provider: rule_based`
- `broker.provider: paper`
- `broker.live_trading_enabled: false`
- `safety.allow_live_trading: false`

この構成では `OPENAI_API_KEY` が未設定でも実行できます。候補銘柄の最終判断とコメント生成はルールベースで完結し、実売買は行いません。J-Quants Freeプランでは12週間遅延データを前提に検証します。

代表的な保存先:

```text
logs/backtests/<profile_id>/YYYY-MM-DD_to_YYYY-MM-DD/
reports/<profile_id>/backtest_YYYY-MM-DD_to_YYYY-MM-DD.md
reports/<profile_id>/backtest_YYYY-MM-DD_to_YYYY-MM-DD.json
reports/backtests/rule_based_90d_summary_YYYY-MM-DD_to_YYYY-MM-DD.md
```

## 9. 分析レポート

```bash
python src/main.py --mode analyze
```

SQLite DBから運用データを集計し、分析レポートを生成します。

保存先:

```text
reports/backtests/analysis_latest.md
reports/backtests/analysis_latest.json
```

確認すること:

- 最新総資産
- 累計損益
- 税引前累計損益
- 税引後累計損益
- 概算税額合計
- 手数料合計
- 勝率
- 最大ドローダウン
- 総取引数
- config_version別集計

## 10. Full Paper Run

立花証券API接続前に、J-Quants実データとPaperBrokerで一連の確認をまとめて実行します。

```bash
python src/main.py --mode full-paper-run --provider jquants --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

例:

```bash
python src/main.py --mode full-paper-run --provider jquants --start-date 2026-03-02 --end-date 2026-03-06
```

処理内容:

- preflight
- init-db 未実行なら自動初期化
- J-Quants healthcheck
- list-stocks
- backtest
- analyze
- release-notes
- 最終サマリ生成

強制条件:

- `broker.provider` は `paper`
- `safety.allow_live_trading` は `false`
- `broker.live_trading_enabled` は `false`

設定が違う場合は実行を中止します。TachibanaBroker / KabuStationBroker は使いません。

最終サマリ保存先:

```text
reports/backtests/full_paper_run_YYYY-MM-DD_to_YYYY-MM-DD.md
reports/backtests/full_paper_run_YYYY-MM-DD_to_YYYY-MM-DD.json
```

確認すること:

- 初期資金
- 最終資産
- 税引前損益
- 税引後損益
- 勝率
- 最大ドローダウン
- 総取引数
- selected_count合計
- no trade日数
- 生成された記事数
- 次に確認すべき課題

このモードは外部APIを叩くため、GitHub Actions CIには含めません。

## 11. note記事公開管理

note記事は自動投稿しません。`articles/drafts/` に生成されたMarkdownを確認し、手動でnoteへ投稿します。

手動投稿後、公開済み記事として記録します。

```bash
python src/main.py --mode publish-article --date YYYY-MM-DD --note-url URL
```

例:

```bash
python src/main.py --mode publish-article --date 2026-03-06 --note-url https://note.com/example/n/example
```

処理内容:

- `articles/drafts/day_YYYY-MM-DD.md` を探す
- front matter に公開情報を追加または更新する
- `articles/published/day_YYYY-MM-DD.md` に保存する
- SQLiteの `articles` テーブルを更新する
- 設定に応じてdraftを削除または保持する

`articles/published/` はGit管理対象です。`articles/drafts/` は生成物のためGit管理しません。

## 12. よくあるトラブル

### Freeプランは12週間遅延

J-Quants Freeプランでは、取得できる株価データが12週間遅延している可能性があります。最新日付でデータがない場合は、過去日付で実行してください。

### 指定日にデータがない

土日祝、休場日、未更新日、J-Quants Freeプランの取得可能範囲外ではデータが取得できない場合があります。別の日付を指定してください。

### J-Quantsレート制限

Freeプランは1分あたり5リクエストを前提にしています。大量の日付を取得する処理では時間がかかる場合があります。Lightプランでは1分あたり60リクエストを想定しています。

### DB未初期化

DBが存在しない、または必須テーブルがない場合は以下を実行してください。

```bash
python src/main.py --mode init-db
```

### selected_count が0

スコア基準や信頼度基準を満たす銘柄がない場合、買付対象なしで正常終了します。新人ディーラー1号は、買うものがない日は買いません。

ただし、設定によりノートレード回避のトップピック採用が有効な場合、65点以上かつ信頼度基準を満たす最上位1銘柄だけを採用することがあります。

### 単元株で買えない

`use_round_lot: true` の場合、100株単位で買付株数を計算します。株価が高く、100株購入に必要な金額が1銘柄上限を超える場合は買付不可になります。

### STOP_TRADING が存在して新規買付停止

以下のファイルが存在すると、新規買付は停止します。

```text
storage/STOP_TRADING
```

初期実装では、損切り・利確などの売却は許可します。新規買付を再開する場合は、意図を確認してからSTOPファイルを削除してください。

### .env が未設定

`.env` がない、または `JQUANTS_API_KEY` が未設定の場合、J-Quants系のコマンドは失敗またはWARNになります。`.env.example` を参考に `.env` を作成してください。

## 13. 実売買について

現時点では実売買しません。

- broker は `paper`
- TachibanaBroker は未実装
- TachibanaDemoBroker は接続準備用スタブ
- TachibanaLiveBroker は未実装
- kabuステーションはMac/Linux環境では優先度低
- 立花証券 e支店 API は今後demo環境から検証予定

実売買へ進む場合でも、以下の安全条件を満たす必要があります。

- `broker.live_trading_enabled: true`
- `safety.allow_live_trading: true`
- `tachibana.environment: live`
- `storage/STOP_TRADING` が存在しない
- PaperBrokerで十分な検証が済んでいる

ただし現時点では、これらを満たしても実発注処理は実装されていません。

## 14. 推奨確認順

1. `pytest`
2. `python src/main.py --mode preflight`
3. `python src/main.py --mode init-db`
4. `python src/main.py --mode healthcheck --provider jquants`
5. `python src/main.py --mode list-stocks --provider jquants`
6. `python src/main.py --mode run-daily --provider jquants --date YYYY-MM-DD`
7. `python src/main.py --mode analyze`
8. `python src/main.py --mode backtest --provider jquants --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
9. `python src/main.py --mode full-paper-run --provider jquants --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
