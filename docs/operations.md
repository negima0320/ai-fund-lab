# 運用手順

このRunbookは、Mac上で `ai-fund-lab` をPaperBroker前提で定期実行するための手順です。実売買は行いません。`tachibana_live` の自動実行例は意図的に載せていません。

## 初回セットアップ

作業ディレクトリへ移動します。

```bash
cd /Users/negishi/work/ai-fund-lab
```

`.env` を作成し、J-QuantsなどのAPIキーを設定します。秘密情報やAPIキーはREADME、docs、Git管理対象ファイルには書きません。

```bash
cp .env.example .env
```

`.env.example` がない場合は `.env` を新規作成し、必要な環境変数だけを書きます。`.env` はGit管理しません。

仮想環境を作成します。

```bash
python3 -m venv .venv
```

依存関係をインストールします。

```bash
.venv/bin/python -m pip install -r requirements.txt
```

DBを初期化します。

```bash
.venv/bin/python src/main.py --mode init-db --profile rookie_dealer_02_v2_1
```

J-Quants接続を確認します。

```bash
.venv/bin/python src/main.py --mode healthcheck --provider jquants --profile rookie_dealer_02_v2_1
```

## 基本運用フロー

毎朝の基本フローは以下です。

1. `preflight`
2. `full-paper-run`
3. `analyze`
4. report確認

例:

```bash
python src/main.py --mode preflight --provider jquants --profile rookie_dealer_02_v2_1
python src/main.py --mode full-paper-run --provider jquants --profile rookie_dealer_02_v2_1 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
python src/main.py --mode analyze --profile rookie_dealer_02_v2_1
```

定期実行では、直接 `python` を呼ばず、`.venv/bin/python` を絶対パスで指定するか、`scripts/run_daily_paper.sh` を呼びます。

```bash
scripts/run_daily_paper.sh
```

分析だけ再実行する場合:

```bash
scripts/run_analyze.sh
```

## 運用スケジュール

運用時刻は `config/operation_schedule.yaml` に明記します。基準市場は東京市場、タイムゾーンは `Asia/Tokyo` です。

基本スケジュール:

- 16:00以降に当日終値ベースのデータ取得を行う
- 16:10に当日終値ベースで銘柄選定を行う
- 16:30にPaperBroker runとレポート作成を行う
- 翌営業日 08:30 に注文候補を確認する
- 翌営業日 08:35 に `tachibana_demo` の自動発注検証を行う
- 09:00〜09:30 に人間が手動で発注判断する

実行ポリシー:

- 銘柄選定は `previous_close`、つまり前営業日または当日確定済み終値を基準にする
- 注文判断は `next_business_day_after_open` を想定する
- `tachibana_demo` では `auto_demo` を使い、テスト環境の自動発注を検証する
- `auto_order_enabled: true` は `tachibana_demo` だけに限定する
- `forbid_live_auto_order: true` を維持する
- live実売買は手動で行う

cron / launchd は16:30の銘柄選定と、08:35の `tachibana_demo` 自動発注検証だけを設定します。`tachibana_live` の自動実行例は作りません。

## PaperBroker実行

PaperBrokerで通し実行します。実売買は行わず、発注先はpaperです。

```bash
.venv/bin/python src/main.py --mode full-paper-run --provider jquants --profile rookie_dealer_02_v2_1 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

実行後、以下を確認します。

- `reports/rookie_dealer_02_v2_1/backtests/`
- `reports/rookie_dealer_02_v2_1/backtests/analysis_latest.md`
- `logs/paper_run.log`

## tachibana_demo自動売買フロー

立花証券のテスト環境だけで、自動発注の流れを検証できます。

夕方の銘柄選定:

```bash
scripts/run_evening_selection.sh
```

このスクリプトは `preflight`、データ取得、screen、score、`preview-orders`、analyzeを実行し、翌営業日の注文候補を `reports/<profile_id>/order_previews/` に保存します。

翌朝のdemo自動発注:

```bash
scripts/run_demo_auto_order.sh
```

このスクリプトは以下を確認してから `tachibana_demo` に注文候補を送ります。

- `tachibana.environment` が `demo`
- `broker.provider` が `tachibana_demo`
- `broker.provider` が `tachibana_live` ではない
- `auto_order_enabled` が `true`
- `forbid_live_auto_order` が `true`
- `preflight` が成功する
- 残高取得
- 保有銘柄取得
- 二重注文チェック
- `max_positions` チェック
- `max_daily_buy_amount` チェック
- `max_single_order_amount` チェック

以下の場合は必ず停止します。

- `env=live`
- `broker=tachibana_live`
- `auto_order_enabled=false`
- `forbid_live_auto_order=true` かつlive broker
- `preflight` 失敗
- cash不足
- 同一銘柄保有中
- `max_positions` 超過
- 当日注文上限超過

発注結果は `logs/demo_orders.log` に保存します。これはテスト環境向けです。live自動売買は禁止です。

## 分析実行

```bash
.venv/bin/python src/main.py --mode analyze --profile rookie_dealer_02_v2_1
```

分析レポート:

- `reports/rookie_dealer_02_v2_1/backtests/analysis_latest.md`
- `reports/rookie_dealer_02_v2_1/backtests/analysis_latest.json`

## レポート確認

毎朝、最低限以下を確認します。

- 日次損益
- 勝率
- PF
- 最大ドローダウン
- Exit Reason Analysis
- Holding Period Optimization
- Walk Forward Validation
- Market Regime Performance Analysis
- Daily Paper Report
- Manual Approval Flow
- `preview_orders`

## cron運用例

Macのcronで平日朝に実行する例です。

```bash
crontab -e
```

平日 16:30 に実行:

```cron
30 16 * * 1-5 cd /Users/negishi/work/ai-fund-lab && /Users/negishi/work/ai-fund-lab/.venv/bin/python src/main.py --mode full-paper-run --provider jquants --profile rookie_dealer_02_v2_1 >> logs/cron.log 2>&1
```

推奨はスクリプト経由です。

```cron
30 16 * * 1-5 cd /Users/negishi/work/ai-fund-lab && scripts/run_evening_selection.sh >> logs/evening_selection.log 2>&1
35 8 * * 1-5 cd /Users/negishi/work/ai-fund-lab && scripts/run_demo_auto_order.sh >> logs/demo_auto_order.log 2>&1
```

注意:

- cronはログインシェルの環境変数を読まない
- `.env` を必ずアプリ側で読み込む
- python は `.venv/bin/python` を絶対パスで指定
- 作業ディレクトリを `cd` で明示
- `logs` ディレクトリが必要
- 自動実行はPaperBrokerだけにする

## launchd運用例

Macではcronよりlaunchd推奨です。plist例は以下にあります。

```text
docs/launchd/com.negima.ai-fund-lab.paper-run.plist
```

配置例:

```bash
mkdir -p ~/Library/LaunchAgents
cp docs/launchd/com.negima.ai-fund-lab.paper-run.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.negima.ai-fund-lab.paper-run.plist
```

停止する場合:

```bash
launchctl unload ~/Library/LaunchAgents/com.negima.ai-fund-lab.paper-run.plist
```

plistには以下を含めます。

- `WorkingDirectory`
- `ProgramArguments`
- `StandardOutPath`
- `StandardErrorPath`
- `StartCalendarInterval`
- `Weekday`
- `Hour`
- `Minute`

## ログ

主な確認先:

- `logs/cron.log`
- `logs/paper_run.log`
- `reports/{profile}/backtests/`
- `reports/{profile}/backtests/analysis_latest.md`

ログが出ていない場合は、cron/launchdの作業ディレクトリ、`.venv/bin/python` の絶対パス、`logs` ディレクトリの有無を確認します。

## 異常時の停止方法

新規買付を止める場合:

```bash
mkdir -p storage
touch storage/STOP_TRADING
```

解除する場合:

```bash
rm storage/STOP_TRADING
```

自動実行そのものを止める場合は、cronの行をコメントアウトするか、launchdのAgentをunloadします。

## 安全運用ルール

- 初期はpaper brokerのみ
- `tachibana_live` は明示的に有効化しない限り使わない
- live発注は手動承認フローが完成するまで禁止
- `.env` にAPIキーを置き、Git管理しない
- READMEやdocsに秘密情報を書かない
- 自動実行例はPaperBrokerだけにする
- `preview_orders` は確認用であり、発注ではない

## よく使うコマンド

状態確認:

```bash
.venv/bin/python src/main.py --mode status --profile rookie_dealer_02_v2_1
```

J-Quants接続確認:

```bash
.venv/bin/python src/main.py --mode healthcheck --provider jquants --profile rookie_dealer_02_v2_1
```

事前チェック:

```bash
.venv/bin/python src/main.py --mode preflight --provider jquants --profile rookie_dealer_02_v2_1
```

PaperBroker通し実行:

```bash
.venv/bin/python src/main.py --mode full-paper-run --provider jquants --profile rookie_dealer_02_v2_1 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

分析:

```bash
.venv/bin/python src/main.py --mode analyze --profile rookie_dealer_02_v2_1
```

profile比較:

```bash
.venv/bin/python src/main.py --mode compare-profiles --profiles rookie_dealer_02_v2_1 rookie_dealer_02_v2_4 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

AI改善サマリ:

```bash
.venv/bin/python src/main.py --mode export-ai-summary --profile rookie_dealer_02_v2_1 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

## 生成物の保存方針

SQLiteは運用データの正本です。DBファイルはGit管理しません。

`logs/` は分析用データとして扱い、Git管理しません。

`data/raw/` はJ-Quants APIの取得キャッシュです。再取得可能なデータであり、Git管理しません。

`data/processed/` は指標、候補銘柄、採点済み候補などの中間生成物です。再生成可能なため、Git管理しません。

`articles/drafts/` はnote投稿前の生成物です。ローカルで確認・編集する下書きとして扱い、Git管理しません。

`articles/published/` は公開済み記事としてGit管理します。

`reports/` は通常の生成レポートをGit管理しません。ただし、バックテストの共有・比較に使う `reports/backtests/` はGit管理対象とします。
