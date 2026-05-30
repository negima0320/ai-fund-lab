# 運用手順

## 日次実行

```bash
python src/main.py --mode demo --days 10
```

現在はダミーデータを使用します。生成物は `logs/`、`reports/`、`articles/drafts/` に保存されます。

ログ、日報、note下書きは月別フォルダに保存します。例: `reports/2026-06/20260601_day_001.md`。

## note投稿

`articles/drafts/YYYY-MM/` に生成されたMarkdownを確認し、必要に応じて人間コメント欄を追記してから手動投稿します。

公開済みの記事は `articles/published/` に移動し、GitHubで管理します。

## 生成物の保存方針

GitHubはコード、設定、ドキュメント、公開記事を保存する場所とします。

SQLiteは運用データの正本とします。将来的に売買履歴、日次資産、採点結果、振り返りなどの運用データはSQLiteへ集約します。

初回またはDBを作り直す場合は以下を実行します。

```bash
python src/main.py --mode init-db
```

SQLite DBは `storage/ai_fund_lab.sqlite3` に保存します。DBファイルはGit管理しません。

`logs/` は分析用データとして扱い、Git管理しません。ローカル実行やバックテストで増えるため、必要に応じてSQLiteまたはCSVへ取り込みます。

`data/raw/` はJ-Quants APIの取得キャッシュです。再取得可能なデータであり、Git管理しません。

`data/processed/` は指標、候補銘柄、採点済み候補などの中間生成物です。再生成可能なため、Git管理しません。

`articles/drafts/` はnote投稿前の生成物です。ローカルで確認・編集する下書きとして扱い、Git管理しません。

`articles/published/` は公開済み記事としてGit管理します。

`reports/` は通常の生成レポートをGit管理しません。ただし、バックテストの共有・比較に使う `reports/backtests/` はGit管理対象とします。

JSON、Markdown、CSVは必要に応じて生成するエクスポート生成物です。運用データの正本はSQLite、公開済み記事の履歴はGitHubで管理します。

## ルール変更

AIは売買ルールを変更できません。AIの改善案は `logs/reflections/` の `suggestions` に保存し、人間がレビューしたうえで `config/rookie_dealer.yaml` と `docs/trading-rules.md` を更新します。
