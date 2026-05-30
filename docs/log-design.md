# ログ設計

ログはJSONまたはMarkdownで保存し、GitHub上で履歴管理できる形にします。

| ログ | 保存先 | 形式 |
| --- | --- | --- |
| スクリーニング実行ログ | `logs/screening/YYYY-MM/` | JSON |
| 候補50銘柄ログ | `logs/screening/YYYY-MM/` | JSON |
| AI採点ログ | `logs/scoring/YYYY-MM/` | JSON |
| 売買判断ログ | `logs/scoring/YYYY-MM/` | JSON |
| 日次トレードログ | `logs/trades/YYYY-MM/` | JSON |
| 仮想注文ログ | `logs/trades/YYYY-MM/` | JSON |
| 保有ポジションログ | `logs/portfolio/YYYY-MM/` | JSON |
| 売却結果ログ | `logs/trades/YYYY-MM/` | JSON |
| 損益ログ | `logs/trades/YYYY-MM/` | JSON |
| ポートフォリオ日次サマリ | `logs/portfolio/YYYY-MM/` | JSON |
| AI振り返りログ | `logs/reflections/YYYY-MM/` | JSON |
| note記事Markdownログ | `articles/drafts/YYYY-MM/` | Markdown |

ファイル名には対象日とDay番号を含めます。例: `20260601_day_001_ai_scores.json`、`20260601_day_001.md`。

日報は `reports/YYYY-MM/YYYYMMDD_day_XXX.md` に保存します。何日のレポートかをファイル名だけで判断できるようにします。

売却ログには `trade_id`、`code`、`name`、`entry_date`、`exit_date`、`holding_days`、`entry_price`、`exit_price`、`shares`、`profit`、`profit_rate`、`exit_reason`、`result` を保存します。

ポートフォリオ日次サマリには `date`、`day`、`cash`、`positions_value`、`total_assets`、`daily_profit`、`cumulative_profit`、`win_rate`、`max_drawdown` を保存します。

秘密情報は `.env` に置き、Git管理しません。
