# AI改善用ログ

AI改善用ログは、人間向けの日報やnote記事とは別に保存する、機械分析用のJSONLデータセットです。後で [ChatGPT](https://chatgpt.com/) や別AIに渡し、新人ディーラー1号の勝ちパターン、負けパターン、改善すべきルール、過剰なフィルター、不足している指標を分析するために使います。

このログは投資助言ではなく、AI Fund Lab内の研究・実験用データです。

## 保存先

AI改善用ログは `analysis_logs/` 配下に保存します。

```text
analysis_logs/<profile_id>/decision_dataset_YYYY-MM-DD.jsonl
analysis_logs/<profile_id>/decision_dataset_START_to_END.jsonl
analysis_logs/<profile_id>/ai_summary_START_to_END.md
```

`analysis_logs/` は `.gitignore` 対象です。APIキー、秘密情報、SQLite DB、raw/processed data、通常ログと同じくGit管理しません。

## JSONLの考え方

`decision_dataset_*.jsonl` は1行1レコードのJSONL形式です。各候補銘柄ごとに1レコードを作ります。

selected銘柄だけでなく、rejected銘柄も含めます。AI改善では「なぜ買ったか」だけでなく、「なぜ買わなかったか」「買わなかった銘柄がその後どうなったか」も重要だからです。

主な構造は以下です。

- `market_context`: 地合い、値上がり銘柄比率、平均騰落率、売買代金
- `sector_context`: 業種名、業種モメンタム、業種順位、コメント
- `technical_features`: 終値、移動平均線、RSI、MACD、ボリンジャーバンド、ATR、出来高倍率、売買代金
- `candlestick_features`: ローソク足タイプ、実体、ヒゲ、終値位置、ギャップ、ローソク足シグナル
- `news_features`: ニューススコア、ニュース件数、ポジティブ/ネガティブ件数
- `rule_based_score`: 総合点、テクニカル点、ニュース点、財務点、信頼度、順位、理由
- `ai_decision`: OpenAI AI Decisionを使ったか、モデル、AI理由、AIリスク
- `decision`: selected、action、見送り理由、safety結果、注文作成・約定状態
- `position_context`: 既存保有との関係
- `future_result`: 売却結果、利益率、MFE/MAE、1/3/5営業日後価格

raw news本文は含めません。必要な場合も、タイトルや集計値までに留めます。

## future_result

`future_result` は、判断時点では空でも構いません。

売却済みの取引がDBに保存されている場合は、以下を埋めます。

- `exit_date`
- `exit_reason`
- `holding_days`
- `gross_profit`
- `gross_profit_rate`
- `net_profit`
- `net_profit_rate`
- `max_favorable_excursion`
- `max_adverse_excursion`
- `price_after_1d`
- `price_after_3d`
- `price_after_5d`

これにより、判断時点の情報と結果を1レコード内で紐づけて分析できます。

## export-ai-dataset

指定期間のDBデータからJSONLデータセットを生成します。

```bash
python src/main.py --mode export-ai-dataset --profile rookie_dealer_01 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

出力先:

```text
analysis_logs/rookie_dealer_01/decision_dataset_START_to_END.jsonl
```

DBから `screening_results`、`scoring_results`、`ai_decisions`、`trades`、`portfolio_snapshots`、`market_contexts` を読み、selected銘柄とrejected銘柄の両方を出力します。

## export-ai-summary

指定期間のAI改善用サマリを生成します。

```bash
python src/main.py --mode export-ai-summary --profile rookie_dealer_01 --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

出力先:

```text
analysis_logs/rookie_dealer_01/ai_summary_START_to_END.md
```

含まれる内容:

- 期間
- 総取引数
- 勝率
- 最大ドローダウン
- スコア帯別勝率
- RSI帯別勝率
- 出来高倍率別勝率
- ローソク足シグナル別勝率
- market_regime別勝率
- sector別勝率
- 利確/損切り/期限切れ件数
- [ChatGPT](https://chatgpt.com/) に改善案を聞くためのプロンプト雛形

## プロンプト雛形

`ai_summary` の最後には以下の文を含めます。

```text
以下のAIファンド売買ログを分析し、勝ちパターン・負けパターン・改善すべきルール・過剰なフィルター・不足している指標を提案してください。ただし、過学習を避け、再現性の高い改善案だけを出してください。
```

## DB記録

エクスポート履歴は `ai_analysis_exports` テーブルに保存します。

- `profile_id`
- `start_date`
- `end_date`
- `dataset_path`
- `summary_path`
- `record_count`
- `created_at`

## 注意

- 秘密情報を含めない
- APIキーを含めない
- raw news本文を含めない
- `analysis_logs/` はGit管理しない
- AI改善案は研究・実験用であり、投資助言ではない
