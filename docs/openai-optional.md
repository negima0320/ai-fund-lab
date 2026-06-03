# OpenAI Optional Usage

OpenAI連携は任意です。`OPENAI_API_KEY` が未設定でも、データ取得、screening、scoring、PaperBroker、backtest、feature analysisは動作します。

## Current Role

OpenAIは、現在の主スコア式の必須componentではありません。

現在のスコア計算は `src/scoring.py::score_real_candidates()` が行い、technical score、relative strength、investor context、各種penalty/adjustmentを使います。旧式の「AIがニュース30点・財務20点を固定採点する」仕様は現行の `total_score` では使っていません。

## Fallback

OpenAIが未設定または失敗した場合:

- commentaryはrule-basedへfallback
- AI decisionが必要な箇所は安全側にfallback
- scoring本体は停止しない

## Where Outputs May Appear

- `logs/ai_decision/<profile_id>/ai_decision_YYYY-MM-DD.json`
- article/commentary generation
- optional analysis text

## Safety

OpenAI出力は売買ルールを勝手に変更しません。改善案やコメントとして保存される場合でも、profile YAMLや実装を変更しない限り、選定・買付ロジックには反映されません。

このプロジェクトは投資助言ではなく、研究・検証・PaperBroker前提です。

