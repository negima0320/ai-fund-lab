# Analysis and AI Logs

この文書は分析ログの現在の位置づけです。OpenAI利用は任意で、スコアリング本体はPython実装です。

## Daily Logs

```text
logs/screening/<profile_id>/screening_YYYY-MM-DD.json
logs/scoring/<profile_id>/scoring_YYYY-MM-DD.json
logs/ai_decision/<profile_id>/ai_decision_YYYY-MM-DD.json
logs/market_context/<profile_id>/market_context_YYYY-MM-DD.json
logs/trades/<profile_id>/trades_YYYY-MM-DD.json
logs/portfolio/<profile_id>/portfolio_YYYY-MM-DD.json
logs/safety/<profile_id>/safety_YYYY-MM-DD.json
logs/reflections/<profile_id>/reflections_YYYY-MM-DD.json
```

## Backtest Logs

```text
logs/backtests/<profile_id>/<START>_to_<END>/
  backtest_summary.json
  scoring_YYYY-MM-DD.json
  trades_YYYY-MM-DD.json
  portfolio_YYYY-MM-DD.json
  summary.csv
  trades.csv
```

## Scoring Logs

Scoring logs contain current score components such as:

- `technical_score`
- `ma_score`
- `volume_score`
- `rsi_score`
- `candlestick_score`
- `sector_score`
- `relative_strength_score`
- `investor_context_score`
- `market_context_score`
- `winner_loser_rule_score`
- `penalty_score`
- `total_score`
- `score_components`

They do not use the old fixed `news_score=30` / `financial_score=20` formula.

## Feature Analysis Reports

`reports/<profile_id>/backtests/feature_analysis.md/json` is the main audit report. Important sections include:

- Score Formula Audit
- Feature Activation Audit
- Backtest Result Integrity Audit
- Score Integrity Audit
- Market Filter Audit
- Candidate Universe Audit
- Screening Audit
- Scored Candidate Audit
- Selected Candidate Audit
- Trade Market Audit
- Affordable Fallback Buy Audit
- Compounding / Capital Flow Audit
- Monthly Performance Audit

## AI Decision Logs

`logs/ai_decision/` is optional. If OpenAI is unavailable, rule-based fallback can be used. AI comments are not the source of truth for score calculation.

