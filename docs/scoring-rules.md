# Scoring Rules

この文書は現在のスコアリングの短い参照です。詳細は [scoring_spec.md](scoring_spec.md) を参照してください。

## Current Formula

現在の正は `src/scoring.py::score_real_candidates()` です。

```text
total_score =
  max(0,
    technical_score
    + relative_strength_score
    + investor_context_score
    + market_context_score
    + winner_loser_rule_score
    + penalty_score
  )
```

旧仕様の「100点満点、ニュース30点、財務20点」は現在の実装では使っていません。ニュース固定加点や財務固定加点が残っているprofileは `validate-config` で警告対象です。

## Components

| component | Current behavior |
| --- | --- |
| `technical_score` | 最大50。MA、出来高、RSI、ローソク足、sector補正 |
| `relative_strength_score` | profileでdata/scoringの両方が有効な場合に加算 |
| `investor_context_score` | profileでdata/scoringの両方が有効な場合に加減点 |
| `market_context_score` | 現在0固定。地合いは主に選定、risk、dynamic exposure、監査で利用 |
| `winner_loser_rule_score` | 勝ち負け分析由来の実験的な加減点 |
| `penalty_score` | RSI過熱、affordabilityなどの減点 |

## Selection Is Separate From Scoring

`total_score` が付いても、必ず買付候補になるわけではありません。選定では以下も見ます。

- `selection.min_score`
- 市場別min score override
- `confidence`
- `selection.max_selected`
- market filter
- volume filter
- RSI filter
- earnings filter
- investor context filter
- conditional selection
- top-pick fallback

買付時にはさらに100株単位、現金、`allocation_limit`、保有上限、`target_exposure`、`min_cash_buffer` を見ます。

