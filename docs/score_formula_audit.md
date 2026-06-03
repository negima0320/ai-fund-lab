# Score Formula Audit

`feature_analysis.md` / `feature_analysis.json` の `Score Formula Audit` と `Score Effective Range Audit` の読み方です。

## Purpose

Score auditは、profile設定と実際の `score_components` が一致しているかを確認するための監査です。

確認する主な点:

- active componentだけが `total_score` に入っているか
- 古い固定componentが残っていないか
- `score_components.component_total` と `total_score` が一致するか
- profileのmin scoreが有効レンジに対して過度に高すぎないか
- fallback/top-pick selectionを通常min score違反と誤判定していないか

## Current Formula

現在のスコア式は次です。

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

`market_context_score` は現在0固定です。財務サマリはaudit/future candidateとして扱い、通常の `total_score` には加算しません。

## Theoretical Range vs Effective Range

`technical_score` は最大50です。relative strengthやinvestor contextなどが有効なprofileでは理論上の上限が広がります。ただし、実際の候補はscreeningで絞られ、RSI/出来高/市場区分などの条件もあるため、観測される `total_score` は理論上限より低くなります。

`Score Effective Range Audit` では、実際のscored candidatesから以下を見ます。

- observed min / max / average
- selected score range
- min_score以上の候補数
- profile別追加componentの発火数
- `invalid_below_threshold_selected_count`

## Integrity Interpretation

`allow_top_pick_when_no_selection` によるtop-pick採用は、profile設定上の正常なfallbackです。通常min score未満で選ばれていても、fallbackとして記録されていればScore Integrityの異常とは扱いません。

警告・エラー対象は、通常選定なのに有効なmin scoreを下回ったケースです。

Key fields:

| field | Meaning |
| --- | --- |
| `selected_below_regular_min_score_count` | 通常min score未満で選ばれた件数。fallbackを含む場合があります |
| `fallback_top_pick_selected_count` | no-trade回避fallbackとして選ばれた件数 |
| `invalid_below_threshold_selected_count` | profile設定で許可されない閾値未満選定。strict-integrityの失敗対象 |

## Feature Activation Audit

`features.*` と `scoring.use_*` は分けて見ます。

| audit field | Meaning |
| --- | --- |
| `data_enabled` | 特徴量生成・API/cache利用が有効 |
| `scoring_enabled` | `total_score` への加算が有効 |
| `actual_trigger_count` | 実際に値が発火した件数 |
| `registry_enabled` | `profile_registry.yaml` 上の識別用feature |

data-only profileでは `data_enabled=true` でも `scoring_enabled=false` です。この場合、値が保存・監査されても `total_score` には入りません。

