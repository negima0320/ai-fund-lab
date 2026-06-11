# Phase 13 Horizon-Aware Selection / Valuation / Exit Redesign

Last updated: `2026-06-11`

## Executive Summary

Phase 11 and Phase 12 confirmed that the newer Valuation / Downside direction
has real signal, but also exposed a deeper architecture problem:

```text
Stock Selection AI      -> short horizon, mainly 5d / 10d / swing composite
Valuation / Downside    -> 20d opportunity and downside-aware quality
Exit / Hold             -> still partly fixed-horizon and threshold-driven
Capital Allocation      -> can raise utilization, but DD can explode
```

Phase 13 should redesign the stack around explicit horizons:

```text
5d / 10d / 20d / 40d
```

The goal is not just to improve ranking. The goal is to align:

```text
candidate generation
valuation
capital allocation
exit / hold
```

so that each layer knows which time horizon it is serving.

## Phase 11 Summary

Phase 11 replaced direct PM multiplier redevelopment with:

```text
Valuation Engine
↓
Capital Allocation Engine
```

The Valuation Engine was intended to answer:

```text
Is this stock currently attractive relative to price?
```

Planned outputs:

```text
opportunity_score
expected_upside
expected_downside
confidence
```

Key findings:

- Phase 11-A built a leakage-safe valuation dataset: `930,243` rows, `4,234`
  unique codes, `55` features, `5` labels, `leakage_risk=low`.
- Phase 11-B showed classification was more useful than regression:
  precision@top10% `0.1998` vs base positive rate `0.0997`.
- Phase 11-D showed Valuation improved BUY quality and profit in 2025-only
  limited testing, but DD worsened.
- Phase 11-E/F showed Opportunity Exit + Stop Loss could control DD in 2025,
  but introduced turnover and threshold dependency.
- Phase 11-I strict OOS preserved rank lift but failed strategy adoption
  thresholds.
- Phase 11-B2 identified the core issue: strict OOS valuation top5 improved
  top-decile rate `8.85% -> 24.00%`, but downside_bad_rate worsened
  `13.58% -> 37.94%`.
- Phase 11-B3 added a Downside model and confirmed downside risk can be
  detected, but simple combined scores either cut too much opportunity or left
  too much downside.

Phase 11 conclusion:

```text
Opportunity signal is real.
Opportunity alone is not enough.
Downside must be modeled separately.
```

## Phase 12 Summary

Phase 12 researched Dynamic Capital Allocation using:

```text
Opportunity
+
Downside
+
Confidence / relative rank
```

The intended behavior:

```text
Opportunity high + Downside low  -> buy larger
Opportunity high + Downside high -> buy smaller
Opportunity low                  -> avoid
```

Major findings:

- Phase 12-A/A2/A3 found `A3_3_rank_medium_floor_zero` could preserve
  opportunity while reducing downside:
  - weighted top-decile rate: `0.2614`
  - weighted downside_bad_rate: `0.1432`
  - weighted opportunity value: `0.0683`
- Phase 12-B/B2 showed raw weights are safe but underuse capital, while
  normalized weights use capital but worsen DD.
- Phase 12-B5 found the Opportunity Exit was too aggressive. Relaxing proba
  drop threshold improved:
  - net profit: `71,922`
  - PF: `2.1827`
  - DD: `-3.24%`
  - average holding days: `13.34`
  - capital utilization: `16.13%`
- Phase 12-C showed normalized allocation can reach high utilization and
  profit, but DD worsened:
  - profit: `306,382`
  - PF: `2.0680`
  - DD: `-18.88%`
  - utilization: `90.76%`
- Phase 12-C2/C3/C4 attributed much of the DD to single-name concentration.
  Blunt concentration caps reduced concentration but damaged PF/DD, implying
  concentration is not always bad; the system cannot yet distinguish good
  exposure from bad exposure.
- Phase 12-D1 found winning-to-losing conversion:
  - peak `>= +5%` then final loss: `7` trades
  - realized loss: `-157,341`
  - estimated recoverable peak-to-final leakage: `328,000`
- Phase 12-D2 confirmed Valuation + Downside improve BUY quality, while Stock
  Selection top5 alone does not beat the universe.
- Phase 12-D3 confirmed the 2025 inputs are strict OOS by existing artifact
  evidence:

```text
stock_selection_strict_oos_for_2025: true
valuation_strict_oos_for_2025: true
downside_strict_oos_for_2025: true
phase12_results_trustworthy: true
leakage_risk: low
blocking_issues: 0
```

- Phase 12-E1/E2 showed Stock Selection is clean from a leakage perspective but
  misaligned with Phase 12's 20d objective.

## Confirmed Findings

### Valuation Works

Valuation top5 improved top-decile capture:

```text
candidate_universe top-decile rate: 0.1053
stock_selection_rank_score_top5: 0.0885
opportunity_top5: 0.2400
```

### Downside Works

Opportunity alone catches high downside. Downside penalty can reduce it:

```text
opportunity_top5 downside_bad_rate: 0.3794
A3_3 weighted downside_bad_rate: 0.1432
```

### Stock Selection Top5 Is Not Valid For Phase 12

Phase 12-E1:

```text
stock_selection_adds_value: false
stock_selection_top5_valid: false
stock_selection_prefilter_hurts_valuation: true
```

Stock Selection AI itself is not "bad" in an absolute sense. The issue is that
its current score is not aligned to the Phase 12 target.

### Candidate Strength Is Closer To Phase 12

Phase 12-E2 found:

```text
candidate_strength =
expected_max_return_20d
+
swing_success_probability_20d
-
bad_entry_probability_10d
```

This is closer to a 20d opportunity / swing target than `stock_selection_rank_score`.

### Exit / Hold Is A Major Bottleneck

Phase 12-D1 showed the system sometimes lets winners become losers. That is a
Hold / Exit design problem, not just a BUY or allocation problem.

## Open Risks

- Phase 12 still has no adoption-ready strategy.
- Dynamic normalized allocation can create high utilization but large DD.
- Blunt concentration caps remove both bad exposure and good exposure.
- Opportunity Exit thresholds are fragile.
- Stop-loss can be too late after a trade has already had meaningful profit.
- Stock Selection prefilter may discard Valuation's best candidates.
- 40d labels / long-horizon behavior are not yet fully integrated into the
  research artifact set.
- Phase 13 must not blur horizons again by mixing short-horizon scores with
  medium-horizon exit decisions.

## Why Phase 13 Is Needed

The root problem is horizon mismatch.

Current stack:

```text
Stock Selection AI
  5d / 10d return, 10d upside, 10d bad-entry, 20d max-return / swing

Valuation Engine
  20d opportunity / downside

Exit / Hold
  fixed holding days + opportunity threshold + stop loss

Capital Allocation
  opportunity/downside weighted, but not horizon-aware
```

This creates conflicting behavior:

```text
short-horizon selector
↓
20d valuation
↓
fixed-ish exit
↓
allocation forced to compensate for mismatched signals
```

Phase 13 exists to make the horizon explicit before implementing the next
generation of models or strategy checks.

## Phase 13 Concept

Phase 13 redesigns the system as horizon-aware:

```text
Horizon-Aware Candidate Generation
↓
Horizon-Aware Valuation
↓
Dynamic Capital Allocation
↓
Hold / Exit Decision
```

Each layer should declare:

```text
horizon
input scores
labels
decision rule
failure mode
```

Example:

| horizon | candidate role | valuation role | exit role |
| --- | --- | --- | --- |
| 5d | short momentum / entry timing | short upside | quick invalidation |
| 10d | swing entry | upside / bad-entry | early risk guard |
| 20d | core opportunity | opportunity + downside | hold vs profit protect |
| 40d | trend continuation | longer-horizon opportunity | extended hold / trailing lock |

## Architecture Diagram

```text
                       ┌──────────────────────────┐
                       │ Prediction-time features │
                       │ Price / Candle / Volume  │
                       │ Financial / Market       │
                       └─────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ Stock Selection  │   │ Valuation Engine │   │ Downside Engine  │
   │ horizon-specific │   │ horizon-specific │   │ horizon-specific │
   │ 5d/10d/20d/40d   │   │ opportunity      │   │ risk / drawdown  │
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │                      │                      │
            └──────────────┬───────┴──────────────┬───────┘
                           ▼                      ▼
                ┌──────────────────────┐  ┌──────────────────────┐
                │ Candidate Generation │  │ Allocation Engine    │
                │ no/top50/top100      │  │ score + cap + cash   │
                └──────────┬───────────┘  └──────────┬───────────┘
                           │                         │
                           └────────────┬────────────┘
                                        ▼
                              ┌────────────────┐
                              │ Hold / Exit AI │
                              │ hold / protect │
                              │ stop / sell    │
                              └────────────────┘
```

## Phase 13 Roadmap

### Phase 13-A: Horizon Reality Audit

Purpose:

```text
Find which horizon actually explains profit, opportunity, downside, and hold value.
```

Compare:

- Stock Selection scores
- `candidate_strength`
- Opportunity
- Opportunity + Downside

Evaluate:

- `future_return_5d`
- `future_return_10d`
- `future_return_20d`
- `future_return_40d`
- future max return
- future drawdown
- top-decile rate
- downside bad rate

Output:

- horizon quality table
- score monotonicity by horizon
- recommendation for candidate generation horizon
- recommendation for exit/hold horizon

### Phase 13-B: Candidate Generation Redesign

Purpose:

```text
Decide whether Stock Selection should be removed, widened, or replaced.
```

Compare:

- no Stock Selection prefilter
- Stock Selection top50
- Stock Selection top100
- Candidate Strength top50
- Candidate Strength top100
- Valuation only
- Valuation + Downside

Rule:

```text
Top5 Stock Selection prefilter should not be the default unless it clearly adds value.
```

### Phase 13-C: Horizon-Aware Valuation Prototype

Purpose:

```text
Separate 5d / 10d / 20d / 40d opportunity and downside definitions.
```

Research questions:

- Is 20d still the best valuation horizon?
- Does 40d identify the trades that should be held longer?
- Does 5d/10d help entry timing rather than final selection?
- Can downside be horizon-specific?

No production overwrite. Any new model must be research-only and separate from
existing current models.

### Phase 13-D: Hold / Exit AI Dataset Audit

Purpose:

```text
Design labels for HOLD / SELL / PROFIT_PROTECT / STOP.
```

Candidate labels:

- hold was correct
- sell was correct
- profit protection was needed
- break-even guard was needed
- trailing lock would have helped
- stop-loss was too late

This phase should directly use the Phase 12-D1 finding that peak-profit trades
sometimes became final losses.

### Phase 13-E: Integrated Strategy Prototype

Purpose:

```text
Combine horizon-aware candidate generation, valuation, dynamic allocation,
and hold/exit logic in a limited-year strategy prototype.
```

Scope:

- limited year only
- small number of variants
- no production profile changes
- no model overwrite

## Success Criteria

Research-level targets:

```text
PF >= 2.0
DD <= -10%
capital_utilization >= 60%
net_profit positive after 0.2% one-way cost
```

Long-term system target:

```text
annualized return >= 50%
PF >= 2.0
DD <= -10%
capital utilization >= 60%
```

Phase 13-A/B success is not measured by backtest profit. It is measured by
clear evidence about which horizon and candidate-generation path should be used.

## Implementation Constraints

Until explicitly approved for a later phase:

- no new AI training
- no full backtest
- no 2023-2026 all-period backtest
- no prediction regeneration
- no profile addition
- no profile modification
- no existing model overwrite
- no J-Quants API refetch
- no OpenAI API
- no long-running variant sweeps

Allowed for Phase 13-A:

- existing artifacts
- existing reports
- 2025-limited audit
- small pytest
- md/json report

Leakage rule:

```text
future columns may be labels/evaluation only.
future columns must not be selection, allocation, or exit features.
backtest/trade/cash/portfolio/selected/bought/current PM data must not be model features.
```

## Next Action

Proceed to:

```text
Phase13-A Horizon Reality Audit
```

Recommended first implementation:

```text
src/ml/phase13a_horizon_reality_audit.py
scripts/ml/run_phase13a_horizon_reality_audit.py
tests/test_ml_phase13a_horizon_reality_audit.py
reports/ml/phase13a_horizon_reality_audit_2025.md
reports/ml/phase13a_horizon_reality_audit_2025.json
```

The audit should answer:

```text
Which horizon should drive candidate generation?
Which horizon should drive valuation?
Which horizon should drive hold/exit decisions?
Should Stock Selection be removed, widened, or rebuilt?
```

