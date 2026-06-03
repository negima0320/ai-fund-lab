# Documentation Audit Report

Updated: 2026-06-03

This report records the documentation棚卸し performed against the current implementation. The source of truth is the code, profile YAMLs, profile registry, CLI output, and generated report fields. Existing docs were treated as untrusted until checked.

## Verified Sources

| Area | Verified source |
| --- | --- |
| CLI modes/options | `python3 src/main.py --mode help`, `src/main.py` |
| Profile loading | `src/profile_loader.py`, `config/profiles/*.yaml` |
| Experiment registry | `config/profile_registry.yaml`, `src/profile_registry.py`, `src/main.py` |
| Scoring formula | `src/scoring.py::score_real_candidates()` |
| Selection rules | `src/scoring.py::_apply_selection_rules()` |
| Paper execution/capital rules | `src/paper_trade.py` |
| Feature reports | `src/feature_analysis.py` |
| Broker status | `src/broker.py`, `src/demo_auto_order.py` |
| J-Quants plan/fallback | `config/jquants.yaml`, `src/jquants_plan.py`, provider/cache code |

## Main Corrections

| Old or risky wording found | Current implementation | Documentation action |
| --- | --- | --- |
| Project described as heading toward automatic real trading without enough caveats | PaperBroker is the normal execution path. Tachibana is read-only guarded. KabuStation is a stub. Live auto trading is blocked/not operated | README and broker docs now state research/PaperBroker premise and avoid investment advice wording |
| Old score model described as 100 points with news 30 and financial 20 | `score_real_candidates()` uses technical max 50 plus active profile components such as relative strength, investor context, affordability/winner-loser adjustments, and penalties. News/fixed financial score is not used in current total score | Scoring docs rewritten. Old formula is marked as obsolete, not current |
| `features.*` and `scoring.use_*` treated as the same switch | `features.*` enables data/feature generation; `scoring.use_*` decides whether the generated feature affects `total_score` | README and scoring docs now separate data_enabled/scoring_enabled |
| Profile examples focused on `rookie_dealer_01` only | Default loader is `rookie_dealer_01`, but active experiments are v2 profiles. Registry baseline is `rookie_dealer_02_v2_1`; many current experiments compare to `rookie_dealer_02_v2_26` | README clarifies default vs experiment baseline |
| Registry role semantics not fully documented | `baseline`, `experiment`, `deprecated`; experiments must point to `compare_to` and pass registry validation for `run-experiments` | README and operations docs updated |
| Market filter docs implied Prime-only as universal | Market sections are profile-dependent through `market_filter.allowed_sections`. Unknown is blocked when `allow_unknown_market=false` | Decision flow and scoring docs updated |
| Standard/Growth handling was unclear | Standard/Growth are allowed only in configured profiles. Audits separate market filter, screening, scoring, selection, and trade stages | Decision flow and report docs updated |
| `final_assets` and `net_cumulative_profit` could be read as the same reconciliation target | Capital flow audit reconciles `final_assets` using realized + unrealized + external cash flow. `net_cumulative_profit` is a report metric and may not equal mark-to-market final profit | README and operations docs updated |
| Logs/cache hierarchy was stale | CLI help lists current `data/raw`, `data/cache/jquants`, `data/processed/common`, profile runtime processed path, backtest logs, reports, and storage | `docs/log-design.md` rewritten |
| Tachibana docs overstated future/live flow or called everything unimplemented | Read-only broker and guarded demo flow exist; actual external order placing remains disabled | `docs/tachibana-plan.md` updated |
| KabuStation docs could imply near-term implementation | Current KabuStation broker is a disabled stub | `docs/kabu-station-plan.md` updated |
| J-Quants plan notes were incomplete | Free/light capability matrix and fallback behavior are implemented in `src/jquants_plan.py` and `config/jquants.yaml` | `docs/jquants_plan_matrix.md` updated |

## Current CLI Checks

These lightweight commands were used or should remain valid:

```bash
python3 src/main.py --mode help
python3 src/main.py --mode list-profiles
python3 src/main.py --mode validate-config
python3 src/main.py --mode profile-info --profile rookie_dealer_02_v2_51
```

`validate-config` currently succeeds with warnings rather than errors when the local config has known warnings such as unregistered legacy profile files or `auto_order_enabled: true` in safety-reviewed demo settings.

## Documentation Boundary

The docs intentionally do not:

- include API keys, account identifiers, or secrets
- present strategy outputs as investment advice
- claim live automatic trading is available
- claim Standard/Growth experiments are profitable or production-ready
- claim financial statements are currently part of the main `total_score`
- claim old news/financial fixed scoring is active

