# System Flow

現在のAI Fund Labの主要フローです。

## Daily / Backtest Pipeline

```text
profile load
  -> listed info / prices / calendar load
  -> indicators
  -> market context
  -> candidate universe
  -> market filter
  -> screening
  -> scoring
  -> selection
  -> PaperBroker simulation
  -> logs / DB / reports
  -> feature analysis / experiment summary
```

## Source of Truth

- Config: `config/profiles/*.yaml`, `config/profile_registry.yaml`
- Scoring: `src/scoring.py`
- Screening: `src/real_screening.py`
- Paper execution: `src/paper_trade.py`
- Feature reports: `src/feature_analysis.py`
- CLI orchestration: `src/main.py`

## Runtime Paths

- Raw/provider cache: `data/raw/`, `data/cache/jquants/`
- Processed cache: `data/processed/common/`, `data/processed/<profile_id>/`
- Daily logs: `logs/<kind>/<profile_id>/`
- Backtest logs: `logs/backtests/<profile_id>/<START>_to_<END>/`
- Reports: `reports/<profile_id>/backtests/`, `reports/experiments/`

## AI / OpenAI Boundary

OpenAI is optional. The deterministic scoring and selection path is implemented in Python. AI commentary or decision logs may be generated when enabled, but AI output does not rewrite rules unless code/config is explicitly changed.

## Safety Boundary

PaperBroker is the normal execution path. Tachibana is read-only guarded, and KabuStation is a stub. Live automatic trading is not enabled.

