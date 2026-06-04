# ML Smoke Test

Lightweight smoke command for the local ML pipeline.

```bash
python3 scripts/ml/smoke_ml_pipeline.py --date 2026-05-29
```

What it does:

- builds one day of features from `data/cache/jquants/`
- updates labels available as of the same date
- separately checks whether labels for the target date itself can be generated, without saving them
- skips prediction with a warning when `models/ml/current/` is missing
- prints rows and columns for generated parquet files

It does not call J-Quants API, OpenAI API, `data/processed/`, training, backtests, or trading integration.
