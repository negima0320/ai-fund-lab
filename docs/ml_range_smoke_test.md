# ML Range Smoke Test

ML Phase 11 adds a short multi-day smoke command that runs the minimum cached-data loop:

1. Generate daily labels.
2. Generate daily features.
3. Build and time-split the dataset.
4. Train the four current LightGBM models.
5. Predict each processed day.
6. Evaluate predictions against labels.
7. Save a multi-day Markdown summary under `reports/ml/`.

Run the small default check:

```bash
python3 scripts/ml/smoke_ml_range.py \
  --start 2026-05-09 \
  --end 2026-05-15 \
  --train-end 2026-05-13 \
  --valid-end 2026-05-15
```

This command only reads existing `data/cache/jquants/` cache and existing ML parquet roots. It does not call J-Quants API, does not use `data/processed/`, and does not run full-period training.

The summary prints processed/skipped dates, feature and label row counts, dataset/train/valid/test rows, total prediction and joined evaluation rows, risk-label bad-entry rates, top-10 average `future_10d_return`, and the correlation between `expected_return_10d` and `future_10d_return`.
