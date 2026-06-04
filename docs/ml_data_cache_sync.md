# ML Raw Price Cache Sync

Convert existing raw price snapshots into the ML J-Quants cache layout.

```bash
python3 scripts/ml/sync_raw_prices_to_jquants_cache.py --date 2026-05-20 --dry-run
python3 scripts/ml/sync_raw_prices_to_jquants_cache.py --date 2026-05-20
python3 scripts/ml/sync_raw_prices_to_jquants_cache.py --start 2026-05-01 --end 2026-05-31 --dry-run
```

Input:

- `data/raw/prices_YYYY-MM-DD.json`

Output:

- `data/cache/jquants/prices/YYYY-MM-DD.json`

The output payload is:

```json
{
  "records": [
    {
      "date": "2026-05-20",
      "code": "1001",
      "open": 100.0,
      "high": 105.0,
      "low": 99.0,
      "close": 104.0,
      "volume": 1000.0,
      "turnover_value": 104000.0
    }
  ]
}
```

The tool does not call APIs or read `data/processed/`. Existing cache files are not overwritten unless `--overwrite` is supplied.
