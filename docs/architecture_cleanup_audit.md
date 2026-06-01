# Architecture Cleanup Audit

This note summarizes the current duplication hotspots after the incremental additions around J-Quants, experiments, analysis, logs, and caches.

## Current Hotspots

### `src/main.py` owns too many responsibilities

`src/main.py` is now the central owner for CLI dispatch, date resolution, J-Quants preload orchestration, cache probing, backtest loops, experiment execution, smoke tests, clean commands, report rendering, and many file utilities. This makes feature routing bugs likely because the same concern can be patched in multiple places.

Recommended split:

- `backtest_runner.py`: date resolution, price history preparation, daily backtest loop
- `jquants_cache.py`: cache paths, usable-cache checks, empty/unsupported range markers
- `jquants_smoke.py`: smoke-test endpoint dispatch and result rendering
- `experiment_runner.py`: registry-driven batch execution and verdict rendering
- `report_paths.py`: report/log/output path conventions

### Price cache has two active formats

There are two valid price cache sources:

- Legacy raw cache: `data/raw/prices_YYYY-MM-DD.json`
- J-Quants endpoint cache: `data/cache/jquants/prices/YYYY-MM-DD.json`

Backtest date detection now reads both, but this is still a compatibility bridge. A single price-cache repository should become the source of truth.

Recommended direction:

- Keep `data/cache/jquants/prices/YYYY-MM-DD.json` as the canonical API cache.
- Treat `data/raw/prices_YYYY-MM-DD.json` as legacy input only.
- Put all price cache reads behind one function or class.
- Stop writing new price API results to two unrelated locations.

### Processed files must not decide data availability

`ensure_price_history_for_backtest` previously returned early when generic `data/processed/indicators_YYYY-MM-DD.json` files existed. That mixed processing outputs into raw data availability and could hide missing price cache for extended end dates.

Fixed:

- Price-history preparation no longer short-circuits based on processed indicator files.
- It checks price caches and missing dates directly.

### Light API caches repeat the same pattern manually

The following endpoints use similar cache/read/write/retry/empty handling:

- `topix_prices`
- `investor_types`
- `earnings_calendar`
- `financial_statements`

Much of the policy is duplicated between `src/data_provider.py` and orchestration wrappers in `src/main.py`.

Recommended direction:

- Define one endpoint cache contract:
  - cache path
  - record extractor
  - records usable predicate
  - empty response policy
  - fallback policy
  - log fields
- Keep HTTP fetching in `JQuantsDataProvider`.
- Keep backtest preload decisions outside the provider.

### Date range logic is spread across several paths

The system now has multiple related dates:

- requested start/end
- effective trade start/end
- indicator fetch start
- TOPIX lookback start
- earnings filter fetch window
- investor context fetch window
- latest available price date

The audit fields added to backtest summaries help detect mismatches, but the calculation itself should be centralized.

Recommended direction:

- Create a `BacktestDateRange` value object.
- Pass it through backtest, preloads, report summary, and experiments.
- Avoid recomputing period boundaries inside each feature.

### Logs and reports are also mixed

Daily backtest logs, scoring logs, J-Quants API logs, reports, article drafts, and experiment summaries are written from several layers.

Recommended direction:

- Keep API logs endpoint-oriented.
- Keep scoring logs profile/date-oriented.
- Keep analysis reports derived from DB/logs only.
- Avoid generating article/report details during backtest unless explicitly enabled.

## Recommended Cleanup Order

1. Canonicalize price cache reads and writes.
2. Move J-Quants endpoint cache policy into a small shared module.
3. Extract backtest date resolution into a single object.
4. Move run-experiments out of `main.py`.
5. Move smoke-test endpoint handling out of `main.py`.
6. Make processed cache invalidation explicit with profile/config/data-source fingerprints.

This order keeps behavior stable while reducing the duplicated logic that has caused recent date, cache, and feature-activation issues.
