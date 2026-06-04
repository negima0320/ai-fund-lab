# J-Quants Data Fetching Inventory

This document summarizes the current J-Quants API fetch/cache boundary. It is intentionally limited to raw/cache acquisition paths and does not cover `data/processed/`, ML features, training, or prediction.

## Common Boundary

- HTTP client: `src/data_provider.py` `JQuantsDataProvider`
- Reusable cache/fetch service: `src/data_provider.py` `JQuantsDataService`
- Endpoint registry: `src/data_provider.py` `JQUANTS_ENDPOINTS`
- Base URL: `https://api.jquants.com/v2`
- Auth: `x-api-key` from `JQUANTS_API_KEY`
- Pagination: `JQuantsDataProvider._get_paginated_records`
- Cache payload shape: `{"fetched_at": "...", "records": [...]}` for service/provider caches

## API Calls

| Logical endpoint | J-Quants path | Current caller functions |
| --- | --- | --- |
| `listed_info` | `/equities/master` | `JQuantsDataProvider.get_listed_stocks`; `main.run_list_stocks`; `main._check_jquants_health`; `main.build_jquants_smoke_test` via `JQuantsDataService.fetch_listed_info_cached`; `screening.generate_screening_log` through `BaseDataProvider` |
| `prices` | `/equities/bars/daily` | `JQuantsDataProvider.get_daily_prices`; `JQuantsDataProvider.get_daily_prices_range`; `main.run_fetch_prices`; `main.fetch_price_history` via `fetch_daily_prices_with_rate_limit_retry`; `main.build_jquants_smoke_test` via `JQuantsDataService.fetch_daily_prices_cached`; `screening.generate_screening_log` through `BaseDataProvider` |
| `financial_statements` | `/fins/summary` | `JQuantsDataProvider.fetch_financial_statements`; `JQuantsDataProvider.fetch_financial_statements_cached`; `main._load_financial_statements_for_period`; `main.build_jquants_smoke_test` via `JQuantsDataService.fetch_financial_statements_cached` |
| `earnings_calendar` | `/equities/earnings-calendar` | `JQuantsDataProvider.fetch_earnings_calendar`; `JQuantsDataProvider.fetch_earnings_calendar_cached`; `JQuantsDataProvider.fetch_earnings_calendar_period_cached`; `main._load_earnings_calendar_for_date`; `main._load_earnings_calendar_for_period`; `main.build_jquants_smoke_test` via `JQuantsDataService` |
| `trading_calendar` | `/markets/calendar` | `JQuantsDataProvider.get_trading_calendar`; `main.build_jquants_smoke_test` via `JQuantsDataService.fetch_trading_calendar_cached` |
| `investor_types` | `/equities/investor-types` | `JQuantsDataProvider.fetch_investor_types`; `JQuantsDataProvider.fetch_investor_types_cached`; `main._load_investor_context_for_date`; `main._load_investor_context_for_period`; `main.build_jquants_smoke_test` via `JQuantsDataService.fetch_investor_types_cached` |
| `topix_prices` | `/indices/bars/daily/topix` | `JQuantsDataProvider.get_topix_prices`; `JQuantsDataProvider.fetch_topix_prices`; `JQuantsDataProvider.fetch_topix_prices_cached`; `main._load_topix_prices_for_period`; `main.build_jquants_smoke_test` via `JQuantsDataService.fetch_topix_prices_cached` |

## Save Locations

| Data | Save path | Naming | Cache use |
| --- | --- | --- | --- |
| Listed stock raw snapshot | `data/raw/listed_stocks_jquants.json` | Fixed filename | Written by `main.run_list_stocks`; read by master loaders |
| Prime stock raw snapshot | `data/raw/prime_stocks_jquants.json` | Fixed filename | Written by `main.run_list_stocks`; fallback master path |
| Listed info API cache | `data/cache/jquants/listed_info/*.json` | `YYYY-MM-DD.json` | Used by `JQuantsDataService.fetch_listed_info_cached` and smoke test |
| Daily prices raw snapshot | `data/raw/prices_YYYY-MM-DD.json` | One file per ISO date | Written by `main.run_fetch_prices` and `main.cache_price_snapshot`; read by backtest/selection loaders |
| Daily prices API cache | `data/cache/jquants/prices/*.json` | `YYYY-MM-DD.json` | Used by `JQuantsDataService.fetch_daily_prices_cached`; also read as fallback by `load_cached_prime_prices_from_jquants_cache` |
| Financial statements | `data/cache/jquants/financial_statements/*.json` | `YYYY-MM-DD_to_YYYY-MM-DD.json` | Used by `JQuantsDataProvider.fetch_financial_statements_cached` and service pass-through |
| Earnings calendar | `data/cache/jquants/earnings_calendar/*.json` | `YYYY-MM-DD.json` or `YYYY-MM-DD_to_YYYY-MM-DD.json` | Used by provider cached methods and service pass-through |
| Trading calendar | `data/cache/jquants/trading_calendar/*.json` | `YYYY-MM-DD_to_YYYY-MM-DD.json` | Used by `JQuantsDataService.fetch_trading_calendar_cached` and smoke test |
| Investor types | `data/cache/jquants/investor_types/*.json` | `YYYY-MM-DD_to_YYYY-MM-DD.json` | Used by provider cached methods and service pass-through |
| TOPIX prices | `data/cache/jquants/topix_prices/*.json` | `YYYY-MM-DD_to_YYYY-MM-DD.json` | Used by provider cached methods and service pass-through |
| Empty ranges | `data/cache/jquants/empty_ranges.json` | Fixed filename | Records empty API ranges by logical endpoint |
| Unsupported ranges/days | `data/cache/jquants/unsupported_ranges.json`, raw no-data/unsupported day helpers | Fixed/range entries | Used by historical price fetch guards |

## Commonization Policy

1. Keep `JQuantsDataProvider` as the low-level HTTP boundary: auth, rate limiting, pagination, response extraction, and endpoint-specific normalization.
2. Use `JQuantsDataService` as the reusable acquisition boundary for callers that need persisted J-Quants data. New ML/AI prediction code should depend on this service rather than calling `_get_paginated_records` or writing cache files directly.
3. Keep raw application snapshots in `data/raw/` for existing backtests and daily operations. Keep API response caches in `data/cache/jquants/`.
4. Add new endpoints by extending `JQUANTS_ENDPOINTS`, then adding a service method with a stable cache filename. Avoid duplicating path strings in `src/main.py`.
5. Do not write or read `data/processed/` from the J-Quants acquisition layer.

## Light Verification

- Static import/compile check: `PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m py_compile src/data_provider.py src/main.py`
- Focused J-Quants tests: `PYTHONPYCACHEPREFIX=/private/tmp/ai-fund-lab-pycache python3 -m pytest tests/test_jquants_api_cache.py tests/test_investor_context.py tests/test_relative_strength.py`
- Smoke command that may call J-Quants, so run only when API access is intended: `python3 src/main.py --mode jquants-smoke-test --endpoint listed_info --provider jquants`
