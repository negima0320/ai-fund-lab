# Runbook

このrunbookは軽量確認とPaperBroker検証用です。実売買やJ-Quants取得を伴う長時間処理は、目的とキャッシュ状態を確認してから実行します。

## Quick Checks

```bash
.venv/bin/python src/main.py --mode help
.venv/bin/python src/main.py --mode validate-config
.venv/bin/python src/main.py --mode list-profiles
.venv/bin/python src/main.py --mode profile-info --profile rookie_dealer_02_v2_51
```

## Before Long Backtests

1. 空き容量を確認する
2. `data/cache/jquants/` と `data/processed/common/` を不用意に削除しない
3. `validate-config --strict` を確認する
4. 対象profileがregistryでbaseのexperimentとして認識されているか確認する
5. `--skip-price-fetch` を使う場合、raw/cacheが揃っているか確認する

## Backtest / Experiment

```bash
.venv/bin/python src/main.py --mode backtest \
  --profile rookie_dealer_02_v2_26 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --skip-price-fetch \
  --fast-analysis \
  --strict-integrity
```

```bash
.venv/bin/python src/main.py --mode run-experiments \
  --base-profile rookie_dealer_02_v2_26 \
  --profiles rookie_dealer_02_v2_26 rookie_dealer_02_v2_51 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --skip-price-fetch \
  --fast-analysis \
  --summary-only \
  --strict-integrity
```

## Report Regeneration

```bash
.venv/bin/python src/feature_analysis.py --profile rookie_dealer_02_v2_51
PYTHONPATH=. .venv/bin/python -m src.feature_analysis --profile rookie_dealer_02_v2_51
```

## Key Files to Inspect

- `reports/<profile_id>/backtests/feature_analysis.md`
- `reports/<profile_id>/backtests/feature_analysis.json`
- `logs/backtests/<profile_id>/<START>_to_<END>/backtest_summary.json`
- `logs/backtests/<profile_id>/<START>_to_<END>/summary.csv`
- `logs/backtests/<profile_id>/<START>_to_<END>/trades.csv`
- `reports/experiments/<START>_to_<END>/<base>/experiment_summary.md`

## Stop Trading Flag

`storage/STOP_TRADING` がある場合、新規買付停止の安全フラグとして扱います。

## Broker Policy

- 通常はPaperBroker
- Tachibanaはread-only guarded
- KabuStationはstub
- live自動売買は使わない

