# Operations

このRunbookは、AI Fund Labを研究・検証・PaperBroker前提で使うための手順です。投資助言ではなく、live自動売買は行いません。

## First Setup

```bash
cd /Users/negishi/work/ai-fund-lab
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/main.py --mode init-db --profile rookie_dealer_02_v2_1
.venv/bin/python src/main.py --mode validate-config
```

`.env` にはJ-Quants等の秘密情報を置きます。APIキーや口座情報はdocsやGit管理対象ファイルに書きません。

## Lightweight Health Checks

```bash
.venv/bin/python src/main.py --mode help
.venv/bin/python src/main.py --mode status --profile rookie_dealer_02_v2_1
.venv/bin/python src/main.py --mode list-profiles
.venv/bin/python src/main.py --mode profile-info --profile rookie_dealer_02_v2_38
.venv/bin/python src/main.py --mode validate-config
```

`validate-config --strict` はwarningも失敗扱いにするため、長期検証前の最終確認に使います。

## Backtest

単体profileの検証:

```bash
.venv/bin/python src/main.py --mode backtest \
  --profile rookie_dealer_02_v2_26 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --skip-price-fetch \
  --fast-analysis \
  --strict-integrity
```

主な出力:

- `logs/backtests/<profile_id>/<START>_to_<END>/backtest_summary.json`
- `logs/backtests/<profile_id>/<START>_to_<END>/summary.csv`
- `logs/backtests/<profile_id>/<START>_to_<END>/trades.csv`
- `reports/<profile_id>/backtests/analysis_latest.md`
- `reports/<profile_id>/backtests/feature_analysis.md`
- `reports/<profile_id>/backtests/selection_quality.md`

## Run Experiments

registry上のbaseとexperimentを同一期間で比較します。

```bash
.venv/bin/python src/main.py --mode run-experiments \
  --base-profile rookie_dealer_02_v2_26 \
  --profiles rookie_dealer_02_v2_26 rookie_dealer_02_v2_38 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --skip-price-fetch \
  --fast-analysis \
  --summary-only \
  --strict-integrity
```

`--profiles` にはbase profileを含めても構いません。`run-experiments` は、指定profileが `config/profile_registry.yaml` 上でbaseのexperimentとして認識されるか検証します。`deprecated` profileは一括実験対象から外れます。

主な出力:

- `reports/experiments/<START>_to_<END>/<base_profile>/experiment_summary.md`
- `reports/experiments/<START>_to_<END>/<base_profile>/experiment_summary.json`
- `reports/experiments/<START>_to_<END>/<base_profile>/compare_profiles.md`
- `reports/experiments/<START>_to_<END>/<base_profile>/compare_profiles.json`

## Feature Analysis Regeneration

既存成果物から `feature_analysis.*` だけ再生成できます。

```bash
.venv/bin/python src/feature_analysis.py --profile rookie_dealer_02_v2_38
PYTHONPATH=. .venv/bin/python -m src.feature_analysis --profile rookie_dealer_02_v2_38
```

期間を指定する場合:

```bash
.venv/bin/python src/feature_analysis.py \
  --profile rookie_dealer_02_v2_38 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD
```

## Reading Experiment Summary

`experiment_summary.md/json` の主な項目です。

| field | Meaning |
| --- | --- |
| `final_assets` | 最終総資産 |
| `net_cumulative_profit` | レポート上の累積損益指標。最終資産検算にはasset reconciliationを優先 |
| `realized_profit_total` | 決済済み実現損益 |
| `unrealized_profit_total` | 期末保有分の含み損益 |
| `profit_factor` | gross profit / gross loss |
| `selection_diff_count` | baseと選定が変わった件数 |
| `outcome_diff_count` | baseと取引・損益結果が変わった件数 |
| `practical_effect` | 差分が実質的に出たか |
| `verdict` | `candidate` / `needs_review` / `rejected` / `no_practical_effect` |
| `verdict_reason` | 判定理由 |

## Conditional Hold Extension Checks

最大保有期間到達時の条件付き保有延長は、`conditional_hold_extension` を持つprofileで検証します。直近のprofile:

- `rookie_dealer_02_v2_59`: v2.58同条件で、保有銘柄indicator補完修正後の検証
- `rookie_dealer_02_v2_60`: v2.59からrelative strength閾値だけを `60` から `5` に緩和

設定確認:

```bash
.venv/bin/python src/main.py --mode validate-config --profile rookie_dealer_02_v2_60
.venv/bin/python src/main.py --mode profile-info --profile rookie_dealer_02_v2_60
```

短期比較を行う場合:

```bash
.venv/bin/python src/main.py --mode run-experiments \
  --base-profile rookie_dealer_02_v2_26 \
  --profiles rookie_dealer_02_v2_26 rookie_dealer_02_v2_60 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --skip-price-fetch \
  --fast-analysis
```

確認する主な出力:

- `logs/backtests/<profile>/<START>_to_<END>/backtest_summary.json`
- `reports/profile_comparisons/compare_<period>_...md/json`
- `conditional_hold_extension_count`
- `conditional_hold_extension_rejected_reason_breakdown`
- `Conditional Hold Extension Rejected Detail`

## Integrity and Capital Flow

`feature_analysis.md` で重要な監査:

- `Backtest Result Integrity Audit`
- `Score Integrity Audit`
- `Market Filter Audit`
- `Compounding / Capital Flow Audit`
- `Monthly Performance Audit`
- `Capital Utilization Audit`
- `Affordable Fallback Buy Audit`
- `Allocation Strategy Audit`
- `Trade Market Audit`

`Compounding / Capital Flow Audit` では `asset_reconciliation` を確認します。

```text
final_assets ≒ initial_capital
  + realized_profit_total
  + unrealized_profit_total
  + external_cash_flow_total
```

`final_assets` と `initial_capital + net_cumulative_profit` が一致しないだけでは、必ずしも資金フロー異常ではありません。`capital_flow_status` と `asset_reconciliation.status` を見ます。

## PaperBroker and Live Safety

通常運用はPaperBrokerです。TachibanaはRead-only brokerとして残高・保有・注文・約定参照IFがありますが、実API発注は `LiveTradingDisabledError` で止まります。KabuStationは現時点ではスタブです。

`storage/STOP_TRADING` を置くと新規買付を止める安全フラグとして扱います。

## Cleanup

大きいファイルは主に以下に溜まります。

- `data/raw/prices_*.json`
- `data/processed/<profile_id>/`
- `data/processed/common/`
- `logs/backtests/<profile_id>/`
- `reports/experiments/`

削除前に `storage-audit` / `cleanup-storage` / `inspect-cache` / `clean-*` 系modeを確認してください。

```bash
.venv/bin/python src/main.py --mode storage-audit
.venv/bin/python src/main.py --mode inspect-cache --profile rookie_dealer_02_v2_38
```

不要になったprofile成果物は、retired profile専用のdry-runで対象を確認します。raw価格、`data/processed/common/`、`data/cache/jquants/` は対象外です。

```bash
.venv/bin/python src/main.py --mode cleanup-retired-profiles --verbose
```

実際に削除する場合のみ `--apply` を付けます。

```bash
.venv/bin/python src/main.py --mode cleanup-retired-profiles --apply --verbose
```
