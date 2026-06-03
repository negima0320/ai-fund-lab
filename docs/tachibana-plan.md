# Tachibana Broker Status

Tachibana連携は現在 **Read-only / guarded demo検証** の状態です。実売買の自動発注は運用していません。

## Current Implementation

Source: `src/broker.py`, `src/demo_auto_order.py`

| Component | Status |
| --- | --- |
| `PaperBroker` | 実装済み。通常の検証・backtest・paper runで使用 |
| `TachibanaBroker` | Read-only broker。口座、残高、保有、注文、約定参照IF |
| `TachibanaDemoBroker` | Read-only派生。demo用guard pathはあるが、外部実発注は無効 |
| `TachibanaLiveBroker` | Read-only派生。注文送信は無効 |
| `place_buy_order` / `place_sell_order` | `LiveTradingDisabledError` で停止 |

## Demo Auto Order Guard

`demo_auto_order.py` には、注文候補をdemo環境へ流す前の安全確認があります。

主なguard:

- `broker.provider` が `tachibana_demo`
- `execution_mode` が `auto_demo`
- `forbid_live_auto_order: true`
- live環境・live brokerを拒否
- preflight成功
- cash/position/order limit確認
- duplicate holding/order確認

ただし、現行broker実装ではTachibanaへの外部注文送信はread-only guardで止まります。ドキュメントや運用上、demo自動発注がproduction-readyであるとは扱いません。

## Live Trading Policy

- live自動売買は禁止
- live brokerへの自動発注例はdocsに載せない
- 実注文を出すコードは安全ロックなしに有効化しない
- PaperBrokerと既存成果物分析を研究・検証の主経路とする

## What Must Change Before Any Real Trading Review

これは実装済み機能ではなく、将来レビュー項目です。

- broker API仕様の再確認
- 注文送信の手動承認フロー
- 誤発注防止limit
- 注文取消/失敗時処理
- 監査ログ
- 少額・手動承認・段階的検証
- legal/compliance確認

