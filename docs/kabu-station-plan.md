# KabuStation Status

KabuStation連携は現在、候補後退・スタブ扱いです。現行の研究・検証フローでは使いません。

## Current Status

| Area | Status |
| --- | --- |
| Broker implementation | disabled stub |
| Order sending | 未実装 |
| Backtest/PaperBroker dependency | なし |
| Recommended operation | 使用しない |

KabuStationに関する過去の計画メモは、現在の実装済み機能ではありません。実売買や自動発注が可能であるとは扱わないでください。

## Current Broker Priority

1. `PaperBroker`: research/backtest/paper run
2. `TachibanaBroker`: read-only account/order/execution reference path
3. `KabuStation`: stub / future candidate

