# kabuステーションAPI接続計画

## 目的

kabuステーションAPIを使い、将来的にAIファンド1号の売買判断を実売買へ接続できるようにする。

ただし、現時点では未実装です。実API接続も実発注も行いません。

## 現在の状態

現在は `PaperBroker` のみを使用します。

`KabuStationBrokerStub` は将来用のスタブであり、呼び出されても必ず例外を出します。

## Broker差し替え設計

売買実行の出口は `src/broker.py` に分離しています。

- `BaseBroker`: 共通インターフェース
- `PaperBroker`: 仮想売買
- `KabuStationBrokerStub`: 将来のkabuステーションAPI用スタブ

将来的には `broker.provider: paper` を `broker.provider: kabu_station` に切り替えることで、売買実行部分だけを差し替える設計です。

## 実売買前に必要な安全条件

実売買へ進むには、最低限以下が必要です。

- `broker.live_trading_enabled: true`
- `safety.allow_live_trading: true`
- `kabu_station.enabled: true`
- `storage/STOP_TRADING` が存在しない

現時点では、これらの条件がそろっても `KabuStationBrokerStub` が例外を出すため、実売買は行われません。

## 実装予定

- APIトークン取得
- 現物買い注文
- 現物売り注文
- 注文状態確認
- 約定確認
- 口座残高取得
- 保有銘柄取得

## 実売買移行時の注意

- 最初は最小金額で検証する
- 1日1注文など強い制限を入れる
- 必ず手動監視する
- 誤発注防止のため、注文前ログと注文後ログを保存する
- セーフティガードを通らない注文は発行しない
- APIキー、パスワード、口座情報はGit管理しない
