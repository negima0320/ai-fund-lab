# 立花証券 e支店 API 接続計画

## 採用候補にした理由

AI Fund Lab では、将来の実売買API候補として立花証券 e支店 API を第一候補にします。

理由は以下です。

- Mac/Linuxから利用しやすい
- 現物取引に対応している
- デモ環境がある
- ローカルMac運用や自動売買システムと相性がよい

## 現時点の状態

現時点では未実装です。

実API接続、認証、注文送信、約定確認、口座情報取得は行いません。

当面は `PaperBroker` で仮想売買を検証し、売買ルール、ログ、セーフティガード、注文プレビューを固めます。

## 認証方式

立花証券 e支店 API は単純なAPIキー方式ではありません。

v4r9では公開鍵暗号化方式を利用する前提で設計します。

- デモ環境URL: `https://demo-kabuka.e-shiten.jp/e_api_v4r9/`
- 本番環境URL: `https://kabuka.e-shiten.jp/e_api_v4r9/`
- e支店の利用設定画面でAPI利用を有効化する必要があります
- 秘密キー・公開キーの作成と登録が必要です
- 実装前に公式マニュアルの認証機能を精読します

秘密鍵、パスワード、第二パスワードはGit管理しません。

認証方式の詳細メモは `docs/tachibana-auth-v4r9.md` に分離しています。実装前にこのメモと公式マニュアルを確認します。

## Broker差し替え設計

売買実行の出口は `src/broker.py` に分離しています。

- `PaperBroker`: 現在使用する仮想売買Broker
- `TachibanaDemoBrokerStub`: 立花証券デモ環境用の将来スタブ
- `TachibanaLiveBrokerStub`: 立花証券ライブ環境用の将来スタブ
- `KabuStationBrokerStub`: 代替候補として残すkabuステーション用スタブ

次の段階では `TachibanaDemoBroker` を実装し、デモ環境で注文形式、認証、レスポンス、エラー処理を検証します。

最終段階で `TachibanaLiveBroker` を検討します。

## 実売買前の安全条件

立花証券ライブ環境を使う場合、最低限以下が必要です。

- `broker.live_trading_enabled: true`
- `safety.allow_live_trading: true`
- `tachibana.environment: live`
- `storage/STOP_TRADING` が存在しない
- preflightが重大なエラーを出していない
- 注文プレビューを人間が確認している

現時点では、これらの条件がそろっても `TachibanaLiveBrokerStub` は例外を出すため、実売買は行われません。

## 誤発注防止方針

- 最初はデモ環境で検証する
- live移行時も最小金額、少数注文から始める
- 1日1注文など強い制限を設定する
- 注文前に必ず `safety.py` を通す
- 注文前に `preview-orders` で内容を確認する
- STOP_TRADING ファイルで即時停止できるようにする
- 注文、約定、拒否、エラーをすべてSQLiteとログに保存する

## 実装予定

- デモ環境ログイン
- デモ環境トークン管理
- 現物買い注文
- 現物売り注文
- 注文状態確認
- 約定確認
- 口座残高取得
- 保有銘柄取得
- APIエラー分類
- レート制限対応

## 次に必要な調査項目

- デモ環境ログイン方式
- セッション管理
- 口座情報取得
- 現物買い注文API
- 現物売り注文API
- 注文取消API
- 約定照会API
- API仕様書のURLまたはローカル保存場所
- 二段階認証の扱い
- 自動売買時の制約
- 公開鍵暗号化方式の認証リクエスト形式
- 秘密キー形式と鍵ファイルの安全な保存方法

## 接続前ヘルスチェック

現時点では実API通信を行わず、設定と認証情報の有無だけを確認します。

```bash
python src/main.py --mode tachibana-healthcheck --env demo
```

出力先は以下です。

- `reports/backtests/tachibana_healthcheck_latest.md`
- `reports/backtests/tachibana_healthcheck_latest.json`

認証情報の値は表示・保存しません。

## 注意

本計画は実売買の準備であり、投資助言ではありません。

実売買へ進む場合は、必ず小さく始め、手動監視を行い、誤発注防止を最優先にします。
