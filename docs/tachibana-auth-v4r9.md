# Tachibana e支店 API v4r9 認証メモ

この文書はTachibana連携の認証・安全設計メモです。現在の実装はread-only brokerであり、実API発注は有効化されていません。

## 前提

- APIキー方式ではなく、v4r9では公開鍵暗号化方式を前提にする
- デモ環境URL: `https://demo-kabuka.e-shiten.jp/e_api_v4r9/`
- 本番環境URL: `https://kabuka.e-shiten.jp/e_api_v4r9/`
- e支店側でAPI利用設定が必要
- 秘密キー・公開キーの生成と登録が必要
- 認証方式の詳細は公式マニュアルを確認してから実装・検証する

## 環境変数候補

秘密情報はdocsやGit管理対象ファイルには書きません。

- `TACHIBANA_USER_ID`
- `TACHIBANA_PASSWORD`
- `TACHIBANA_SECOND_PASSWORD`
- `TACHIBANA_PRIVATE_KEY_PATH`
- `TACHIBANA_PUBLIC_KEY_ID`

## Current Implementation Status

| Area | Status |
| --- | --- |
| account/balance/position/order/execution read IF | read-only pathあり |
| token/login implementation | 未完成・要公式仕様確認 |
| buy/sell order sending | 無効。`LiveTradingDisabledError` で停止 |
| demo auto-order guard | safety check pathあり。ただし外部発注はread-only guardで止まる |
| live auto trading | 禁止・未運用 |

## Safety Notes

- 秘密鍵はGit管理しない
- パスワードや秘密鍵内容をログ出力しない
- live自動売買は有効化しない
- PaperBrokerで十分に検証しても、実注文機能は別途安全設計・手動承認・少額検証・法務/規約確認が必要
- demo環境の検証結果をもってlive自動化済みとは扱わない

