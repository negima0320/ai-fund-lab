# 立花証券 e支店 API v4r9 認証メモ

## 前提

- APIキー方式ではない
- v4r9では公開鍵暗号化方式を前提にする
- デモ環境URL:
  https://demo-kabuka.e-shiten.jp/e_api_v4r9/
- 本番環境URL:
  https://kabuka.e-shiten.jp/e_api_v4r9/
- e支店側でAPI利用設定が必要
- 秘密キー・公開キーの生成と登録が必要
- 認証方式の詳細は公式マニュアルを確認してから実装する

## 環境変数

- TACHIBANA_USER_ID
- TACHIBANA_PASSWORD
- TACHIBANA_SECOND_PASSWORD
- TACHIBANA_PRIVATE_KEY_PATH
- TACHIBANA_PUBLIC_KEY_ID

## 実装予定

- 秘密鍵読み込み
- 公開鍵ID読み込み
- 認証リクエスト生成
- ログイン
- セッション管理
- ログアウト
- 口座情報取得
- 現物買い注文
- 現物売り注文
- 注文照会
- 約定照会

## 注意

- 秘密鍵はGit管理しない
- パスワードや秘密鍵内容をログ出力しない
- 実売買はPaperBrokerで十分検証してから
- demo環境で検証後にlive環境へ進む
