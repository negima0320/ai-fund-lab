# OpenAI API は任意です

AI Fund Lab は、`OPENAI_API_KEY` が未設定でも通常動作します。OpenAI APIは追加機能であり、未設定時やAPI失敗時は `rule_based` 処理へフォールバックします。

APIキーの値は、ログ、DB、README、生成記事には出力しません。

## OpenAIを使わない場合

- ルールベースで銘柄選定する
- テクニカル、ニューススコア、財務スコア、market_contextなどで判断する
- APIコストがかからない
- 再現性が高い
- コメントはテンプレート寄りになる
- AI Decisionによる柔軟な市場判断は使えない

設定例です。

```yaml
ai_decision:
  enabled: false
  provider: openai
  fallback_to_rule_based: true

ai_commentary:
  enabled: true
  provider: rule_based
  fallback_to_rule_based: true
```

この状態では、OpenAI API呼び出しは行いません。

## OpenAIを使う場合

- 候補銘柄をまとめてOpenAIに渡して最終判断できる
- market_contextを踏まえた見送り判断が可能
- note記事や振り返りコメントが自然になる
- APIコストが発生する
- API失敗時はrule_basedへフォールバックする
- 銘柄ごとの個別API呼び出しは禁止

AI Decisionを使う場合の例です。

```yaml
ai_decision:
  enabled: true
  provider: openai
  fallback_to_rule_based: true
  daily_call_limit: 3
```

コメント生成だけOpenAIにする場合の例です。

```yaml
ai_commentary:
  enabled: true
  provider: openai
  fallback_to_rule_based: true
```

`OPENAI_API_KEY` が未設定の場合、どちらも自動的に `rule_based` へフォールバックします。

## 推奨

初期開発・バックテスト:

`rule_based`

記事品質を上げたい時:

`ai_commentary` のみ `openai`

最終判断もAIにさせたい時:

`ai_decision` を `openai`

## preflightでの扱い

`preflight` では、OpenAI APIキー未設定をエラーにしません。

- `ai_decision.enabled=false`: `SKIP`
- `ai_decision.enabled=true` かつ `OPENAI_API_KEY` なし: `WARN` + fallback確認
- `ai_commentary.provider=rule_based`: `SKIP`
- `ai_commentary.provider=openai` かつ `OPENAI_API_KEY` なし: `WARN` + fallback確認

OpenAI接続確認は、OpenAI系機能が有効で、かつ `OPENAI_API_KEY` が設定されている場合だけ行います。

## 注意

OpenAI出力は投資助言ではなく、AI Fund Lab内の実験用判断です。

OpenAI APIを使う場合でも、売買ルール、Safety Guard、Brokerの安全設計はPython側が管理します。AIが売買ルールを勝手に変更することはありません。
