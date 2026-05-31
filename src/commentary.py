"""Rookie dealer commentary provider interface.

The rule-based provider is the default and remains the safe fallback. The
OpenAI provider is optional and only used when explicitly enabled in config.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseCommentaryProvider(ABC):
    @abstractmethod
    def generate_daily_comment(
        self,
        portfolio_summary: dict[str, Any],
        selected_candidates: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_buy_comment(self, candidate: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_sell_comment(self, trade: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_no_trade_comment(self, reason: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_reflection_comment(self, closed_trade: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_note_title(self, summary: dict[str, Any], paper_trade_log: dict[str, Any]) -> str:
        raise NotImplementedError


class RuleBasedCommentaryProvider(BaseCommentaryProvider):
    def generate_daily_comment(
        self,
        portfolio_summary: dict[str, Any],
        selected_candidates: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> str:
        try:
            buys = [trade for trade in trades if trade.get("action") == "BUY"]
            sells = [trade for trade in trades if trade.get("action") == "SELL"]
            skipped = [trade for trade in trades if trade.get("action") == "SKIP_BUY"]
            if sells:
                return "本日は売却条件に到達した銘柄を処理しました。感情は考慮しません。ルールに従い判断します。"
            if buys:
                return "本日は採点基準を満たした銘柄を買付しました。統計的優位性を優先し、売却条件は翌営業日以降に確認します。"
            if skipped and selected_candidates:
                return "選定銘柄はありましたが、単元株または資金制約により買付を見送りました。現時点では見送りが妥当です。"
            if not selected_candidates:
                return self.generate_no_trade_comment("採用基準を満たす銘柄がありません。")
            if float(portfolio_summary.get("daily_profit", 0)) < 0:
                return "資産は減少しましたが、規律を優先します。感情は考慮しません。"
            return "本日は大きな売買判断を行いません。ルールに従い、次の条件到達を待ちます。"
        except Exception:
            return "ルールに従い判断します。"

    def generate_buy_comment(self, candidate: dict[str, Any]) -> str:
        try:
            technical = _format_value(candidate.get("technical_score"), "点")
            confidence = _format_value(candidate.get("confidence"), "")
            technical_note = _technical_note(candidate)
            return (
                f"{candidate.get('code')} {candidate.get('name')}は、テクニカル{technical}、"
                f"信頼度{confidence}です。"
                f"{technical_note}ルールに従い買付候補とします。"
            )
        except Exception:
            return "ルールに従い買付候補とします。"

    def generate_sell_comment(self, trade: dict[str, Any]) -> str:
        reason = trade.get("exit_reason", "")
        if "利確" in reason:
            return "利益確定条件に到達したため、欲張らずに確定します。"
        if "損切り" in reason:
            return "損切り条件に到達したため、規律を優先します。"
        if "最大保有" in reason:
            return "最大保有期間に到達しました。保有理由が消失したため売却します。"
        return "売却条件に到達したため、ルールに従い売却します。"

    def generate_no_trade_comment(self, reason: str) -> str:
        return f"{reason} 感情は考慮しません。現時点では見送りが妥当です。"

    def generate_reflection_comment(self, closed_trade: dict[str, Any]) -> str:
        result = closed_trade.get("result")
        if result == "WIN":
            return "ルール通りに利益を確定できました。ただし、再現性を確認するまでは過信しません。"
        if result == "LOSS":
            return "損失は発生しましたが、損切りは資金防衛です。規律を優先します。"
        return "損益は中立です。資金効率の観点から、次回以降も記録を継続します。"

    def generate_note_title(self, summary: dict[str, Any], paper_trade_log: dict[str, Any]) -> str:
        closed = paper_trade_log["closed_trades"]
        if any("利確" in trade.get("exit_reason", "") for trade in closed):
            return "新人ディーラー、初めての利確"
        if any("損切り" in trade.get("exit_reason", "") for trade in closed):
            return "ルール通りに損切りしました"
        if not paper_trade_log["orders"] and not closed:
            return "本日は静観です"
        if summary["daily_profit"] > 0:
            return "資産は増えたが、油断は禁物です"
        if summary["daily_profit"] < 0:
            return "資産は減少、ルール確認を継続します"
        return "ルールに従い、淡々と初日を終えました"


class OpenAICommentaryProvider(BaseCommentaryProvider):
    def __init__(self, config: dict[str, Any], fallback: RuleBasedCommentaryProvider | None = None) -> None:
        self.config = config
        self.fallback = fallback or RuleBasedCommentaryProvider()
        self.settings = config.get("ai_commentary", {})
        self.model = self.settings.get("model", "gpt-4.1-mini")
        self.fallback_to_rule_based = bool(self.settings.get("fallback_to_rule_based", True))
        self._client: Any = None

    def generate_daily_comment(
        self,
        portfolio_summary: dict[str, Any],
        selected_candidates: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> str:
        return self._generate_or_fallback(
            "日次コメント",
            {
                "portfolio_summary": portfolio_summary,
                "selected_candidates": selected_candidates,
                "trades": trades,
            },
            lambda: self.fallback.generate_daily_comment(portfolio_summary, selected_candidates, trades),
        )

    def generate_buy_comment(self, candidate: dict[str, Any]) -> str:
        return self._generate_or_fallback("買付理由コメント", {"candidate": candidate}, lambda: self.fallback.generate_buy_comment(candidate))

    def generate_sell_comment(self, trade: dict[str, Any]) -> str:
        return self._generate_or_fallback("売却理由コメント", {"trade": trade}, lambda: self.fallback.generate_sell_comment(trade))

    def generate_no_trade_comment(self, reason: str) -> str:
        return self._generate_or_fallback("買付対象なしコメント", {"reason": reason}, lambda: self.fallback.generate_no_trade_comment(reason))

    def generate_reflection_comment(self, closed_trade: dict[str, Any]) -> str:
        return self._generate_or_fallback(
            "売却後の振り返りコメント",
            {"closed_trade": closed_trade},
            lambda: self.fallback.generate_reflection_comment(closed_trade),
        )

    def generate_note_title(self, summary: dict[str, Any], paper_trade_log: dict[str, Any]) -> str:
        return self._generate_or_fallback(
            "note記事の一言タイトル",
            {"summary": summary, "paper_trade_log": paper_trade_log},
            lambda: self.fallback.generate_note_title(summary, paper_trade_log),
        )

    def _generate_or_fallback(self, task: str, payload: dict[str, Any], fallback: Any) -> str:
        try:
            return self._generate(task, payload)
        except Exception as exc:
            return fallback()

    def _generate(self, task: str, payload: dict[str, Any]) -> str:
        client = self._get_client()
        prompt = self._build_prompt(task, payload)
        response = client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=220,
        )
        text = getattr(response, "output_text", "")
        if not text:
            raise RuntimeError("OpenAI response text is empty.")
        return text.strip()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        _load_dotenv_if_available()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY が未設定です")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("openai パッケージがインストールされていません") from exc
        self._client = OpenAI(api_key=api_key)
        return self._client

    def _build_prompt(self, task: str, payload: dict[str, Any]) -> str:
        output_instruction = "出力は日本語で1〜3文、Markdownなし。"
        if task == "note記事の一言タイトル":
            output_instruction = "出力は日本語で20文字前後の短いタイトル1つ。Markdownなし。"
        return (
            "あなたはAIファンド1号の「新人ディーラー1号」です。\n"
            "証券会社の研修を終えたばかりの新人ディーラーとして、教科書通りでお堅く、"
            "データ重視、感情なし、ルール厳守の口調で運用ログを説明してください。\n"
            "損切りをためらわず、利確をためらわず、投資助言ではなく運用ログの説明だけを行います。\n\n"
            "禁止事項:\n"
            "- 断定的な投資助言\n"
            "- 必ず上がる、絶対に儲かる等の表現\n"
            "- 売買ルール変更の実行\n"
            "- 過度な煽り表現\n\n"
            f"生成対象: {task}\n"
            f"{output_instruction}\n"
            f"入力データ:\n{json.dumps(_safe_payload(payload), ensure_ascii=False, indent=2)}"
        )


def build_commentary_provider(config: dict[str, Any] | None = None) -> BaseCommentaryProvider:
    config = config or {}
    settings = config.get("ai_commentary", {})
    if not settings.get("enabled", True):
        return RuleBasedCommentaryProvider()
    provider_name = settings.get("provider", "rule_based")
    if provider_name == "openai":
        return OpenAICommentaryProvider(config)
    return RuleBasedCommentaryProvider()


def generate_daily_comment(
    portfolio_summary: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> str:
    return build_commentary_provider(config).generate_daily_comment(portfolio_summary, selected_candidates, trades)


def generate_buy_comment(candidate: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    return build_commentary_provider(config).generate_buy_comment(candidate)


def generate_sell_comment(trade: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    return build_commentary_provider(config).generate_sell_comment(trade)


def generate_no_trade_comment(reason: str, config: dict[str, Any] | None = None) -> str:
    return build_commentary_provider(config).generate_no_trade_comment(reason)


def generate_reflection_comment(closed_trade: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    return build_commentary_provider(config).generate_reflection_comment(closed_trade)


def generate_note_title(
    summary: dict[str, Any],
    paper_trade_log: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    return build_commentary_provider(config).generate_note_title(summary, paper_trade_log)


def _format_value(value: Any, suffix: str) -> str:
    if value is None or value == "":
        return "未算出"
    return f"{value}{suffix}"


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= 5000:
        return payload
    return {"summary": "入力データが長いため要約対象のみを送信", "truncated_payload": text[:5000]}


def _technical_note(candidate: dict[str, Any]) -> str:
    signals = candidate.get("candlestick_signals") or []
    notes = []
    if candidate.get("candle_type"):
        notes.append(f"ローソク足は{candidate.get('candle_type')}です")
    close = _as_float(candidate.get("close") or candidate.get("entry_price") or candidate.get("price"))
    ma5 = _as_float(candidate.get("ma5"))
    ma25 = _as_float(candidate.get("ma25"))
    if close is not None and ma5 is not None and ma25 is not None and close > ma5 > ma25:
        notes.append("close > ma5 > ma25 の上昇配列を確認します")
    if "long_upper_shadow_warning" in signals:
        notes.append("上ヒゲが長く売り圧力に注意します")
    if "volume_confirmed_breakout" in signals:
        notes.append("出来高を伴う短期ブレイクを評価します")
    return "。".join(notes) + "。" if notes else ""


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
