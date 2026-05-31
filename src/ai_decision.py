"""AI final decision providers for scored candidates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class AIDecisionError(RuntimeError):
    """Raised when an AI decision provider cannot complete."""


@dataclass
class AIDecisionResult:
    decision: dict[str, Any]
    log: dict[str, Any]


class BaseAIDecisionProvider:
    def decide(
        self,
        target_date: str,
        config_version: str,
        market_context: dict[str, Any],
        portfolio_summary: dict[str, Any],
        scored_candidates: list[dict[str, Any]],
    ) -> AIDecisionResult:
        raise NotImplementedError


class RuleBasedDecisionProvider(BaseAIDecisionProvider):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def decide(
        self,
        target_date: str,
        config_version: str,
        market_context: dict[str, Any],
        portfolio_summary: dict[str, Any],
        scored_candidates: list[dict[str, Any]],
    ) -> AIDecisionResult:
        selected = [item for item in scored_candidates if item.get("selected")]
        rejected = [item for item in scored_candidates if not item.get("selected")]
        decision = {
            "date": target_date,
            "decision_summary": "ルールベース採点結果を最終判断として採用しました。",
            "selected": [
                {
                    "code": item.get("code", ""),
                    "name": item.get("name", ""),
                    "ai_rank": index,
                    "ai_score": float(item.get("total_score", 0) or 0),
                    "action": "BUY",
                    "reason": item.get("selection_reason") or item.get("selected_reason") or item.get("reason") or "ルールベース選定",
                    "risk": "ルールベース判断のため、個別リスク評価は簡易です。",
                    "confidence": float(item.get("confidence", 0) or 0),
                }
                for index, item in enumerate(selected, start=1)
            ],
            "rejected": [
                {
                    "code": item.get("code", ""),
                    "name": item.get("name", ""),
                    "reason": item.get("rejected_reason") or item.get("reason") or "ルールベースで落選",
                }
                for item in rejected
            ],
            "no_trade_reason": None if selected else "ルールベース基準を満たす買付候補がありません。",
            "rookie_comment": "ルールに従い判断します。感情は考慮しません。",
        }
        return AIDecisionResult(
            decision=decision,
            log={
                "provider": "rule_based",
                "model": None,
                "fallback_used": False,
                "prompt": None,
                "response": decision,
                "token_usage": {},
                "estimated_cost": None,
            },
        )


class OpenAIDecisionProvider(BaseAIDecisionProvider):
    def __init__(self, config: dict[str, Any], root: Path):
        self.config = config
        self.root = root
        self.ai_config = config.get("ai_decision", {})
        self.model = self.ai_config.get("model", "gpt-4.1-mini")
        self.fallback = RuleBasedDecisionProvider(config)

    def decide(
        self,
        target_date: str,
        config_version: str,
        market_context: dict[str, Any],
        portfolio_summary: dict[str, Any],
        scored_candidates: list[dict[str, Any]],
    ) -> AIDecisionResult:
        fallback_to_rule_based = bool(self.ai_config.get("fallback_to_rule_based", True))
        try:
            self._load_env()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise AIDecisionError("OPENAI_API_KEY is not set")
            if self._daily_call_count(target_date) >= int(self.ai_config.get("daily_call_limit", 3)):
                raise AIDecisionError("daily_call_limit exceeded")

            prompt_payload = self._build_prompt_payload(target_date, config_version, market_context, portfolio_summary, scored_candidates)
            prompt = self._build_prompt(prompt_payload)
            response_text, usage = self._call_openai(api_key, prompt)
            decision = self._parse_response(response_text)
            return AIDecisionResult(
                decision=decision,
                log={
                    "provider": "openai",
                    "model": self.model,
                    "fallback_used": False,
                    "prompt": prompt if self.ai_config.get("save_prompt", True) else None,
                    "response": decision if self.ai_config.get("save_response", True) else None,
                    "raw_response": response_text if self.ai_config.get("save_response", True) else None,
                    "token_usage": usage,
                    "estimated_cost": _estimate_cost(usage),
                },
            )
        except Exception as exc:
            fallback_result = self.fallback.decide(target_date, config_version, market_context, portfolio_summary, scored_candidates)
            fallback_result.log.update(
                {
                    "provider": "openai",
                    "model": self.model,
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                    "fallback_forced": not fallback_to_rule_based,
                    "prompt": fallback_result.log.get("prompt"),
                }
            )
            return fallback_result

    def _load_env(self) -> None:
        env_path = self.root / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def _daily_call_count(self, target_date: str) -> int:
        path = self.root / "logs" / "ai_decision" / f"ai_decision_{target_date}.json"
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return 0
        if payload.get("provider") == "openai" and not payload.get("fallback_used"):
            return 1
        return 0

    def _build_prompt_payload(
        self,
        target_date: str,
        config_version: str,
        market_context: dict[str, Any],
        portfolio_summary: dict[str, Any],
        scored_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        max_candidates = int(self.ai_config.get("max_candidates", 50))
        return {
            "date": target_date,
            "config_version": config_version,
            "market_context": market_context,
            "portfolio_summary": portfolio_summary,
            "rules": {
                "max_selected": int(self.ai_config.get("max_selected", 5)),
                "min_score": float(self.ai_config.get("min_score", 65)),
                "action_allowed": ["BUY"],
                "do_not_change_trading_rules": True,
            },
            "candidates": [_compact_candidate(item) for item in scored_candidates[:max_candidates]],
        }

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        schema = {
            "date": "YYYY-MM-DD",
            "decision_summary": "string",
            "selected": [
                {
                    "code": "string",
                    "name": "string",
                    "ai_rank": 1,
                    "ai_score": 0,
                    "action": "BUY",
                    "reason": "string",
                    "risk": "string",
                    "confidence": 0.0,
                }
            ],
            "rejected": [{"code": "string", "name": "string", "reason": "string"}],
            "no_trade_reason": "string or null",
            "rookie_comment": "string",
        }
        return "\n".join(
            [
                "あなたはAI Fund Labの新人ディーラー1号です。",
                "証券会社の研修を終えたばかりの新人ディーラーとして、教科書通り、データ重視、感情なし、ルール厳守で短期売買候補を判断します。",
                "損切りをためらわず、利確をためらわず、塩漬けしません。",
                "これは投資助言ではなく、AI Fund Lab内の実験用の内部判断です。",
                "禁止: 売買ルール変更、断定的表現、必ず上がる等の表現、低スコア銘柄の無理な採用、JSON以外の出力。",
                "total_score が min_score 未満の銘柄は原則選ばないでください。買うべき銘柄なしの判断も許可します。",
                "ローソク足タイプ、ローソク足シグナル、close > ma5 > ma25 などの移動平均線の関係、出来高確認を最終判断材料として重視してください。",
                "selected は最大5件、action は BUY のみです。",
                "以下のJSON Schema相当の形だけで返してください。",
                json.dumps(schema, ensure_ascii=False),
                "入力データ:",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    def _call_openai(self, api_key: str, prompt: str) -> tuple[str, dict[str, int]]:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise AIDecisionError("openai package is not installed") from exc

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        usage = getattr(completion, "usage", None)
        return content, {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise AIDecisionError("OpenAI response is not valid JSON") from exc
        if not isinstance(decision, dict):
            raise AIDecisionError("OpenAI response root must be an object")
        decision.setdefault("selected", [])
        decision.setdefault("rejected", [])
        decision.setdefault("no_trade_reason", None)
        decision.setdefault("rookie_comment", "")
        if len(decision["selected"]) > int(self.ai_config.get("max_selected", 5)):
            decision["selected"] = decision["selected"][: int(self.ai_config.get("max_selected", 5))]
        for item in decision["selected"]:
            item["action"] = "BUY"
            item["ai_score"] = max(0, min(100, float(item.get("ai_score", 0) or 0)))
            item["confidence"] = max(0, min(1, float(item.get("confidence", 0) or 0)))
        return decision


def build_ai_decision_provider(config: dict[str, Any], root: Path) -> BaseAIDecisionProvider:
    ai_config = config.get("ai_decision", {})
    if ai_config.get("enabled") and ai_config.get("provider") == "openai":
        return OpenAIDecisionProvider(config, root)
    return RuleBasedDecisionProvider(config)


def apply_ai_decision(
    scoring_log: dict[str, Any],
    decision_result: AIDecisionResult,
    config: dict[str, Any],
) -> dict[str, Any]:
    decision = decision_result.decision
    selected_by_code = {item.get("code"): item for item in decision.get("selected", [])}
    rejected_by_code = {item.get("code"): item for item in decision.get("rejected", [])}
    min_score = float(config.get("ai_decision", {}).get("min_score", 65))

    for item in scoring_log.get("scores", []):
        code = item.get("code")
        ai_selected = selected_by_code.get(code)
        item["ai_decision_enabled"] = bool(config.get("ai_decision", {}).get("enabled", False))
        if ai_selected and float(item.get("total_score", 0) or 0) >= min_score:
            item["selected"] = True
            item["ai_reason"] = ai_selected.get("reason", "")
            item["ai_risk"] = ai_selected.get("risk", "")
            item["ai_confidence"] = ai_selected.get("confidence")
            item["ai_score"] = ai_selected.get("ai_score")
            item["ai_rank"] = ai_selected.get("ai_rank")
            item["selection_reason"] = ai_selected.get("reason", "AI最終判断により採用")
            item["selected_reason"] = item["selection_reason"]
            item["rejected_reason"] = ""
            item["reason"] = item["selection_reason"]
        else:
            item["selected"] = False
            rejected = rejected_by_code.get(code, {})
            reason = rejected.get("reason") or item.get("rejected_reason") or item.get("reason") or "AI最終判断により落選"
            if ai_selected and float(item.get("total_score", 0) or 0) < min_score:
                reason = "AI選定候補だが、min_score未満のためPython側で落選"
            item["rejected_reason"] = reason
            item["reason"] = reason

    scoring_log["selected"] = [item for item in scoring_log.get("scores", []) if item.get("selected")]
    scoring_log["rejected"] = [item for item in scoring_log.get("scores", []) if not item.get("selected")]
    scoring_log["ai_decision"] = {
        "enabled": bool(config.get("ai_decision", {}).get("enabled", False)),
        "provider": decision_result.log.get("provider"),
        "model": decision_result.log.get("model"),
        "decision_summary": decision.get("decision_summary", ""),
        "rookie_comment": decision.get("rookie_comment", ""),
        "no_trade_reason": decision.get("no_trade_reason"),
        "fallback_used": bool(decision_result.log.get("fallback_used", False)),
    }
    return scoring_log


def build_ai_decision_log(
    target_date: str,
    config_version: str,
    decision_result: AIDecisionResult,
    candidates_count: int,
    selected_count: int,
) -> dict[str, Any]:
    decision = decision_result.decision
    usage = decision_result.log.get("token_usage") or {}
    return {
        "date": target_date,
        "config_version": config_version,
        "provider": decision_result.log.get("provider"),
        "model": decision_result.log.get("model"),
        "candidates_count": candidates_count,
        "selected_count": selected_count,
        "decision_summary": decision.get("decision_summary", ""),
        "rookie_comment": decision.get("rookie_comment", ""),
        "fallback_used": bool(decision_result.log.get("fallback_used", False)),
        "fallback_reason": decision_result.log.get("fallback_reason"),
        "prompt": decision_result.log.get("prompt"),
        "response": decision_result.log.get("response"),
        "token_usage": usage,
        "token_input": usage.get("input_tokens"),
        "token_output": usage.get("output_tokens"),
        "estimated_cost": decision_result.log.get("estimated_cost"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "code",
        "name",
        "sector_name",
        "sector_momentum_score",
        "sector_rank",
        "sector_comment",
        "sector_score_adjustment",
        "close",
        "total_score",
        "technical_score",
        "trend_score",
        "ma_score",
        "volume_score",
        "rsi_score",
        "candlestick_score",
        "market_context_score",
        "sector_score",
        "penalty_score",
        "score_components",
        "confidence",
        "rank",
        "reason",
        "volume_ratio",
        "rsi",
        "turnover_value",
        "five_day_volatility",
        "macd_hist",
        "bb_position",
        "atr",
        "candle_type",
        "candlestick_signals",
        "fallback",
    ]
    return {key: item.get(key) for key in keys}


def _estimate_cost(usage: dict[str, int]) -> float | None:
    if not usage:
        return None
    return None
