from __future__ import annotations

from copy import deepcopy

import pytest

from ai_decision import OpenAIDecisionProvider, RuleBasedDecisionProvider, build_ai_decision_provider
from commentary import OpenAICommentaryProvider, RuleBasedCommentaryProvider, build_commentary_provider
from main import _check_openai_configuration


def test_openai_missing_key_is_preflight_warning_not_failure() -> None:
    results: list[dict] = []

    _check_openai_configuration(results, api_key_set=False, fallback_enabled=False)

    assert results
    assert results[-1]["status"] == "WARN"
    assert "OPENAI_API_KEY is not set" in results[-1]["message"]


def test_ai_decision_disabled_uses_rule_based_without_openai_call(config_copy: dict, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("OpenAI should not be called when ai_decision.enabled is false")

    config_copy["ai_decision"]["enabled"] = False
    monkeypatch.setattr(OpenAIDecisionProvider, "_call_openai", fail_if_called)

    provider = build_ai_decision_provider(config_copy, tmp_path)
    result = provider.decide("2026-03-06", "test", {}, {}, [])

    assert isinstance(provider, RuleBasedDecisionProvider)
    assert result.log["provider"] == "rule_based"
    assert result.log["fallback_used"] is False


def test_rule_based_commentary_does_not_use_openai(config_copy: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("OpenAI commentary should not be called when provider is rule_based")

    config_copy["ai_commentary"]["provider"] = "rule_based"
    monkeypatch.setattr(OpenAICommentaryProvider, "_generate", fail_if_called)

    provider = build_commentary_provider(config_copy)
    comment = provider.generate_no_trade_comment("基準未満")

    assert isinstance(provider, RuleBasedCommentaryProvider)
    assert "基準未満" in comment


def test_ai_decision_openai_without_key_falls_back_to_rule_based(config_copy: dict, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("OpenAI should not be called when OPENAI_API_KEY is missing")

    config_copy["ai_decision"]["enabled"] = True
    config_copy["ai_decision"]["provider"] = "openai"
    config_copy["ai_decision"]["fallback_to_rule_based"] = True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(OpenAIDecisionProvider, "_call_openai", fail_if_called)

    provider = build_ai_decision_provider(config_copy, tmp_path)
    result = provider.decide(
        "2026-03-06",
        "test",
        {},
        {},
        [{"code": "1234", "name": "テスト", "selected": True, "total_score": 80, "confidence": 0.8}],
    )

    assert result.log["provider"] == "openai"
    assert result.log["fallback_used"] is True
    assert "OPENAI_API_KEY is not set" in result.log["fallback_reason"]
    assert result.decision["selected"][0]["code"] == "1234"


def test_openai_commentary_without_key_falls_back_to_rule_based(config_copy: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    config_copy = deepcopy(config_copy)
    config_copy["ai_commentary"]["provider"] = "openai"
    config_copy["ai_commentary"]["fallback_to_rule_based"] = True
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("commentary._load_dotenv_if_available", lambda: None)

    provider = build_commentary_provider(config_copy)
    comment = provider.generate_no_trade_comment("基準未満")

    assert isinstance(provider, OpenAICommentaryProvider)
    assert "基準未満" in comment
