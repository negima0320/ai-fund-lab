"""AI reflection log templates for closed trades."""

from __future__ import annotations

from typing import Any

from commentary import generate_reflection_comment


def generate_reflections(paper_trade_log: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    reflections = []
    for trade in paper_trade_log["closed_trades"]:
        reflections.append(
            {
                "trade_id": trade["trade_id"],
                "code": trade["code"],
                "name": trade["name"],
                "buy_reason": trade["buy_reason"],
                "sell_reason": trade["exit_reason"],
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "holding_days": trade["holding_days"],
                "profit_rate": trade["profit_rate"],
                "result": trade["result"],
                "reflection_comment": generate_reflection_comment(trade, config),
                "good_points": _good_points(trade),
                "bad_points": _bad_points(trade),
                "lesson_for_next_trade": _lesson(trade),
                "suggestions": _suggestions(trade),
                "rule_change_note": "AIは売買ルールを変更しません。改善案はsuggestionsとして保存するのみです。",
            }
        )

    return {
        "run_id": paper_trade_log["run_id"],
        "date": paper_trade_log["date"],
        "dealer_id": config["dealer"]["id"],
        "reflections": reflections,
    }


def _good_points(trade: dict[str, Any]) -> list[str]:
    if trade["result"] == "WIN":
        return ["利確ルールに従って利益を確定できた", "買付理由と売却理由を明確に記録できた"]
    return ["損切りルールに従い、損失拡大を避けた", "塩漬けを回避した"]


def _bad_points(trade: dict[str, Any]) -> list[str]:
    if trade["result"] == "WIN":
        return ["利確後の継続上昇余地は未検証"]
    if trade["result"] == "LOSS":
        return ["エントリー時点の下落リスク評価に改善余地"]
    return ["値動きが限定的で資金効率の検証が必要"]


def _lesson(trade: dict[str, Any]) -> str:
    if trade["result"] == "WIN":
        return "短期売買では利益確定条件に達した時点で迷わず記録し、再現可能性を確認する。"
    if trade["result"] == "LOSS":
        return "損切りは失敗ではなく、ルールに基づく資金防衛として扱う。"
    return "値幅が出ない銘柄では機会費用も確認する。"


def _suggestions(trade: dict[str, Any]) -> list[str]:
    if trade["result"] == "LOSS":
        return ["候補選定時に直近急騰後の反落リスクを別項目で観察する"]
    if trade["result"] == "WIN":
        return ["利確後の値動きを追跡し、現行ルールの妥当性検証データを増やす"]
    return ["横ばい銘柄の資金拘束期間を分析する"]
