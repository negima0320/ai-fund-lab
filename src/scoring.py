"""AI scoring placeholder for rookie dealer."""

from __future__ import annotations

import random
from typing import Any

from candlestick import detect_candlestick_signals
from earnings_calendar import EARNINGS_FILTER_REJECTED_REASON, earnings_filter_result
from market_sections import allowed_market_sections, market_section_counts, market_section_from_row, normalize_market_section
from market_regime import classify_market_regime


def score_candidates(screening_log: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    run_id = screening_log["run_id"]
    scored = []

    for candidate in screening_log["candidates"]:
        rng = random.Random(f"{run_id}:{candidate['code']}:scoring")
        technical = _technical_score(candidate, config, rng)
        total = technical
        confidence = round(min(0.98, max(0.30, total / 100 + rng.uniform(-0.08, 0.08))), 2)

        scored.append(
            {
                "code": candidate["code"],
                "name": candidate["name"],
                "market": candidate["market"],
                "sector": candidate["sector"],
                "close_price": candidate["close_price"],
                "technical_score": technical,
                "total_score": total,
                "confidence": confidence,
                "selected": False,
                "selection_reason": "",
                "rejection_reason": "",
                "score_comment": _score_comment(total, confidence),
            }
        )

    earnings_matched_count = sum(1 for item in scored if item.get("earnings_info_found") or item.get("earnings_candidate_date"))
    earnings_rejected_count = sum(1 for item in scored if item.get("earnings_filter_blocked"))
    for item in scored:
        item["earnings_pipeline_matched_candidates"] = earnings_matched_count
        item["earnings_pipeline_rejected_candidates"] = earnings_rejected_count

    scored.sort(key=lambda item: (item["total_score"], item["confidence"]), reverse=True)
    max_positions = int(config["portfolio"]["max_positions"])
    confidence_min = float(config["scoring"]["confidence_min_for_buy"])

    selected_count = 0
    for item in scored:
        if selected_count < max_positions and item["confidence"] >= confidence_min:
            item["selected"] = True
            item["selection_reason"] = "総合点と信頼度が上位で、短期売買候補として採用"
            selected_count += 1
        else:
            item["rejection_reason"] = _rejection_reason(item, selected_count, max_positions, confidence_min)

    return {
        "run_id": run_id,
        "date": screening_log["date"],
        "dealer_id": config["dealer"]["id"],
        "scoring_policy": {
            "technical_max": config["scoring"]["technical_max"],
            "confidence_min_for_buy": confidence_min,
        },
        "scores": scored,
        "selected": [item for item in scored if item["selected"]],
        "rejected": [item for item in scored if not item["selected"]],
    }


def build_trade_decisions(scoring_log: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    decisions = []
    for item in scoring_log["scores"]:
        decisions.append(
            {
                "code": item["code"],
                "name": item["name"],
                "decision": "BUY" if item["selected"] else "PASS",
                "reason": item["selection_reason"] or item["rejection_reason"],
                "total_score": item["total_score"],
                "technical_score": item.get("technical_score"),
                "sector_name": item.get("sector_name"),
                "sector_momentum_score": item.get("sector_momentum_score"),
                "sector_rank": item.get("sector_rank"),
                "sector_comment": item.get("sector_comment"),
                "sector_score_adjustment": item.get("sector_score_adjustment"),
                "candle_type": item.get("candle_type"),
                "candlestick_signals": item.get("candlestick_signals", []),
                "candlestick_score": item.get("candlestick_score"),
                "trend_score": item.get("trend_score"),
                "volume_score": item.get("volume_score"),
                "rsi_score": item.get("rsi_score"),
                "ma5": item.get("ma5"),
                "ma25": item.get("ma25"),
                "volume_ratio": item.get("volume_ratio"),
                "macd_hist": item.get("macd_hist"),
                "bb_position": item.get("bb_position"),
                "atr": item.get("atr"),
                "stock_return_5d": item.get("stock_return_5d"),
                "stock_return_10d": item.get("stock_return_10d"),
                "stock_return_20d": item.get("stock_return_20d"),
                "benchmark_source": item.get("benchmark_source"),
                "benchmark_return_5d": item.get("benchmark_return_5d"),
                "benchmark_return_10d": item.get("benchmark_return_10d"),
                "benchmark_return_20d": item.get("benchmark_return_20d"),
                "relative_strength_5d": item.get("relative_strength_5d"),
                "relative_strength_10d": item.get("relative_strength_10d"),
                "relative_strength_20d": item.get("relative_strength_20d"),
                "relative_strength_score": item.get("relative_strength_score"),
                "market_filter_applied": item.get("market_filter_applied", False),
                "market_regime": item.get("market_regime"),
                "market_filter_reason": item.get("market_filter_reason", ""),
                "earnings_filter_checked": item.get("earnings_filter_checked", False),
                "earnings_filter_blocked": item.get("earnings_filter_blocked", False),
                "earnings_filter_reason": item.get("earnings_filter_reason", ""),
                "earnings_announcement_date": item.get("earnings_announcement_date"),
                "investor_context_source": item.get("investor_context_source"),
                "investor_context_week": item.get("investor_context_week"),
                "overseas_net_buy": item.get("overseas_net_buy"),
                "overseas_net_buy_4w_sum": item.get("overseas_net_buy_4w_sum"),
                "overseas_net_buy_4w_trend": item.get("overseas_net_buy_4w_trend"),
                "overseas_buy_sell_ratio": item.get("overseas_buy_sell_ratio"),
                "individual_net_buy": item.get("individual_net_buy"),
                "institution_net_buy": item.get("institution_net_buy"),
                "trust_bank_net_buy": item.get("trust_bank_net_buy"),
                "proprietary_net_buy": item.get("proprietary_net_buy"),
                "investor_context_score": item.get("investor_context_score"),
                "confidence": item["confidence"],
                "rule_snapshot": {
                    "max_positions": config["portfolio"]["max_positions"],
                    "max_allocation_per_symbol": config["portfolio"]["max_allocation_per_symbol"],
                    "stop_loss_pct": config["risk"]["stop_loss_pct"],
                    "take_profit_pct": config["risk"]["take_profit_pct"],
                    "max_holding_business_days": config["risk"]["max_holding_business_days"],
                    "ai_rule_change_allowed": config["risk"]["ai_rule_change_allowed"],
                },
            }
        )

    return {
        "run_id": scoring_log["run_id"],
        "date": scoring_log["date"],
        "dealer_id": config["dealer"]["id"],
        "decisions": decisions,
    }


def score_real_candidates(
    candidates: list[dict[str, Any]],
    target_date: str,
    config: dict[str, Any],
    source_provider: str,
    market_context: dict[str, Any] | None = None,
    dynamic_exposure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scored = []
    selection_config = _selection_config(config)
    volume_filter = _volume_filter_config(config)
    rsi_volume_hot_zone_filter = _rsi_volume_hot_zone_filter_config(config)
    affordability_filter = _affordability_filter_config(config)
    winner_loser_rule_adjustment = _winner_loser_rule_adjustment_config(config)
    market_filter = _market_filter_config(config)
    market_regime = str((market_context or {}).get("market_regime") or "neutral")
    advance_ratio = _optional_float((market_context or {}).get("advance_ratio"))
    market_average_change_rate = _optional_float((market_context or {}).get("average_change_rate"))
    classified_market_regime = classify_market_regime(advance_ratio, market_average_change_rate, market_regime)
    dynamic_context = dynamic_exposure_context or {}
    dynamic_market_context = dynamic_context.get("market_context") if isinstance(dynamic_context.get("market_context"), dict) else {}
    dynamic_regime = str(dynamic_context.get("regime") or "") or classify_market_regime(
        _optional_float(dynamic_market_context.get("advance_ratio")),
        _optional_float(dynamic_market_context.get("average_change_rate")),
        dynamic_market_context.get("market_regime"),
    )
    earnings_calendar_records_count = len(config.get("_earnings_calendar_records") or [])
    earnings_pipeline = {}
    earnings_metadata = config.get("_earnings_calendar_metadata")
    if isinstance(earnings_metadata, dict) and isinstance(earnings_metadata.get("pipeline"), dict):
        earnings_pipeline = dict(earnings_metadata["pipeline"])

    for candidate in candidates:
        technical_parts = _real_technical_score_parts(candidate)
        sector_adjustment = _sector_score_adjustment(candidate)
        technical = max(0, min(50, technical_parts["technical_score"] + sector_adjustment))
        technical_parts["technical_score"] = technical
        technical_parts["sector_adjustment"] = sector_adjustment
        relative_strength_score = _relative_strength_score_for_candidate(candidate, config)
        investor_context = _investor_context_for_candidate(config)
        investor_context_raw_score = _investor_context_raw_score_for_candidate(candidate, config, investor_context)
        investor_context_score = _investor_context_score_for_candidate(candidate, config, investor_context)
        market_context_score = 0
        total_before_selection_adjustment = technical + relative_strength_score + investor_context_score + market_context_score
        rsi_selection = _rsi_selection_adjustment(candidate.get("rsi"), selection_config)
        volume_selection = _volume_selection_adjustment(candidate.get("volume_ratio"), volume_filter)
        hot_zone_selection = _rsi_volume_hot_zone_adjustment(
            candidate.get("rsi"),
            candidate.get("volume_ratio"),
            rsi_volume_hot_zone_filter,
        )
        affordability = _affordability_adjustment(candidate, affordability_filter, config)
        winner_loser_rule = _winner_loser_rule_adjustment(candidate, winner_loser_rule_adjustment)
        total_penalty = rsi_selection["penalty"] + affordability["penalty"]
        total = max(0, total_before_selection_adjustment - total_penalty + winner_loser_rule["score_adjustment"])
        score_components = _score_components(
            technical_parts=technical_parts,
            market_context_score=market_context_score,
            relative_strength_score=relative_strength_score,
            investor_context_score=investor_context_score,
            penalty_score=-total_penalty,
            winner_loser_rule_score=winner_loser_rule["score_adjustment"],
            total_score=total,
        )
        confidence = _real_confidence(candidate, technical)
        score_reason = _real_score_reason(candidate, technical_parts, total)
        if relative_strength_score:
            score_reason = f"{score_reason}、relative_strength_score={relative_strength_score}"
        if investor_context_score:
            score_reason = f"{score_reason}、investor_context_score={investor_context_score}"
        elif investor_context_raw_score:
            score_reason = f"{score_reason}、investor_context_score={investor_context_raw_score}"
        if rsi_selection["penalty"]:
            score_reason = f"{score_reason}、RSIが{selection_config['max_rsi_for_new_position']:.0f}を超えたため減点{rsi_selection['penalty']:.0f}点"
        if affordability["penalty"]:
            score_reason = (
                f"{score_reason}、100株購入金額が{affordability['preferred_round_lot_amount']:.0f}円を超えたため"
                f"資金効率ペナルティ{affordability['penalty']:.0f}点"
            )
        if winner_loser_rule["score_adjustment"]:
            direction = "加点" if winner_loser_rule["score_adjustment"] > 0 else "減点"
            score_reason = f"{score_reason}、{winner_loser_rule['reason']}のため{direction}{abs(winner_loser_rule['score_adjustment']):.0f}点"
        if rsi_selection["excluded"]:
            score_reason = f"{score_reason}、RSI過熱のため新規買付見送り"
        if volume_selection["excluded"]:
            if volume_selection["reason"] == "volume_ratio_above_max":
                score_reason = f"{score_reason}、出来高倍率が{volume_selection['max_threshold']:.1f}を超えたため新規買付見送り"
            else:
                score_reason = f"{score_reason}、出来高倍率が{volume_selection['threshold']:.1f}未満のため新規買付見送り"
        if hot_zone_selection["excluded"]:
            score_reason = f"{score_reason}、RSIと出来高倍率が過熱ゾーンのため新規買付見送り"
        earnings_result = earnings_filter_result(candidate, target_date, config)
        if earnings_result["blocked"]:
            score_reason = f"{score_reason}、{EARNINGS_FILTER_REJECTED_REASON}"
        scored.append(
            {
                "code": candidate["code"],
                "name": candidate["name"],
                "sector_name": candidate.get("sector_name", ""),
                "section": candidate.get("section", "Unknown"),
                "market_section": candidate.get("market_section", candidate.get("section", "Unknown")),
                "listing_market": candidate.get("listing_market", candidate.get("section", "Unknown")),
                "sector_momentum_score": candidate.get("sector_momentum_score"),
                "sector_rank": candidate.get("sector_rank"),
                "sector_comment": candidate.get("sector_comment", ""),
                "sector_score_adjustment": sector_adjustment,
                "sector17_code": candidate.get("sector17_code"),
                "sector17_name": candidate.get("sector17_name"),
                "sector33_code": candidate.get("sector33_code"),
                "sector33_name": candidate.get("sector33_name"),
                "scale_category": candidate.get("scale_category"),
                "margin_type": candidate.get("margin_type"),
                "product_category": candidate.get("product_category"),
                "date": candidate["date"],
                "open": candidate.get("open"),
                "high": candidate.get("high"),
                "low": candidate.get("low"),
                "close": candidate["close"],
                "volume": candidate.get("volume"),
                "adjusted_open": candidate.get("adjusted_open"),
                "adjusted_high": candidate.get("adjusted_high"),
                "adjusted_low": candidate.get("adjusted_low"),
                "adjusted_close": candidate.get("adjusted_close"),
                "adjusted_volume": candidate.get("adjusted_volume"),
                "adjusted_price_usage": candidate.get("adjusted_price_usage"),
                "limit_up_flag": candidate.get("limit_up_flag"),
                "limit_down_flag": candidate.get("limit_down_flag"),
                "ma5": candidate.get("ma5"),
                "ma25": candidate.get("ma25"),
                "volume_ratio": candidate.get("volume_ratio"),
                "rsi": candidate.get("rsi"),
                "macd": candidate.get("macd"),
                "macd_signal": candidate.get("macd_signal"),
                "macd_hist": candidate.get("macd_hist"),
                "bb_upper": candidate.get("bb_upper"),
                "bb_middle": candidate.get("bb_middle"),
                "bb_lower": candidate.get("bb_lower"),
                "bb_position": candidate.get("bb_position"),
                "atr": candidate.get("atr"),
                "turnover_value": candidate.get("turnover_value"),
                "direct_turnover_value": candidate.get("direct_turnover_value"),
                "direct_turnover_value_source": candidate.get("direct_turnover_value_source"),
                "five_day_volatility": candidate.get("five_day_volatility"),
                "five_day_change_rate": candidate.get("five_day_change_rate"),
                "stock_return_5d": candidate.get("stock_return_5d"),
                "stock_return_10d": candidate.get("stock_return_10d"),
                "stock_return_20d": candidate.get("stock_return_20d"),
                "benchmark_source": candidate.get("benchmark_source"),
                "benchmark_return_5d": candidate.get("benchmark_return_5d"),
                "benchmark_return_10d": candidate.get("benchmark_return_10d"),
                "benchmark_return_20d": candidate.get("benchmark_return_20d"),
                "relative_strength_5d": candidate.get("relative_strength_5d"),
                "relative_strength_10d": candidate.get("relative_strength_10d"),
                "relative_strength_20d": candidate.get("relative_strength_20d"),
                "relative_strength_score": relative_strength_score,
                "affordability_filter_enabled": affordability["enabled"],
                "round_lot_amount": affordability["round_lot_amount"],
                "preferred_round_lot_amount": affordability["preferred_round_lot_amount"],
                "price_band_penalty": affordability["penalty"],
                "price_band_penalty_reason": affordability["reason"],
                "winner_loser_rule_adjustment_enabled": winner_loser_rule["enabled"],
                "winner_loser_rule_triggered": winner_loser_rule["triggered"],
                "winner_loser_rule_name": winner_loser_rule["rule_name"],
                "winner_loser_rule_score": winner_loser_rule["score_adjustment"],
                "winner_loser_rule_reason": winner_loser_rule["reason"],
                "topix_records_loaded": candidate.get("topix_records_loaded"),
                "topix_api_calls": candidate.get("topix_api_calls"),
                "topix_cache_path": candidate.get("topix_cache_path"),
                "relative_strength_feature_enabled": candidate.get("relative_strength_feature_enabled"),
                "relative_strength_scoring_enabled": candidate.get("relative_strength_scoring_enabled"),
                "relative_strength_benchmark_provider_called": candidate.get("relative_strength_benchmark_provider_called"),
                "relative_strength_cache_exists": candidate.get("relative_strength_cache_exists"),
                "relative_strength_calculated": candidate.get("relative_strength_calculated"),
                "investor_context_source": investor_context.get("investor_context_source"),
                "investor_context_week": investor_context.get("investor_context_week"),
                "overseas_net_buy": investor_context.get("overseas_net_buy"),
                "overseas_net_buy_4w_sum": investor_context.get("overseas_net_buy_4w_sum"),
                "overseas_net_buy_4w_trend": investor_context.get("overseas_net_buy_4w_trend"),
                "overseas_buy_sell_ratio": investor_context.get("overseas_buy_sell_ratio"),
                "individual_net_buy": investor_context.get("individual_net_buy"),
                "institution_net_buy": investor_context.get("institution_net_buy"),
                "trust_bank_net_buy": investor_context.get("trust_bank_net_buy"),
                "proprietary_net_buy": investor_context.get("proprietary_net_buy"),
                "investor_context_score": investor_context_raw_score,
                "candle_type": candidate.get("candle_type", "unknown"),
                "candle_body_rate": candidate.get("candle_body_rate"),
                "upper_shadow_rate": candidate.get("upper_shadow_rate"),
                "lower_shadow_rate": candidate.get("lower_shadow_rate"),
                "close_position_in_range": candidate.get("close_position_in_range"),
                "gap_rate": candidate.get("gap_rate"),
                "candlestick_signals": technical_parts["candlestick_signals"],
                "candlestick_score": technical_parts["candlestick_score"],
                "trend_score": technical_parts["trend_score"],
                "ma_score": technical_parts["trend_score"],
                "volume_score": technical_parts["volume_score"],
                "rsi_score": technical_parts["rsi_score"],
                "market_context_score": score_components["market_context_score"],
                "sector_score": score_components["sector_score"],
                "penalty_score": score_components["penalty_score"],
                "score_components": score_components,
                "score_components_total": score_components["component_total"],
                "score_components_match": score_components["matches_total_score"],
                "total_score": total,
                "rsi_selection_penalty": rsi_selection["penalty"],
                "affordability_penalty": affordability["penalty"],
                "winner_loser_rule_adjustment": winner_loser_rule["score_adjustment"],
                "rsi_selection_excluded": rsi_selection["excluded"],
                "rsi_filter_threshold": rsi_selection["threshold"],
                "volume_filter_excluded": volume_selection["excluded"],
                "volume_filter_threshold": volume_selection["threshold"],
                "volume_filter_max_threshold": volume_selection["max_threshold"],
                "volume_filter_reason": volume_selection["reason"],
                "rsi_volume_hot_zone_excluded": hot_zone_selection["excluded"],
                "rsi_volume_hot_zone_reason": hot_zone_selection["reason"],
                "rsi_volume_hot_zone_min_rsi": hot_zone_selection["min_rsi"],
                "rsi_volume_hot_zone_min_volume_ratio": hot_zone_selection["min_volume_ratio"],
                "rsi_volume_hot_zone_max_volume_ratio": hot_zone_selection["max_volume_ratio"],
                "technical_score": technical,
                "confidence": confidence,
                "rank": 0,
                "selected": False,
                "reason": score_reason,
                "score_reason": score_reason,
                "selection_reason": "",
                "selected_reason": "",
                "rejected_reason": "",
                "conditional_selection_checked": False,
                "conditional_selection_matched": False,
                "conditional_selection_reason": "",
                "market_filter_applied": False,
                "market_regime": market_regime,
                "advance_ratio": advance_ratio,
                "market_average_change_rate": market_average_change_rate,
                "classified_market_regime": classified_market_regime,
                "dynamic_exposure_regime": dynamic_regime,
                "dynamic_exposure_source_date": dynamic_context.get("source_date", ""),
                "dynamic_exposure_source_date_mode": "previous_trading_day",
                "dynamic_exposure_source_lag_days": dynamic_context.get("lag_days"),
                "dynamic_exposure_source_fallback_used": bool(dynamic_context.get("fallback_used", False)),
                "dynamic_exposure_same_day_context_used": bool(dynamic_context.get("same_day_used", False)),
                "market_filter_reason": "",
                "earnings_filter_checked": earnings_result["checked"],
                "earnings_filter_blocked": earnings_result["blocked"],
                "earnings_filter_reason": earnings_result["reason"],
                "earnings_announcement_date": earnings_result["earnings_date"],
                "earnings_calendar_records_count": earnings_calendar_records_count,
                "earnings_info_found": earnings_result.get("info_found", False),
                "earnings_candidate_date": earnings_result.get("candidate_earnings_date"),
                "earnings_days_until_earnings": earnings_result.get("days_until_earnings"),
                "earnings_pipeline_feature_enabled": earnings_pipeline.get("feature_enabled", False),
                "earnings_pipeline_fetch_start": earnings_pipeline.get("fetch_start"),
                "earnings_pipeline_fetch_end": earnings_pipeline.get("fetch_end"),
                "earnings_pipeline_cache_path": earnings_pipeline.get("cache_path"),
                "earnings_pipeline_cache_exists": earnings_pipeline.get("cache_exists", False),
                "earnings_pipeline_cache_records": earnings_pipeline.get("cache_records", earnings_calendar_records_count),
                "earnings_pipeline_cache_loaded": earnings_pipeline.get("cache_loaded", False),
                "earnings_pipeline_index_built": earnings_pipeline.get("index_built", False),
                "earnings_pipeline_candidate_matching_called": earnings_result["checked"],
                "earnings_pipeline_records_loaded": earnings_pipeline.get("earnings_records_loaded", earnings_calendar_records_count),
                "earnings_pipeline_matched_candidates": None,
                "earnings_pipeline_rejected_candidates": None,
                "earnings_pipeline_reason": earnings_pipeline.get("reason", ""),
                "source_provider": source_provider,
                "fallback": candidate.get("fallback", False),
            }
        )

    scored.sort(key=lambda item: (item["total_score"], item["confidence"]), reverse=True)
    market_filter_summary = _apply_selection_rules(scored, selection_config, market_filter, market_regime)

    return {
        "date": target_date,
        "dealer_id": config["dealer"]["id"],
        "source_provider": source_provider,
        "scoring_policy": {
            "technical_max": 50,
            "technical_breakdown": {
                "trend_score": 15,
                "volume_score": 10,
                "rsi_score": 10,
                "candlestick_score": 15,
                "sector_score": 5,
            },
            "score_formula": _score_formula_label(config),
            "min_total_score_for_selection": selection_config["min_score"],
            "min_confidence_for_selection": selection_config["min_confidence"],
            "max_selected": selection_config["max_selected"],
            "max_rsi_for_new_position": selection_config["max_rsi_for_new_position"],
            "reject_overheated_rsi": selection_config["reject_overheated_rsi"],
            "volume_filter_enabled": volume_filter["enabled"],
            "min_volume_ratio": volume_filter["min_volume_ratio"],
            "max_volume_ratio": volume_filter["max_volume_ratio"],
            "rsi_volume_hot_zone_filter": rsi_volume_hot_zone_filter,
            "affordability_filter": affordability_filter,
        },
        "selection_config": selection_config,
        "market_context": market_context or {},
        "dynamic_exposure_context": dynamic_context,
        "market_filter": market_filter_summary,
        "scores": scored,
        "selected": [item for item in scored if item["selected"]],
        "rejected": [item for item in scored if not item["selected"]],
    }


def _apply_selection_rules(
    scored: list[dict[str, Any]],
    selection_config: dict[str, Any],
    market_filter: dict[str, Any],
    market_regime: str,
) -> dict[str, Any]:
    allowed_sections = set(market_filter.get("allowed_sections") or {"TSEPrime", "TSEStandard", "TSEGrowth"})
    allow_unknown_market = bool(market_filter.get("allow_unknown_market", "allowed_sections" not in market_filter))
    risk_off = market_filter["enabled"] and market_regime == "risk_off"
    selected_count = 0
    max_selected = selection_config["max_selected"]
    min_score = selection_config["min_score"]
    if risk_off:
        max_selected = min(max_selected, market_filter["risk_off_max_buy_orders"])
        min_score = max(min_score, market_filter["risk_off_min_score"])

    for index, item in enumerate(scored, start=1):
        item["rank"] = index
        item["market_regime"] = market_regime
        if risk_off:
            item["market_filter_applied"] = True
            item["market_filter_reason"] = "risk_offのため買付抑制"

        investor_filter_result = _investor_context_filter_result(item, selection_config)
        item["investor_context_filter_checked"] = investor_filter_result["checked"]
        item["investor_context_filter_blocked"] = investor_filter_result["blocked"]
        item["investor_context_filter_reason"] = investor_filter_result["reason"]

        conditional_result = _conditional_selection_result(item, selection_config, market_regime)
        item["conditional_selection_checked"] = conditional_result["checked"]
        item["conditional_selection_matched"] = conditional_result["matched"]
        item["conditional_selection_reason"] = conditional_result["reason"]
        section = market_section_from_row(item)
        item_min_score = _effective_regular_min_score(item, selection_config, min_score)
        item["effective_min_score"] = item_min_score
        item["market_min_score_override_applied"] = item_min_score != min_score
        market_section_blocked = (
            section == "Unknown" and not allow_unknown_market
        ) or (section != "Unknown" and section not in allowed_sections)
        item["market_section_filter_checked"] = True
        item["market_section_filter_blocked"] = market_section_blocked
        item["market_section_filter_reason"] = "market_filter_excluded" if market_section_blocked else ""

        if market_section_blocked:
            item["selected"] = False
            item["rejected_reason"] = "market_filter_excluded"
            item["reason"] = item["rejected_reason"]
        elif investor_filter_result["blocked"]:
            item["rejected_reason"] = investor_filter_result["reason"]
            item["reason"] = item["rejected_reason"]
        elif item.get("earnings_filter_blocked"):
            item["rejected_reason"] = EARNINGS_FILTER_REJECTED_REASON
            item["reason"] = item["rejected_reason"]
        elif item.get("rsi_volume_hot_zone_excluded"):
            item["rejected_reason"] = "rsi_volume_hot_zone"
            item["reason"] = item["rejected_reason"]
        elif _meets_regular_selection(item, {**selection_config, "min_score": item_min_score}) and selected_count < max_selected:
            item["selected"] = True
            reason = "スコア基準を満たしたため採用"
            if item.get("market_min_score_override_applied"):
                reason = f"市場別スコア基準{item_min_score:.0f}点を満たしたため採用"
            if risk_off:
                reason = f"{reason}（risk_offだが高スコアのため限定採用）"
            item["selection_reason"] = reason
            item["selected_reason"] = item["selection_reason"]
            item["reason"] = item["selection_reason"]
            selected_count += 1
        elif conditional_result["matched"] and selected_count < max_selected:
            item["selected"] = True
            item["selection_reason"] = "conditional selected: 低スコア例外条件を満たしたため採用"
            item["selected_reason"] = item["selection_reason"]
            item["reason"] = item["selection_reason"]
            selected_count += 1
        else:
            if conditional_result["matched"] and selected_count >= max_selected:
                item["rejected_reason"] = "上位候補だが最大採用数を超えたため落選"
            elif conditional_result["checked"] and not conditional_result["matched"]:
                item["rejected_reason"] = f"conditional rejected: {conditional_result['reason']}"
            else:
                item["rejected_reason"] = _real_rejected_reason(item, selected_count, {**selection_config, "min_score": item_min_score, "max_selected": max_selected})
            if risk_off and (item["total_score"] < item_min_score or selected_count >= max_selected):
                item["rejected_reason"] = "risk_offのため買付抑制"
            item["reason"] = item["rejected_reason"]

    top_pick_allowed = selection_config["allow_top_pick_when_no_selection"]
    if risk_off and market_filter["risk_off_disable_top_pick"]:
        top_pick_allowed = False

    if selected_count == 0 and top_pick_allowed:
        for item in scored:
            if (
                not item.get("investor_context_filter_blocked")
                and not item.get("earnings_filter_blocked")
                and not item.get("market_section_filter_blocked")
                and not item.get("rsi_volume_hot_zone_excluded")
                and _meets_top_pick_selection(item, selection_config)
                and not _is_conditional_low_score_candidate(item, selection_config)
            ):
                item["selected"] = True
                item["selection_reason"] = f"通常基準{selection_config['min_score']:.0f}点には届かなかったが、ノートレード回避ルールにより最上位候補として採用"
                item["selected_reason"] = item["selection_reason"]
                item["rejected_reason"] = ""
                item["reason"] = item["selection_reason"]
                selected_count = 1
                break

    return {
        "enabled": market_filter["enabled"],
        "applied": risk_off,
        "market_regime": market_regime,
        "risk_off_buy_policy": market_filter["risk_off_buy_policy"],
        "risk_off_max_buy_orders": market_filter["risk_off_max_buy_orders"],
        "risk_off_min_score": market_filter["risk_off_min_score"],
        "risk_off_disable_top_pick": market_filter["risk_off_disable_top_pick"],
        "allowed_sections": sorted(allowed_sections),
        "allow_unknown_market": allow_unknown_market,
        "candidate_market_counts": market_section_counts(scored),
        "selected_market_counts": market_section_counts([item for item in scored if item.get("selected")]),
        "market_filter_excluded_count": sum(1 for item in scored if item.get("market_section_filter_blocked")),
        "reason": "risk_offのため買付抑制" if risk_off else "",
    }


def _selection_config(config: dict[str, Any]) -> dict[str, Any]:
    selection = config.get("selection", {})
    max_rsi = _optional_float(selection.get("max_rsi_for_new_position"))
    conditional = selection.get("conditional_selection", {})
    investor_filter = config.get("investor_context_filter", {})
    low_score_range = conditional.get("low_score_range", {}) if isinstance(conditional, dict) else {}
    allow_if = conditional.get("allow_if", {}) if isinstance(conditional, dict) else {}
    return {
        "min_score": float(selection.get("min_score", 70)),
        "market_min_score_overrides": _market_min_score_overrides(selection),
        "fallback_min_score": float(selection.get("fallback_min_score", 65)),
        "min_confidence": float(selection.get("min_confidence", config["scoring"].get("confidence_min_for_buy", 0.7))),
        "allow_top_pick_when_no_selection": bool(selection.get("allow_top_pick_when_no_selection", True)),
        "top_pick_min_score": float(selection.get("top_pick_min_score", 65)),
        "max_selected": int(selection.get("max_selected", config["portfolio"].get("max_positions", 5))),
        "max_rsi_for_new_position": max_rsi,
        "reject_overheated_rsi": bool(selection.get("reject_overheated_rsi", False)),
        "conditional_selection": {
            "enabled": bool(conditional.get("enabled", False)) if isinstance(conditional, dict) else False,
            "low_score_range": {
                "min": float(low_score_range.get("min", selection.get("fallback_min_score", 65))),
                "max": float(low_score_range.get("max", selection.get("min_score", 70) - 1)),
            },
            "allow_if": {
                "min_volume_ratio": _optional_float(allow_if.get("min_volume_ratio")) if isinstance(allow_if, dict) else None,
                "required_candlestick_signals": list(allow_if.get("required_candlestick_signals", [])) if isinstance(allow_if, dict) else [],
                "min_rsi": _optional_float(allow_if.get("min_rsi")) if isinstance(allow_if, dict) else None,
                "max_rsi": _optional_float(allow_if.get("max_rsi")) if isinstance(allow_if, dict) else None,
                "allowed_market_regimes": list(allow_if.get("allowed_market_regimes", [])) if isinstance(allow_if, dict) else [],
            },
        },
        "investor_context_filter": {
            "enabled": bool(investor_filter.get("enabled", False)) if isinstance(investor_filter, dict) else False,
            "reject_below": float(investor_filter.get("reject_below", 0)) if isinstance(investor_filter, dict) else 0.0,
            "reason": str(investor_filter.get("reason", "investor_context_negative")) if isinstance(investor_filter, dict) else "investor_context_negative",
        },
    }


def _market_min_score_overrides(selection: dict[str, Any]) -> dict[str, float]:
    raw = selection.get("market_min_score_overrides") or selection.get("min_score_by_market_section") or {}
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, float] = {}
    for key, value in raw.items():
        section = normalize_market_section(key)
        if section == "Unknown":
            continue
        try:
            overrides[section] = float(value)
        except (TypeError, ValueError):
            continue
    return overrides


def _effective_regular_min_score(
    item: dict[str, Any],
    selection_config: dict[str, Any],
    active_min_score: float,
) -> float:
    overrides = selection_config.get("market_min_score_overrides", {})
    if not isinstance(overrides, dict) or not overrides:
        return active_min_score
    section = market_section_from_row(item)
    override = overrides.get(section)
    if override is None:
        return active_min_score
    base_min_score = float(selection_config.get("min_score", active_min_score) or active_min_score)
    if active_min_score > base_min_score:
        return max(float(override), active_min_score)
    return float(override)


def _market_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    market_filter = config.get("market_filter", {})
    return {
        "enabled": bool(market_filter.get("enabled", True)),
        "risk_off_buy_policy": str(market_filter.get("risk_off_buy_policy", "conservative")),
        "risk_off_max_buy_orders": int(market_filter.get("risk_off_max_buy_orders", 1)),
        "risk_off_min_score": float(market_filter.get("risk_off_min_score", 75)),
        "risk_off_disable_top_pick": bool(market_filter.get("risk_off_disable_top_pick", True)),
        "allowed_sections": allowed_market_sections(config),
        "allow_unknown_market": bool(market_filter.get("allow_unknown_market", False)),
    }


def _volume_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    volume_filter = config.get("volume_filter", {})
    return {
        "enabled": bool(volume_filter.get("enabled", False)),
        "min_volume_ratio": float(volume_filter.get("min_volume_ratio", 0.0)),
        "max_volume_ratio": _optional_float(volume_filter.get("max_volume_ratio")),
    }


def _rsi_volume_hot_zone_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    hot_zone = config.get("rsi_volume_hot_zone_filter", {})
    return {
        "enabled": bool(hot_zone.get("enabled", False)) if isinstance(hot_zone, dict) else False,
        "min_rsi": float(hot_zone.get("min_rsi", 60.0)) if isinstance(hot_zone, dict) else 60.0,
        "min_volume_ratio": float(hot_zone.get("min_volume_ratio", 3.0)) if isinstance(hot_zone, dict) else 3.0,
        "max_volume_ratio": float(hot_zone.get("max_volume_ratio", 5.0)) if isinstance(hot_zone, dict) else 5.0,
        "reason": str(hot_zone.get("reason", "rsi_volume_hot_zone")) if isinstance(hot_zone, dict) else "rsi_volume_hot_zone",
    }


def _affordability_filter_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("affordability_filter", {})
    if not isinstance(policy, dict):
        policy = {}
    return {
        "enabled": bool(policy.get("enabled", False)),
        "preferred_round_lot_amount": _optional_float(policy.get("preferred_round_lot_amount")),
        "penalty_points": float(policy.get("penalty_points", 3.0)),
        "reason": str(policy.get("reason", "price_band_penalty")),
    }


def _affordability_adjustment(candidate: dict[str, Any], policy: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    lot_size = int(
        (config.get("capital_utilization_policy", {}) or {}).get("buy_lot_size")
        or (config.get("trading", {}) or {}).get("round_lot_size")
        or 100
    )
    price = _candidate_round_lot_price(candidate)
    round_lot_amount = round(price * lot_size, 2) if price is not None else None
    threshold = policy.get("preferred_round_lot_amount")
    enabled = bool(policy.get("enabled"))
    if not enabled or threshold is None or round_lot_amount is None or round_lot_amount <= threshold:
        return {
            "enabled": enabled,
            "round_lot_amount": round_lot_amount,
            "preferred_round_lot_amount": threshold,
            "penalty": 0.0,
            "reason": "",
        }
    return {
        "enabled": enabled,
        "round_lot_amount": round_lot_amount,
        "preferred_round_lot_amount": threshold,
        "penalty": float(policy.get("penalty_points") or 0.0),
        "reason": str(policy.get("reason") or "price_band_penalty"),
    }


def _winner_loser_rule_adjustment_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = config.get("winner_loser_rule_adjustment", {})
    if not isinstance(policy, dict):
        policy = {}
    return {
        "enabled": bool(policy.get("enabled", False)),
        "rule_name": str(policy.get("rule_name") or ""),
        "score_adjustment": float(policy.get("score_adjustment", 0) or 0),
        "volume_ratio_min": _optional_float(policy.get("volume_ratio_min")),
        "volume_ratio_max": _optional_float(policy.get("volume_ratio_max")),
        "sector_name": str(policy.get("sector_name") or ""),
        "reason": str(policy.get("reason") or policy.get("rule_name") or "winner_loser_rule_adjustment"),
    }


def _winner_loser_rule_adjustment(candidate: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(policy.get("enabled", False))
    result = {
        "enabled": enabled,
        "triggered": False,
        "rule_name": str(policy.get("rule_name") or ""),
        "score_adjustment": 0.0,
        "reason": "",
    }
    if not enabled:
        return result

    volume_ratio = _optional_float(candidate.get("volume_ratio"))
    min_volume_ratio = policy.get("volume_ratio_min")
    max_volume_ratio = policy.get("volume_ratio_max")
    if min_volume_ratio is not None:
        if volume_ratio is None or volume_ratio < float(min_volume_ratio):
            return result
    if max_volume_ratio is not None:
        if volume_ratio is None or volume_ratio > float(max_volume_ratio):
            return result

    sector_name = str(policy.get("sector_name") or "")
    if sector_name and str(candidate.get("sector_name") or "") != sector_name:
        return result

    result["triggered"] = True
    result["score_adjustment"] = float(policy.get("score_adjustment") or 0.0)
    result["reason"] = str(policy.get("reason") or policy.get("rule_name") or "winner_loser_rule_adjustment")
    return result


def _candidate_round_lot_price(candidate: dict[str, Any]) -> float | None:
    for key in [
        "entry_candidate_price",
        "signal_close_price",
        "close",
        "adjusted_close",
        "adjusted_price",
        "entry_price",
        "open",
    ]:
        value = _optional_float(candidate.get(key))
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _score_components(
    technical_parts: dict[str, Any],
    market_context_score: float,
    penalty_score: float,
    total_score: float,
    relative_strength_score: float = 0,
    investor_context_score: float = 0,
    winner_loser_rule_score: float = 0,
) -> dict[str, Any]:
    ma_score = float(technical_parts.get("trend_score") or 0)
    rsi_score = float(technical_parts.get("rsi_score") or 0)
    volume_score = float(technical_parts.get("volume_score") or 0)
    candlestick_score = float(technical_parts.get("candlestick_score") or 0)
    base_technical_score = ma_score + rsi_score + volume_score + candlestick_score
    adjusted_technical_score = float(technical_parts.get("technical_score") or base_technical_score)
    sector_score = adjusted_technical_score - base_technical_score
    components = {
        "ma_score": ma_score,
        "rsi_score": rsi_score,
        "volume_score": volume_score,
        "candlestick_score": candlestick_score,
        "market_context_score": float(market_context_score or 0),
        "relative_strength_score": float(relative_strength_score or 0),
        "investor_context_score": float(investor_context_score or 0),
        "winner_loser_rule_score": float(winner_loser_rule_score or 0),
        "sector_score": sector_score,
        "penalty_score": float(penalty_score or 0),
    }
    component_total = round(sum(components.values()), 2)
    components["component_total"] = component_total
    components["total_score"] = round(float(total_score or 0), 2)
    components["matches_total_score"] = abs(component_total - components["total_score"]) <= 0.01
    if not components["matches_total_score"]:
        components["component_difference"] = round(components["total_score"] - component_total, 2)
    return components


def _meets_regular_selection(item: dict[str, Any], selection_config: dict[str, Any]) -> bool:
    if item.get("investor_context_filter_blocked"):
        return False
    if item.get("rsi_selection_excluded"):
        return False
    if item.get("volume_filter_excluded"):
        return False
    if item.get("rsi_volume_hot_zone_excluded"):
        return False
    return item["total_score"] >= selection_config["min_score"] and item["confidence"] >= selection_config["min_confidence"]


def _conditional_selection_result(item: dict[str, Any], selection_config: dict[str, Any], market_regime: str) -> dict[str, Any]:
    conditional = selection_config.get("conditional_selection", {})
    if not conditional.get("enabled"):
        return {"checked": False, "matched": False, "reason": ""}
    if item.get("investor_context_filter_blocked"):
        return {"checked": True, "matched": False, "reason": str(item.get("investor_context_filter_reason") or "investor_context_negative")}
    if not _is_conditional_low_score_candidate(item, selection_config):
        return {"checked": False, "matched": False, "reason": ""}
    if item.get("rsi_selection_excluded"):
        return {"checked": True, "matched": False, "reason": "RSI過熱のため新規買付見送り"}
    if item.get("volume_filter_excluded"):
        if item.get("volume_filter_reason") == "volume_ratio_above_max":
            return {"checked": True, "matched": False, "reason": "volume_ratio_above_max"}
        return {"checked": True, "matched": False, "reason": "出来高倍率不足のため新規買付見送り"}
    if item.get("rsi_volume_hot_zone_excluded"):
        return {"checked": True, "matched": False, "reason": "rsi_volume_hot_zone"}
    if item["confidence"] < selection_config["min_confidence"]:
        return {"checked": True, "matched": False, "reason": "信頼度基準を満たさないため落選"}

    allow_if = conditional.get("allow_if", {})
    failures = []
    volume_ratio = _optional_float(item.get("volume_ratio"))
    min_volume_ratio = allow_if.get("min_volume_ratio")
    if min_volume_ratio is not None and (volume_ratio is None or volume_ratio < min_volume_ratio):
        failures.append(f"volume_ratioが{min_volume_ratio:.1f}未満")

    signals = set(str(signal) for signal in item.get("candlestick_signals", []) if signal)
    required_signals = [str(signal) for signal in allow_if.get("required_candlestick_signals", [])]
    missing_signals = [signal for signal in required_signals if signal not in signals]
    if missing_signals:
        failures.append("必要なcandlestick_signal不足")

    rsi = _optional_float(item.get("rsi"))
    min_rsi = allow_if.get("min_rsi")
    max_rsi = allow_if.get("max_rsi")
    if min_rsi is not None and (rsi is None or rsi < min_rsi):
        failures.append(f"RSIが{min_rsi:.0f}未満")
    if max_rsi is not None and (rsi is None or rsi >= max_rsi):
        failures.append(f"RSIが{max_rsi:.0f}以上")

    allowed_market_regimes = [str(value) for value in allow_if.get("allowed_market_regimes", [])]
    if allowed_market_regimes and market_regime not in allowed_market_regimes:
        failures.append("market_regimeが許可対象外")

    if failures:
        return {"checked": True, "matched": False, "reason": "、".join(failures)}
    return {"checked": True, "matched": True, "reason": "低スコア例外条件をすべて満たした"}


def _is_conditional_low_score_candidate(item: dict[str, Any], selection_config: dict[str, Any]) -> bool:
    conditional = selection_config.get("conditional_selection", {})
    if not conditional.get("enabled"):
        return False
    score = _optional_float(item.get("total_score"))
    if score is None:
        return False
    low_score_range = conditional.get("low_score_range", {})
    min_score = float(low_score_range.get("min", selection_config["fallback_min_score"]))
    max_score = float(low_score_range.get("max", selection_config["min_score"] - 1))
    return min_score <= score <= max_score


def _meets_top_pick_selection(item: dict[str, Any], selection_config: dict[str, Any]) -> bool:
    if item.get("investor_context_filter_blocked"):
        return False
    if item.get("rsi_selection_excluded"):
        return False
    if item.get("volume_filter_excluded"):
        return False
    if item.get("rsi_volume_hot_zone_excluded"):
        return False
    return item["total_score"] >= selection_config["top_pick_min_score"] and item["confidence"] >= selection_config["min_confidence"]


def _rsi_selection_adjustment(rsi_value: Any, selection_config: dict[str, Any]) -> dict[str, Any]:
    max_rsi = selection_config.get("max_rsi_for_new_position")
    if max_rsi is None:
        return {"penalty": 0.0, "excluded": False, "threshold": None}
    try:
        rsi = float(rsi_value)
    except (TypeError, ValueError):
        return {"penalty": 0.0, "excluded": False, "threshold": max_rsi}
    penalty = 0.0
    if rsi > max_rsi and not selection_config.get("reject_overheated_rsi"):
        penalty = min((rsi - max_rsi) * 2, 10)
    return {
        "penalty": round(penalty, 2),
        "excluded": bool(selection_config.get("reject_overheated_rsi")) and rsi > max_rsi,
        "threshold": max_rsi,
    }


def _volume_selection_adjustment(volume_ratio_value: Any, volume_filter: dict[str, Any]) -> dict[str, Any]:
    threshold = volume_filter.get("min_volume_ratio")
    max_threshold = volume_filter.get("max_volume_ratio")
    if not volume_filter.get("enabled"):
        return {"excluded": False, "threshold": threshold, "max_threshold": max_threshold, "reason": ""}
    try:
        volume_ratio = float(volume_ratio_value)
    except (TypeError, ValueError):
        return {"excluded": True, "threshold": threshold, "max_threshold": max_threshold, "reason": "volume_ratio_below_min"}
    if volume_ratio < float(threshold):
        return {"excluded": True, "threshold": threshold, "max_threshold": max_threshold, "reason": "volume_ratio_below_min"}
    if max_threshold is not None and volume_ratio > float(max_threshold):
        return {"excluded": True, "threshold": threshold, "max_threshold": max_threshold, "reason": "volume_ratio_above_max"}
    return {
        "excluded": False,
        "threshold": threshold,
        "max_threshold": max_threshold,
        "reason": "",
    }


def _rsi_volume_hot_zone_adjustment(rsi_value: Any, volume_ratio_value: Any, hot_zone_filter: dict[str, Any]) -> dict[str, Any]:
    min_rsi = hot_zone_filter.get("min_rsi")
    min_volume_ratio = hot_zone_filter.get("min_volume_ratio")
    max_volume_ratio = hot_zone_filter.get("max_volume_ratio")
    result = {
        "excluded": False,
        "reason": "",
        "min_rsi": min_rsi,
        "min_volume_ratio": min_volume_ratio,
        "max_volume_ratio": max_volume_ratio,
    }
    if not hot_zone_filter.get("enabled"):
        return result
    rsi = _optional_float(rsi_value)
    volume_ratio = _optional_float(volume_ratio_value)
    if rsi is None or volume_ratio is None:
        return result
    if rsi >= float(min_rsi) and volume_ratio >= float(min_volume_ratio) and volume_ratio <= float(max_volume_ratio):
        result["excluded"] = True
        result["reason"] = str(hot_zone_filter.get("reason") or "rsi_volume_hot_zone")
    return result


def _real_technical_score_parts(candidate: dict[str, Any]) -> dict[str, Any]:
    trend_score = 0.0
    volume_score = 0.0
    rsi_score = 0.0
    candlestick_score = 0.0
    volume_ratio = float(candidate.get("volume_ratio") or 0)
    turnover_value = float(candidate.get("turnover_value") or 0)
    rsi = float(candidate.get("rsi") or 0)
    close = float(candidate["close"])
    ma5 = float(candidate["ma5"])
    ma25 = float(candidate["ma25"])
    signals = list(candidate.get("candlestick_signals") or detect_candlestick_signals(candidate))

    if close > ma5:
        trend_score += 5
    if ma5 > ma25:
        trend_score += 5
    ma_spread = (ma5 - ma25) / ma25 if ma25 else 0.0
    trend_score += max(0.0, 1 - abs(ma_spread - 0.03) / 0.15) * 5

    volume_score += min(volume_ratio, 2.5) / 2.5 * 6
    volume_score += min(turnover_value / 2_000_000_000, 1.0) * 4

    if 50 <= rsi <= 65:
        rsi_score = 10
    elif 40 <= rsi < 50:
        rsi_score = 6 + (rsi - 40) / 10 * 4
    elif 65 < rsi <= 70:
        rsi_score = 8 - (rsi - 65) / 5 * 2
    elif 30 <= rsi < 40:
        rsi_score = (rsi - 30) / 10 * 6
    elif 70 < rsi <= 80:
        rsi_score = max(0.0, 6 - (rsi - 70) / 10 * 6)

    if _has_candlestick_data(candidate):
        if "bullish_candle" in signals:
            candlestick_score += 4
        if "strong_bullish_candle" in signals:
            candlestick_score += 5
        if "long_lower_shadow_support" in signals:
            candlestick_score += 3
        if "ma_reclaim" in signals:
            candlestick_score += 2
        if "volume_confirmed_breakout" in signals:
            candlestick_score += 3
        if "long_upper_shadow_warning" in signals:
            candlestick_score -= 4
        if "overheated_warning" in signals:
            candlestick_score -= 5
    else:
        candlestick_score = 12

    if candidate.get("fallback"):
        candlestick_score -= 2

    trend_score = round(max(0, min(15, trend_score)))
    volume_score = round(max(0, min(10, volume_score)))
    rsi_score = round(max(0, min(10, rsi_score)))
    candlestick_score = round(max(0, min(15, candlestick_score)))
    technical_score = max(0, min(50, trend_score + volume_score + rsi_score + candlestick_score))
    return {
        "technical_score": technical_score,
        "trend_score": trend_score,
        "volume_score": volume_score,
        "rsi_score": rsi_score,
        "candlestick_score": candlestick_score,
        "candlestick_signals": signals,
    }


def _real_technical_score(candidate: dict[str, Any]) -> int:
    return _real_technical_score_parts(candidate)["technical_score"]


def _real_confidence(candidate: dict[str, Any], technical_score: int) -> float:
    confidence = 0.45 + technical_score / 100
    if candidate.get("fallback"):
        confidence -= 0.08
    required_fields = ["close", "volume", "ma5", "ma25", "rsi", "volume_ratio", "turnover_value", "five_day_volatility"]
    missing = sum(1 for field in required_fields if candidate.get(field) is None)
    confidence -= missing * 0.04
    return round(max(0.1, min(0.95, confidence)), 2)


def _relative_strength_score_for_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> float:
    scoring = config.get("scoring", {})
    if not config.get("features", {}).get("relative_strength"):
        return 0.0
    if not scoring.get("use_relative_strength_score"):
        return 0.0
    weight = _optional_float(scoring.get("relative_strength_score_weight"))
    max_score = 10.0 if weight is None else max(0.0, min(10.0, weight))
    raw_score = _optional_float(candidate.get("relative_strength_score")) or 0.0
    return round(max(0.0, min(max_score, raw_score)), 2)


def _investor_context_for_candidate(config: dict[str, Any]) -> dict[str, Any]:
    context = config.get("_investor_context")
    return context if isinstance(context, dict) else {}


def _investor_context_raw_score_for_candidate(candidate: dict[str, Any], config: dict[str, Any], context: dict[str, Any]) -> float:
    if not config.get("features", {}).get("investor_context"):
        return 0.0
    raw_score = _optional_float(candidate.get("investor_context_score"))
    if raw_score is None:
        raw_score = _optional_float(context.get("investor_context_score"))
    raw_score = raw_score or 0.0
    return round(max(-3.0, min(5.0, raw_score)), 2)


def _investor_context_score_for_candidate(candidate: dict[str, Any], config: dict[str, Any], context: dict[str, Any]) -> float:
    scoring = config.get("scoring", {})
    if not config.get("features", {}).get("investor_context"):
        return 0.0
    if not scoring.get("use_investor_context_score"):
        return 0.0
    weight = _optional_float(scoring.get("investor_context_score_weight"))
    max_score = 5.0 if weight is None else max(0.0, min(5.0, weight))
    raw_score = _investor_context_raw_score_for_candidate(candidate, config, context)
    return round(max(-3.0, min(max_score, raw_score)), 2)


def _investor_context_filter_result(item: dict[str, Any], selection_config: dict[str, Any]) -> dict[str, Any]:
    filter_config = selection_config.get("investor_context_filter", {})
    if not filter_config.get("enabled"):
        return {"checked": False, "blocked": False, "reason": ""}
    score = _optional_float(item.get("investor_context_score"))
    if score is None:
        return {"checked": True, "blocked": False, "reason": ""}
    reject_below = float(filter_config.get("reject_below", 0.0))
    blocked = score < reject_below
    return {
        "checked": True,
        "blocked": blocked,
        "reason": str(filter_config.get("reason") or "investor_context_negative") if blocked else "",
    }


def _score_formula_label(config: dict[str, Any]) -> str:
    parts = ["technical_score"]
    if config.get("scoring", {}).get("use_relative_strength_score"):
        parts.append("relative_strength_score")
    if config.get("scoring", {}).get("use_investor_context_score"):
        parts.append("investor_context_score")
    if bool((config.get("winner_loser_rule_adjustment", {}) or {}).get("enabled", False)):
        parts.append("winner_loser_rule_score")
    parts.extend(["market_context_score", "penalty_score"])
    return " + ".join(parts)


def _real_score_reason(candidate: dict[str, Any], technical_parts: dict[str, Any], total_score: int) -> str:
    technical_score = technical_parts["technical_score"]
    signals = technical_parts.get("candlestick_signals", [])
    reasons = [
        f"technical_score={technical_score}",
        f"trend_score={technical_parts['trend_score']}",
        f"volume_score={technical_parts['volume_score']}",
        f"rsi_score={technical_parts['rsi_score']}",
        f"candlestick_score={technical_parts['candlestick_score']}",
    ]
    if "bullish_candle" in signals and "ma_reclaim" in signals:
        reasons.append("陽線かつ終値が高値圏で、5日線を上回っているため短期資金流入を評価")
    elif "long_upper_shadow_warning" in signals:
        reasons.append("出来高増加はあるが、上ヒゲが長く高値圏で売り圧力を確認")
    if "ma_trend_alignment" in signals:
        reasons.append("close > ma5 > ma25 の上昇配列を確認")
    if "overheated_warning" in signals:
        reasons.append("RSI高水準かつ上ヒゲを伴う失速に注意")
    if technical_parts.get("sector_adjustment"):
        direction = "加点" if technical_parts["sector_adjustment"] > 0 else "減点"
        reasons.append(f"業種モメンタムにより{direction}{technical_parts['sector_adjustment']:+.0f}点")
    if candidate.get("fallback"):
        reasons.append("fallback候補のため減点")
    reasons.append(f"total_score={total_score}")
    return "、".join(reasons)


def _has_candlestick_data(candidate: dict[str, Any]) -> bool:
    return all(candidate.get(field) is not None for field in ["candle_body_rate", "upper_shadow_rate", "lower_shadow_rate", "close_position_in_range"])


def _sector_score_adjustment(candidate: dict[str, Any]) -> int:
    score = candidate.get("sector_momentum_score")
    if score is None:
        return 0
    try:
        raw = (float(score) - 50.0) / 10.0
    except (TypeError, ValueError):
        return 0
    return round(max(-5.0, min(5.0, raw)))


def _real_rejected_reason(
    item: dict[str, Any],
    selected_count: int,
    selection_config: dict[str, Any],
) -> str:
    if item.get("rsi_selection_excluded"):
        return "RSI過熱のため新規買付見送り"
    if item.get("volume_filter_excluded"):
        if item.get("volume_filter_reason") == "volume_ratio_above_max":
            return "volume_ratio_above_max"
        return "出来高倍率不足のため新規買付見送り"
    if item.get("rsi_volume_hot_zone_excluded"):
        return "rsi_volume_hot_zone"
    if item.get("investor_context_filter_blocked"):
        return str(item.get("investor_context_filter_reason") or "investor_context_negative")
    if item["confidence"] < selection_config["min_confidence"]:
        return "信頼度基準を満たさないため落選"
    if item["total_score"] < selection_config["top_pick_min_score"]:
        return f"トップピック基準{selection_config['top_pick_min_score']:.0f}点を下回るため落選"
    if item["total_score"] < selection_config["min_score"]:
        return f"通常基準{selection_config['min_score']:.0f}点には届かないため落選"
    if selected_count >= selection_config["max_selected"]:
        return "上位候補だが最大採用数を超えたため落選"
    return "採用条件を満たさなかったため"


def _technical_score(candidate: dict[str, Any], config: dict[str, Any], rng: random.Random) -> int:
    max_score = int(config["scoring"]["technical_max"])
    momentum_part = min(max(candidate["momentum_5d"], 0), 0.12) / 0.12 * 24
    volume_part = min(candidate["volume_ratio_20d"], 3.0) / 3.0 * 16
    volatility_part = max(0, 0.08 - candidate["volatility_20d"]) / 0.08 * 10
    return min(max_score, round(momentum_part + volume_part + volatility_part + rng.uniform(-3, 3)))


def _score_comment(total: int, confidence: float) -> str:
    if total >= 80 and confidence >= 0.75:
        return "教科書的には優先検討に値する水準です。"
    if total >= 65:
        return "候補としては悪くありませんが、過信は禁物です。"
    return "今回は見送りが妥当です。"


def _rejection_reason(
    item: dict[str, Any],
    selected_count: int,
    max_positions: int,
    confidence_min: float,
) -> str:
    if item["confidence"] < confidence_min:
        return "信頼度が採用基準に届かないため落選"
    if selected_count >= max_positions:
        return "最大保有銘柄数の上限に達したため落選"
    return "総合順位が採用圏外のため落選"
