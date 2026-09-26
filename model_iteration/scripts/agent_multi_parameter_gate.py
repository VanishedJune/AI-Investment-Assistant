# -*- coding: utf-8 -*-
"""Agent多量化参数联合判断与发布门禁。

本模块不生成方向或仓位，只把客观参数整理为固定契约，并检查Agent填写的
七项证据解释、交易情景和动作是否自洽。不按满足项数决定入场资格，也不因
周线三项均未满足而否决逆势修复机会。冠军模型与B0207计算不调用本模块。
v3另要求连续观察证据及面向持有期价格收益的分析，不将当前强弱当作预测。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Mapping


GATE_VERSION = "agent-multi-parameter-v3"
AGENT_DIRECTIONS = {"bullish", "neutral", "bearish"}
STRUCTURAL_TRENDS = {"bullish", "neutral", "bearish"}
MOMENTUM_STATES = {"strengthening", "slowing", "repairing", "deteriorating", "mixed"}
TRADE_ACTIONS = {
    "open", "add", "hold", "hold_with_exit", "reduce", "exit", "avoid_new",
    "receive_protection_reinvestment",
}
ENTRY_SETUPS = {"none", "trend_continuation", "pullback_entry", "countertrend_rebound"}
EVIDENCE_KEYS = (
    "weekly_first_positive",
    "weekly_second_nonnegative",
    "daily_first_positive",
    "daily_second_nonnegative",
    "daily_close_above_ma20",
    "weekly_close_above_ma20",
    "volume_confirmed",
)


def empty_evidence_review() -> dict[str, str]:
    """只生成待Agent填写的清单，不替Agent解释任何客观条件。"""
    return {key: "" for key in EVIDENCE_KEYS}


def empty_forward_analysis(evidence: Mapping[str, object], start: str, end: str) -> dict:
    """只复制信息边界/参考价/周期；所有判断留空，不能用指标代填预测。"""
    return {
        "objective": "holding_period_price_return",
        "information_cutoff": evidence["data_as_of"],
        "horizon": {"start": start, "end": end},
        "reference_raw_close": evidence["daily"][-1]["raw_close"],
        "current_state": "",
        "observed_change": {"daily": "", "weekly": ""},
        "expected_price_direction": "",
        "return_thesis": "",
        "indicator_price_distinction": "",
        "entry_quality": "",
        "priced_in_risk": "",
        "upside_basis": "",
        "downside_basis": "",
        "cost_and_execution": "",
        "opportunity_cost": "",
        "scenarios": {key: {"condition": "", "price_path": "", "action": ""}
                      for key in ("up", "sideways", "down")},
        "invalidation": "",
        "review_date": "",
    }


TIMEFRAME_FIELDS = (
    "close",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "rsi14",
    "dif_first_raw",
    "dif_second_raw",
    "dif_first_normalized",
    "dif_second_normalized",
)


class AgentGateError(ValueError):
    """Agent联合判断违反客观门禁。"""


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise AgentGateError(f"{label}必须是有限数字而非布尔值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentGateError(f"{label}必须是有限数字") from exc
    if not math.isfinite(number):
        raise AgentGateError(f"{label}必须是有限数字")
    return number


def _timeframe_evidence(values: Mapping[str, object], label: str) -> dict:
    source_keys = {
        "close": "adj_close",
        "ma5": "sma5",
        "ma10": "sma10",
        "ma20": "sma20",
        "ma60": "sma60",
        "rsi14": "rsi14",
        "dif_first_raw": "dif_first_raw",
        "dif_second_raw": "dif_second_raw",
        "dif_first_normalized": "dif_first_normalized",
        "dif_second_normalized": "dif_second_normalized",
    }
    return {
        target: _finite_number(values.get(source), f"{label}.{source}")
        for target, source in source_keys.items()
    }


def build_quantitative_evidence(feature_item: Mapping[str, object]) -> dict:
    daily = feature_item.get("daily")
    weekly = feature_item.get("weekly")
    volume = feature_item.get("volume")
    relative = feature_item.get("relative_strength")
    if not isinstance(daily, Mapping) or not isinstance(weekly, Mapping):
        raise AgentGateError("缺少日K或周K客观参数")
    if not isinstance(volume, Mapping) or not isinstance(relative, Mapping):
        raise AgentGateError("缺少成交量或相对强弱客观参数")
    return {
        "daily": _timeframe_evidence(daily, "daily"),
        "weekly": _timeframe_evidence(weekly, "weekly"),
        "volume_to_20d_median": _finite_number(
            volume.get("volume_to_20d_median"), "volume.volume_to_20d_median"
        ),
        "return_20d": _finite_number(relative.get("return_20d"), "relative_strength.return_20d"),
        "return_60d": _finite_number(relative.get("return_60d"), "relative_strength.return_60d"),
    }


def validate_quantitative_evidence(
    submitted: object, feature_item: Mapping[str, object], code: str
) -> dict:
    if not isinstance(submitted, Mapping):
        raise AgentGateError(f"{code}缺少quantitative_evidence")
    expected = build_quantitative_evidence(feature_item)
    if set(submitted) != set(expected):
        raise AgentGateError(f"{code}量化证据字段不完整或含额外字段")
    for timeframe in ("daily", "weekly"):
        actual_timeframe = submitted.get(timeframe)
        if not isinstance(actual_timeframe, Mapping) or set(actual_timeframe) != set(TIMEFRAME_FIELDS):
            raise AgentGateError(f"{code}.{timeframe}量化证据字段不完整")
        for field in TIMEFRAME_FIELDS:
            actual = _finite_number(actual_timeframe.get(field), f"{code}.{timeframe}.{field}")
            if abs(actual - expected[timeframe][field]) > 1e-10:
                raise AgentGateError(f"{code}.{timeframe}.{field}与客观参数快照不一致")
    for field in ("volume_to_20d_median", "return_20d", "return_60d"):
        actual = _finite_number(submitted.get(field), f"{code}.{field}")
        if abs(actual - expected[field]) > 1e-10:
            raise AgentGateError(f"{code}.{field}与客观参数快照不一致")
    return expected


def gate_diagnostics(evidence: Mapping[str, object]) -> dict:
    daily = evidence["daily"]
    weekly = evidence["weekly"]
    volume = float(evidence["volume_to_20d_median"])
    return_20d = float(evidence["return_20d"])
    observations = {
        "weekly_first_positive": weekly["dif_first_normalized"] > 0,
        "weekly_second_nonnegative": weekly["dif_second_normalized"] >= 0,
        "daily_first_positive": daily["dif_first_normalized"] > 0,
        "daily_second_nonnegative": daily["dif_second_normalized"] >= 0,
        "daily_close_above_ma20": daily["close"] > daily["ma20"],
        "weekly_close_above_ma20": weekly["close"] > weekly["ma20"],
        "volume_confirmed": volume >= 1.0,
    }
    severe_deterioration = (
        weekly["dif_second_normalized"] <= -1.0
        and daily["dif_first_normalized"] < 0
        and daily["dif_second_normalized"] < 0
        and volume < 0.60
    )
    overheated_thin_volume = daily["rsi14"] >= 70.0 and return_20d >= 0.10 and volume < 1.0
    return {
        "evaluation_mode": "mandatory_evidence_review_no_vote",
        "checklist_observations": observations,
        # 这是必须评估抄底机会的背景标记，不是买入信号或否决条件。
        "weekly_all_unfavorable": not any(
            observations[key]
            for key in ("weekly_first_positive", "weekly_second_nonnegative", "weekly_close_above_ma20")
        ),
        "severe_deterioration": severe_deterioration,
        "overheated_thin_volume": overheated_thin_volume,
    }


def _require_text(values: Mapping[str, object], field: str, code: str) -> None:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AgentGateError(f"{code}.{field}必须填写非空证据说明")


def _validate_evidence_review(decision: Mapping[str, object], code: str) -> None:
    review = decision.get("evidence_review")
    if not isinstance(review, Mapping) or set(review) != set(EVIDENCE_KEYS):
        raise AgentGateError(f"{code}.evidence_review必须逐项覆盖七项证据，不得按数量替代分析")
    for key in EVIDENCE_KEYS:
        _require_text(review, key, f"{code}.evidence_review")


def _iso_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise AgentGateError(f"{label}必须为YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise AgentGateError(f"{label}必须为YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise AgentGateError(f"{label}必须为YYYY-MM-DD")
    return parsed


def validate_forward_analysis(
    decision: Mapping[str, object], code: str, *, expected_dates: tuple[str, str, str] | None = None,
) -> None:
    """检查预测对象、时间、方向和入场理由契约；不声称验证了文字的预测能力。"""
    evidence = decision.get("forward_evidence")
    analysis = decision.get("forward_analysis")
    if not isinstance(evidence, Mapping) or not isinstance(analysis, Mapping):
        raise AgentGateError(f"{code}缺少forward_evidence或forward_analysis，不能用当前状态代替未来判断")
    cutoff = _iso_date(evidence.get("data_as_of"), f"{code}.forward_evidence.data_as_of")
    if analysis.get("information_cutoff") != cutoff.isoformat():
        raise AgentGateError(f"{code}未来判断的信息截止日不一致")
    if analysis.get("objective") != "holding_period_price_return":
        raise AgentGateError(f"{code}预测对象必须为持有期价格收益，不能用MACD变色或金叉代替")
    horizon = analysis.get("horizon")
    if not isinstance(horizon, Mapping) or set(horizon) != {"start", "end"}:
        raise AgentGateError(f"{code}.horizon必须明确起止日期")
    start = _iso_date(horizon["start"], f"{code}.horizon.start")
    end = _iso_date(horizon["end"], f"{code}.horizon.end")
    if not cutoff < start < end:
        raise AgentGateError(f"{code}预测必须发生在信息截止日之后且持有期非空")
    if expected_dates is not None and (cutoff.isoformat(), start.isoformat(), end.isoformat()) != expected_dates:
        raise AgentGateError(f"{code}预测截止日/持有期与本次报告周期不一致")
    review = _iso_date(analysis.get("review_date"), f"{code}.review_date")
    if not cutoff < review <= end:
        raise AgentGateError(f"{code}复核日期必须晚于截止日且不晚于持有期结束")
    observed = analysis.get("observed_change")
    if not isinstance(observed, Mapping) or set(observed) != {"daily", "weekly"}:
        raise AgentGateError(f"{code}必须分列日周连续变化解释observed_change")
    for timeframe, count in (("daily", 5), ("weekly", 4)):
        points = evidence.get(timeframe)
        if not isinstance(points, list) or len(points) != count or any(not isinstance(p, Mapping) for p in points):
            raise AgentGateError(f"{code}.{timeframe}缺少连续观察证据")
        dates = [_iso_date(point.get("date"), f"{code}.{timeframe}.date") for point in points]
        if dates != sorted(set(dates)) or any(day > cutoff for day in dates):
            raise AgentGateError(f"{code}.{timeframe}观察日期重复、乱序或含未来数据")
        if timeframe == "daily" and dates[-1] != cutoff:
            raise AgentGateError(f"{code}.daily连续观察未覆盖截止日")
        _require_text(observed, timeframe, f"{code}.observed_change")
    reference = _finite_number(analysis.get("reference_raw_close"), f"{code}.reference_raw_close")
    last_close = _finite_number(evidence["daily"][-1].get("raw_close"), f"{code}.latest_raw_close")
    if reference <= 0 or abs(reference - last_close) > 1e-10:
        raise AgentGateError(f"{code}判断必须以已知最新原始收盘价为参考，不得冒充未来成交价")
    direction_map = {"bullish": "up", "neutral": "sideways", "bearish": "down"}
    direction = decision.get("agent_direction")
    if not isinstance(direction, str) or direction not in direction_map:
        raise AgentGateError(f"{code}.agent_direction非法")
    if analysis.get("expected_price_direction") != direction_map[direction]:
        raise AgentGateError(f"{code}Agent方向必须对应所声明持有期的预期价格方向")
    quality = analysis.get("entry_quality")
    if not isinstance(quality, str) or quality not in {"attractive", "wait", "unattractive", "uncertain"}:
        raise AgentGateError(f"{code}.entry_quality必须明确当前入场质量")
    if decision.get("new_position_eligible") is True and quality != "attractive":
        raise AgentGateError(f"{code}允许本次新建/加仓必须说明当前入场具有吸引力，不能仅因状态多头")
    for field in (
        "current_state", "return_thesis", "indicator_price_distinction", "priced_in_risk",
        "upside_basis", "downside_basis", "cost_and_execution", "opportunity_cost", "invalidation",
    ):
        _require_text(analysis, field, f"{code}.forward_analysis")
    scenarios = analysis.get("scenarios")
    if not isinstance(scenarios, Mapping) or set(scenarios) != {"up", "sideways", "down"}:
        raise AgentGateError(f"{code}必须给出持有期价格上涨、震荡、下跌三路径")
    for name, scenario in scenarios.items():
        if not isinstance(scenario, Mapping):
            raise AgentGateError(f"{code}.{name}情景必须包含条件、价格路径及动作")
        for field in ("condition", "price_path", "action"):
            _require_text(scenario, field, f"{code}.scenarios.{name}")


def validate_agent_decision(
    *,
    code: str,
    evidence: Mapping[str, object],
    decision: Mapping[str, object],
    current_weight_pct: float,
    recommended_weight_pct: float,
    expected_dates: tuple[str, str, str] | None = None,
    protection_only: bool = False,
) -> dict:
    direction = str(decision.get("agent_direction") or "")
    structural = str(decision.get("structural_trend") or "")
    momentum = str(decision.get("momentum_state") or "")
    action = str(decision.get("trade_action") or "")
    if direction not in AGENT_DIRECTIONS:
        raise AgentGateError(f"{code}.agent_direction非法")
    if structural not in STRUCTURAL_TRENDS:
        raise AgentGateError(f"{code}.structural_trend非法")
    if momentum not in MOMENTUM_STATES:
        raise AgentGateError(f"{code}.momentum_state非法")
    if action not in TRADE_ACTIONS:
        raise AgentGateError(f"{code}.trade_action非法")
    if not isinstance(decision.get("new_position_eligible"), bool):
        raise AgentGateError(f"{code}.new_position_eligible必须为布尔值")
    if not isinstance(decision.get("staged_entry"), bool):
        raise AgentGateError(f"{code}.staged_entry必须为布尔值")
    if not str(decision.get("multi_parameter_conclusion") or "").strip():
        raise AgentGateError(f"{code}缺少multi_parameter_conclusion")
    if not isinstance(decision.get("holding_inertia_note"), str):
        raise AgentGateError(f"{code}.holding_inertia_note必须为字符串")
    _validate_evidence_review(decision, code)
    _require_text(decision, "position_size_reason", code)
    setup = decision.get("entry_setup")
    if not isinstance(setup, str) or setup not in ENTRY_SETUPS:
        raise AgentGateError(f"{code}.entry_setup非法")
    if not isinstance(decision.get("countertrend_thesis"), str):
        raise AgentGateError(f"{code}.countertrend_thesis必须为字符串")

    current = _finite_number(current_weight_pct, f"{code}.current_weight_pct")
    recommended = _finite_number(recommended_weight_pct, f"{code}.recommended_weight_pct")
    diagnostics = gate_diagnostics(evidence)
    increasing = recommended > current + 1e-8
    decreasing = recommended < current - 1e-8

    if protection_only:
        # V70 scope is checked by agent_v70_protection.validate_plan. Do not
        # let legacy entry thresholds override R007 or generate a new signal.
        if decision["new_position_eligible"] or action in {"open", "add"}:
            raise AgentGateError(f"{code}V70看空保护不得新建或普通加仓")
        if increasing:
            if current <= 1e-8 or action != "receive_protection_reinvestment":
                raise AgentGateError(f"{code}保护回投资金只能进入保护前已有持仓")
        if decreasing and action not in {"reduce", "exit"}:
            raise AgentGateError(f"{code}保护减持与动作不一致")
        if not decreasing and not increasing and current > 1e-8:
            if action not in {"hold", "hold_with_exit"}:
                raise AgentGateError(f"{code}已有仓位不变时必须保持")
            _require_text(decision, "holding_inertia_note", code)
        if current <= 1e-8 and action != "avoid_new":
            raise AgentGateError(f"{code}无仓位时不得生成保护交易")
        validate_forward_analysis(decision, code, expected_dates=expected_dates)
        return diagnostics

    # 周线弱势必须被解释，允许Agent在尚未周线翻多时判断日线领先的抄底机会。
    # 仅检查情景/理由完整性，不把任何导数正负或成交量比单独升级为新门槛。
    if diagnostics["weekly_all_unfavorable"] or setup == "countertrend_rebound":
        _require_text(decision, "countertrend_thesis", code)
    if decision["new_position_eligible"]:
        if direction != "bullish" or setup == "none":
            raise AgentGateError(f"{code}新建仓资格必须有明确方向和入场情景")
        if diagnostics["weekly_all_unfavorable"] and setup != "countertrend_rebound":
            raise AgentGateError(f"{code}周线三项均未满足时，拟入场须说明countertrend_rebound情景，而非冒称顺势")
        for field in (
            "entry_trigger", "cancel_or_exit", "timeframe_relationship",
            "volume_confirmation", "evidence_for", "evidence_against",
        ):
            _require_text(decision, field, code)
    elif setup != "none":
        raise AgentGateError(f"{code}不具备本次新建仓资格时entry_setup应为none；潜在机会写入证据说明")

    if increasing:
        expected_actions = {"open"} if current <= 1e-8 else {"add"}
        if action not in expected_actions:
            raise AgentGateError(f"{code}建议增仓但trade_action不是{next(iter(expected_actions))}")
        if direction != "bullish" or decision.get("new_position_eligible") is not True:
            raise AgentGateError(f"{code}建议增仓必须明确看多且通过新建仓资格")
    elif decreasing:
        if action not in {"reduce", "exit"}:
            raise AgentGateError(f"{code}建议减仓但trade_action不是reduce或exit")
    elif current > 1e-8:
        if action not in {"hold", "hold_with_exit"}:
            raise AgentGateError(f"{code}维持已有仓位必须明确hold或hold_with_exit")
        if not str(decision.get("holding_inertia_note") or "").strip():
            raise AgentGateError(f"{code}维持原仓位必须披露持仓惯性是否影响判断")
    elif action != "avoid_new":
        raise AgentGateError(f"{code}当前及建议均为0时trade_action必须为avoid_new")

    if diagnostics["severe_deterioration"]:
        if direction == "bullish" or momentum != "deteriorating":
            raise AgentGateError(f"{code}触发多参数强恶化门禁，不得标记看多且动量必须为deteriorating")
        if decision.get("new_position_eligible") is not False:
            raise AgentGateError(f"{code}触发多参数强恶化门禁，不得具备新建仓资格")
        if increasing or action in {"open", "add"}:
            raise AgentGateError(f"{code}触发多参数强恶化门禁，不得新增或加仓")

    if increasing and diagnostics["overheated_thin_volume"]:
        if decision.get("staged_entry") is not True:
            raise AgentGateError(f"{code}高RSI、高20日涨幅且量能不足，必须分步建仓")
        if recommended - current > 10.0 + 1e-8:
            raise AgentGateError(f"{code}追涨风险门禁下单次建议增幅不得超过10个百分点")
    validate_forward_analysis(decision, code, expected_dates=expected_dates)
    return diagnostics
