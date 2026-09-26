"""V70 protection-only publication contract, not a new signal or backtest.

The Agent supplies the reviewed rule and counterevidence. This module checks
scope and arithmetic; it cannot establish that a narrative predicts returns.
No file access, model calls, trade execution, or historical state mutation.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

POLICY = "agent-v70-bearish-protection-with-holding-peak-stop-reinvest-prior-holdings"
LEGACY_POLICY = "agent-v70-bearish-protection-with-holding-peak-stop"
LEGACY_POLICIES = {
    LEGACY_POLICY,
    "agent-v70-bearish-protection",
}
BEARISH_RULES = {"R002", "R003", "R008"}
STOP_LOSS_THRESHOLD = 0.10


class ProtectionError(ValueError):
    pass


def resolve_versions(rules_text, state, memory):
    """Read current authority pointers; legacy progress fields are not authority."""
    match = re.search(r"^#\s+ETF统一训练规则[（(]v(\d+)[)）]", rules_text, re.M)
    if not match or match.group(1) != "70":
        raise ProtectionError("规则文档未明确为V70，禁止猜测版本")
    authority = state.get('authority') or {}
    active = (memory.get('governance') or {}).get('active_versions') or {}
    policy = memory.get('current_policy') or {}
    versions = (authority.get('current_effective_policy_version'),
                authority.get('current_effective_rule_version'), active.get('policy_version'),
                active.get('rule_document_version'), policy.get('policy_version'),
                policy.get('rule_document_version'))
    if any(str(value) != '70' for value in versions):
        raise ProtectionError("V70规则/策略权威指针不一致，禁止自动升级")
    version = memory.get('memory_version')
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ProtectionError("当前记忆版本非法")
    if any(str(v) != str(version) for v in (active.get('memory_version'),
            authority.get('current_effective_memory_version'))):
        raise ProtectionError("当前记忆版本与权威指针不一致")
    return dict(rule_version='v70', policy_version='70', state_version=str(version), memory_version=str(version))


def empty_review():
    # Unknown is not false: prior executed protection must be reviewed first.
    return {"bearish_rule": "", "rule_reason": "", "r007_review": "",
            "counterevidence": "", "invalidation": "", "interval_status": "unknown"}


def empty_holding_peak_stop_review():
    """Return an intentionally incomplete review; publication must fill it."""
    return {
        "held_before_decision": None,
        "holding_start_date": "",
        "holding_peak_raw_close": None,
        "signal_raw_close": None,
        "drawdown_from_holding_peak": None,
        "threshold_pct": 10.0,
        "triggered": None,
        "interval_status": "unknown",
        "evidence_note": "",
    }


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtectionError("保护比例必须为有限数字")
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ProtectionError("保护比例必须在0至100之间")
    return float(value)


def _plan(weights, cash, order):
    if not isinstance(weights, Mapping) or set(weights) != set(order):
        raise ProtectionError("保护组合必须完整覆盖固定8只ETF")
    values = {code: _number(weights[code]) for code in order}
    cash = _number(cash)
    if abs(sum(values.values()) + cash - 100) > 1e-8:
        raise ProtectionError("保护组合与现金必须合计100%")
    return values, cash


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtectionError(label + "必须为有限数字")
    value = float(value)
    if not math.isfinite(value):
        raise ProtectionError(label + "必须为有限数字")
    return value


def _validate_holding_peak_stop(review, code):
    if not isinstance(review, Mapping):
        raise ProtectionError(code + "缺少持有期高点回撤止损复核")
    held = review.get("held_before_decision")
    if not isinstance(held, bool):
        raise ProtectionError(code + "必须明确决策前是否连续持有")
    threshold = _finite_number(review.get("threshold_pct"), code + "止损阈值")
    if abs(threshold - STOP_LOSS_THRESHOLD * 100.0) > 1e-8:
        raise ProtectionError(code + "持有期高点回撤止损阈值必须为10%")
    note = review.get("evidence_note")
    if not isinstance(note, str) or not note.strip():
        raise ProtectionError(code + "缺少持有期高点止损证据说明")
    status = review.get("interval_status")
    if status not in {"new", "already_applied", "not_triggered"}:
        raise ProtectionError(code + "持有期高点止损区间状态非法")
    triggered = review.get("triggered")
    if not isinstance(triggered, bool):
        raise ProtectionError(code + "必须明确持有期高点止损是否触发")
    if not held:
        if triggered or status != "not_triggered":
            raise ProtectionError(code + "决策前未持有时不得触发持有期止损")
        return {"triggered": False, "new": False}

    start = review.get("holding_start_date")
    if not isinstance(start, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", start):
        raise ProtectionError(code + "缺少连续持有起始日")
    peak = _finite_number(review.get("holding_peak_raw_close"), code + "持有期最高日收盘")
    current = _finite_number(review.get("signal_raw_close"), code + "信号日收盘")
    drawdown = _finite_number(review.get("drawdown_from_holding_peak"), code + "持有期高点回撤")
    if peak <= 0 or current <= 0:
        raise ProtectionError(code + "持有期最高收盘和信号日收盘必须大于0")
    expected = current / peak - 1.0
    if abs(drawdown - expected) > 1e-8:
        raise ProtectionError(code + "持有期高点回撤与价格不一致")
    expected_triggered = expected < -STOP_LOSS_THRESHOLD
    if triggered != expected_triggered:
        raise ProtectionError(code + "持有期高点止损触发判断错误")
    if triggered and status not in {"new", "already_applied"}:
        raise ProtectionError(code + "已触发止损时必须说明区间是否首次触发")
    if not triggered and status != "not_triggered":
        raise ProtectionError(code + "未触发止损时区间状态必须为not_triggered")
    return {"triggered": triggered, "new": triggered and status == "new"}


def validate_plan(*, policy, base_kind, base_weights, base_cash,
                  weights, cash, rows, order, allow_legacy=False):
    # Frozen reports may retain either pre-2026-09-13 protection identifier.
    # They are accepted only when the caller has explicitly established a
    # historical analysis date.  New analysis must always use POLICY.
    legacy = policy in LEGACY_POLICIES and allow_legacy is True
    if policy != POLICY and not legacy:
        raise ProtectionError("新分析必须使用Agent V70看空保护加持有期高点10%止损")
    if base_kind not in {"confirmed_portfolio", "b0207_cycle_baseline"}:
        raise ProtectionError("必须明确保护起点：实际持仓或B0207周期基线")
    base, before_cash = _plan(base_weights, base_cash, order)
    target, after_cash = _plan(weights, cash, order)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ProtectionError("缺少逐ETF的V70保护复核")
    if [row.get("code") for row in rows] != list(order):
        raise ProtectionError("V70保护复核必须按固定ETF顺序完整列出")
    new_stop_count = 0
    for row in rows:
        code = row["code"]
        review = row.get("v70_protection_review")
        if not isinstance(review, Mapping):
            raise ProtectionError(code + "缺少V70保护复核")
        for field in ("rule_reason", "r007_review", "counterevidence", "invalidation"):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ProtectionError(code + "缺少V70保护证据:" + field)
        status = review.get("interval_status")
        if not isinstance(status, str) or status not in {"new", "already_applied", "not_triggered", "unknown"}:
            raise ProtectionError(code + "保护区间状态非法")
        rule = review.get("bearish_rule")
        if not isinstance(rule, str) or rule not in BEARISH_RULES | {""}:
            raise ProtectionError(code + "R007不是独立看空触发规则；使用R002/R003/R008或空值")
        if (status in {"new", "already_applied"} and not rule) or (status == "not_triggered" and rule):
            raise ProtectionError(code + "看空规则与保护区间状态矛盾")
        stop = ({"triggered": False, "new": False} if legacy else
                _validate_holding_peak_stop(row.get("holding_peak_stop_review"), code))
        if stop["new"]:
            new_stop_count += 1
        if row.get("new_position_eligible") is not False or row.get("trade_action") in {"open", "add"}:
            raise ProtectionError(code + "Agent保护不得新建或普通加仓")
        delta = target[code] - base[code]
        action = row.get("trade_action")
        if delta > 1e-8:
            if legacy or base[code] <= 1e-8 or action != "receive_protection_reinvestment":
                raise ProtectionError(code + "保护回投资金只能进入保护前已经持有的其他ETF")
        elif delta < -1e-8:
            new_v70 = (rule in BEARISH_RULES and status == "new" and
                       row.get("agent_direction") in {"bearish", "看空"})
            if not (new_v70 or stop["new"]):
                raise ProtectionError(code + "减持须由新V70看空区间或新10%持有期高点止损触发")
            if stop["new"] and target[code] > 1e-8:
                raise ProtectionError(code + "持有期高点回撤严格超过10%时保护建议必须清仓100%")
            expected_action = "exit" if target[code] <= 1e-8 else "reduce"
            if action != expected_action:
                raise ProtectionError(code + "保护动作与剩余比例不一致")
        else:
            if stop["new"]:
                raise ProtectionError(code + "持有期高点回撤严格超过10%时必须提出保护减持")
            allowed = {"hold", "hold_with_exit"} if base[code] > 1e-8 else {"avoid_new"}
            if action not in allowed:
                raise ProtectionError(code + "没有减持时必须保持或不新建")
    released = sum(max(0.0, base[c] - target[c]) for c in order)
    reinvested = sum(max(0.0, target[c] - base[c]) for c in order)
    if reinvested - released > 1e-8:
        raise ProtectionError("保护回投比例不得超过保护卖出释放比例")
    eligible = [c for c in order if base[c] > 1e-8 and target[c] >= base[c] - 1e-8]
    if not legacy and released > 1e-8 and eligible and abs(reinvested - released) > 1e-8:
        raise ProtectionError("存在其他原有持仓时，保护卖出款必须全部回投")
    if not legacy and reinvested > 1e-8:
        recipients = [c for c in order if target[c] > base[c] + 1e-8]
        if set(recipients) != set(eligible):
            raise ProtectionError("保护回投必须覆盖所有未被保护卖出的原有持仓")
        denominator = sum(base[c] for c in eligible)
        if denominator <= 0:
            raise ProtectionError("保护回投缺少保护前已经持有的接收ETF")
        for code in recipients:
            expected = reinvested * base[code] / denominator
            if abs((target[code] - base[code]) - expected) > 1e-6:
                raise ProtectionError("保护回投必须按其他原有持仓比例分配")
    if abs(after_cash - before_cash - released + reinvested) > 1e-8:
        raise ProtectionError("保护卖出、原有持仓回投与现金变化不守恒")
    return {"policy": policy, "base_kind": base_kind,
            "cash_increase_pct": after_cash - before_cash,
            "reinvested_pct": reinvested,
            "new_holding_peak_stop_triggers": new_stop_count,
            "automatic_execution": False}
