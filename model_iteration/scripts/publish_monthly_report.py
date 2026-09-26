# -*- coding: utf-8 -*-
"""校验Agent建议、用户确认并原子发布四周决策月报。

发布器不计算信号或仓位。它验证B0207机械基线未被篡改、Agent完成了以周K
为主日K为辅的独立分析、用户明确确认了最终目标，然后才把冻结结果写入锁定
HTML模板。任何一层仍为pending或来源不一致均失败关闭。
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

from scripts.agent_v70_protection import validate_plan, ProtectionError
from scripts.agent_multi_parameter_gate import (
    GATE_VERSION,
    AgentGateError,
    validate_agent_decision,
    validate_quantitative_evidence,
)
from scripts.agent_forward_evidence import ForwardEvidenceError, load_forward_evidence
from scripts.monthly_details import DetailsError, latest_context, render_details, with_agent_column
from scripts.investment_priority import CONFIG
from scripts.monthly_policy import (
    MonthlyPolicyError,
    PROJECT_ROOT,
    ProjectLock,
    SHANGHAI,
    canonical_json_bytes,
    file_meta,
    iteration_model_reference,
    parse_iso_date,
    pretty_json_bytes,
    resolve_trading_cycle,
)

REPORT_DIR = PROJECT_ROOT / "app" / "reports" / "monthly"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ICON_PATH = PROJECT_ROOT / "app" / "assets" / "ai-research-owl-avatar-v1.ico"
REQUIRED_STRATEGY_FIELDS = {
    "code",
    "name",
    "baseline_priority",
    "champion_id",
    "mechanical_champion_position",
    "mechanical_bullish",
    "mechanical_baseline_pct",
    "current_weight_pct",
    "raw_close",
    "quantitative_evidence",
    "forward_evidence",
    "forward_analysis",
    "evidence_review",
    "entry_setup",
    "countertrend_thesis",
    "position_size_reason",
    "weekly_evidence",
    "daily_evidence",
    "timeframe_relationship",
    "volume_confirmation",
    "relative_advantage",
    "evidence_for",
    "evidence_against",
    "fixed_priority",
    "agent_direction",
    "structural_trend",
    "momentum_state",
    "trade_action",
    "new_position_eligible",
    "staged_entry",
    "holding_inertia_note",
    "multi_parameter_conclusion",
    "agent_recommendation",
    "agent_recommended_weight_pct",
    "position_target_pct",
    "entry_trigger",
    "execution_time",
    "cancel_or_exit",
    "risk_note",
}


class PublishError(MonthlyPolicyError):
    """Fail-closed monthly publication error."""


def _project_path(relative: str, *, must_exist: bool = True) -> Path:
    candidate = (PROJECT_ROOT / Path(relative)).resolve()
    root = PROJECT_ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise PublishError(f"候选路径越界: {relative}")
    if must_exist and not candidate.is_file():
        raise PublishError(f"候选文件不存在: {relative}")
    return candidate


def _forbidden_provenance_keys(value: object, found: set[str] | None = None) -> set[str]:
    result = set() if found is None else found
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"decision_maker", "agent_overrides"}:
                result.add(key)
            _forbidden_provenance_keys(child, result)
    elif isinstance(value, list):
        for child in value:
            _forbidden_provenance_keys(child, result)
    return result


def _parse_generated_at(report: dict) -> tuple[datetime, str]:
    raw = str(report.get("generated_at") or "")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise PublishError("generated_at必须是带时区的ISO-8601时间") from exc
    if parsed.tzinfo is None:
        raise PublishError("generated_at必须包含时区")
    shanghai = parsed.astimezone(SHANGHAI)
    generated_date = shanghai.date().isoformat()
    if report.get("generated_date") != generated_date:
        raise PublishError("generated_date必须等于generated_at对应的上海日期")
    return shanghai, generated_date


def _parse_confirmation_time(value: object) -> datetime:
    raw = str(value or "")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise PublishError("user_decision.confirmed_at必须是带时区的ISO-8601时间") from exc
    if parsed.tzinfo is None:
        raise PublishError("user_decision.confirmed_at必须包含时区")
    return parsed.astimezone(SHANGHAI)


def _validate_weight_plan(weights: object, cash: object, label: str) -> tuple[dict[str, float], float]:
    if not isinstance(weights, dict) or set(weights) != set(CONFIG["priority"]):
        raise PublishError(f"{label}必须完整包含8只ETF且不得增加未知标的")

    def pct(value: object, field: str) -> float:
        if isinstance(value, bool):
            raise PublishError(f"{field}不是合法百分比")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PublishError(f"{field}不是合法百分比") from exc
        if not math.isfinite(result) or result < 0 or result > 100:
            raise PublishError(f"{field}必须位于0%至100%")
        return result

    normalized = {code: pct(weights[code], f"{label}.{code}") for code in CONFIG["priority"]}
    cash_pct = pct(cash, f"{label}.cash")
    if abs(sum(normalized.values()) + cash_pct - 100.0) > 1e-8:
        raise PublishError(f"{label}的8只ETF与现金必须合计100%")
    return normalized, cash_pct


def _plans_equal(left: Mapping[str, float], left_cash: float, right: Mapping[str, float], right_cash: float) -> bool:
    return all(abs(float(left[code]) - float(right[code])) <= 1e-8 for code in CONFIG["priority"]) and abs(
        float(left_cash) - float(right_cash)
    ) <= 1e-8


def validate_candidate(candidate: dict) -> tuple[dict, dict, Path, str]:
    if candidate.get("schema_version") != "monthly-decision-candidate-v2":
        raise PublishError("候选schema_version必须为monthly-decision-candidate-v2")
    if candidate.get("status") != "user_confirmed_ready_for_publish":
        raise PublishError("候选尚未完成Agent建议和用户最终确认，禁止修改正式HTML")
    snapshot_path = _project_path(str(candidate.get("policy_snapshot_path") or ""))
    snapshot_meta = file_meta(snapshot_path)
    if candidate.get("policy_snapshot_size_bytes") != snapshot_meta["size_bytes"]:
        raise PublishError("候选引用的策略快照大小不一致")
    if candidate.get("policy_snapshot_modified_utc") != snapshot_meta["modified_utc"]:
        raise PublishError("候选引用的策略快照修改时间不一致")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if (
        snapshot.get("schema_version") != "monthly-policy-baseline-v2"
        or snapshot.get("policy_id") != "B0207"
        or snapshot.get("policy_role") != "mechanical_baseline_only"
    ):
        raise PublishError("策略快照身份非法")

    feature_path = _project_path(str(snapshot.get("feature_snapshot_path") or ""))
    feature_meta = file_meta(feature_path)
    if feature_meta["size_bytes"] != snapshot.get("feature_snapshot_size_bytes"):
        raise PublishError("B0207基线绑定的日周参数快照大小不一致")
    if feature_meta["modified_utc"] != snapshot.get("feature_snapshot_modified_utc"):
        raise PublishError("B0207基线绑定的日周参数快照修改时间不一致")
    feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_instruments = feature_payload.get("instruments") or {}
    if set(feature_instruments) != set(CONFIG["priority"]):
        raise PublishError("绑定的日周参数快照未完整覆盖8只ETF")

    template_info = candidate.get("template") or {}
    template_path = _project_path(str(template_info.get("path") or ""))
    template_meta = file_meta(template_path)
    if template_info.get("size_bytes") != template_meta["size_bytes"]:
        raise PublishError("锁定HTML模板大小不一致")
    if template_info.get("modified_utc") != template_meta["modified_utc"]:
        raise PublishError("锁定HTML模板修改时间不一致")

    report = candidate.get("report")
    if not isinstance(report, dict) or report.get("schema_version") != "monthly-report-v2":
        raise PublishError("候选缺少monthly-report-v2报告对象")
    forbidden = _forbidden_provenance_keys(report)
    if forbidden:
        raise PublishError(f"正式月报含废弃溯源字段: {sorted(forbidden)}")
    provenance = {
        "decision_authority": "user_explicit_confirmation",
        "policy_id": "B0207",
        "policy_role": "mechanical_baseline",
        "signal_source": "current_champion_models",
        "allocation_engine": "investment_priority_v3",
        "recommendation_maker": "Agent",
        "agent_role": "v70_bearish_plus_holding_peak_stop_only",
        "timeframe_policy": "weekly_primary_daily_confirmation",
    }
    for key, expected in provenance.items():
        if report.get(key) != expected:
            raise PublishError(f"报告溯源字段{key}必须为{expected}")
    if report.get("report_status") not in {"user_confirmed", "user_modified"}:
        raise PublishError("正式月报必须记录用户已确认或已修改最终目标")
    if report.get("policy_snapshot_size_bytes") != snapshot_meta["size_bytes"]:
        raise PublishError("报告未绑定同一策略快照")
    if report.get("policy_snapshot_modified_utc") != snapshot_meta["modified_utc"]:
        raise PublishError("报告未绑定同一策略快照")
    if report.get("feature_snapshot_path") != snapshot.get("feature_snapshot_path"):
        raise PublishError("报告引用的日周参数快照路径与B0207基线不一致")
    if report.get("feature_snapshot_size_bytes") != feature_meta["size_bytes"]:
        raise PublishError("报告引用的日周参数快照大小不一致")
    if report.get("feature_snapshot_modified_utc") != feature_meta["modified_utc"]:
        raise PublishError("报告引用的日周参数快照修改时间不一致")
    for report_key, snapshot_key in (
        ("data_run_id", "data_run_id"),
        ("data_as_of", "data_as_of"),
        ("decision_date", "decision_date"),
        ("execution_date", "execution_date"),
        ("next_decision_date", "next_decision_date"),
    ):
        if report.get(report_key) != snapshot.get(snapshot_key):
            raise PublishError(f"报告{report_key}与策略快照不一致")
    generated_at, generated_date = _parse_generated_at(report)
    try:
        latest_context(PROJECT_ROOT, report)
    except (DetailsError, ValueError, KeyError) as exc:
        raise PublishError(f"8只ETF详情来源检查失败: {exc}") from exc

    current_state = json.loads((PROJECT_ROOT / "portfolio_state.json").read_text(encoding="utf-8"))
    current_portfolio = current_state.get("last_confirmed_portfolio") or {}
    portfolio_meta = file_meta(PROJECT_ROOT / "portfolio_state.json")
    state_metadata_changed = (
        snapshot.get("last_confirmed_portfolio_revision")
        != str(current_state.get("decision_revision") or "")
        or snapshot.get("last_confirmed_portfolio_size_bytes") != portfolio_meta["size_bytes"]
        or snapshot.get("last_confirmed_portfolio_modified_utc") != portfolio_meta["modified_utc"]
    )
    if state_metadata_changed and report.get("last_confirmed_portfolio") != current_portfolio:
        raise PublishError("策略快照生成后真实持仓事实已变化，必须重新prepare")
    if report.get("last_confirmed_portfolio") != current_portfolio:
        raise PublishError("报告真实持仓与portfolio_state.json不一致")
    configured_next = (current_state.get("policy_cycle") or {}).get("next_decision_date")
    snapshot_nominal = snapshot.get("nominal_decision_date") or snapshot.get("decision_date")
    if configured_next and configured_next != snapshot_nominal:
        current_report = current_state.get("current_monthly_report") or {}
        if (current_report.get("decision_date") != snapshot_nominal
                or current_report.get("report_date") != report.get("generated_date")):
            raise PublishError("策略快照不是portfolio_state.json当前待处理或已发布的同一决策周期")
    if report.get("execution_protection") != snapshot.get("price_protection"):
        raise PublishError("报告价格保护必须逐项引用客观基线快照")

    baseline_weights, baseline_cash = _validate_weight_plan(
        snapshot.get("mechanical_baseline_weights_pct"),
        snapshot.get("mechanical_baseline_cash_pct"),
        "B0207机械基线",
    )

    analysis = report.get("agent_analysis") or {}
    if analysis.get("status") != "recommendation_ready":
        raise PublishError("Agent分析尚未形成可提交用户的建议")
    if analysis.get("multi_parameter_gate_version") != GATE_VERSION:
        raise PublishError(f"Agent分析必须使用{GATE_VERSION}联合判断门禁")
    for field in (
        "weekly_primary_confirmed",
        "weekly_first_second_joint_reviewed",
        "daily_confirmation_reviewed",
        "all_quantitative_fields_reviewed",
        "trend_action_separated",
        "future_return_and_entry_reviewed",
        "holding_inertia_reviewed",
        "all_8_compared",
        "current_holding_and_no_action_compared",
    ):
        if analysis.get(field) is not True:
            raise PublishError(f"Agent分析未完成强制检查: {field}")
    priority_slots = snapshot.get("priority_slot_analysis") or {}
    if priority_slots:
        if analysis.get("first_priority_slot_reviewed") is not True:
            raise PublishError("Agent必须分别复核159915与021528自身数据后才能提交第一优先级槽位")
        if analysis.get("priority_slot_analysis") != priority_slots:
            raise PublishError("第一优先级槽位资料与B0207客观快照不一致")
    try:
        forward_evidence = load_forward_evidence(
            PROJECT_ROOT, data_as_of=report["data_as_of"], data_run_id=report["data_run_id"],
            instruments={code: feature_instruments[code] for code in CONFIG["priority"]},
        )
    except ForwardEvidenceError as exc:
        raise PublishError(str(exc)) from exc
    iteration_ref = analysis.get("iteration_model_reference")
    if not isinstance(iteration_ref, dict):
        raise PublishError("Agent分析必须包含iteration_model_reference（ETF做题记录迭代模型）")
    if iteration_ref.get("read") is not True:
        raise PublishError("Agent必须确认已读取ETF做题记录迭代模型三份权威文件")
    if not str(iteration_ref.get("how_applied") or "").strip():
        raise PublishError("Agent必须说明ETF做题记录迭代模型如何应用于本次建议")
    current_iteration_ref = iteration_model_reference()
    for key in (
        "rule_size_bytes",
        "rule_modified_utc",
        "state_size_bytes",
        "state_modified_utc",
        "memory_size_bytes",
        "memory_modified_utc",
        "rule_version",
        "policy_version",
        "state_version",
        "memory_version",
    ):
        if str(iteration_ref.get(key) or "") != str(current_iteration_ref.get(key) or ""):
            raise PublishError(
                f"iteration_model_reference.{key}与ETF做题记录当前迭代模型不一致"
            )
    fixed_priority_order = analysis.get("fixed_priority_order")
    if fixed_priority_order != list(CONFIG["priority"]):
        raise PublishError("fixed_priority_order必须逐项遵守B0207生产配置，不得由Agent重排")
    agent_weights, agent_cash = _validate_weight_plan(
        analysis.get("recommended_weights_pct"), analysis.get("recommended_cash_pct"), "Agent建议"
    )
    # This entry publishes a nominal-cycle candidate. Agent may protect the
    # B0207 allocation, not substitute an independently constructed portfolio.
    if analysis.get("protection_base") != "b0207_cycle_baseline":
        raise PublishError("名义决策日候选的Agent保护起点必须为B0207周期基线")
    try:
        validate_plan(
            policy=analysis.get("protection_policy"), base_kind=analysis.get("protection_base"),
            base_weights=baseline_weights, base_cash=baseline_cash,
            weights=agent_weights, cash=agent_cash,
            rows=report.get("decision_snapshot", {}).get("strategies"), order=CONFIG["priority"],
        )
    except ProtectionError as exc:
        raise PublishError(str(exc)) from exc
    for field in ("portfolio_thesis", "no_action_comparison"):
        if not str(analysis.get(field) or "").strip():
            raise PublishError(f"Agent分析缺少{field}")
    scenarios = analysis.get("scenario_analysis")
    if not isinstance(scenarios, dict) or any(
        not str(scenarios.get(key) or "").strip() for key in ("down", "sideways", "up")
    ):
        raise PublishError("Agent必须完成下跌、震荡、上涨三路径推演")

    expected_deviations = {
        code for code in CONFIG["priority"] if abs(agent_weights[code] - baseline_weights[code]) > 1e-8
    }
    if abs(agent_cash - baseline_cash) > 1e-8:
        expected_deviations.add("CASH")
    deviations = analysis.get("baseline_deviations")
    if not isinstance(deviations, list):
        raise PublishError("baseline_deviations必须为数组")
    actual_deviations: set[str] = set()
    for entry in deviations:
        if not isinstance(entry, dict):
            raise PublishError("baseline_deviations条目格式错误")
        code = str(entry.get("code") or "")
        if code not in {*CONFIG["priority"], "CASH"} or code in actual_deviations:
            raise PublishError("baseline_deviations代码非法或重复")
        expected_baseline = baseline_cash if code == "CASH" else baseline_weights[code]
        expected_agent = agent_cash if code == "CASH" else agent_weights[code]
        try:
            declared_baseline = float(entry.get("baseline_pct"))
            declared_agent = float(entry.get("recommended_pct"))
        except (TypeError, ValueError) as exc:
            raise PublishError(f"{code}偏离记录的百分比非法") from exc
        if not math.isfinite(declared_baseline) or not math.isfinite(declared_agent):
            raise PublishError(f"{code}偏离记录的百分比非法")
        if abs(declared_baseline - expected_baseline) > 1e-8:
            raise PublishError(f"{code}偏离记录的B0207基线不正确")
        if abs(declared_agent - expected_agent) > 1e-8:
            raise PublishError(f"{code}偏离记录的Agent建议不正确")
        if not str(entry.get("reason") or "").strip():
            raise PublishError(f"{code}偏离B0207基线但没有说明原因")
        actual_deviations.add(code)
    if actual_deviations != expected_deviations:
        raise PublishError("Agent建议与B0207基线的全部差异必须逐项披露")

    comparison = report.get("decision_comparison") or {}
    if comparison.get("status") != "ready_for_user_feedback":
        raise PublishError("基线与Agent建议对比表尚未完成，不能提交用户或发布")
    expected_order = [*CONFIG["priority"], "CASH"]
    if comparison.get("display_order") != expected_order or comparison.get("user_feedback_required") is not True:
        raise PublishError("对比表必须按8只ETF逐只列出并在末行单列现金")
    comparison_rows = comparison.get("rows")
    if not isinstance(comparison_rows, list) or [row.get("code") for row in comparison_rows] != expected_order:
        raise PublishError("对比表不得合并ETF、遗漏ETF或改变固定展示顺序")
    deviation_reasons = {str(entry["code"]): str(entry["reason"]) for entry in deviations}
    comparison_by_code: dict[str, dict] = {}
    for row in comparison_rows:
        if not isinstance(row, dict):
            raise PublishError("对比表条目格式错误")
        code = str(row["code"])
        expected_name = "现金" if code == "CASH" else str(feature_instruments[code].get("name") or code)
        if code == "159915" and priority_slots.get("159915"):
            expected_name = str(priority_slots["159915"].get("display_name") or expected_name)
        if row.get("name") != expected_name:
            raise PublishError(f"{code}对比表名称与参数快照不一致")
        expected_baseline = baseline_cash if code == "CASH" else baseline_weights[code]
        expected_recommended = agent_cash if code == "CASH" else agent_weights[code]
        try:
            shown_baseline = float(row.get("mechanical_baseline_pct"))
            shown_recommended = float(row.get("agent_recommended_pct"))
            shown_difference = float(row.get("difference_pct"))
        except (TypeError, ValueError) as exc:
            raise PublishError(f"{code}对比表仓位不是合法数字") from exc
        if any(not math.isfinite(value) for value in (shown_baseline, shown_recommended, shown_difference)):
            raise PublishError(f"{code}对比表仓位含NaN或无穷值")
        if abs(shown_baseline - expected_baseline) > 1e-8:
            raise PublishError(f"{code}对比表机械基线不正确")
        if abs(shown_recommended - expected_recommended) > 1e-8:
            raise PublishError(f"{code}对比表Agent建议不正确")
        if abs(shown_difference - (expected_recommended - expected_baseline)) > 1e-8:
            raise PublishError(f"{code}对比表差值不正确")
        reason = str(row.get("agent_reason") or "").strip()
        if not reason:
            raise PublishError(f"{code}对比表缺少Agent原因")
        if code in deviation_reasons and reason != deviation_reasons[code]:
            raise PublishError(f"{code}对比表原因与baseline_deviations不一致")
        comparison_by_code[code] = row

    user = report.get("user_decision") or {}
    user_status = user.get("status")
    if user_status not in {"confirmed_agent_recommendation", "modified_by_user"}:
        raise PublishError("用户尚未明确确认或修改最终目标")
    if user.get("confirmation_source") != "explicit_user_instruction":
        raise PublishError("最终目标必须来自用户明确指令")
    confirmed_at = _parse_confirmation_time(user.get("confirmed_at"))
    if confirmed_at > generated_at:
        raise PublishError("报告生成时间不得早于用户确认时间")
    if not str(user.get("confirmation_note") or "").strip():
        raise PublishError("用户最终决定必须保留确认记录")
    final_weights, final_cash = _validate_weight_plan(
        user.get("final_target_weights_pct"), user.get("final_cash_pct"), "用户最终目标"
    )
    same_as_agent = _plans_equal(agent_weights, agent_cash, final_weights, final_cash)
    if user_status == "confirmed_agent_recommendation" and not same_as_agent:
        raise PublishError("用户选择确认Agent建议时，最终目标必须与Agent建议一致")
    if user_status == "modified_by_user" and same_as_agent:
        raise PublishError("用户未改变Agent建议时不得标记为modified_by_user")
    expected_report_status = "user_confirmed" if user_status == "confirmed_agent_recommendation" else "user_modified"
    if report.get("report_status") != expected_report_status:
        raise PublishError("report_status与用户决定类型不一致")

    decision_snapshot = report.get("decision_snapshot") or {}
    plan = decision_snapshot.get("portfolio_plan") or {}
    if plan.get("mechanical_baseline_weights_pct") != snapshot.get("mechanical_baseline_weights_pct"):
        raise PublishError("报告中的B0207机械基线被改写")
    if plan.get("mechanical_baseline_cash_pct") != snapshot.get("mechanical_baseline_cash_pct"):
        raise PublishError("报告中的B0207机械现金基线被改写")
    plan_agent, plan_agent_cash = _validate_weight_plan(
        plan.get("agent_recommended_weights_pct"), plan.get("agent_recommended_cash_pct"), "报告Agent建议"
    )
    if not _plans_equal(plan_agent, plan_agent_cash, agent_weights, agent_cash):
        raise PublishError("portfolio_plan中的Agent建议与agent_analysis不一致")
    plan_final, plan_final_cash = _validate_weight_plan(
        plan.get("target_weights_pct"), plan.get("cash_pct"), "报告用户目标"
    )
    if not _plans_equal(plan_final, plan_final_cash, final_weights, final_cash):
        raise PublishError("portfolio_plan中的最终目标与用户确认不一致")
    strategies = decision_snapshot.get("strategies")
    if not isinstance(strategies, list) or len(strategies) != 8:
        raise PublishError("报告必须完整包含8只ETF策略")
    if [item.get("code") for item in strategies] != list(CONFIG["priority"]):
        raise PublishError("报告ETF顺序必须与B0207生产配置一致")
    current_weights = current_portfolio.get("weights_pct") or {}
    priority_by_code = {code: index for index, code in enumerate(fixed_priority_order, 1)}
    for expected_priority, item in enumerate(strategies, 1):
        missing = REQUIRED_STRATEGY_FIELDS - set(item)
        if missing:
            raise PublishError(f"{item.get('code')}策略缺少字段: {sorted(missing)}")
        code = item["code"]
        expected_name = str(feature_instruments[code].get("name") or code)
        if code == "159915" and priority_slots.get("159915"):
            expected_name = str(priority_slots["159915"].get("display_name") or expected_name)
        if item["name"] != expected_name or item["name"] != comparison_by_code[code]["name"]:
            raise PublishError(f"{code}名称与日周参数快照或对比表不一致")
        if code == "159915" and priority_slots.get("159915"):
            slot = priority_slots["159915"]
            if item.get("alternative_fund_analysis") != slot.get("fund_analysis"):
                raise PublishError("021528独立净值分析与第一优先级槽位快照不一致")
            if item.get("priority_slot", {}).get("members") != ["159915", "021528"]:
                raise PublishError("第一优先级槽位必须合并159915与021528且不得重复计权")
        if item["baseline_priority"] != expected_priority:
            raise PublishError(f"{code}机械基线顺序与B0207不一致")
        champion = snapshot["champions"][code]
        champion_contract = {
            "champion_id": champion["champion_id"],
            "mechanical_champion_position": champion["position"],
            "mechanical_bullish": champion["bullish"],
        }
        for key, expected in champion_contract.items():
            if item[key] != expected:
                raise PublishError(f"{code}的{key}与冠军策略快照不一致")
        if item["mechanical_baseline_pct"] != snapshot["mechanical_baseline_weights_pct"][code]:
            raise PublishError(f"{code}机械基线仓位与快照不一致")
        if abs(float(item["current_weight_pct"]) - float(current_weights.get(code, 0))) > 1e-8:
            raise PublishError(f"{code}当前真实仓位与portfolio_state.json不一致")
        reference_close = snapshot["price_protection"][code]["reference_raw_close"]
        if abs(float(item["raw_close"]) - float(reference_close)) > 1e-8:
            raise PublishError(f"{code}参考收盘价与客观快照不一致")
        if item["fixed_priority"] != priority_by_code[code]:
            raise PublishError(f"{code}固定结构优先级与生产配置不一致")
        if abs(float(item["agent_recommended_weight_pct"]) - agent_weights[code]) > 1e-8:
            raise PublishError(f"{code}逐项Agent建议仓位与组合建议不一致")
        if abs(float(item["position_target_pct"]) - final_weights[code]) > 1e-8:
            raise PublishError(f"{code}逐项最终目标与用户确认不一致")
        try:
            quantitative_evidence = validate_quantitative_evidence(
                item["quantitative_evidence"], feature_instruments[code], code
            )
            if item["forward_evidence"] != forward_evidence[code]:
                raise AgentGateError(f"{code}.forward_evidence与截止日前日周连续观察不一致")
            validate_agent_decision(
                code=code,
                evidence=quantitative_evidence,
                decision=item,
                current_weight_pct=float(baseline_weights[code]),
                recommended_weight_pct=agent_weights[code],
                expected_dates=(report["data_as_of"], report["execution_date"], report["next_decision_date"]),
                protection_only=True,
            )
        except AgentGateError as exc:
            raise PublishError(str(exc)) from exc
        for field in (
            "weekly_evidence",
            "daily_evidence",
            "timeframe_relationship",
            "volume_confirmation",
            "relative_advantage",
            "evidence_for",
            "evidence_against",
            "agent_recommendation",
            "entry_trigger",
            "execution_time",
            "cancel_or_exit",
            "risk_note",
        ):
            if not str(item.get(field) or "").strip():
                raise PublishError(f"{code}缺少Agent填写的{field}")
        weekly_evidence = str(item["weekly_evidence"])
        if "一阶" not in weekly_evidence or "二阶" not in weekly_evidence:
            raise PublishError(f"{code}周K证据必须同时包含DIF一阶和二阶联合判断")
        first_value = re.search(r"一阶(?:导数|导|变化|斜率)?\s*(?:为|=|：|:)?\s*[+\-−]?\d+(?:\.\d+)?", weekly_evidence)
        second_value = re.search(r"二阶(?:导数|导|变化|斜率)?\s*(?:为|=|：|:)?\s*[+\-−]?\d+(?:\.\d+)?", weekly_evidence)
        if first_value is None or second_value is None:
            raise PublishError(f"{code}周K证据必须明确写出DIF一阶和二阶数值")
    return snapshot, report, template_path, generated_date


def _style_block(source: str) -> str:
    match = re.search(r"<style>[\s\S]*?</style>", source, flags=re.IGNORECASE)
    if not match:
        raise PublishError("锁定模板缺少style块")
    return match.group(0)


def _script_block(source: str) -> str:
    matches = re.findall(r"<script[\s\S]*?</script>", source, flags=re.IGNORECASE)
    if len(matches) != 1:
        raise PublishError("锁定模板必须恰好包含一个达标标记脚本")
    return matches[0]


def _tag_sequence(source: str) -> list[tuple[str, str]]:
    return [(match.group(1), match.group(2).lower()) for match in re.finditer(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>", source)]


def _replace_once(source: str, pattern: str, replacement: str, label: str, *, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=flags)
    if count != 1:
        raise PublishError(f"锁定模板无法唯一更新{label}")
    return updated


def _standard_monthly_labels(source: str, report: dict, generated_date: str) -> str:
    result = source
    result = _replace_once(result, r"<title>[^<]*</title>", f"<title>投资决策月报｜{generated_date}</title>", "页面标题")
    result = _replace_once(result, r"<h1>[^<]*</h1>", "<h1>8只ETF月度交易决策</h1>", "主标题")
    result = _replace_once(
        result,
        r'<div class="subtitle">[\s\S]*?</div>',
        '<div class="subtitle">B0207四周完整调仓 · V70看空减持70%＋持有期高点回撤超10%清仓 · 用户确认执行</div>',
        "副标题",
    )
    stamp = (
        f'<div class="stamp"><b>生成 {report["generated_at"]} · 数据截止 {report["data_as_of"]}</b>'
        f'决策日 {report["decision_date"]} · 执行日 {report["execution_date"]} 09:45<br>'
        f'下次决策 {report["next_decision_date"]} · '
        f'policy B0207 · run {report["data_run_id"]}</div>'
    )
    result = _replace_once(result, r'<div class="stamp">[\s\S]*?</div>', stamp, "日期戳")
    result = _replace_once(
        result,
        r'(<span class="section-no">01</span><h2>)[\s\S]*?(</h2>)',
        r"\g<1>本决策周期8只ETF联合决策摘要\g<2>",
        "第一部分标题",
    )
    result = re.sub(
        r"<th>(?:Agent决策|B0207目标|政策状态|用户确认目标)</th>",
        "<th>用户确认目标</th>",
        result,
        count=1,
    )
    result = re.sub(
        r"<th>(?:本周唯一触发|本周期执行条件|本周期执行触发|本决策周期执行触发)</th>",
        "<th>本决策周期执行触发</th>",
        result,
        count=1,
    )
    result = result.replace("Agent判断：", "Agent保护判断：")
    result = result.replace("Agent验收：", "Agent保护判断：")
    result = result.replace("本周", "本决策周期")
    result = _replace_once(
        result,
        r"<footer>[\s\S]*?</footer>",
        '<footer><span>B0207：四周完整调仓｜V70看空参考减持70%｜持有期高点回撤超10%建议清仓100%｜最终执行：用户确认</span><span>仅供本地研究，不构成投资建议</span></footer>',
        "页脚",
    )
    decision_key = report["decision_date"].replace("-", "")
    result = _replace_once(
        result,
        r"var KEY='(?:weekly|monthly)_mark_\d{8}';",
        f"var KEY='monthly_mark_{decision_key}';",
        "达标标记存储键",
    )
    return result


def render_html(template: str, candidate: dict, report: dict, generated_date: str) -> str:
    result = _standard_monthly_labels(template, report, generated_date)
    replacements = candidate.get("html_replacements")
    if not isinstance(replacements, list):
        raise PublishError("html_replacements必须为数组")
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict) or not isinstance(replacement.get("from"), str) or not isinstance(replacement.get("to"), str):
            raise PublishError(f"html_replacements[{index}]格式错误")
        old = replacement["from"]
        new = replacement["to"]
        expected_count = int(replacement.get("count", 1))
        actual_count = result.count(old)
        if actual_count != expected_count:
            raise PublishError(
                f"html_replacements[{index}]源片段数量错误 expected={expected_count} actual={actual_count}"
            )
        result = result.replace(old, new, expected_count)

    # Always rebuild the detail matrix after textual replacements. Stale HTML
    # cannot survive a partial replacement or masquerade as current analysis.
    try:
        result = render_details(result, PROJECT_ROOT, report, candidate.get("policy_snapshot_path"))
    except (DetailsError, ValueError, KeyError) as exc:
        raise PublishError(f"8只ETF详情刷新失败: {exc}") from exc

    # Compare against the shared, explicitly authorized presentation migration;
    # arbitrary candidate CSS/structure changes remain blocked.
    presentation_template = render_details(template, PROJECT_ROOT, report, candidate.get("policy_snapshot_path"))
    if _style_block(result) != _style_block(presentation_template):
        raise PublishError("CSS黄金块发生变化，发布已阻断")
    if _tag_sequence(result) != _tag_sequence(presentation_template):
        raise PublishError("HTML标签序列发生变化，排版锁被触发")
    before_script = re.sub(r"(?:weekly|monthly)_mark_\d{8}", "REPORT_MARK_KEY", _script_block(template))
    after_script = re.sub(r"(?:weekly|monthly)_mark_\d{8}", "REPORT_MARK_KEY", _script_block(result))
    if before_script != after_script:
        raise PublishError("达标标记脚本除存储键外发生变化")
    verify_html(result, report, candidate)
    return result


def verify_html(source: str, report: dict, candidate: dict) -> None:
    counts = {
        "section": len(re.findall(r"<section(?:\s|>)", source, flags=re.IGNORECASE)),
        "main": len(re.findall(r"<main(?:\s|>)", source, flags=re.IGNORECASE)),
        "svg": len(re.findall(r"<svg(?:\s|>)", source, flags=re.IGNORECASE)),
        "mark": source.count('class="mark"'),
        "yes": source.count('class="yes"'),
        "no": source.count('class="no"'),
        "alloc": source.count('class="alloc"'),
        "script": len(re.findall(r"<script(?:\s|>)", source, flags=re.IGNORECASE)),
    }
    # Current reports render 159915 and 021528 as separate analysis rows
    # sharing one allocation slot and one rank cell. Frozen pre-fund reports
    # retain eight review controls.
    review_controls = 9 if 'data-code="021528"' in source else 8
    portfolio = report.get("last_confirmed_portfolio") or {}
    weights = portfolio.get("weights_pct") or {}
    members = portfolio.get("growth_slot_member_weights_pct")
    if isinstance(members, dict):
        held_count = sum(1 for value in members.values() if isinstance(value, (int, float)) and value > 0)
        held_count += sum(1 for code, value in weights.items()
                          if code != "159915" and isinstance(value, (int, float)) and value > 0)
    else:
        held_count = sum(1 for value in weights.values() if isinstance(value, (int, float)) and value > 0)
    expected = {"section": 4, "main": 1, "svg": 1, "mark": review_controls,
                "yes": review_controls, "no": review_controls, "script": 1}
    structural_counts = {key: counts[key] for key in expected}
    if structural_counts != expected or counts["alloc"] < 1:
        raise PublishError(f"月报固定结构不一致: {counts}")
    lower = source.lower()
    if "/api/" in lower or "weekly_mark_" in source or "周报" in source or "本周" in source:
        raise PublishError("月报HTML含动态API或旧周报运行标记")
    if "月K" in source or "月线" in source:
        raise PublishError("月报HTML不得恢复月K或月线分析")
    legacy_schedule_tokens = ("2026-08-17", "8月17日")
    stale = [token for token in legacy_schedule_tokens if token in source]
    if stale:
        raise PublishError(f"月报HTML残留旧决策周期: {stale}")
    for value in (
        report["generated_at"],
        report["generated_date"],
        report["data_as_of"],
        report["decision_date"],
        report["execution_date"],
        report["next_decision_date"],
    ):
        if value not in source:
            raise PublishError(f"月报HTML缺少日期字段: {value}")
    if f"monthly_mark_{report['decision_date'].replace('-', '')}" not in source:
        raise PublishError("月报达标标记键不正确")
    for item in report["decision_snapshot"]["strategies"]:
        if item["code"] not in source:
            raise PublishError(f"月报HTML缺少ETF代码: {item['code']}")
        for field in (
            "weekly_evidence",
            "daily_evidence",
            "timeframe_relationship",
            "volume_confirmation",
            "relative_advantage",
            "evidence_for",
            "evidence_against",
            "agent_recommendation",
            "entry_trigger",
            "execution_time",
            "cancel_or_exit",
            "risk_note",
        ):
            value = str(item[field])
            escaped = html_lib.escape(value, quote=False)
            if value not in source and escaped not in source:
                raise PublishError(f"月报HTML未展示{item['code']}的{field}")
    analysis = report.get("agent_analysis") or {}
    for note in analysis.get("risk_notes") or []:
        value = str(note)
        if value not in source and html_lib.escape(value, quote=False) not in source:
            raise PublishError("月报HTML未展示Agent全局风险说明")
    for field in ("portfolio_thesis", "no_action_comparison"):
        value = str(analysis[field])
        if value not in source and html_lib.escape(value, quote=False) not in source:
            raise PublishError(f"月报HTML未展示Agent的{field}")
    for entry in analysis.get("baseline_deviations") or []:
        value = str(entry["reason"])
        if value not in source and html_lib.escape(value, quote=False) not in source:
            raise PublishError("月报HTML未披露Agent偏离B0207机械基线的原因")
    confirmation_note = str((report.get("user_decision") or {})["confirmation_note"])
    if confirmation_note not in source and html_lib.escape(confirmation_note, quote=False) not in source:
        raise PublishError("月报HTML未展示用户最终确认记录")
    for assertion in candidate.get("html_assertions") or []:
        if not isinstance(assertion, str) or assertion not in source:
            raise PublishError(f"月报HTML断言失败: {assertion!r}")


def _powershell() -> str:
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"


def enable_windows_acl_inheritance(path: Path) -> None:
    """Make atomically published files readable through the project ACL.

    Files created inside the private staging directory keep that directory's
    protected ACL when ``os.replace`` moves them into the report tree.  Enable
    inheritance after every move so the interactive Windows user can open the
    HTML/shortcut and the application can read the paired JSON/state files.
    """
    if os.name != "nt":
        return
    completed = subprocess.run(
        ["icacls.exe", str(path), "/inheritance:e"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PublishError(f"无法恢复月报文件的项目访问权限: {path} {detail}")


def create_shortcut(link_path: Path, target_path: Path) -> None:
    if not ICON_PATH.is_file():
        raise PublishError("猫头鹰ICO缺失")
    launcher = Path(os.environ.get("WINDIR", r"C:\Windows")) / "explorer.exe"
    if not launcher.is_file():
        raise PublishError("Windows 资源管理器缺失，无法创建可打开的月报快捷入口")
    env = os.environ.copy()
    def encoded(value: Path) -> str:
        return base64.b64encode(str(value).encode("utf-16le")).decode("ascii")
    env.update({
        "MONTHLY_LINK_B64": encoded(link_path),
        "MONTHLY_TARGET_B64": encoded(target_path),
        "MONTHLY_LAUNCHER_B64": encoded(launcher),
        "MONTHLY_ICON_B64": encoded(ICON_PATH),
        "MONTHLY_WORKDIR_B64": encoded(SCRIPTS_DIR),
    })
    command = (
        "$d={param($n)[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String([Environment]::GetEnvironmentVariable($n)))};"
        "$link=&$d 'MONTHLY_LINK_B64';$target=&$d 'MONTHLY_TARGET_B64';"
        "$launcher=&$d 'MONTHLY_LAUNCHER_B64';$icon=&$d 'MONTHLY_ICON_B64';$workdir=&$d 'MONTHLY_WORKDIR_B64';"
        "$w=New-Object -ComObject WScript.Shell;"
        "$s=$w.CreateShortcut($link);"
        "$s.TargetPath=$launcher;"
        "$s.Arguments='\"'+$target+'\"';"
        "$s.WorkingDirectory=$workdir;"
        "$s.IconLocation=$icon+',0';"
        "$s.Save()"
    )
    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-Command", command],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not link_path.is_file():
        raise PublishError(f"创建快捷入口失败: {completed.stderr.strip()}")


def shortcut_properties(link_path: Path) -> dict:
    env = os.environ.copy()
    env["MONTHLY_LINK_B64"] = base64.b64encode(str(link_path).encode("utf-16le")).decode("ascii")
    command = (
        "$link=[Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($env:MONTHLY_LINK_B64));"
        "$w=New-Object -ComObject WScript.Shell;"
        "$s=$w.CreateShortcut($link);"
        "[pscustomobject]@{TargetPath=$s.TargetPath;Arguments=$s.Arguments;IconLocation=$s.IconLocation;WorkingDirectory=$s.WorkingDirectory}|"
        "ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-Command", command],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise PublishError(f"读取快捷入口失败: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise PublishError("快捷入口属性输出无法解析") from exc


def verify_shortcut(link_path: Path, target_path: Path) -> None:
    props = shortcut_properties(link_path)
    launcher = Path(os.environ.get("WINDIR", r"C:\Windows")) / "explorer.exe"
    if Path(props.get("TargetPath", "")).resolve() != launcher.resolve():
        raise PublishError("快捷入口未通过Windows打开月报")
    arguments = str(props.get("Arguments") or "").strip().strip('"')
    if Path(arguments).resolve() != target_path.resolve():
        raise PublishError("快捷入口月报路径不正确")
    icon_value = str(props.get("IconLocation") or "").split(",", 1)[0]
    if Path(icon_value).resolve() != ICON_PATH.resolve():
        raise PublishError("快捷入口未使用固定猫头鹰ICO")


def _legacy_archive_moves(first_monthly: bool, new_link: Path, generated_date: str) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    if first_monthly:
        weekly_sources: list[Path] = []
        weekly_sources.extend(sorted(SCRIPTS_DIR.glob("投资决策周报*.html")))
        weekly_sources.extend(sorted(PROJECT_ROOT.glob("投资决策周报*.lnk")))
        weekly_report_dir = PROJECT_ROOT / "app" / "reports" / "weekly"
        weekly_sources.extend(sorted(weekly_report_dir.glob("????-??-??.json")) if weekly_report_dir.is_dir() else [])
        latest_weekly = weekly_report_dir / "latest.json"
        if latest_weekly.is_file():
            weekly_sources.append(latest_weekly)

        archive_date = "0000-00-00"
        for source in weekly_sources:
            match = re.search(r"(\d{4}-\d{2}-\d{2})", source.name)
            if match:
                archive_date = max(archive_date, match.group(1))
        if archive_date == "0000-00-00":
            archive_date = generated_date
        archive_root = PROJECT_ROOT / "backup" / "weekly_legacy" / archive_date
        for source in weekly_sources:
            if source.exists():
                moves.append((source, archive_root / source.relative_to(PROJECT_ROOT)))

    shortcut_archive = PROJECT_ROOT / "backup" / "monthly_shortcuts" / generated_date
    for source in PROJECT_ROOT.glob("投资决策月报*.lnk"):
        if source.resolve() != new_link.resolve():
            moves.append((source, shortcut_archive / source.name))
    return moves


def _advanced_portfolio_state(snapshot: dict, report: dict, generated_date: str) -> bytes:
    """推进报告周期状态；已确认目标必须已经登记为真实持仓。"""

    path = PROJECT_ROOT / "portfolio_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    confirmed_before = canonical_json_bytes(state.get("last_confirmed_portfolio") or {})
    next_nominal = parse_iso_date(str(snapshot["next_decision_date"]))
    next_execution, next_required = resolve_trading_cycle(next_nominal)
    cadence = int(CONFIG["decision_cycle_days"])
    cycle = state.setdefault("policy_cycle", {})
    cycle.update(
        {
            "policy_id": "B0207",
            "cadence_days": cadence,
            "next_decision_date": next_nominal.isoformat(),
            "effective_execution_date": next_execution.isoformat(),
            "execution_time": snapshot["execution_time"],
            "required_data_as_of": next_required.isoformat(),
            "subsequent_nominal_dates": [
                (next_nominal + timedelta(days=cadence)).isoformat(),
                (next_nominal + timedelta(days=2 * cadence)).isoformat(),
            ],
            "schedule_status": "provisional_until_local_trading_calendar_validation",
        }
    )
    state["last_updated"] = generated_date
    state["data_as_of"] = snapshot["data_as_of"]
    state["data_run_id"] = snapshot["data_run_id"]
    state["decision_revision"] = f"{snapshot['decision_date']}-agent-user-confirmed-monthly-release"
    plan = report["decision_snapshot"]["portfolio_plan"]
    actual = state.get("last_confirmed_portfolio") or {}
    actual_weights = actual.get("weights_pct") or {}
    actual_cash = actual.get("cash_pct")
    if not _plans_equal(
        {code: float(actual_weights.get(code, 0.0)) for code in CONFIG["priority"]},
        float(actual_cash),
        plan["target_weights_pct"],
        plan["cash_pct"],
    ):
        raise PublishError("下周已确认持仓尚未登记为真实持仓，禁止发布为待成交目标")
    state.pop("active_policy_target", None)
    state["active_mechanical_baseline"] = {
        "policy_id": "B0207",
        "role": "mechanical_reference_not_final_target",
        "decision_date": snapshot["decision_date"],
        "data_as_of": snapshot["data_as_of"],
        "policy_snapshot_size_bytes": report["policy_snapshot_size_bytes"],
        "policy_snapshot_modified_utc": report["policy_snapshot_modified_utc"],
        "weights_pct": snapshot["mechanical_baseline_weights_pct"],
        "cash_pct": snapshot["mechanical_baseline_cash_pct"],
    }
    state["active_user_confirmed_target"] = {
        "decision_date": snapshot["decision_date"],
        "data_as_of": snapshot["data_as_of"],
        "agent_recommended_weights_pct": plan["agent_recommended_weights_pct"],
        "agent_recommended_cash_pct": plan["agent_recommended_cash_pct"],
        "target_weights_pct": plan["target_weights_pct"],
        "cash_pct": plan["cash_pct"],
        "user_confirmation": report["user_decision"],
        "status": "published_and_recorded_as_actual",
        "actual_portfolio_event_id": actual.get("source_event_id"),
    }
    state["user_final_target"] = report.get("user_final_target")
    state["user_decision"] = report.get("user_decision")
    preferences = state.setdefault("report_preferences", {})
    preferences["current_formal_template"] = "desktop_monthly_v1_fixed_layout"
    current = state.setdefault("current_monthly_report", {})
    current.update(
        {
            "status": "formal_report_published_actual_confirmed",
            "monitoring_html_status": "all_current_fields_bound_to_latest_completed_common_session",
            "formal_publication_status": "published",
            "report_date": generated_date,
            "html": f"scripts/投资决策月报{generated_date}.html",
            "json": "app/reports/monthly/latest.json",
            "shortcut": f"投资决策月报{generated_date}.lnk",
            "data_as_of": snapshot["data_as_of"],
            "decision_date": snapshot["decision_date"],
            "execution_date": snapshot["execution_date"],
            "next_decision_date": snapshot["next_decision_date"],
            "required_data_as_of": next_required.isoformat(),
            "execution_rule": "正式月报已发布；用户确认的下周持仓已按项目规则登记为真实持仓，成交明细可后补。",
        }
    )
    execution = state.setdefault("execution_status", {})
    execution["status"] = "user_target_confirmed_and_recorded_as_actual"
    execution.pop("pending_target", None)
    execution["completed_target"] = {
        "weights_pct": plan["target_weights_pct"],
        "cash_pct": plan["cash_pct"],
        "actual_portfolio_event_id": actual.get("source_event_id"),
        "fill_details_status": "price_shares_fees_pending",
    }
    execution.setdefault("unconfirmed_fills", [])
    legacy = state.setdefault("legacy_weekly_snapshot", {})
    legacy.update(
        {
            "status": "archived_read_only",
            "html": "backup/weekly_legacy/2026-08-10/scripts/投资决策周报2026-08-10.html",
            "json": "backup/weekly_legacy/2026-08-10/app/reports/weekly/2026-08-10.json",
        }
    )
    if canonical_json_bytes(state.get("last_confirmed_portfolio") or {}) != confirmed_before:
        raise PublishError("发布状态推进不得修改真实持仓")
    return pretty_json_bytes(state)


def _rollback(outputs: list[tuple[Path, Path | None]], moved: list[tuple[Path, Path]]) -> None:
    for source, destination in reversed(moved):
        if destination.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, source)
    for target, backup in reversed(outputs):
        if target.exists():
            target.unlink()
        if backup and backup.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(backup, target)
            enable_windows_acl_inheritance(target)


def publish(candidate_path: Path, *, check_only: bool = False) -> dict:
    with ProjectLock(PROJECT_ROOT / ".monthly-pipeline.lock"):
        return _publish_locked(candidate_path, check_only=check_only)


def _publish_locked(candidate_path: Path, *, check_only: bool = False) -> dict:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    snapshot, report, template_path, generated_date = validate_candidate(candidate)
    template = template_path.read_text(encoding="utf-8")
    rendered = render_html(template, candidate, report, generated_date)
    report_payload = pretty_json_bytes(report)

    stage_root = Path(tempfile.mkdtemp(prefix=".monthly_publish_", dir=str(PROJECT_ROOT)))
    try:
        stage_json = stage_root / f"{generated_date}.json"
        stage_html = stage_root / f"投资决策月报{generated_date}.html"
        stage_link = stage_root / f"投资决策月报{generated_date}.lnk"
        stage_state = stage_root / "portfolio_state.json"
        stage_json.write_bytes(report_payload)
        stage_html.write_text(rendered, encoding="utf-8", newline="")
        stage_state.write_bytes(_advanced_portfolio_state(snapshot, report, generated_date))
        final_html = SCRIPTS_DIR / stage_html.name
        create_shortcut(stage_link, final_html)
        verify_shortcut(stage_link, final_html)
        if stage_json.read_bytes() != report_payload:
            raise PublishError("暂存JSON内容校验失败")
        verify_html(stage_html.read_text(encoding="utf-8"), report, candidate)
        if check_only:
            return {
                "status": "CHECK_PASSED",
                "generated_date": generated_date,
                "policy_snapshot_size_bytes": candidate["policy_snapshot_size_bytes"],
                "policy_snapshot_modified_utc": candidate["policy_snapshot_modified_utc"],
                "template_size_bytes": candidate["template"]["size_bytes"],
                "template_modified_utc": candidate["template"]["modified_utc"],
            }

        dated_json = REPORT_DIR / f"{generated_date}.json"
        latest_json = REPORT_DIR / "latest.json"
        final_link = PROJECT_ROOT / stage_link.name
        legacy_weekly_active = any(
            path.exists()
            for path in (
                PROJECT_ROOT / "app" / "reports" / "weekly" / "latest.json",
                PROJECT_ROOT / "app" / "reports" / "weekly" / "2026-08-10.json",
                SCRIPTS_DIR / "投资决策周报2026-08-10.html",
                PROJECT_ROOT / "投资决策周报2026-08-10.lnk",
            )
        )
        moves = _legacy_archive_moves(legacy_weekly_active, final_link, generated_date)
        failure_stage = os.environ.get("MONTHLY_PUBLISH_FAIL_STAGE", "")
        output_specs = [
            (stage_json, dated_json, "json"),
            (stage_html, final_html, "html"),
            (stage_link, final_link, "shortcut"),
        ]
        rollback_outputs: list[tuple[Path, Path | None]] = []
        completed_moves: list[tuple[Path, Path]] = []
        try:
            for staged, target, stage_name in output_specs:
                target.parent.mkdir(parents=True, exist_ok=True)
                backup = None
                if target.exists():
                    backup = stage_root / "rollback" / target.relative_to(PROJECT_ROOT)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, backup)
                rollback_outputs.append((target, backup))
                os.replace(staged, target)
                enable_windows_acl_inheritance(target)
                if failure_stage == stage_name:
                    raise OSError(f"injected failure after {stage_name}")

            # latest.json是同一报告字节，放在归档前；任一后续失败均回滚。
            latest_staged = stage_root / "latest.json"
            latest_staged.write_bytes(report_payload)
            latest_backup = None
            if latest_json.exists():
                latest_backup = stage_root / "rollback" / latest_json.relative_to(PROJECT_ROOT)
                latest_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(latest_json, latest_backup)
            rollback_outputs.append((latest_json, latest_backup))
            os.replace(latest_staged, latest_json)
            enable_windows_acl_inheritance(latest_json)
            if failure_stage == "latest":
                raise OSError("injected failure after latest")

            state_path = PROJECT_ROOT / "portfolio_state.json"
            state_backup = stage_root / "rollback" / "portfolio_state.json"
            state_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(state_path, state_backup)
            rollback_outputs.append((state_path, state_backup))
            os.replace(stage_state, state_path)
            enable_windows_acl_inheritance(state_path)
            if failure_stage == "state":
                raise OSError("injected failure after state")

            for source, destination in moves:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if source.is_file() and source.read_bytes() == destination.read_bytes():
                        # The legacy item is already archived byte-for-byte.
                        # Keep both copies rather than deleting or overwriting.
                        continue
                    raise PublishError(f"归档目标已存在且内容不同: {destination}")
                os.replace(source, destination)
                completed_moves.append((source, destination))
            if failure_stage == "archive":
                raise OSError("injected failure after archive")

            if latest_json.read_bytes() != dated_json.read_bytes():
                raise PublishError("发布后latest.json与日期JSON不一致")
            verify_shortcut(final_link, final_html)
            if failure_stage == "final_verify":
                raise OSError("injected failure after final verification")
        except Exception:
            _rollback(rollback_outputs, completed_moves)
            raise
        return {
            "status": "PUBLISHED",
            "json": dated_json.relative_to(PROJECT_ROOT).as_posix(),
            "html": final_html.relative_to(PROJECT_ROOT).as_posix(),
            "shortcut": final_link.relative_to(PROJECT_ROOT).as_posix(),
            "policy_snapshot_size_bytes": candidate["policy_snapshot_size_bytes"],
            "policy_snapshot_modified_utc": candidate["policy_snapshot_modified_utc"],
            "report_size_bytes": dated_json.stat().st_size,
            "report_modified_utc": file_meta(dated_json)["modified_utc"],
            "html_size_bytes": final_html.stat().st_size,
            "html_modified_utc": file_meta(final_html)["modified_utc"],
            "decision_date": snapshot["decision_date"],
        }
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="发布用户明确确认后的四周决策月报")
    parser.add_argument("--candidate", required=True, help="Agent已分析且用户已确认最终目标的候选JSON")
    parser.add_argument("--check", action="store_true", help="只完成全部暂存校验，不发布")
    args = parser.parse_args()
    try:
        candidate_path = Path(args.candidate)
        if not candidate_path.is_absolute():
            candidate_path = (Path.cwd() / candidate_path).resolve()
        result = publish(candidate_path, check_only=args.check)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (PublishError, MonthlyPolicyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
