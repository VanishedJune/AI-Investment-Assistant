# -*- coding: utf-8 -*-
"""最近一年三策略PK：B0207、最新日周Agent代理、创业板智能定投。

本脚本只生成研究影子结果，不修改生产策略、正式月报或真实持仓。
Agent轨使用事先写入本文件的确定性代理协议，单次决策只读取信号日及以前数据，
下一可交易日开盘执行。规则设计吸收了已知历史反例，因此结果不是独立样本外证据。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts import compare_b0207_agent_gem_1y as base  # noqa: E402
from scripts.investment_priority import CONFIG  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import WINDOW_Y3, gem_dca_realistic  # noqa: E402


OUT = ROOT.parent / "champion_vs_dca"
ORDER = list(CONFIG["priority"])
STEP = 5

LATEST_AGENT_PROTOCOL = {
    "id": "agent-weekly-primary-first-second-joint-v2",
    "scope": "research_shadow_only",
    "policy_source": "投资策略.md@2026-08-11",
    "decision_frequency_weeks": 4,
    "fixed_priority_order": ORDER,
    "baseline": "B0207 production allocation",
    "weekly_primary": ["w_dif1", "w_dif2"],
    "daily_secondary": ["d_dif1", "d_dif2", "mom20"],
    "held_position": {
        "joint_deterioration": "retain 75% of B0207 weight",
        "otherwise": "retain 100% of B0207 weight",
    },
    "new_position": {
        "confirmed": "15% when w_dif1>0 and d_dif1>0 and (w_dif2>=0 or d_dif2>=0)",
        "early_repair": "10% when w_dif1<=0, w_dif2>0, d_dif1>0 and mom20>=-2%",
    },
    "joint_deterioration": "w_dif1<0,w_dif2<0,d_dif1<0,d_dif2<0,mom20<-2%",
    "rounding": "floor_to_5_pct",
    "over_100": "remove 5pct from the lowest fixed structural priority first",
    "cash": "only the remaining unallocated balance",
    "future_labels_allowed": False,
    "known_history_used_to_design_rule": True,
}


def floor_step(value: float) -> int:
    return max(0, min(100, int(math.floor((value + 1e-12) / STEP)) * STEP))


def evidence_score(features: dict[str, float]) -> float:
    """仅作审计排序数值，不用于重排固定结构优先级。"""
    return (
        4.0 * (features["w_dif1"] > 0)
        + 2.0 * (features["w_dif2"] >= 0)
        + 1.5 * (features["d_dif1"] > 0)
        + 0.5 * (features["d_dif2"] >= 0)
        + 0.5 * (features["mom20"] >= -0.02)
    )


def latest_agent_overlay(base_target: dict, feature_map: dict[str, dict[str, float]]) -> tuple[dict, list[dict]]:
    weights: dict[str, int] = {}
    audit: list[dict] = []

    for code in ORDER:
        features = feature_map[code]
        baseline_weight = int(base_target["weights"].get(code, 0))
        joint_deterioration = (
            features["w_dif1"] < 0
            and features["w_dif2"] < 0
            and features["d_dif1"] < 0
            and features["d_dif2"] < 0
            and features["mom20"] < -0.02
        )
        confirmed = (
            features["w_dif1"] > 0
            and features["d_dif1"] > 0
            and (features["w_dif2"] >= 0 or features["d_dif2"] >= 0)
        )
        early_repair = (
            features["w_dif1"] <= 0
            and features["w_dif2"] > 0
            and features["d_dif1"] > 0
            and features["mom20"] >= -0.02
        )

        if baseline_weight > 0:
            if joint_deterioration:
                raw_weight = baseline_weight * 0.75
                reason = "B0207持仓但日周一二阶共同恶化，保留75%而非清仓"
                treatment = "reduced"
            else:
                raw_weight = float(baseline_weight)
                reason = "B0207持仓；未满足日周共同恶化，保持机械基线"
                treatment = "retained"
        elif confirmed:
            raw_weight = 15.0
            reason = "B0207为0，但周线主方向与日线确认成立，建立15%观察仓"
            treatment = "added_confirmed"
        elif early_repair:
            raw_weight = 10.0
            reason = "B0207为0；周一仍负但周二转正且日一转正，建立10%早期修复仓"
            treatment = "added_early_repair"
        else:
            raw_weight = 0.0
            reason = "B0207为0，且未满足确认或早期修复条件"
            treatment = "blocked"

        weight = floor_step(raw_weight)
        weights[code] = weight
        audit.append(
            {
                "code": code,
                "name": NAMES[code],
                "baseline_weight_pct": baseline_weight,
                "agent_weight_before_priority_cap_pct": weight,
                "treatment": treatment,
                "reason": reason,
                "features": {
                    key: round(float(features[key]), 12)
                    for key in ("w_dif1", "w_dif2", "d_dif1", "d_dif2", "mom20")
                },
                "evidence_score": evidence_score(features),
            }
        )

    while sum(weights.values()) > 100:
        candidate = next((code for code in reversed(ORDER) if weights[code] >= STEP), None)
        if candidate is None:
            raise RuntimeError("最新Agent代理无法按固定优先级压缩至100%")
        weights[candidate] -= STEP

    cash_pct = 100 - sum(weights.values())
    if cash_pct < 0 or cash_pct % STEP or any(value % STEP for value in weights.values()):
        raise RuntimeError("最新Agent代理违反5%步长或总和约束")

    final_lookup = {item["code"]: item for item in audit}
    for code in ORDER:
        final_lookup[code]["agent_weight_after_priority_cap_pct"] = weights[code]
        if weights[code] < final_lookup[code]["agent_weight_before_priority_cap_pct"]:
            final_lookup[code]["priority_cap_note"] = "总和超过100%，按固定结构优先级从后向前削减"

    return {"weights": weights, "cash_pct": cash_pct}, audit


def validate_decisions(decisions: list[dict]) -> None:
    for item in decisions:
        if item["data_used_through"] != item["signal_date"]:
            raise RuntimeError(f"未来数据边界失败: {item['signal_date']}")
        if item["execution_date"] <= item["signal_date"]:
            raise RuntimeError(f"执行日期未晚于信号日期: {item['signal_date']}")
        target = item["final_target"]
        total = int(target["cash_pct"]) + sum(int(value) for value in target["weights"].values())
        if total != 100 or int(target["cash_pct"]) % STEP:
            raise RuntimeError(f"目标合计错误: {item['signal_date']}={total}")
        if any(int(value) < 0 or int(value) > 100 or int(value) % STEP for value in target["weights"].values()):
            raise RuntimeError(f"目标权重错误: {item['signal_date']}")


def pct(value: float) -> str:
    return f"{float(value):.2%}"


def turnover_display(metrics: dict) -> str:
    value = metrics.get("turnover")
    return "不可用" if value is None else f"{float(value):.4f}"


def main() -> int:
    # simulate()在原模块内查找这两个全局，因此这里显式冻结最新协议与覆盖函数。
    base.AGENT_PROTOCOL = LATEST_AGENT_PROTOCOL
    base.agent_overlay = latest_agent_overlay

    etfs = {code: load_etf(code) for code in CODES}
    set_workspace("159915")
    _, _, _, _, usable, _ = load_aligned_data("159915")
    dates = [item.date().isoformat() for item in usable["signal"]]
    execs = [None if item is None else str(np.datetime64(item).astype("datetime64[D]")) for item in usable["exec"]]
    start, end = range_idx(dates, *WINDOW_Y3)

    mechanical, mechanical_decisions, mechanical_weekly = base.simulate(etfs, dates, execs, start, end, agent=False)
    agent, agent_decisions, agent_weekly = base.simulate(etfs, dates, execs, start, end, agent=True)
    dca = gem_dca_realistic(etfs, dates, execs, start, end)
    dca["decision_count"] = len(mechanical_weekly)
    dca["weeks"] = len(mechanical_weekly)
    dca["turnover"] = None
    dca["turnover_status"] = "UNAVAILABLE_LEGACY_DCA_HELPER"

    validate_decisions(mechanical_decisions)
    validate_decisions(agent_decisions)
    if not (mechanical["invested"] == agent["invested"] == dca["invested"]):
        raise RuntimeError("三组资金流不一致")

    rows = [
        ("B0207机械基线", mechanical),
        ("B0207+最新策略Agent代理", agent),
        ("创业板ETF智能定投", dca),
    ]
    winner = max(rows, key=lambda item: item[1]["final_value"])[0]
    protocol_hash = base.canonical_hash(LATEST_AGENT_PROTOCOL)
    manifest = json.loads((ROOT.parent / "data_manifest.json").read_text(encoding="utf-8"))
    gap = agent["final_value"] - mechanical["final_value"]
    drawdown_gap = agent["max_drawdown"] - mechanical["max_drawdown"]
    sharpe_gap = agent["sharpe"] - mechanical["sharpe"]
    exposure_gap = agent["avg_position"] - mechanical["avg_position"]

    payload = {
        "schema_version": "b0207-agent-latest-gem-pk-v1",
        "status": "POST_HOC_RULE_REVISION_NON_PIT",
        "research_only": True,
        "window": {
            "requested": [WINDOW_Y3[0], WINDOW_Y3[1]],
            "signal_start": dates[start],
            "signal_end": dates[end],
            "weeks": mechanical["weeks"],
        },
        "account": {
            "initial_capital": 0,
            "weekly_contribution_per_available_etf": base.WEEKLY,
            "lot": base.LOT,
            "commission_rate": base.trade_cost.__globals__["COMMISSION_RATE"],
            "minimum_commission": base.trade_cost.__globals__["MIN_COMMISSION"],
            "cash_apr": 0.0175,
        },
        "data_run_id": manifest.get("run_id"),
        "data_as_of": manifest.get("as_of"),
        "agent_protocol": LATEST_AGENT_PROTOCOL,
        "agent_protocol_sha256": protocol_hash,
        "no_future_label_contract": {
            "decision_features_end_at_signal_date": True,
            "execution_is_next_tradeable_open": True,
            "future_returns_passed_to_agent": False,
            "rule_design_used_known_history": True,
        },
        "results": {label: metrics for label, metrics in rows},
        "comparison": {
            "agent_minus_mechanical_final_value": gap,
            "agent_minus_mechanical_max_drawdown": drawdown_gap,
            "agent_minus_mechanical_sharpe": sharpe_gap,
            "agent_minus_mechanical_average_exposure": exposure_gap,
        },
        "winner_by_final_value": winner,
        "mechanical_decisions": mechanical_decisions,
        "agent_decisions": agent_decisions,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "b0207_agent_latest_gem_pk_1y"
    json_path = OUT / f"{stem}.json"
    csv_path = OUT / f"{stem}.csv"
    decision_csv = OUT / f"{stem}_decisions.csv"
    md_path = OUT / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["策略", "期末资产", "累计投入", "累计盈利", "累计收益", "TWR", "IRR", "Sharpe", "最大回撤", "平均持仓", "换手", "决策次数"])
        for label, metrics in rows:
            writer.writerow([
                label,
                round(metrics["final_value"], 2),
                round(metrics["invested"], 2),
                round(metrics["profit"], 2),
                pct(metrics["cum_return"]),
                pct(metrics["twr"]),
                pct(metrics["annual_return_irr"]),
                f"{metrics['sharpe']:.3f}",
                pct(metrics["max_drawdown"]),
                pct(metrics["avg_position"]),
                turnover_display(metrics),
                metrics["decision_count"],
            ])

    with decision_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["信号日", "执行日", "策略", *[f"{code}_{NAMES[code]}" for code in ORDER], "现金"])
        for label, decisions in (("B0207机械基线", mechanical_decisions), ("B0207+最新策略Agent代理", agent_decisions)):
            for item in decisions:
                target = item["final_target"]
                writer.writerow([item["signal_date"], item["execution_date"], label, *[target["weights"][code] for code in ORDER], target["cash_pct"]])

    lines = [
        "# 最近一年三策略PK：B0207 vs 最新策略Agent代理 vs 创业板智能定投",
        "",
        "> 状态：POST_HOC_RULE_REVISION_NON_PIT｜研究影子对比，不修改生产策略、正式月报或真实持仓。",
        "",
        f"- 信号区间：{dates[start]}至{dates[end]}，共{mechanical['weeks']}个有效周度持有区间、{len(agent_decisions)}个四周决策点。",
        f"- 三组累计投入均为{mechanical['invested']:,.2f}元；统一使用复利、100份整手、佣金、现金利息和下一可交易日开盘成交。",
        f"- 最新Agent代理协议SHA-256：`{protocol_hash}`；逐期目标函数没有接收未来收益。",
        "- 重要限制：规则设计已经吸收2026年6月末等已知反例，且当前冠军历史PIT证据不完整，因此本结果不能宣称独立样本外。",
        "",
        "## PK结果",
        "",
        "| 策略 | 期末资产 | 累计投入 | 累计盈利 | 累计收益 | TWR | IRR | Sharpe | 最大回撤 | 平均持仓 | 换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in rows:
        lines.append(
            f"| {label} | {metrics['final_value']:,.2f} | {metrics['invested']:,.2f} | {metrics['profit']:,.2f} | "
            f"{pct(metrics['cum_return'])} | {pct(metrics['twr'])} | {pct(metrics['annual_return_irr'])} | "
            f"{metrics['sharpe']:.3f} | {pct(metrics['max_drawdown'])} | {pct(metrics['avg_position'])} | {turnover_display(metrics)} |"
        )
    lines += [
        "",
        f"按期末资产，本次领先者为：**{winner}**。",
        "",
        "## 最新Agent代理相对B0207",
        "",
        f"- 期末资产差额：{gap:,.2f}元。",
        f"- 最大回撤差额：{drawdown_gap:.2%}；Sharpe差额：{sharpe_gap:.3f}。",
        f"- 平均持仓差额：{exposure_gap:.2%}。",
        "",
        "## 最新策略代理规则",
        "",
        "1. B0207为起点，每四周决策一次，固定ETF结构优先级不重排。",
        "2. 周K一阶和二阶联合为主，日K一阶、二阶和20日动量为辅。",
        "3. B0207持仓只有日周一二阶共同恶化且20日动量低于-2%时降至原权重75%，其余保持。",
        "4. B0207为0但周一、日一为正且周二或日二非负时建立15%确认仓。",
        "5. 周一仍负、周二转正、日一转正且20日动量不低于-2%时建立10%早期修复仓。",
        "6. 向下取整至5%；总和超过100%时从固定结构优先级最低者开始每次削减5%，余额才留现金。",
        "",
        "## 解释边界",
        "",
        "- 这不是伪称Agent过去真实保存过13份判断，而是把当前策略转为确定性代理后逐期核算。",
        "- 每次计算只读取当时可见的日周参数，下一交易日开盘执行；但规则本身由已知历史反例启发，存在设计期过拟合风险。",
        "- 创业板智能定投旧函数没有输出可比成交额换手，换手显示为不可用，不能解释为没有交易。",
        "- 只有从下一决策日起冻结真实Agent建议并向前累计，才能形成可归因的样本外证据。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"md": str(md_path), "json": str(json_path), "csv": str(csv_path), "decisions": str(decision_csv), "winner": winner}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
