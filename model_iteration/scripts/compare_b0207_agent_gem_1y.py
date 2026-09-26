# -*- coding: utf-8 -*-
"""最近一年：B0207机械基线、B0207+Agent覆盖、创业板智能定投PK。

这是研究回放，不改写生产策略、真实持仓或正式月报。Agent轨采用事先冻结的
确定性覆盖协议；每个信号只读取当周已完成的日/周特征，下一交易日开盘执行。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts.champion_chain_vs_dca import _apply_ca  # noqa: E402
from scripts.investment_priority import CONFIG, allocate  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CASH_FACTOR,
    LOT,
    WEEKLY,
    WINDOW_Y3,
    _metrics,
    gem_dca_realistic,
    trade_cost,
)

OUT = ROOT.parent / "champion_vs_dca"
ORDER = list(CONFIG["priority"])
STEP = 5

AGENT_PROTOCOL = {
    "id": "agent-weekly-primary-daily-secondary-v1",
    "scope": "research_shadow_only",
    "decision_frequency_weeks": 4,
    "baseline": "B0207 production allocation",
    "weekly_primary": ["w_dif1", "w_dif2"],
    "daily_secondary": ["d_dif1", "d_dif2", "mom20"],
    "held_adjustments": {
        "weekly_down_accelerating_and_daily_down": 0.50,
        "weekly_down_other": 0.75,
        "weekly_up_daily_down": 1.00,
        "weekly_up_daily_up_but_not_accelerating": 1.00,
        "weekly_up_accelerating_and_daily_up": 1.15,
    },
    "new_position": {
        "weight_pct": 5,
        "requires": "w_dif1>0 and w_dif2>=0 and d_dif1>0 and mom20>0",
    },
    "rounding": "floor_to_5_pct",
    "cash": "unallocated reductions remain cash",
    "over_100": "remove 5pct repeatedly from weakest evidence; later priority loses first on ties",
    "future_labels_allowed": False,
}


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def strength(features: dict[str, float]) -> float:
    return (
        2.0 * (features["w_dif1"] > 0)
        + 1.0 * (features["w_dif2"] >= 0)
        + 0.5 * (features["d_dif1"] > 0)
        + 0.25 * (features["d_dif2"] >= 0)
        + 0.1 * (features["mom20"] > 0)
    )


def floor_step(value: float) -> int:
    return max(0, min(100, int(math.floor((value + 1e-12) / STEP)) * STEP))


def agent_overlay(base: dict, feature_map: dict[str, dict[str, float]]) -> tuple[dict, list[dict]]:
    weights: dict[str, int] = {}
    audit: list[dict] = []
    for code in ORDER:
        f = feature_map[code]
        base_weight = int(base["weights"].get(code, 0))
        weekly_up = f["w_dif1"] > 0
        weekly_accel = f["w_dif2"] >= 0
        daily_up = f["d_dif1"] > 0
        if base_weight > 0:
            if not weekly_up and not weekly_accel and not daily_up:
                multiplier, reason = 0.50, "周线向下且继续恶化，日线未确认"
            elif not weekly_up:
                multiplier, reason = 0.75, "周线仍向下，日线只作缓冲"
            elif not daily_up:
                multiplier, reason = 1.00, "周线向上，日线回落不先减仓"
            elif weekly_accel:
                multiplier, reason = 1.15, "周线向上且加速，日线确认"
            else:
                multiplier, reason = 1.00, "周线向上但未加速"
            raw = base_weight * multiplier
        else:
            confirmed = weekly_up and weekly_accel and daily_up and f["mom20"] > 0
            multiplier = None
            raw = float(AGENT_PROTOCOL["new_position"]["weight_pct"] if confirmed else 0)
            reason = "周线主趋势和日线共同确认，建立5%观察仓" if confirmed else "不满足新增仓位确认"
        weight = floor_step(raw)
        weights[code] = weight
        audit.append(
            {
                "code": code,
                "name": NAMES[code],
                "baseline_weight_pct": base_weight,
                "agent_weight_before_cap_pct": weight,
                "multiplier": multiplier,
                "reason": reason,
                "features": {key: round(float(f[key]), 12) for key in ("w_dif1", "w_dif2", "d_dif1", "d_dif2", "mom20")},
                "evidence_strength": strength(f),
            }
        )

    while sum(weights.values()) > 100:
        candidates = [code for code in ORDER if weights[code] >= STEP]
        if not candidates:
            raise RuntimeError("Agent覆盖权重无法压缩到100%")
        weakest = min(candidates, key=lambda code: (strength(feature_map[code]), -ORDER.index(code)))
        weights[weakest] -= STEP
    cash = 100 - sum(weights.values())
    if cash < 0 or cash % STEP or any(value % STEP for value in weights.values()):
        raise RuntimeError("Agent覆盖权重违反5%步长或总和约束")
    return {"weights": weights, "cash_pct": cash}, audit


def rebalance(shares: dict[str, float], cash: float, prices: dict[str, float], target: dict) -> tuple[dict[str, float], float, float]:
    value = sum(shares[c] * prices[c] for c in shares) + cash
    desired: dict[str, float] = {}
    for code in shares:
        price = prices[code]
        weight = float(target["weights"].get(code, 0)) / 100.0
        desired[code] = int(weight * value / price // LOT) * LOT if price > 0 else shares[code]

    traded = 0.0
    for code in ORDER:
        delta = desired[code] - shares[code]
        if delta >= 0 or prices[code] <= 0:
            continue
        amount = -delta * prices[code]
        fee = trade_cost(amount)
        cash += amount - fee
        traded += amount
        shares[code] = desired[code]

    for code in ORDER:
        delta = desired[code] - shares[code]
        if delta <= 0 or prices[code] <= 0:
            continue
        lots = int(delta // LOT) * LOT
        while lots > 0:
            amount = lots * prices[code]
            fee = trade_cost(amount)
            if amount + fee <= cash + 1e-9:
                break
            lots -= LOT
        if lots <= 0:
            continue
        amount = lots * prices[code]
        fee = trade_cost(amount)
        cash -= amount + fee
        traded += amount
        shares[code] += lots
    if cash < -1e-6:
        raise RuntimeError(f"账户出现负现金: {cash}")
    return shares, max(0.0, cash), traded


def weekly_deploy(shares: dict[str, float], cash: float, prices: dict[str, float], inflow: float) -> tuple[dict[str, float], float, float]:
    holdings = sum(shares[c] * prices[c] for c in shares if prices[c] > 0)
    if holdings <= 0:
        return shares, cash, 0.0
    traded = 0.0
    for code in ORDER:
        if shares[code] <= 0 or prices[code] <= 0:
            continue
        allocation = inflow * (shares[code] * prices[code] / holdings)
        lots = int(allocation / prices[code] // LOT) * LOT
        if lots <= 0:
            continue
        amount = lots * prices[code]
        fee = trade_cost(amount)
        if amount + fee > cash:
            continue
        cash -= amount + fee
        shares[code] += lots
        traded += amount
    return shares, cash, traded


def simulate(etfs: dict, dates: list[str], execs: list[str | None], start: int, end: int, *, agent: bool) -> tuple[dict, list[dict], list[dict]]:
    cash = 0.0
    shares = {code: 0.0 for code in ORDER}
    first_exec = next((ex for ex in execs[start : end + 1] if ex), None)
    applied = {
        code: {
            key: bool(first_exec and str(event.get("pay_date") or event.get("ex_date")) < first_exec)
            for key, event in etfs[code]["ca_events"]
        }
        for code in ORDER
    }
    twr: list[float] = []
    flows: list[tuple[int, float]] = []
    exposures: list[float] = []
    navs: list[float] = []
    decisions: list[dict] = []
    weekly_rows: list[dict] = []
    invested = 0.0
    traded_total = 0.0
    prev_after = None

    for i in range(start, end + 1):
        signal_date = dates[i]
        execution_date = execs[i]
        if execution_date is None:
            continue
        cash *= CASH_FACTOR
        n_available = sum(1 for code in ORDER if etfs[code]["start_date"] <= signal_date)
        if n_available == 0:
            continue
        prices = {code: float(etfs[code]["price_map"].get(execution_date, 0.0)) for code in ORDER}
        for code in ORDER:
            if prices[code] > 0 or shares[code] != 0:
                shares[code], cash = _apply_ca(shares[code], cash, etfs[code]["ca_events"], execution_date, applied[code])
        nav_before = sum(shares[code] * prices[code] for code in ORDER) + cash
        if prev_after is not None:
            twr.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_available
        cash += inflow
        invested += inflow
        flows.append((i - start, -inflow))

        decision = (i - start) % int(AGENT_PROTOCOL["decision_frequency_weeks"]) == 0
        if decision:
            positions = {code: max(0.0, min(1.0, float(etfs[code]["by_date"].get(signal_date, 0.0)))) for code in ORDER}
            base = allocate({code: positions[code] > 0 for code in ORDER}, positions)
            feature_map = {code: etfs[code]["feat_by_date"][signal_date] for code in ORDER}
            if agent:
                target, audit = agent_overlay(base, feature_map)
            else:
                target, audit = {"weights": dict(base["weights"]), "cash_pct": int(base["cash_pct"])}, []
            shares, cash, traded = rebalance(shares, cash, prices, target)
            traded_total += traded
            decisions.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "data_used_through": signal_date,
                    "baseline_target": {"weights": dict(base["weights"]), "cash_pct": int(base["cash_pct"])},
                    "final_target": target,
                    "agent_audit": audit,
                }
            )
        else:
            shares, cash, traded = weekly_deploy(shares, cash, prices, inflow)
            traded_total += traded

        nav_after = sum(shares[code] * prices[code] for code in ORDER) + cash
        exposure = sum(shares[code] * prices[code] for code in ORDER) / nav_after if nav_after > 0 else 0.0
        exposures.append(exposure)
        navs.append(nav_after)
        prev_after = nav_after
        weekly_rows.append(
            {
                "signal_date": signal_date,
                "execution_date": execution_date,
                "nav": nav_after,
                "cash": cash,
                "exposure": exposure,
            }
        )
    metrics = _metrics(twr, flows, prev_after or 0.0, invested, float(np.mean(exposures)) if exposures else 0.0,
                       traded_total / sum(navs) if navs else 0.0)
    metrics["decision_count"] = len(decisions)
    metrics["weeks"] = len(weekly_rows)
    return metrics, decisions, weekly_rows


def pct(value: float) -> str:
    return f"{float(value):.2%}"


def main() -> int:
    etfs = {code: load_etf(code) for code in CODES}
    set_workspace("159915")
    _, _, _, _, usable, _ = load_aligned_data("159915")
    dates = [item.date().isoformat() for item in usable["signal"]]
    execs = [None if item is None else str(np.datetime64(item).astype("datetime64[D]")) for item in usable["exec"]]
    start, end = range_idx(dates, *WINDOW_Y3)

    baseline, baseline_decisions, baseline_weekly = simulate(etfs, dates, execs, start, end, agent=False)
    agent, agent_decisions, agent_weekly = simulate(etfs, dates, execs, start, end, agent=True)
    gem = gem_dca_realistic(etfs, dates, execs, start, end)
    gem["decision_count"] = len(baseline_weekly)
    gem["weeks"] = len(baseline_weekly)
    # The legacy DCA helper does not expose comparable traded-notional data.
    # Its placeholder 0 must not be presented as evidence of zero turnover.
    gem["turnover"] = None
    gem["turnover_status"] = "UNAVAILABLE_LEGACY_DCA_HELPER"

    rows = [
        ("B0207机械基线", baseline),
        ("B0207+Agent周主日辅覆盖", agent),
        ("创业板ETF智能定投", gem),
    ]
    winner = max(rows, key=lambda item: item[1]["final_value"])[0]
    agent_gap = agent["final_value"] - baseline["final_value"]
    drawdown_improvement = agent["max_drawdown"] - baseline["max_drawdown"]
    sharpe_gap = agent["sharpe"] - baseline["sharpe"]
    exposure_gap = agent["avg_position"] - baseline["avg_position"]
    cash_examples = sorted(
        agent_decisions,
        key=lambda item: (-item["final_target"]["cash_pct"], item["signal_date"]),
    )[:4]
    protocol_hash = canonical_hash(AGENT_PROTOCOL)
    manifest_path = ROOT.parent / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    payload = {
        "schema_version": "b0207-agent-gem-pk-1y-v1",
        "status": "NON_PIT_REPLAY",
        "research_only": True,
        "window": {"requested": list(WINDOW_Y3), "signal_start": dates[start], "signal_end": dates[end], "weeks": baseline["weeks"]},
        "account": {
            "initial_capital": 0,
            "weekly_contribution_per_available_etf": WEEKLY,
            "lot": LOT,
            "commission_rate": 0.00025,
            "minimum_commission": 5,
            "cash_apr": 0.0175,
        },
        "data_run_id": manifest.get("run_id"),
        "data_as_of": manifest.get("as_of"),
        "agent_protocol": AGENT_PROTOCOL,
        "agent_protocol_sha256": protocol_hash,
        "no_future_label_contract": {
            "decision_features_end_at_signal_date": True,
            "execution_is_next_available_raw_open": True,
            "future_returns_not_passed_to_target_function": True,
            "limitation": "当前冠军历史PIT可得性未被一级证据完全证明；因此不得解释为严格样本外业绩。",
        },
        "results": {label: metrics for label, metrics in rows},
        "winner_by_final_value": winner,
        "baseline_decisions": baseline_decisions,
        "agent_decisions": agent_decisions,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "b0207_agent_gem_pk_1y.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = OUT / "b0207_agent_gem_pk_1y.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["策略", "期末资产", "累计投入", "累计盈利", "累计收益", "TWR", "IRR", "Sharpe", "最大回撤", "平均持仓", "换手", "决策次数"])
        for label, metrics in rows:
            writer.writerow([
                label, round(metrics["final_value"], 2), round(metrics["invested"], 2), round(metrics["profit"], 2),
                pct(metrics["cum_return"]), pct(metrics["twr"]), pct(metrics["annual_return_irr"]),
                f"{metrics['sharpe']:.3f}", pct(metrics["max_drawdown"]), pct(metrics["avg_position"]),
                "不可用" if metrics["turnover"] is None else f"{metrics['turnover']:.4f}",
                metrics["decision_count"],
            ])

    decision_csv = OUT / "b0207_agent_gem_pk_1y_decisions.csv"
    with decision_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["信号日", "执行日", "策略", *[f"{code}_{NAMES[code]}" for code in ORDER], "现金"])
        for kind, decisions in (("B0207机械基线", baseline_decisions), ("B0207+Agent", agent_decisions)):
            for item in decisions:
                target = item["final_target"]
                writer.writerow([item["signal_date"], item["execution_date"], kind,
                                 *[target["weights"][code] for code in ORDER], target["cash_pct"]])

    lines = [
        "# 最近一年三策略PK：B0207 vs B0207+Agent vs 创业板智能定投",
        "",
        "> 状态：NON_PIT_REPLAY｜研究影子对比，不修改生产策略、真实持仓或正式月报。",
        "",
        f"- 信号区间：{dates[start]}至{dates[end]}，共{baseline['weeks']}个周度持有区间。",
        "- 所有策略使用相同资金流、100份整手、佣金与现金利率；决策使用当周已完成数据，下一可交易日开盘执行。",
        f"- Agent覆盖协议已冻结，SHA-256：`{protocol_hash}`；收益没有传入Agent目标函数。",
        "- 重要限制：当前冠军历史PIT证据不完整，因此本报告不能宣称严格样本外，只能比较同一回放口径。",
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
            f"{metrics['sharpe']:.3f} | {pct(metrics['max_drawdown'])} | {pct(metrics['avg_position'])} | "
            f"{'不可用' if metrics['turnover'] is None else format(metrics['turnover'], '.4f')} |"
        )
    lines += [
        "",
        f"按期末资产，本次同口径领先者为：**{winner}**。",
        "",
        "## PK结论",
        "",
        f"- Agent覆盖相对机械基线的期末资产差额为{agent_gap:,.2f}元；最大回撤改善{drawdown_improvement:.2%}，但Sharpe差额为{sharpe_gap:.3f}。",
        f"- Agent平均持仓比机械基线低{-exposure_gap:.2%}。本区间内，减仓后留存现金造成的上涨机会成本，明显高于{drawdown_improvement:.2%}的回撤改善。",
        "- 因此当前Agent覆盖协议不应替换B0207机械基线；若继续研究，应优先减少无明确替代标的时的现金化，而不是增加更多短周期过滤。",
        "- 创业板智能定投仅交易159915：相对持仓成本盈利超过2.5%时缩小当周投入，亏损超过2.5%时放大投入，否则按标准额投入；它不卖出，未投入部分保留现金。",
        "",
        "## Agent高现金典型决策",
        "",
        "| 信号日 | 下一执行日 | 非零目标 | 现金 |",
        "|---|---|---|---:|",
    ]
    for item in cash_examples:
        target = item["final_target"]
        nonzero = "、".join(
            f"{NAMES[code]}{target['weights'][code]}%" for code in ORDER if target["weights"][code] > 0
        ) or "无"
        lines.append(
            f"| {item['signal_date']} | {item['execution_date']} | {nonzero} | {target['cash_pct']}% |"
        )
    lines += [
        "",
        "## Agent覆盖协议（结果揭盲前固定）",
        "",
        "1. B0207目标是起点，不在非决策周自由调仓。",
        "2. 周线DIF一阶、二阶变化为主；日线DIF变化和20日动量只作确认。",
        "3. 周线向下且继续恶化、日线也向下：原目标减半；其他周线向下状态保留75%。",
        "4. 周线向上时，日线回落不先减仓；周线向上且加速、日线确认时放大到115%。",
        "5. B0207未持有的ETF只有在周线主趋势、周线加速、日线和20日动量同时为正时才建立5%观察仓。",
        "6. 所有结果向下取整至5%；未分配部分留现金；总和超过100%时从证据最弱者开始每次减少5%。",
        "",
        "## 解释边界",
        "",
        "- 这不是伪称Agent历史上真实保存过一年判断，而是预注册Agent覆盖协议的逐周影子核算。",
        "- Agent组允许改变B0207研究仓位，但该权限仅存在于本报告，不覆盖《投资策略.md》的生产权限。",
        "- 创业板智能定投的旧核算函数未输出可比的成交额换手数据，因此换手显示为“不可用”，不能解释为没有交易。",
        "- 若要获得可正式归因于Agent的无未来知识成绩，需要从下一决策日起冻结真实周度Proposal并向前积累。",
    ]
    md_path = OUT / "b0207_agent_gem_pk_1y.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"md": str(md_path), "json": str(json_path), "csv": str(csv_path), "decisions": str(decision_csv), "winner": winner}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
