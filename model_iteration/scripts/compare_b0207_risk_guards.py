# -*- coding: utf-8 -*-
"""B0207 + 三类风险保护 对比。

主线：每 4 周按 B0207（核心3，85/15）目标调仓（不变）。
保护1 组合回撤保护：账户时间加权净值从阶段高点回撤超过阈值 → 降仓至
  risk_level（半仓50% / 30%），净值创新高后恢复最近一次决策目标。
保护2 单周急跌保护：任一持仓 ETF 单周（执行价对执行价）跌幅超过阈值 →
  只将该 ETF 卖出至现金，不清其他持仓。
保护3 看空只减仓：非决策周若任一持仓 ETF 冠军仓位=0（看空）→ 只卖出该 ETF，
  不买入其他 ETF。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _irr  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CASH_FACTOR,
    LOT,
    MIN_COMMISSION,
    COMMISSION_RATE,
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    robust_score,
)

WEEKLY = 2000.0
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
OUT = ROOT.parent / "champion_vs_dca"


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def _metrics(twr_rets, flows, final_nav, invested, avg_pos, turnover):
    arr = np.asarray(twr_rets, dtype=float)
    index = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(index)
    mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    return {
        "cum_return": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "annual_return_irr": _irr(flows, final_nav),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "invested": invested,
        "profit": final_nav - invested,
        "avg_position": avg_pos,
        "final_value": final_nav,
        "twr": float(index[-1] - 1.0) if len(index) else 0.0,
        "turnover": turnover,
    }


def cap_weights(weights: dict, level: float) -> dict:
    return {c: int(w * level / 5.0) * 5 for c, w in weights.items() if w > 0}


def simulate_risk(etfs, master_dates, master_execs, start_idx, end_idx, order, params,
                  guard: dict | None = None) -> dict:
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[start_idx : end_idx + 1] if ex is not None), None)
    for c, e in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
            for key, event in e["ca_events"]
        }
    freq = max(1, int(params["rebalance"]))
    floor = float(params["floor"])
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    traded_total = 0.0
    nav_after_list: list[float] = []
    prev_after = None
    twr_index = 1.0
    peak_twr = 1.0
    risk_mode = False
    last_target: dict | None = None
    prev_price = {c: None for c in etfs}
    triggers = 0
    trigger_dates: list[str] = []
    reduced = {c: False for c in etfs}

    def apply_target(target_weights: dict, value: float) -> None:
        nonlocal cash
        for c, e in etfs.items():
            if prices[c] <= 0:
                continue
            weight = float(target_weights.get(c, 0)) / 100.0
            tgt_share = int(weight * value / prices[c] // LOT) * LOT
            delta = tgt_share - shares[c]
            traded = abs(delta) * prices[c]
            nonlocal traded_total
            traded_total += traded
            cost = trade_cost(traded)
            cash -= delta * prices[c] + cost
            shares[c] = tgt_share

    def sell_to_cash(code: str) -> None:
        nonlocal cash, traded_total
        delta = -shares[code]
        if abs(delta) <= 0:
            return
        traded = abs(delta) * prices[code]
        traded_total += traded
        cost = trade_cost(traded)
        cash -= delta * prices[code] + cost
        shares[code] = 0.0

    for i in range(start_idx, end_idx + 1):
        date = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        cash *= CASH_FACTOR
        n_avail = sum(1 for e in etfs.values() if e["start_date"] <= date)
        if n_avail == 0:
            continue
        prices = {c: e["price_map"].get(ex, 0.0) for c, e in etfs.items()}
        for c, e in etfs.items():
            if prices[c] > 0 or shares[c] != 0:
                shares[c], cash = _apply_ca(shares[c], cash, e["ca_events"], ex, applied[c])
        nav_before = sum(shares[c] * prices[c] for c in etfs) + cash
        if prev_after is not None:
            week_ret = nav_before / prev_after - 1.0
            twr_rets.append(week_ret)
            twr_index *= nav_before / prev_after
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        flows.append((i - start_idx, -inflow))
        value = sum(shares[c] * prices[c] for c in etfs) + cash

        weekly_ret = {c: 0.0 for c in etfs}
        for c in etfs:
            if prices[c] > 0 and prev_price[c] is not None and prev_price[c] > 0:
                weekly_ret[c] = prices[c] / prev_price[c] - 1.0

        if (i - start_idx) % freq == 0:
            eff_pos = {}
            signals = {}
            for c, e in etfs.items():
                raw = e["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0
                eff = min(1.0, max(floor, raw))
                eff_pos[c] = eff
                signals[c] = eff > 0.0
            target = allocate_cfg(signals, eff_pos, order, params)
            last_target = dict(target["weights"])
            target_weights = last_target
            if guard and guard["type"] == "drawdown" and risk_mode:
                target_weights = cap_weights(last_target, float(guard["risk_level"]))
            apply_target(target_weights, value)
        else:
            risk_before = risk_mode
            if guard and guard["type"] == "drawdown":
                if twr_index >= peak_twr:
                    peak_twr = twr_index
                    risk_mode = False
                elif twr_index / peak_twr - 1.0 <= -float(guard["threshold"]):
                    risk_mode = True
                if risk_mode and not risk_before:
                    triggers += 1
                    trigger_dates.append(date)
                    if last_target is not None:
                        apply_target(cap_weights(last_target, float(guard["risk_level"])), value)
                elif not risk_mode and risk_before and last_target is not None:
                    apply_target(last_target, value)
            elif guard and guard["type"] == "weekly_drop":
                for c in etfs:
                    if shares[c] > 0 and prices[c] > 0 and weekly_ret[c] <= -float(guard["threshold"]):
                        sell_to_cash(c)
                        triggers += 1
                        trigger_dates.append(date)
            elif guard and guard["type"] == "bearish_sell":
                for c, e in etfs.items():
                    raw = e["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0
                    if shares[c] > 0 and prices[c] > 0 and raw <= 0.0:
                        sell_to_cash(c)
                        triggers += 1
                        trigger_dates.append(date)
            elif guard and guard["type"] == "strong_bearish_sell":
                for c, e in etfs.items():
                    feat = e.get("feat_by_date", {}).get(date, {})
                    strong_bear = (
                        shares[c] > 0
                        and prices[c] > 0
                        and float(feat.get("d_dif1", 1.0)) < 0
                        and float(feat.get("d_dif2", 1.0)) < 0
                        and float(feat.get("w_dif1", 1.0)) < 0
                        and float(feat.get("w_dif2", 1.0)) < 0
                    )
                    if strong_bear:
                        if not reduced[c]:
                            sell_frac = float(guard.get("sell_frac", 0.5))
                            sell_lots = int(shares[c] * sell_frac // LOT) * LOT
                            if sell_lots > 0:
                                traded = sell_lots * prices[c]
                                traded_total += traded
                                cost = trade_cost(traded)
                                cash += sell_lots * prices[c] - cost
                                shares[c] -= sell_lots
                                triggers += 1
                                trigger_dates.append(date)
                            reduced[c] = True
                    else:
                        reduced[c] = False

        for c in etfs:
            if prices[c] > 0:
                prev_price[c] = prices[c]
        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        nav_after_list.append(nav_after)
        prev_after = nav_after

    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    metrics = _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover)
    metrics["triggers"] = triggers
    metrics["trigger_dates"] = trigger_dates
    return metrics


def pct(x) -> str:
    return f"{float(x):.1%}"


def main() -> int:
    etfs = {}
    for code in CODES:
        etfs[code] = load_etf(code)
        print("loaded", code, NAMES[code], "start", etfs[code]["start_date"])

    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    idx = {w: range_idx(master_dates, *b) for w, b in {
        "Y1": WINDOW_Y1, "Y2": WINDOW_Y2, "Y3": WINDOW_Y3, "Y5": WINDOW_5Y,
    }.items()}

    variants = [
        ("B0207 原版（基准，无保护）", None),
        ("保护1：回撤>8%降到50%", {"type": "drawdown", "threshold": 0.08, "risk_level": 0.5}),
        ("保护1：回撤>8%降到30%", {"type": "drawdown", "threshold": 0.08, "risk_level": 0.3}),
        ("保护1：回撤>10%降到50%", {"type": "drawdown", "threshold": 0.10, "risk_level": 0.5}),
        ("保护1：回撤>10%降到30%", {"type": "drawdown", "threshold": 0.10, "risk_level": 0.3}),
        ("保护2：单周跌>8%只减该ETF", {"type": "weekly_drop", "threshold": 0.08}),
        ("保护2：单周跌>10%只减该ETF", {"type": "weekly_drop", "threshold": 0.10}),
        ("保护3：看空触发只减仓不换仓", {"type": "bearish_sell"}),
    ]
    results = []
    for label, guard in variants:
        rec = {"label": label, "guard": guard}
        for w in ("Y1", "Y2", "Y3", "Y5"):
            rec[w] = simulate_risk(etfs, master_dates, master_execs, *idx[w],
                                   B0207_ORDER, B0207_PARAMS, guard=guard)
        rec["score"] = robust_score(rec)
        results.append(rec)
        print(label, "5y", round(rec["Y5"]["cum_return"], 4),
              "score", round(rec["score"], 4), "MDD", round(rec["Y5"]["max_drawdown"], 4),
              "triggers", rec["Y5"]["triggers"])

    # 参考：B0207 改版（每周加仓）来自此前对比结果
    try:
        prev = json.loads((OUT / "allocation_optimization_b0207_bearish_guard.json").read_text(encoding="utf-8"))
        modified = next(r for r in prev["results"] if r["label"] == "B0207 改版（4周决策+每周加仓）")
        modified_row = {
            "label": "B0207 改版（4周决策+每周加仓，参考）",
            **{w: modified[w] for w in ("Y1", "Y2", "Y3", "Y5")},
            "score": modified["score"],
        }
        modified_row["Y5"]["triggers"] = 0
    except Exception:
        modified_row = None

    lines = [
        "# B0207 + 三类风险保护 对比",
        "",
        "主线不变：每 4 周按 B0207（核心3，85/15）目标调仓。",
        "保护1：账户时间加权净值从阶段高点回撤超阈值 → 降至半仓或30%，创新高后恢复最近一次决策目标；",
        "保护2：任一持仓 ETF 单周跌幅超阈值 → 只卖出该 ETF 至现金，不清其他；",
        "保护3：非决策周任一持仓 ETF 冠军仓位=0（看空）→ 只卖出该 ETF，不买入其他。",
        "口径：真实账户（现金1.75%、100份整手、佣金万2.5最低5元、免印花税/过户费）。",
        "",
        "| 方案 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 | 触发次数(5y) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    all_rows = list(results)
    if modified_row:
        all_rows.insert(1, modified_row)
    for r in all_rows:
        y5 = r["Y5"]
        lines.append(
            f"| {r['label']} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['final_value']:.0f} | "
            f"{y5.get('triggers', 0)} |"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "allocation_optimization_b0207_risk_guards.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "allocation_optimization_b0207_risk_guards.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "期末市值", "触发次数(5y)"])
        for r in all_rows:
            y5 = r["Y5"]
            writer.writerow([r["label"], pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), pct(y5["avg_position"]),
                             f"{y5['turnover']:.3f}", f"{y5['final_value']:.0f}", y5.get("triggers", 0)])
    payload = {
        "schema_version": "b0207-risk-guards-v1",
        "order": B0207_ORDER,
        "params": B0207_PARAMS,
        "results": [
            {"label": r["label"], "guard": r.get("guard"),
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]}
            for r in results
        ],
    }
    if modified_row:
        payload["modified_reference"] = modified_row
    (OUT / "allocation_optimization_b0207_risk_guards.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
