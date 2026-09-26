# -*- coding: utf-8 -*-
"""按参考表格式生成/更新 B0207 连续账户收益 xlsx（初始决策 2021-07-31）。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import openpyxl

from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _irr  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CASH_FACTOR,
    LOT,
    MIN_COMMISSION,
    COMMISSION_RATE,
)
from rolling.ledger import set_workspace
from rolling.data import load_aligned_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "champion_vs_dca" / "B0207连续账户_当前策略_初始决策2021-07-31_截至2026-08-07.xlsx"
WEEKLY = 2000.0
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
MOM20_TH = -0.05


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def main() -> int:
    etfs = {c: load_etf(c) for c in CODES}
    set_workspace("159915")
    _, _, _, _, usable_m, _ = load_aligned_data("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    s0 = next(k for k in range(len(master_dates)) if master_dates[k] == "2021-07-30")
    end = len(master_dates) - 1
    print("window:", master_dates[s0], "~", master_dates[end])

    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next(ex for ex in master_execs[s0 : end + 1] if ex is not None)
    for c, et in etfs.items():
        applied[c] = {
            key: (str(ev.get("pay_date") or ev.get("ex_date")) < first_ex)
            for key, ev in et["ca_events"]
        }
    reduced = {c: False for c in etfs}
    commission_total = 0.0
    min_cash = 0.0
    weeks = []
    invested = 0.0
    prev_after = None

    def apply_weights(target: dict, value: float) -> None:
        nonlocal cash, commission_total
        plan: list[tuple[str, float]] = []
        for c, et in etfs.items():
            if prices[c] <= 0:
                continue
            weight = float(target.get(c, 0)) / 100.0
            tgt_share = int(weight * value / prices[c] // LOT) * LOT
            delta = tgt_share - shares[c]
            plan.append((c, delta))
        # 先卖后买
        for c, delta in sorted(plan, key=lambda x: x[1]):
            if delta >= 0:
                continue
            traded = -delta * prices[c]
            cost = trade_cost(traded)
            commission_total += cost
            cash += traded - cost
            shares[c] += delta
        for c, delta in sorted(plan, key=lambda x: -x[1]):
            if delta <= 0:
                continue
            max_lots = int(max(0.0, cash - 5.0) / (prices[c] * (1.0 + COMMISSION_RATE)) // LOT) * LOT
            delta = min(delta, max_lots)
            if delta <= 0:
                continue
            traded = delta * prices[c]
            cost = trade_cost(traded)
            commission_total += cost
            cash -= traded + cost
            shares[c] += delta

    def deploy(inflow: float) -> None:
        nonlocal cash, commission_total
        holdings_value = sum(shares[c] * prices[c] for c in etfs if prices[c] > 0)
        if holdings_value <= 0:
            return
        for c, et in etfs.items():
            if prices[c] <= 0 or shares[c] <= 0:
                continue
            alloc = inflow * (shares[c] * prices[c] / holdings_value)
            add_share = int(alloc / prices[c] // LOT) * LOT
            if add_share <= 0:
                continue
            traded = add_share * prices[c]
            cost = trade_cost(traded)
            commission_total += cost
            cash -= traded + cost
            shares[c] += add_share

    for i in range(s0, end + 1):
        day = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        cash *= CASH_FACTOR
        n_avail = sum(1 for et in etfs.values() if et["start_date"] <= day)
        if n_avail == 0:
            continue
        prices = {c: et["price_map"].get(ex, 0.0) for c, et in etfs.items()}
        for c, et in etfs.items():
            if prices[c] > 0 or shares[c] != 0:
                shares[c], cash = _apply_ca(shares[c], cash, et["ca_events"], ex, applied[c])
        nav_before = sum(shares[c] * prices[c] for c in etfs) + cash
        twr = nav_before / prev_after - 1.0 if prev_after is not None else 0.0
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        value = sum(shares[c] * prices[c] for c in etfs) + cash
        # 周度强看空保护
        for c, et in etfs.items():
            feat = et.get("feat_by_date", {}).get(day, {})
            trig = (
                shares[c] > 0 and prices[c] > 0
                and float(feat.get("d_dif1", 1.0)) < 0
                and float(feat.get("d_dif2", 1.0)) < 0
                and float(feat.get("w_dif1", 1.0)) < 0
                and float(feat.get("w_dif2", 1.0)) < 0
                and float(feat.get("mom20", 0.0)) < MOM20_TH
            )
            if trig:
                if not reduced[c]:
                    sell_lots = int(shares[c] * 0.7 // LOT) * LOT
                    if sell_lots > 0:
                        traded = sell_lots * prices[c]
                        cost = trade_cost(traded)
                        commission_total += cost
                        cash += sell_lots * prices[c] - cost
                        shares[c] -= sell_lots
                    reduced[c] = True
            else:
                reduced[c] = False
        if (i - s0) % int(B0207_PARAMS["rebalance"]) == 0:
            raw = {c: (et["by_date"].get(day, 0.0) if prices[c] > 0 else 0.0)
                   for c, et in etfs.items()}
            floor = float(B0207_PARAMS["floor"])
            eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
            sig = {c: eff[c] > 0.0 for c in etfs}
            target = allocate_cfg(sig, eff, B0207_ORDER, B0207_PARAMS)
            apply_weights(target["weights"], value)
        else:
            deploy(inflow)

        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        weeks.append({
            "i": i, "date": day, "exec": ex, "inflow": inflow,
            "nav_before": nav_before, "nav_after": nav_after,
            "twr": twr, "invested": invested, "cash": cash,
        })
        prev_after = nav_after

    final_nav = weeks[-1]["nav_after"]
    final_cash = weeks[-1]["cash"]
    invested_5y = weeks[-1]["invested"]
    min_cash = min(x["cash"] for x in weeks)

    def window_metrics(ws: int) -> dict:
        w = [x for x in weeks if x["i"] >= ws]
        start_nav = 0.0 if ws == s0 else next((x["nav_before"] for x in weeks if x["i"] == ws), weeks[0]["nav_before"])
        deposits = sum(x["inflow"] for x in w)
        final = w[-1]["nav_after"]
        profit = final - start_nav - deposits
        twrs = [x["twr"] for x in w if x["twr"] != 0.0]
        index = np.cumprod([1.0 + t for t in [x["twr"] for x in w]])
        peak = np.maximum.accumulate(index)
        mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
        twr_total = float(index[-1] - 1.0) if len(index) else 0.0
        flows = [(k, -x["inflow"]) for k, x in enumerate(w)]
        irr = _irr(flows, final)
        return {
            "signal": w[0]["date"], "decision": (date.fromisoformat(w[0]["date"]) + timedelta(days=1)).isoformat(),
            "exec": w[0]["exec"], "start_nav": start_nav, "deposits": deposits,
            "final": final, "profit": profit, "twr": twr_total, "irr": irr, "mdd": mdd,
        }

    w5 = window_metrics(s0)
    w3 = window_metrics(next(k for k in range(len(master_dates)) if master_dates[k] == "2023-07-28"))
    w1 = window_metrics(next(k for k in range(len(master_dates)) if master_dates[k] == "2025-07-25"))
    for name, w in (("5y", w5), ("3y", w3), ("1y", w1)):
        print(name, w["signal"], w["decision"], w["exec"], "start", round(w["start_nav"], 2),
              "deposits", w["deposits"], "final", round(w["final"], 2), "profit", round(w["profit"], 2),
              "twr", round(w["twr"], 4), "irr", round(w["irr"], 4), "mdd", round(w["mdd"], 4))
    print("audit: invested", invested_5y, "final", round(final_nav, 2), "cum", round(final_nav / invested_5y - 1, 4),
          "profit", round(final_nav - invested_5y, 2), "commission", round(commission_total, 2),
          "final_cash", round(final_cash, 2), "min_cash", round(min_cash, 2))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "连续账户收益"
    rows = [
        ["B0207连续账户收益：初始决策日2021-07-31"],
        ["同一账户连续运行，不重新归零；信号使用当时已完成数据，下一交易日开盘执行；数据截止2026-08-07。"],
        [],
        ["区间", "信号日", "决策日", "首次执行日", "起始账户资产", "期间新增资金", "期末资产",
         "扣除资金流后盈利", "累计TWR", "年化IRR", "最大回撤"],
    ]
    for w in (w5, w3, w1):
        rows.append(["近5年" if w is w5 else "近3年" if w is w3 else "近1年",
                     w["signal"], w["decision"], w["exec"], w["start_nav"], w["deposits"],
                     w["final"], w["profit"], w["twr"], w["irr"], w["mdd"]])
    rows += [
        [], [],
        ["资金规则与审计"],
        ["项目", "值"],
        ["策略", "B0207机械基线（含周度强看空保护）"],
        ["账户开始", "2021-07-31首次决策；2021-08-02首次交易"],
        ["供款规则", "每只已上市且当周可用ETF每周2,000元；未上市为0元"],
        ["决策周期", "每28天"],
        ["交易单位", 100],
        ["现金年利率", 0.0175],
        ["佣金", "单边max(成交额×0.025%, 5元)"],
        ["全程最低现金", min_cash],
        ["累计佣金", commission_total],
        ["期末现金", final_cash],
        ["5年累计投入", invested_5y],
        ["5年期末资产", final_nav],
        ["5年投入回报率", final_nav / invested_5y - 1.0],
        ["5年累计盈利", final_nav - invested_5y],
        [],
        ["限制：本地历史行情缺少完整available_at、revision_id及覆盖全部ETF的公司行动PIT证据，因此结果属于研究回测，不等同于真实历史成交业绩。"],
        [],
    ]
    for row in rows:
        ws.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
