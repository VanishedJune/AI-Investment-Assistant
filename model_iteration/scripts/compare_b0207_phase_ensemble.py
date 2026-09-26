# -*- coding: utf-8 -*-
"""B0207 多相位共识（消除单相位运气）对比。

设计：
- 4 个相位 p=0..3，相位 p 每 4 周（相对窗口起点偏移 p 周）按 B0207 规则
  用当周信号重新计算目标权重；两次决策之间保持最近一次目标；
- 每周组合目标 = 4 个相位目标的等权平均（共识）；
- 执行方式对比：
  C1 每周调仓到共识（最平滑，换手最高）；
  C2 每 4 周调仓到共识，其余周按现有持仓比例加仓（与当前正式 weekly_deploy 一致）；
  C1t 每周计算共识，但只有共识相对当前持仓变化 ≥15pp 才调仓，否则按持仓比例加仓。
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
    simulate_realistic,
)

WEEKLY = 2000.0
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
OUT = ROOT.parent / "champion_vs_dca"
THRESHOLD = 0.15


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


def simulate_ensemble(etfs, master_dates, master_execs, start_idx, end_idx, order, params,
                      mode: str, combine: str = "mean") -> dict:
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[start_idx : end_idx + 1] if ex is not None), None)
    for c, e in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
            for key, event in e["ca_events"]
        }
    floor = float(params["floor"])
    phase_targets: dict[int, dict[str, float]] = {p: {c: 0.0 for c in etfs} for p in range(4)}
    initialized = False
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    traded_total = 0.0
    nav_after_list: list[float] = []
    prev_after = None

    def apply_weights(target_weights: dict, value: float) -> None:
        nonlocal cash, traded_total
        for c, e in etfs.items():
            if prices[c] <= 0:
                continue
            weight = float(target_weights.get(c, 0)) / 100.0
            tgt_share = int(weight * value / prices[c] // LOT) * LOT
            delta = tgt_share - shares[c]
            traded = abs(delta) * prices[c]
            traded_total += traded
            cost = trade_cost(traded)
            cash -= delta * prices[c] + cost
            shares[c] = tgt_share

    def deploy_to_holdings(inflow: float) -> None:
        nonlocal cash, traded_total
        holdings_value = sum(shares[c] * prices[c] for c in etfs if prices[c] > 0)
        if holdings_value <= 0:
            return
        for c, e in etfs.items():
            if prices[c] <= 0 or shares[c] <= 0:
                continue
            alloc = inflow * (shares[c] * prices[c] / holdings_value)
            add_share = int(alloc / prices[c] // LOT) * LOT
            if add_share <= 0:
                continue
            traded = add_share * prices[c]
            traded_total += traded
            cost = trade_cost(traded)
            cash -= traded + cost
            shares[c] += add_share

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
            twr_rets.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        flows.append((i - start_idx, -inflow))
        value = sum(shares[c] * prices[c] for c in etfs) + cash

        eff_pos = {}
        signals = {}
        for c, e in etfs.items():
            raw = e["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0
            eff = min(1.0, max(floor, raw))
            eff_pos[c] = eff
            signals[c] = eff > 0.0
        if not initialized:
            target = allocate_cfg(signals, eff_pos, order, params)
            for p in range(4):
                phase_targets[p] = dict(target["weights"])
            initialized = True
        for p in range(4):
            if (i - start_idx - p) % 4 == 0:
                target = allocate_cfg(signals, eff_pos, order, params)
                phase_targets[p] = dict(target["weights"])
        if combine == "median":
            consensus = {c: 0.0 for c in etfs}
            for c in etfs:
                vals = sorted(phase_targets[p].get(c, 0.0) for p in range(4))
                consensus[c] = (vals[1] + vals[2]) / 2.0
        elif combine == "best_guard":
            phase_cash = {p: 100.0 - sum(phase_targets[p].values()) for p in range(4)}
            strongest = min(range(4), key=lambda p: phase_cash[p])
            mean_cash = sum(phase_cash.values()) / 4.0
            if mean_cash >= 50.0:
                consensus = {c: sum(phase_targets[p].get(c, 0.0) for p in range(4)) / 4.0
                             for c in etfs}
            else:
                consensus = dict(phase_targets[strongest])
        else:
            consensus = {c: sum(phase_targets[p].get(c, 0.0) for p in range(4)) / 4.0
                         for c in etfs}

        if mode == "weekly":
            apply_weights(consensus, value)
        elif mode == "monthly":
            if (i - start_idx) % 4 == 0:
                apply_weights(consensus, value)
            else:
                deploy_to_holdings(inflow)
        elif mode == "weekly_threshold":
            current_pct = {
                c: (shares[c] * prices[c] / value * 100.0 if value > 0 and prices[c] > 0 else 0.0)
                for c in etfs
            }
            moved = sum(abs(consensus[c] - current_pct[c]) for c in etfs)
            if moved >= THRESHOLD * 100.0:
                apply_weights(consensus, value)
            else:
                deploy_to_holdings(inflow)

        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        nav_after_list.append(nav_after)
        prev_after = nav_after

    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover)


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

    def run_ensemble(mode: str, combine: str = "mean") -> dict:
        rec = {w: simulate_ensemble(etfs, master_dates, master_execs, *idx[w],
                                    B0207_ORDER, B0207_PARAMS, mode=mode, combine=combine)
               for w in ("Y1", "Y2", "Y3", "Y5")}
        rec["score"] = robust_score(rec)
        return rec

    ref = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w],
                                 B0207_ORDER, B0207_PARAMS, weekly_deploy=True)
           for w in ("Y1", "Y2", "Y3", "Y5")}
    ref["score"] = robust_score(ref)
    ref["label"] = "B0207 改版（单相位，当前正式）"

    variants = [
        ("C1 四相位均值共识·每周调仓", "weekly", "mean"),
        ("C2 四相位均值共识·每月调仓", "monthly", "mean"),
        ("C1t 四相位均值共识·周度阈值15pp", "weekly_threshold", "mean"),
        ("M1 四相位中位数共识·每周调仓", "weekly", "median"),
        ("S1 共识+择优（跟最激进相位，共识现金≥50%防守）", "weekly", "best_guard"),
    ]
    results = [ref]
    for label, mode, combine in variants:
        rec = run_ensemble(mode, combine)
        rec["label"] = label
        results.append(rec)
        print(label, "5y", round(rec["Y5"]["cum_return"], 4),
              "score", round(rec["score"], 4), "MDD", round(rec["Y5"]["max_drawdown"], 4),
              "turnover", round(rec["Y5"]["turnover"], 3))

    lines = [
        "# B0207 多相位共识（消除单相位运气）对比",
        "",
        "4 个相位 p=0..3 每 4 周（错开 1 周）各自按 B0207 重新计算目标；",
        "每周组合目标 = 4 相位目标等权平均（共识）。",
        "C1：每周调仓到均值共识；C2：每 4 周调仓到均值共识；C1t：均值共识变化≥15pp 才调仓；",
        "M1：每周调仓到中位数共识（降低极端相位影响）；",
        "S1：平时跟最激进相位（现金最少），4 相位平均现金≥50% 时切回均值共识防守。",
        "",
        "| 方案 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        y5 = r["Y5"]
        lines.append(
            f"| {r['label']} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['final_value']:.0f} |"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "b0207_phase_ensemble.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "b0207_phase_ensemble.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "期末市值"])
        for r in results:
            y5 = r["Y5"]
            writer.writerow([r["label"], pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), pct(y5["avg_position"]),
                             f"{y5['turnover']:.3f}", f"{y5['final_value']:.0f}"])
    payload = {
        "schema_version": "b0207-phase-ensemble-v1",
        "threshold": THRESHOLD,
        "results": [
            {"label": r["label"], **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]}
            for r in results
        ],
    }
    (OUT / "b0207_phase_ensemble.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
