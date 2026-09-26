# -*- coding: utf-8 -*-
"""S1（共识+择优）多参数微调：以综合评分为最高优先级。

综合评分 = 70%×近一年(2025-08~2026-08) + 20%×近三年(2023-08~2026-08)
          + 10%×近五年(2021-08~2026-08)。

参数：
- guard_threshold：4相位平均现金 ≥ 阈值时进入防守（30%~70%）；
- guard_target：防守时的目标（mean=均值共识 / median=中位数共识 / cash=100%现金）；
- guard_cap：防守时共识暴露上限（None=不设限 / 0.5=最多50%）；
- pick_mode：平时跟随的相位（min_cash=最激进 / median_cash=现金中位相位）；
- freq：调仓频率（1=每周 / 2=每两周），非调仓周按持仓比例加仓。
"""

from __future__ import annotations

import csv
import itertools
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
    OUT_DIR,
    WINDOW_5Y,
    WINDOW_Y3,
)

WINDOW_3Y = ("2023-08-10", "2026-08-07")

WEEKLY = 2000.0
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}


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


def simulate_s1(etfs, master_dates, master_execs, start_idx, end_idx, order, params,
                guard_threshold: float, guard_target: str, guard_cap: float | None,
                pick_mode: str, freq: int) -> dict:
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

        phase_cash = {p: 100.0 - sum(phase_targets[p].values()) for p in range(4)}
        mean_cash = sum(phase_cash.values()) / 4.0
        if guard_target == "cash":
            consensus: dict[str, float] = {c: 0.0 for c in etfs}
        else:
            if guard_target == "median":
                consensus = {c: 0.0 for c in etfs}
                for c in etfs:
                    vals = sorted(phase_targets[p].get(c, 0.0) for p in range(4))
                    consensus[c] = (vals[1] + vals[2]) / 2.0
            else:
                consensus = {c: sum(phase_targets[p].get(c, 0.0) for p in range(4)) / 4.0
                             for c in etfs}
        if mean_cash >= guard_threshold:
            target_weights = consensus
            if guard_cap is not None and guard_target != "cash":
                target_weights = {c: w * guard_cap for c, w in consensus.items()}
        else:
            if pick_mode == "median_cash":
                ranked = sorted(range(4), key=lambda p: phase_cash[p])
                pick = ranked[1]
                tied = phase_cash[ranked[0]] == phase_cash[ranked[1]]
            else:
                min_cash = min(phase_cash.values())
                cands = [p for p in range(4) if phase_cash[p] == min_cash]
                pick = cands[0]
                tied = len(cands) > 1
            # 平手时用共识，避免固定偏向某个相位（相位不变性）
            if tied:
                target_weights = consensus
            else:
                target_weights = dict(phase_targets[pick])

        if (i - start_idx) % freq == 0:
            apply_weights(target_weights, value)
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
        "1y": WINDOW_Y3, "3y": WINDOW_3Y, "5y": WINDOW_5Y,
    }.items()}

    def composite(r: dict) -> float:
        return 0.7 * r["1y"]["cum_return"] + 0.2 * r["3y"]["cum_return"] + 0.1 * r["5y"]["cum_return"]

    grid = list(itertools.product(
        [30, 40, 50, 60, 70],
        ["mean", "median", "cash"],
        [None, 0.5],
        ["min_cash", "median_cash"],
        [1, 2],
    ))
    results = []
    for threshold, g_target, g_cap, pick, freq in grid:
        rec = {"threshold": threshold, "guard_target": g_target, "guard_cap": g_cap,
               "pick_mode": pick, "freq": freq}
        for w in ("1y", "3y", "5y"):
            rec[w] = simulate_s1(etfs, master_dates, master_execs, *idx[w],
                                 B0207_ORDER, B0207_PARAMS, threshold, g_target, g_cap, pick, freq)
        rec["score"] = composite(rec)
        results.append(rec)

    results.sort(key=lambda r: -r["score"])
    print("grid size:", len(grid))
    for r in results[:15]:
        print(f"thr={r['threshold']}% target={r['guard_target']} cap={r['guard_cap']} "
              f"pick={r['pick_mode']} freq={r['freq']} 1y={r['1y']['cum_return']:.4f} "
              f"3y={r['3y']['cum_return']:.4f} 5y={r['5y']['cum_return']:.4f} "
              f"score={r['score']:.4f} MDD={r['5y']['max_drawdown']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "b0207_s1_tuning.md"
    lines = [
        "# S1 多参数微调结果（按综合评分排序，Top 15）",
        "",
        "参数：防守阈值30%~70%；防守目标 mean/median/cash；防守上限 None/50%；",
        "择优方式 min_cash/median_cash；调仓频率 1/2 周。共 120 组。",
        "综合评分 = 70%×近一年 + 20%×近三年 + 10%×近五年。",
        "",
        "| 排名 | 阈值 | 防守目标 | 防守上限 | 择优 | 频率 | 近1年 | 近3年 | 近5年 | 综合分 | IRR | Sharpe | MDD | 换手 |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results[:15], 1):
        y5 = r["5y"]
        lines.append(
            f"| {i} | {r['threshold']}% | {r['guard_target']} | "
            f"{'无' if r['guard_cap'] is None else f'{r['guard_cap']:.0%}'} | "
            f"{r['pick_mode']} | {r['freq']}周 | {pct(r['1y']['cum_return'])} | "
            f"{pct(r['3y']['cum_return'])} | {pct(y5['cum_return'])} | "
            f"{r['score']:.3f} | {pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | "
            f"{pct(y5['max_drawdown'])} | {y5['turnover']:.2f} |"
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "b0207_s1_tuning.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["排名", "阈值", "防守目标", "防守上限", "择优", "频率",
                         "近1年", "近3年", "近5年", "综合分", "IRR", "Sharpe", "MDD", "换手", "期末市值"])
        for i, r in enumerate(results, 1):
            y5 = r["5y"]
            writer.writerow([i, f"{r['threshold']}%", r["guard_target"],
                             "" if r["guard_cap"] is None else f"{r['guard_cap']:.0%}",
                             r["pick_mode"], r["freq"], pct(r["1y"]["cum_return"]),
                             pct(r["3y"]["cum_return"]), pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), f"{y5['turnover']:.3f}",
                             f"{y5['final_value']:.0f}"])
    payload = {
        "schema_version": "b0207-s1-tuning-v1",
        "grid_size": len(grid),
        "results": [
            {"threshold": r["threshold"], "guard_target": r["guard_target"],
             "guard_cap": r["guard_cap"], "pick_mode": r["pick_mode"], "freq": r["freq"],
             **{w: r[w] for w in ("1y", "3y", "5y")}, "score": r["score"]}
            for r in results
        ],
    }
    (OUT_DIR / "b0207_s1_tuning.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
