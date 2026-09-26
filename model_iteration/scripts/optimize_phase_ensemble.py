# -*- coding: utf-8 -*-
"""四相位各自寻优 + 中位数共识。

流程：
1. 对相位 p=0..3，分别在该相位的决策周（每 4 周、错开 p 周）上做完整搜索：
   - 阶段A：全枚举 2160 种顺序（创业板第1、稀土前四、创新药自由）× 基础参数；
   - 阶段B：Top50 顺序 × 60 组细化参数；
   - 按综合分（20%近一年 + 30%近三年 + 50%近五年）选出该相位的最优模型；
2. （本版暂不执行中位数共识，只输出各相位 TOP1。）
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
    simulate_realistic,
)
from scripts.optimize_allocation_core12 import orders_159915_first_rare_top4  # noqa: E402

WEEKLY = 2000.0
WINDOW_3Y = ("2023-08-10", "2026-08-07")
BASE_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
               "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
SEED = 42
PHASE_B_PER_ORDER = 60
TOP_ORDERS = 50


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


def pct(x) -> str:
    return f"{float(x):.1%}"


def composite(r: dict) -> float:
    return 0.2 * r["1y"]["cum_return"] + 0.3 * r["3y"]["cum_return"] + 0.5 * r["5y"]["cum_return"]


def simulate_phase_median(etfs, master_dates, master_execs, start_idx, end_idx,
                          phase_configs: list[dict]) -> dict:
    """4 个相位最优模型的中位数共识：每周调仓。"""
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[start_idx : end_idx + 1] if ex is not None), None)
    for c, e in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
            for key, event in e["ca_events"]
        }
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

        raw_pos = {c: (e["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0)
                   for c, e in etfs.items()}

        def target_for(cfg: dict) -> dict:
            floor = float(cfg["params"]["floor"])
            eff = {c: min(1.0, max(floor, raw_pos[c])) for c in etfs}
            sig = {c: eff[c] > 0.0 for c in etfs}
            return allocate_cfg(sig, eff, cfg["order"], cfg["params"])["weights"]

        if not initialized:
            for p, cfg in enumerate(phase_configs):
                phase_targets[p] = dict(target_for(cfg))
            initialized = True
        for p, cfg in enumerate(phase_configs):
            if (i - start_idx - p) % 4 == 0:
                phase_targets[p] = dict(target_for(cfg))

        consensus: dict[str, float] = {c: 0.0 for c in etfs}
        for c in etfs:
            vals = sorted(phase_targets[p].get(c, 0.0) for p in range(4))
            consensus[c] = (vals[1] + vals[2]) / 2.0
        apply_weights(consensus, value)

        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        nav_after_list.append(nav_after)
        prev_after = nav_after

    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover)


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

    def run(order, params, phase: int) -> dict:
        rec = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params,
                                     weekly_deploy=False, phase_offset=phase)
               for w in ("1y", "3y", "5y")}
        rec["score"] = composite(rec)
        return rec

    orders = orders_159915_first_rare_top4()
    print("orders:", len(orders))
    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]

    phase_winners = []
    for phase in range(4):
        print("=== optimizing phase", phase, "===")
        phase_a = []
        for oi, order in enumerate(orders):
            rec = {"model": {"id": f"A{phase}_{oi:04d}", "order": order, "params": dict(BASE_PARAMS)}}
            rec.update(run(order, BASE_PARAMS, phase))
            phase_a.append(rec)
        top_orders = sorted(phase_a, key=lambda r: -r["score"])[:TOP_ORDERS]
        print("phase A done; top score:", round(top_orders[0]["score"], 4))

        rng = np.random.default_rng(SEED + phase)
        phase_b = []
        for tr in top_orders:
            order = tr["model"]["order"]
            for _ in range(PHASE_B_PER_ORDER):
                core = int(rng.choice(core_choices))
                rem_options = [r for r in rem_choices if r <= 100 - core]
                params = {
                    "core_budget": core,
                    "remainder_budget": int(rng.choice(rem_options)),
                    "floor": float(rng.choice(floor_choices)),
                    "ladder": int(rng.choice(ladder_choices)),
                    "max_single": int(rng.choice(max_choices)),
                    "rebalance": 4,
                    "core_size": 3,
                }
                rec = {"model": {"id": f"B{phase}_{len(phase_b):04d}", "order": order, "params": params}}
                rec.update(run(order, params, phase))
                phase_b.append(rec)
        winner = max(phase_b, key=lambda r: r["score"])
        phase_winners.append(winner)
        print("phase", phase, "winner:", winner["model"]["id"],
              "score", round(winner["score"], 4),
              "1y", round(winner["1y"]["cum_return"], 4),
              "3y", round(winner["3y"]["cum_return"], 4),
              "5y", round(winner["5y"]["cum_return"], 4))

    # 参考：单相位 B0207 改版
    b0207_order = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
    b0207_params = {**BASE_PARAMS}
    ref = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w],
                                 b0207_order, b0207_params, weekly_deploy=True)
           for w in ("1y", "3y", "5y")}
    ref["score"] = composite(ref)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 四相位各自寻优 TOP1（综合分 = 20%近一年 + 30%近三年 + 50%近五年）",
        "",
        "每个相位独立搜索最优（顺序×参数，2160 顺序全枚举 + Top50×60 细化），",
        "按综合分 = 20%近一年 + 30%近三年 + 50%近五年 选出各自最优；",
        "本版暂不执行中位数共识，只输出各相位 TOP1。",
        "",
        "## 各相位最优模型",
        "",
        "| 相位 | 模型 | 顺序 | 核心/剩余 | 下限 | 档位 | 上限 | 近1年 | 近3年 | 近5年 | 综合分 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p, w in enumerate(phase_winners):
        m = w["model"]
        prm = m["params"]
        lines.append(
            f"| {p} | {m['id']} | {'/'.join(m['order'])} | "
            f"{prm['core_budget']}%/{prm['remainder_budget']}% | {prm['floor']:.0%} | "
            f"{prm['ladder']} | {prm['max_single']}% | {pct(w['1y']['cum_return'])} | "
            f"{pct(w['3y']['cum_return'])} | {pct(w['5y']['cum_return'])} | {w['score']:.3f} |"
        )
    lines += [
        "",
        "## 参考对比",
        "",
        "| 方案 | 近1年 | 近3年 | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in (("B0207 改版（单相位）", ref),):
        y5 = r["5y"]
        lines.append(
            f"| {label} | {pct(r['1y']['cum_return'])} | {pct(r['3y']['cum_return'])} | "
            f"{pct(y5['cum_return'])} | {r['score']:.3f} | {pct(y5['annual_return_irr'])} | "
            f"{y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | {pct(y5['avg_position'])} | "
            f"{y5['turnover']:.2f} | {y5['final_value']:.0f} |"
        )
    md = OUT_DIR / "b0207_phase_top1_20_30_50.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "b0207_phase_top1_20_30_50.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "近1年", "近3年", "近5年", "综合分", "IRR", "Sharpe", "MDD", "期末市值"])
        for label, r in (("B0207 改版（单相位）", ref),):
            y5 = r["5y"]
            writer.writerow([label, pct(r["1y"]["cum_return"]), pct(r["3y"]["cum_return"]),
                             pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), f"{y5['final_value']:.0f}"])
    payload = {
        "schema_version": "phase-optimized-ensemble-v1",
        "score_weights": {"1y": 0.2, "3y": 0.3, "5y": 0.5},
        "phase_winners": [
            {"phase": p, "model": w["model"], **{k: w[k] for k in ("1y", "3y", "5y")},
             "score": w["score"]}
            for p, w in enumerate(phase_winners)
        ],
        "reference": {**{k: ref[k] for k in ("1y", "3y", "5y")}, "score": ref["score"]},
    }
    (OUT_DIR / "b0207_phase_top1_20_30_50.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
