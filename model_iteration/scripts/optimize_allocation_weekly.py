# -*- coding: utf-8 -*-
"""v2 补充搜索：每周调仓（rebalance=1）候选模型 Top 列表。

背景：v2 正式搜索空间只包含 2/3/4/6 周再平衡，没有每周调仓候选
（唯一每周模型是当前策略基线）。本脚本按 v2 相同引擎与评价口径，
对 v2 阶段 A 的 Top 50 顺序 × 80 组细化参数（强制 rebalance=1）搜索，
列出按近 5 年累计收益率排序的 Top 10 每周调仓模型。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CURRENT_ORDER,
    DEFAULT_PARAMS,
    OUT_DIR,
    PHASE_B_PER_ORDER,
    TOP_ORDERS,
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    passes_gates,
    robust_score,
    run_windows,
)

SEED = 777


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
    idx = {w: __import__("scripts.optimize_allocation", fromlist=["range_idx"]).range_idx(master_dates, *b)
           for w, b in {"Y1": WINDOW_Y1, "Y2": WINDOW_Y2, "Y3": WINDOW_Y3, "Y5": WINDOW_5Y}.items()}

    # 基线（当前策略，每周调仓）
    base = run_windows(etfs, master_dates, master_execs, idx, CURRENT_ORDER, DEFAULT_PARAMS)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline(weekly):", {w: round(base[w]["cum_return"], 4) for w in ("Y1", "Y2", "Y3", "Y5")})

    # 从 v2 JSON 读取阶段 A（基础参数，4周）的 Top 顺序
    v2 = json.loads((OUT_DIR / "allocation_optimization_v2.json").read_text(encoding="utf-8"))
    phase_a = [m for m in v2["all_models"] if m["id"].startswith("A")]
    gated = [
        m for m in phase_a
        if sum(1 for w in ("Y1", "Y2", "Y3") if m[w]["cum_return"] > 0) >= 2
        and m["Y5"]["cum_return"] >= baseline_5y
    ]
    if not gated:
        gated = phase_a
    top_orders = sorted(gated, key=lambda m: m["score"], reverse=True)[:TOP_ORDERS]
    print("top orders:", len(top_orders))

    rng = np.random.default_rng(SEED)
    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    results: list[dict] = []
    bid = 0
    for tr in top_orders:
        order = tr["order"]
        for _ in range(PHASE_B_PER_ORDER):
            core = int(rng.choice(core_choices))
            rem_options = [r for r in rem_choices if r <= 100 - core]
            params = {
                "core_budget": core,
                "remainder_budget": int(rng.choice(rem_options)),
                "floor": float(rng.choice(floor_choices)),
                "ladder": int(rng.choice(ladder_choices)),
                "max_single": int(rng.choice(max_choices)),
                "rebalance": 1,
            }
            rec = {"id": f"W{bid:04d}", "order": list(order), "params": params}
            rec.update(run_windows(etfs, master_dates, master_execs, idx, order, params))
            rec["score"] = robust_score(rec)
            results.append(rec)
            bid += 1
    print("weekly models evaluated:", len(results))

    candidates = [r for r in results if passes_gates(r, baseline_5y)]
    pool = candidates if candidates else results
    top10 = sorted(pool, key=lambda r: r["Y5"]["cum_return"], reverse=True)[:10]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# v2 补充：每周调仓模型 Top 10（近5年累计收益率排序）",
        "",
        "口径与 v2 完全一致：冠军链信号、核心3、5% 倍数、100 份整手、现金 1.75%、"
        "佣金万2.5最低5元（免印花税/过户费）、稳健年度评分、门槛（至少两年为正且近5年≥基线）。",
        f"搜索：v2 阶段 A Top{TOP_ORDERS} 顺序 × 每顺序 {PHASE_B_PER_ORDER} 组参数（强制每周调仓），"
        f"共 {len(results)} 个模型；通过门槛 {len(candidates)} 个。",
        "",
        "| 排名 | 模型 | 近5年 | 综合分 | Y1 | Y2 | Y3 | IRR | Sharpe | MDD | 换手 | 核心 | 剩余 | 下限 | 档位 | 上限 | 顺序 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, r in enumerate(top10, 1):
        p = r["params"]
        lines.append(
            f"| {rank} | {r['id']} | {r['Y5']['cum_return']:.1%} | {r['score']:.3f} | "
            f"{r['Y1']['cum_return']:.1%} | {r['Y2']['cum_return']:.1%} | {r['Y3']['cum_return']:.1%} | "
            f"{r['Y5']['annual_return_irr']:.1%} | {r['Y5']['sharpe']:.2f} | "
            f"{r['Y5']['max_drawdown']:.1%} | {r['Y5']['turnover']:.2f} | "
            f"{p['core_budget']}% | {p['remainder_budget']}% | {p['floor']:.0%} | "
            f"{p['ladder']} | {p['max_single']}% | {'/'.join(r['order'])} |"
        )
    md_path = OUT_DIR / "allocation_optimization_weekly_top10.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    with (OUT_DIR / "allocation_optimization_weekly_top10.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["排名", "模型ID", "顺序", "核心预算", "剩余预算", "看空下限", "档位变体", "单只上限",
                         "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "最大回撤", "换手", "期末市值"])
        for rank, r in enumerate(top10, 1):
            p = r["params"]
            writer.writerow([rank, r["id"], "/".join(r["order"]), p["core_budget"], p["remainder_budget"],
                             f"{p['floor']:.0%}", p["ladder"], p["max_single"],
                             f"{r['Y1']['cum_return']:.2%}", f"{r['Y2']['cum_return']:.2%}",
                             f"{r['Y3']['cum_return']:.2%}", f"{r['Y5']['cum_return']:.2%}",
                             f"{r['score']:.4f}", f"{r['Y5']['annual_return_irr']:.2%}",
                             f"{r['Y5']['sharpe']:.2f}", f"{r['Y5']['max_drawdown']:.2%}",
                             f"{r['Y5']['turnover']:.3f}", f"{r['Y5']['final_value']:.0f}"])

    json_payload = {
        "schema_version": "weekly-top10-v1",
        "baseline_5y": baseline_5y,
        "top10": [
            {"id": r["id"], "order": r["order"], "params": r["params"],
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]}
            for r in top10
        ],
    }
    (OUT_DIR / "allocation_optimization_weekly_top10.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("saved", md_path)
    for r in top10:
        print(r["id"], f"5y={r['Y5']['cum_return']:.4f}", "score", round(r["score"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
