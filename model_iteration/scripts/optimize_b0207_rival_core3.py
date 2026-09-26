# -*- coding: utf-8 -*-
"""核心3 框架下搜索 B0207 的竞争者（创业板第1、稀土前四、创新药自由）。

搜索空间：
- 159915 创业板固定第 1；
- 516150 稀土必须前四（第 2/3/4 位）；
- 159622 创新药不设约束；
- 核心组=前3（core_size=3），与 B0207 一致；
- 阶段A：全枚举 2160 种顺序 × 基础参数 → Top50；
- 阶段B：Top50 × 每顺序 80 组细化参数 → 找 Top1；
- 账户口径与 B0207 原版一致（weekly_deploy=False，每4周决策）。
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
from scripts.optimize_allocation import range_idx  # noqa: E402
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
    simulate_realistic,
)
from scripts.optimize_allocation_core12 import orders_159915_first_rare_top4  # noqa: E402

SEED = 42
BASE_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
               "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
OUT = ROOT.parent / "champion_vs_dca"


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

    def run(order, params, deploy: bool = False) -> dict:
        rec = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params,
                                     weekly_deploy=deploy)
               for w in ("Y1", "Y2", "Y3", "Y5")}
        rec["score"] = robust_score(rec)
        return rec

    base = run(CURRENT_ORDER, DEFAULT_PARAMS)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline 5y:", round(baseline_5y, 4))

    # B0207 参照：原版 + 当前改版
    b0207_orig = run(B0207_ORDER, BASE_PARAMS, deploy=False)
    b0207_mod = run(B0207_ORDER, BASE_PARAMS, deploy=True)
    b0207_orig["model"] = {"id": "B0207", "order": B0207_ORDER, "params": dict(BASE_PARAMS)}
    b0207_mod["model"] = {"id": "B0207_mod", "order": B0207_ORDER,
                          "params": {**BASE_PARAMS, "weekly_deploy": True}}
    print("B0207 orig 5y:", round(b0207_orig["Y5"]["cum_return"], 4),
          "mod 5y:", round(b0207_mod["Y5"]["cum_return"], 4))

    orders = orders_159915_first_rare_top4()
    print("orders:", len(orders))

    # 阶段 A：全枚举 × 基础参数
    phase_a = []
    for oi, order in enumerate(orders):
        rec = {"model": {"id": f"A{oi:04d}", "order": order, "params": dict(BASE_PARAMS)}}
        rec.update(run(order, BASE_PARAMS))
        rec["score"] = robust_score(rec)
        phase_a.append(rec)
    top_orders = sorted(
        (r for r in phase_a if passes_gates(r, baseline_5y)),
        key=lambda r: r["score"], reverse=True,
    )[:TOP_ORDERS]
    if not top_orders:
        top_orders = sorted(phase_a, key=lambda r: r["score"], reverse=True)[:TOP_ORDERS]
    print("phase A done; top score:", round(top_orders[0]["score"], 4))

    # 阶段 B：细化参数搜索
    rng = np.random.default_rng(SEED)
    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    rebal_choices = [2, 3, 4, 6]
    phase_b = []
    bid = 0
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
                "rebalance": int(rng.choice(rebal_choices)),
                "core_size": 3,
            }
            rec = {"model": {"id": f"B{bid:04d}", "order": order, "params": params}}
            rec.update(run(order, params))
            rec["score"] = robust_score(rec)
            phase_b.append(rec)
            bid += 1
    print("phase B done:", len(phase_b))

    candidates = [r for r in phase_b if passes_gates(r, baseline_5y)]
    if not candidates:
        candidates = phase_b
    champion = max(candidates, key=lambda r: r["score"])
    best_5y = max(candidates, key=lambda r: r["Y5"]["cum_return"])
    # B0207 在候选池中的排名
    b_pool = [r for r in candidates if r["model"]["id"] == "B0207"]
    b0207_in_pool = b_pool[0] if b_pool else None
    if b0207_in_pool is None:
        # 显式加入 B0207 原版到候选池（其参数=基础参数，若未被采样）
        b0207_in_pool = {"model": {"id": "B0207", "order": B0207_ORDER, "params": dict(BASE_PARAMS)}}
        b0207_in_pool.update(b0207_orig)
        b0207_in_pool["score"] = robust_score(b0207_in_pool)
        candidates.append(b0207_in_pool)
    score_rank = next((i for i, r in enumerate(sorted(candidates, key=lambda x: -x["score"]), 1)
                       if r["model"]["id"] == b0207_in_pool["model"]["id"]), None)
    y5_rank = next((i for i, r in enumerate(sorted(candidates, key=lambda x: -x["Y5"]["cum_return"]), 1)
                    if r["model"]["id"] == b0207_in_pool["model"]["id"]), None)

    lines = [
        "# 核心3 框架：B0207 竞争者搜索（创业板第1、稀土前四、创新药自由）",
        "",
        f"搜索：阶段A 全枚举 {len(orders)} 顺序 × 基础参数 + 阶段B Top{TOP_ORDERS} 顺序 × "
        f"每顺序 {PHASE_B_PER_ORDER} 组参数 = {len(phase_a) + len(phase_b)} 个模型；"
        f"通过门槛 {len(candidates)} 个（B0207 原版已显式纳入候选池）。",
        "口径：核心3、每周入金 2000×n 进现金（weekly_deploy=False，与 B0207 原版一致）、"
        "每4周决策、真实账户费率。",
        "",
        "## 对比",
        "",
        "| 方案 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    refs = [
        ("B0207 原版（参照）", b0207_orig),
        ("B0207 改版（当前正式，每周加仓）", b0207_mod),
        ("新冠军 Top1（按综合分）", champion),
        ("候选池近5年最高", best_5y),
    ]
    for label, r in refs:
        y5 = r["Y5"]
        lines.append(
            f"| {label} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['final_value']:.0f} |"
        )
    lines += [
        "",
        f"B0207 原版在候选池中的排名：综合分第 {score_rank} 名，近5年收益第 {y5_rank} 名"
        f"（共 {len(candidates)} 个）。",
        "",
        "## 新冠军参数",
        "",
        f"- 模型：{champion['model']['id']}",
        f"- 优先级顺序：{' → '.join(champion['model']['order'])}",
        f"- 参数：核心预算 {champion['model']['params']['core_budget']}% / 剩余预算 "
        f"{champion['model']['params']['remainder_budget']}% / 看空下限 "
        f"{champion['model']['params']['floor']:.0%} / 档位变体 {champion['model']['params']['ladder']} / "
        f"单只上限 {champion['model']['params']['max_single']}% / 再平衡 "
        f"{champion['model']['params']['rebalance']} 周",
        "",
        "## 候选池近5年最高参数",
        "",
        f"- 模型：{best_5y['model']['id']}",
        f"- 优先级顺序：{' → '.join(best_5y['model']['order'])}",
        f"- 参数：核心预算 {best_5y['model']['params']['core_budget']}% / 剩余预算 "
        f"{best_5y['model']['params']['remainder_budget']}% / 看空下限 "
        f"{best_5y['model']['params']['floor']:.0%} / 档位变体 {best_5y['model']['params']['ladder']} / "
        f"单只上限 {best_5y['model']['params']['max_single']}% / 再平衡 "
        f"{best_5y['model']['params']['rebalance']} 周",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "b0207_rival_core3_optimization.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "b0207_rival_core3_optimization.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "顺序", "核心", "剩余", "下限", "档位", "上限", "频率",
                         "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "MDD", "期末市值"])
        for label, r in refs:
            p = r["model"]["params"]
            writer.writerow([label, "/".join(r["model"]["order"]), p["core_budget"],
                             p["remainder_budget"], pct(p["floor"]), p["ladder"], p["max_single"],
                             p["rebalance"], pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(r["Y5"]["cum_return"]),
                             f"{r['score']:.4f}", pct(r["Y5"]["annual_return_irr"]),
                             f"{r['Y5']['sharpe']:.2f}", pct(r["Y5"]["max_drawdown"]),
                             f"{r['Y5']['final_value']:.0f}"])
    payload = {
        "schema_version": "b0207-rival-core3-v1",
        "orders": len(orders),
        "phase_a": len(phase_a),
        "phase_b": len(phase_b),
        "candidates": len(candidates),
        "b0207_orig": b0207_orig,
        "b0207_mod": b0207_mod,
        "champion": {"model": champion["model"], **{w: champion[w] for w in ("Y1", "Y2", "Y3", "Y5")},
                     "score": champion["score"]},
        "best_5y": {"model": best_5y["model"], **{w: best_5y[w] for w in ("Y1", "Y2", "Y3", "Y5")},
                    "score": best_5y["score"]},
        "b0207_rank": {"score": score_rank, "y5": y5_rank, "pool": len(candidates)},
    }
    (OUT / "b0207_rival_core3_optimization.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md_path)
    print("champion:", champion["model"]["id"], "score", round(champion["score"], 4),
          "5y", round(champion["Y5"]["cum_return"], 4))
    print("b0207 rank: score", score_rank, "y5", y5_rank, "of", len(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
