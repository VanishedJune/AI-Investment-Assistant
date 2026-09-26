# -*- coding: utf-8 -*-
"""核心组=1 / 核心组=2 的分配优化，并与当前核心3 B0207 对比。

规则：
- 159915 创业板固定排在优先级第 1 位；
- 516150 稀土必须排在前四（第 2/3/4 位）；159622 创新药不设约束；
- core_size=1：核心组=第 1 名（创业板），预算 85%；看空时依次递补
  2-3 名、4-5 名、6-7 名、第 8 名；
- core_size=2：核心组=前 2 名，预算 85%；看空时依次递补 3-4、5-6、7-8 名；
- 剩余 15% 只在核心组活跃时分配给核心组之外的看多 ETF（最多 2 只）；
- 执行口径与当前 B0207 改版一致：每 4 周决策、weekly_deploy=True
  （非决策周入金立即按现有持仓比例加仓）；
- 评价：稳健年度评分 70/20/10、门槛（至少两年为正且近5年≥基线）、真实账户。
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

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402
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

SEED = 42
BASE_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
               "ladder": 0, "max_single": 100, "rebalance": 4}
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]


def orders_159915_first_rare_top4() -> list[list[str]]:
    """159915 第1 + 516150 前四（第2/3/4位），其余（含159622）自由排列。"""
    rest = [c for c in CODES if c not in ("159915", "516150")]
    orders: set[tuple[str, ...]] = set()
    for pos in (1, 2, 3):
        base: list[str | None] = [None] * 8
        base[0] = "159915"
        base[pos] = "516150"
        free = [i for i in range(8) if base[i] is None]
        for p in itertools.permutations(rest):
            order = list(base)
            for idx, code in zip(free, p):
                order[idx] = code
            orders.add(tuple(order))
    return [list(o) for o in sorted(orders)]


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

    def run(order, params, deploy: bool) -> dict:
        rec = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params,
                                     weekly_deploy=deploy)
               for w in ("Y1", "Y2", "Y3", "Y5")}
        rec["score"] = robust_score(rec)
        return rec

    base = run(CURRENT_ORDER, DEFAULT_PARAMS, False)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline 5y:", round(baseline_5y, 4))

    # 当前 B0207 改版（核心3，weekly_deploy=True）作为对比基准
    b0207_ref = run(B0207_ORDER, {**BASE_PARAMS, "core_size": 3}, True)
    # 核心4 V3B0207 参考（同样用 weekly_deploy 口径重算）
    v3_champ = json.loads((OUT_DIR / "allocation_optimization_v3.json").read_text(encoding="utf-8"))["champion"]
    v3_ref = run(v3_champ["order"], {**v3_champ["params"], "core_size": 4}, True)

    orders = orders_159915_first_rare_top4()
    print("orders (159915 first, 516150 top4):", len(orders))

    results = {}
    for core_size in (2, 1):
        print("=== core_size =", core_size, "===")
        params_a = {**BASE_PARAMS, "core_size": core_size}
        phase_a: list[dict] = []
        for oi, order in enumerate(orders):
            rec = {"model": {"id": f"A{core_size}_{oi:04d}", "order": order,
                             "params": dict(params_a), "core_size": core_size}}
            rec.update(run(order, params_a, True))
            rec["score"] = robust_score(rec)
            phase_a.append(rec)
        top_orders = sorted(
            (r for r in phase_a if passes_gates(r, baseline_5y)),
            key=lambda r: r["score"], reverse=True,
        )[:TOP_ORDERS]
        if not top_orders:
            top_orders = sorted(phase_a, key=lambda r: r["score"], reverse=True)[:TOP_ORDERS]
        print("phase A done; top score:", round(top_orders[0]["score"], 4))

        rng = np.random.default_rng(SEED + core_size)
        core_choices = [70, 75, 80, 85]
        rem_choices = [15, 20, 25, 30]
        floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
        ladder_choices = [0, 1, 2, 3]
        max_choices = [60, 70, 80, 100]
        rebal_choices = [2, 3, 4, 6]
        phase_b: list[dict] = []
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
                    "core_size": core_size,
                }
                rec = {"model": {"id": f"B{core_size}_{bid:04d}", "order": order,
                                 "params": params, "core_size": core_size}}
                rec.update(run(order, params, True))
                rec["score"] = robust_score(rec)
                phase_b.append(rec)
                bid += 1
        candidates = [r for r in phase_b if passes_gates(r, baseline_5y)]
        if not candidates:
            candidates = phase_b
        champion = max(candidates, key=lambda r: r["score"])
        in_sample = max(phase_b, key=lambda r: r["Y5"]["cum_return"])
        results[core_size] = {
            "champion": champion,
            "in_sample_best": in_sample,
            "phase_a": phase_a,
            "phase_b": phase_b,
        }
        print("champion:", champion["model"]["id"],
              "5y", round(champion["Y5"]["cum_return"], 4),
              "score", round(champion["score"], 4))

    # ---------- 落盘 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 核心组=1 / 核心组=2 优化 Top1 vs 当前核心3 B0207",
        "",
        "约束：159915 创业板固定优先级第 1；516150 稀土前四；159622 创新药不设约束；"
        "执行口径与当前 B0207 改版一致"
        "（每 4 周决策 + 每周入金立即按现有持仓比例加仓，weekly_deploy=True）。",
        f"基线（当前策略，门槛参照）：近5年 {pct(base['Y5']['cum_return'])}。",
        "",
        "## 对比（真实账户口径）",
        "",
        "| 方案 | 核心组 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ref_rows = [
        ("当前 B0207（核心3）", 3, B0207_ORDER, b0207_ref),
        ("V3B0207（核心4，参考）", 4, v3_champ["order"], v3_ref),
    ]
    for cs in (1, 2):
        ch = results[cs]["champion"]
        ref_rows.append((f"核心{cs} Top1", cs, ch["model"]["order"], ch))
    for label, cs, order, r in ref_rows:
        y5 = r["Y5"]
        lines.append(
            f"| {label} | {cs} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['final_value']:.0f} |"
        )

    for cs in (1, 2):
        ch = results[cs]["champion"]
        p = ch["model"]["params"]
        lines += [
            "",
            f"## 核心{cs} Top1 参数：{ch['model']['id']}",
            "",
            f"- 优先级顺序：{' → '.join(ch['model']['order'])}",
            f"- 参数：核心预算 {p['core_budget']}% / 剩余预算 {p['remainder_budget']}% / "
            f"看空下限 {p['floor']:.0%} / 档位变体 {p['ladder']} / 单只上限 {p['max_single']}% / "
            f"再平衡 {p['rebalance']} 周 / 核心组 {p['core_size']}",
        ]
    lines += ["", "## 在样本内近5年最高（仅参考）", ""]
    for cs in (1, 2):
        ib = results[cs]["in_sample_best"]
        lines.append(
            f"| 核心{cs} in-sample | {ib['model']['id']} | {pct(ib['Y5']['cum_return'])} | "
            f"{ib['score']:.3f} | {'/'.join(ib['model']['order'])} |"
        )
    md_path = OUT_DIR / "allocation_optimization_core1_2_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "schema_version": "core1-2-optimization-v1",
        "baseline_5y": baseline_5y,
        "b0207_ref": {w: b0207_ref[w] for w in ("Y1", "Y2", "Y3", "Y5")},
        "v3_ref": {w: v3_ref[w] for w in ("Y1", "Y2", "Y3", "Y5")},
        "core1": {"champion": results[1]["champion"]["model"],
                  **{w: results[1]["champion"][w] for w in ("Y1", "Y2", "Y3", "Y5")},
                  "score": results[1]["champion"]["score"]},
        "core2": {"champion": results[2]["champion"]["model"],
                  **{w: results[2]["champion"][w] for w in ("Y1", "Y2", "Y3", "Y5")},
                  "score": results[2]["champion"]["score"]},
    }
    (OUT_DIR / "allocation_optimization_core1_2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (OUT_DIR / "allocation_optimization_core1_2.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "核心组", "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe",
                         "MDD", "平均持仓", "换手", "期末市值", "顺序"])
        for label, cs, order, r in ref_rows:
            y5 = r["Y5"]
            writer.writerow([label, cs, pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), pct(y5["avg_position"]),
                             f"{y5['turnover']:.3f}", f"{y5['final_value']:.0f}", "/".join(order)])

    print("saved", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
