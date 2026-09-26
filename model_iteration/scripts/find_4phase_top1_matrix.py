# -*- coding: utf-8 -*-
"""4 相位 TOP1 + 4×4 矩阵。

约束：159915 创业板第1；518600 黄金与 516150 稀土均在前四（第2~4位）。
锚点：P0=2021-08-13（B0207），P1=+1周，P2=+2周，P3=+3周。
频率固定4周；资金流=每周入金2000×n、非决策周现金（weekly_deploy=False）；
主指标=近5年累计收益率（总价值/总投入-1）。
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
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    BASE_PARAMS,
    OUT_DIR,
    PHASE_B_SEED,
    WINDOW_5Y,
    simulate_realistic,
)

PHASE_B_PER_ORDER = 80
TOP_ORDERS = 50
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}


def pct(x) -> str:
    return f"{float(x):.1%}"


def build_orders() -> list[list[str]]:
    rest = [c for c in CODES if c not in ("159915", "518600", "516150")]
    orders: set[tuple[str, ...]] = set()
    for slot_a, slot_b in itertools.permutations((1, 2, 3), 2):
        base: list[str | None] = [None] * 8
        base[0] = "159915"
        base[slot_a] = "518600"
        base[slot_b] = "516150"
        free = [i for i in range(8) if base[i] is None]
        for rest_order in itertools.permutations(rest):
            order = list(base)
            for idx, code in zip(free, rest_order):
                order[idx] = code
            orders.add(tuple(order))
    return [list(o) for o in sorted(orders)]


def main() -> int:
    etfs = {c: load_etf(c) for c in CODES}
    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    idx5 = range_idx(master_dates, *WINDOW_5Y)

    def run5(order, params, offset: int) -> dict:
        return simulate_realistic(etfs, master_dates, master_execs, *idx5, order, params,
                                  weekly_deploy=False, phase_offset=offset)

    orders = build_orders()
    print("orders:", len(orders))

    champions = {0: {"id": "B0207", "order": list(B0207_ORDER), "params": dict(B0207_PARAMS)}}
    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]

    for p in (1, 2, 3):
        print("=== optimizing P", p, "===")
        phase_a = []
        for oi, order in enumerate(orders):
            score = run5(order, BASE_PARAMS, p)["cum_return"]
            phase_a.append({"order": order, "score": score})
        top = sorted(phase_a, key=lambda r: -r["score"])[:TOP_ORDERS]
        print("phase A top score:", round(top[0]["score"], 4))
        rng = np.random.default_rng(PHASE_B_SEED)
        phase_b = []
        for tr in top:
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
                    "rebalance": 4,
                    "core_size": 3,
                }
                score = run5(order, params, p)["cum_return"]
                phase_b.append({"order": order, "params": params, "score": score})
        best = max(phase_b, key=lambda r: r["score"])
        champions[p] = {"id": f"P{p}_TOP1", "order": best["order"], "params": best["params"]}
        print("P", p, "champion score:", round(best["score"], 4),
              "params:", best["params"])

    # 4×4 矩阵：行=模型，列=相位
    matrix = {}
    for i in range(4):
        matrix[i] = {}
        for j in range(4):
            m = run5(champions[i]["order"], champions[i]["params"], j)
            matrix[i][j] = m
            print(f"model P{i} @ phase P{j}: cum={m['cum_return']:.4f} "
                  f"irr={m['annual_return_irr']:.4f} mdd={m['max_drawdown']:.4f}")

    row_mean = {i: float(np.mean([matrix[i][j]["cum_return"] for j in range(4)])) for i in range(4)}
    col_mean = {j: float(np.mean([matrix[i][j]["cum_return"] for i in range(4)])) for j in range(4)}
    diag_ok = {i: matrix[i][i]["cum_return"] >= max(matrix[i][j]["cum_return"] for j in range(4))
               for i in range(4)}

    lines = [
        "# 4 相位 TOP1 + 4×4 矩阵（近5年累计收益率）",
        "",
        "约束：创业板第1，黄金与稀土均前四；频率固定4周；每周入金2000×n、非决策周现金；",
        "主指标=近5年累计收益率（总价值/总投入-1）；P0=B0207（现有），P1/P2/P3各自寻优。",
        "",
        "## 各相位 TOP1 模型",
        "",
        "| 相位 | 模型 | 5年累计收益率 | 核心/剩余 | 下限 | 档位 | 上限 | 顺序 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for i in range(4):
        ch = champions[i]
        p = ch["params"]
        score = matrix[i][i]["cum_return"]
        lines.append(
            f"| P{i} | {ch['id']} | {pct(score)} | {p['core_budget']}%/{p['remainder_budget']}% | "
            f"{p['floor']:.0%} | {p['ladder']} | {p['max_single']}% | {'/'.join(ch['order'])} |"
        )
    lines += ["", "## 4×4 矩阵（单元格=近5年累计收益率）", "",
              "| 模型\\相位 | P0 | P1 | P2 | P3 | 行均值 | 对角最优 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for i in range(4):
        cells = [pct(matrix[i][j]["cum_return"]) for j in range(4)]
        lines.append(f"| {champions[i]['id']} | " + " | ".join(cells) +
                     f" | {pct(row_mean[i])} | {'是' if diag_ok[i] else '否'} |")
    lines.append("| 列均值 | " + " | ".join(pct(col_mean[j]) for j in range(4)) + " | - | - |")
    lines += ["", "## 说明", "",
              "- 行均值=该模型在4个相位下的平均收益（相位敏感度）；",
              "- 列均值=该相位下4个模型平均收益（相位整体好坏）；",
              "- 对角最优=模型在自己相位上是否强于其他相位。"]
    md = OUT_DIR / "b0207_4phase_top1_matrix.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "b0207_4phase_top1_matrix.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["模型", "相位", "累计收益率", "TWR", "IRR", "Sharpe", "MDD", "平均持仓", "期末总价值"])
        for i in range(4):
            for j in range(4):
                m = matrix[i][j]
                writer.writerow([champions[i]["id"], f"P{j}", pct(m["cum_return"]), pct(m["twr"]),
                                 pct(m["annual_return_irr"]), f"{m['sharpe']:.2f}",
                                 pct(m["max_drawdown"]), pct(m["avg_position"]),
                                 f"{m['final_value']:.0f}"])
    payload = {
        "schema_version": "4phase-top1-matrix-v1",
        "champions": {str(i): {"id": champions[i]["id"], "order": champions[i]["order"],
                               "params": champions[i]["params"]} for i in range(4)},
        "matrix": {str(i): {str(j): matrix[i][j] for j in range(4)} for i in range(4)},
        "row_mean": row_mean, "col_mean": col_mean, "diag_ok": diag_ok,
    }
    (OUT_DIR / "b0207_4phase_top1_matrix.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
