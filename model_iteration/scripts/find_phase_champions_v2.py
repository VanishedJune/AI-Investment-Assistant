# -*- coding: utf-8 -*-
"""按 v2 完全同一协议，为 4 个相位各找冠军。

协议（与产出 B0207 的 v2 完全一致，只增加 phase_offset）：
- 顺序空间：159915 前二、159622 前四、516150 前四（1440 种）；
- 阶段A：全枚举 × 基础参数（80/20/0%/档位0/100%/4周），评分=0.7×Y3+0.2×Y2+0.1×Y1，
  门槛=至少两年为正 且 近5年≥基线，取 Top50；
- 阶段B：种子 2026，Top50 × 每顺序 80 组细化参数（核心70-85/剩余15-30/下限0-40%/
  档位0-3/上限60-100/频率2-6周），门槛同上，取综合分最高者为该相位冠军；
- 模拟使用 weekly_deploy=False，并施加 phase_offset=p（决策周错开 p 周）。
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
    BASE_PARAMS,
    CURRENT_ORDER,
    DEFAULT_PARAMS,
    OUT_DIR,
    PHASE_B_PER_ORDER,
    PHASE_B_SEED,
    TOP_ORDERS,
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    all_valid_orders,
    passes_gates,
    robust_score,
    simulate_realistic,
)


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

    def run(order, params, phase: int) -> dict:
        rec = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params,
                                     weekly_deploy=False, phase_offset=phase)
               for w in ("Y1", "Y2", "Y3", "Y5")}
        rec["score"] = robust_score(rec)
        return rec

    base = run(CURRENT_ORDER, DEFAULT_PARAMS, 0)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline 5y:", round(baseline_5y, 4))

    orders = all_valid_orders()
    print("orders:", len(orders))
    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    rebal_choices = [2, 3, 4, 6]

    champions = []
    for phase in range(4):
        print("=== phase", phase, "===")
        phase_a = []
        for oi, order in enumerate(orders):
            rec = {"model": {"id": f"A{phase}_{oi:04d}", "order": order, "params": dict(BASE_PARAMS)}}
            rec.update(run(order, BASE_PARAMS, phase))
            phase_a.append(rec)
        top_orders = sorted(
            (r for r in phase_a if passes_gates(r, baseline_5y)),
            key=lambda r: r["score"], reverse=True,
        )[:TOP_ORDERS]
        if not top_orders:
            top_orders = sorted(phase_a, key=lambda r: r["score"], reverse=True)[:TOP_ORDERS]
        print("phase A done; top score:", round(top_orders[0]["score"], 4) if top_orders else None)

        rng = np.random.default_rng(PHASE_B_SEED)
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
                    "rebalance": int(rng.choice(rebal_choices)),
                    "core_size": 3,
                }
                rec = {"model": {"id": f"B{phase}_{len(phase_b):04d}", "order": order, "params": params}}
                rec.update(run(order, params, phase))
                phase_b.append(rec)
        candidates = [r for r in phase_b if passes_gates(r, baseline_5y)]
        if not candidates:
            candidates = phase_b
        champion = max(candidates, key=lambda r: r["score"])
        champions.append(champion)
        print("phase", phase, "champion:", champion["model"]["id"],
              "score", round(champion["score"], 4),
              "Y1", round(champion["Y1"]["cum_return"], 4),
              "Y2", round(champion["Y2"]["cum_return"], 4),
              "Y3", round(champion["Y3"]["cum_return"], 4),
              "Y5", round(champion["Y5"]["cum_return"], 4))

    # 校验：相位0冠军是否与 v2 全局冠军 B0207 相同
    v2 = json.loads((OUT_DIR / "allocation_optimization_v2.json").read_text(encoding="utf-8"))
    b0207 = v2["champion"]
    c0 = champions[0]["model"]
    same = c0["order"] == b0207["order"] and c0["params"] == b0207["params"]
    print("phase0 == v2 B0207 config:", same)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 4 相位冠军（v2 同一协议）",
        "",
        "协议与产出 B0207 的 v2 完全一致（1440 顺序、非重叠年度评分 70/20/10、门槛、Top50×80、种子2026），",
        "仅对相位 1/2/3 施加 phase_offset 错开决策周。",
        f"相位0冠军是否等于 v2 全局冠军 B0207 配置：{'是' if same else '否'}",
        "",
        "## 各相位冠军",
        "",
        "| 相位 | 模型 | 顺序 | 核心/剩余 | 下限 | 档位 | 上限 | 频率 | Y1 | Y2 | Y3 | 近5年 | 综合分 | 门槛 | IRR | Sharpe | MDD | 期末市值 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p, ch in enumerate(champions):
        m = ch["model"]
        prm = m["params"]
        gate = "是" if passes_gates(ch, baseline_5y) else "否"
        y5 = ch["Y5"]
        lines.append(
            f"| {p} | {m['id']} | {'/'.join(m['order'])} | "
            f"{prm['core_budget']}%/{prm['remainder_budget']}% | {prm['floor']:.0%} | "
            f"{prm['ladder']} | {prm['max_single']}% | {prm['rebalance']}周 | "
            f"{pct(ch['Y1']['cum_return'])} | {pct(ch['Y2']['cum_return'])} | "
            f"{pct(ch['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {ch['score']:.3f} | {gate} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{y5['final_value']:.0f} |"
        )
    md = OUT_DIR / "b0207_phase_champions_v2_standard.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "b0207_phase_champions_v2_standard.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["相位", "模型", "顺序", "核心", "剩余", "下限", "档位", "上限", "频率",
                         "Y1", "Y2", "Y3", "近5年", "综合分", "门槛", "IRR", "Sharpe", "MDD", "期末市值"])
        for p, ch in enumerate(champions):
            m = ch["model"]
            prm = m["params"]
            y5 = ch["Y5"]
            writer.writerow([p, m["id"], "/".join(m["order"]), prm["core_budget"],
                             prm["remainder_budget"], f"{prm['floor']:.0%}", prm["ladder"],
                             prm["max_single"], prm["rebalance"], pct(ch["Y1"]["cum_return"]),
                             pct(ch["Y2"]["cum_return"]), pct(ch["Y3"]["cum_return"]),
                             pct(y5["cum_return"]), f"{ch['score']:.4f}",
                             "是" if passes_gates(ch, baseline_5y) else "否",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), f"{y5['final_value']:.0f}"])
    payload = {
        "schema_version": "phase-champions-v2-standard-v1",
        "baseline_5y": baseline_5y,
        "phase0_equals_v2_b0207": same,
        "champions": [
            {"phase": p, "model": ch["model"], **{w: ch[w] for w in ("Y1", "Y2", "Y3", "Y5")},
             "score": ch["score"], "gate": bool(passes_gates(ch, baseline_5y))}
            for p, ch in enumerate(champions)
        ],
        "v2_b0207": b0207,
    }
    (OUT_DIR / "b0207_phase_champions_v2_standard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
