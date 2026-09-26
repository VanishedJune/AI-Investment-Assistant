# -*- coding: utf-8 -*-
"""B0207 主线不变 + 周度看空保护 对比。

规则：
- 主线：每 4 周按 B0207 目标权重调仓一次（不变）；
- 新增周度看空保护：每周分析冠军信号，只有“当前持仓中任一 ETF 冠军仓位=0
  （看空）”时才立即按当时 B0207 目标调仓；否则持仓不动（不因普通信号变化交易）；
- weekly_deploy 变体：非决策周且无看空触发时，是否仍按现有持仓比例加仓。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    robust_score,
    simulate_realistic,
)

B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
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

    variants = [
        ("B0207 原版（仅4周调仓）", False, False, "any"),
        ("B0207 改版（4周决策+每周加仓）", True, False, "any"),
        ("B0207 + 看空保护(任一持仓看空)", False, True, "any"),
        ("B0207 + 看空保护(连续2周)", False, True, "confirm2"),
        ("B0207 + 看空保护(持仓过半看空)", False, True, "majority"),
        ("B0207 + 看空保护(全部持仓看空)", False, True, "all"),
    ]
    results = []
    for label, deploy, guard, mode in variants:
        rec = {"label": label, "weekly_deploy": deploy, "bearish_guard": guard, "guard_mode": mode}
        total_triggers = 0
        for w in ("Y1", "Y2", "Y3", "Y5"):
            stats = {"triggers": 0}
            rec[w] = simulate_realistic(etfs, master_dates, master_execs, *idx[w],
                                        B0207_ORDER, B0207_PARAMS,
                                        weekly_deploy=deploy, bearish_guard=guard,
                                        guard_stats=stats, guard_mode=mode)
            total_triggers += stats["triggers"]
        rec["score"] = robust_score(rec)
        rec["triggers_5y"] = total_triggers
        results.append(rec)
        print(label, "5y", round(rec["Y5"]["cum_return"], 4),
              "score", round(rec["score"], 4), "triggers", total_triggers)

    lines = [
        "# B0207 + 周度看空保护 对比",
        "",
        "主线不变：每 4 周按 B0207（核心3，85/15）目标调仓。",
        "新增约束：每周分析冠军信号，仅当当前持仓中任一 ETF 冠军仓位=0（看空）时",
        "立即按当时 B0207 目标调仓；否则持仓不动。",
        "口径：真实账户（现金1.75%、100份整手、佣金万2.5最低5元、免印花税/过户费）。",
        "",
        "| 方案 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 | 看空触发次数(5y) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        y5 = r["Y5"]
        lines.append(
            f"| {r['label']} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['final_value']:.0f} | "
            f"{r['triggers_5y']} |"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "allocation_optimization_b0207_bearish_guard.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "allocation_optimization_b0207_bearish_guard.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "期末市值", "看空触发次数(5y)"])
        for r in results:
            y5 = r["Y5"]
            writer.writerow([r["label"], pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), pct(y5["avg_position"]),
                             f"{y5['turnover']:.3f}", f"{y5['final_value']:.0f}", r["triggers_5y"]])
    payload = {
        "schema_version": "b0207-bearish-guard-v1",
        "order": B0207_ORDER,
        "params": B0207_PARAMS,
        "results": [
            {"label": r["label"], "weekly_deploy": r["weekly_deploy"], "bearish_guard": r["bearish_guard"],
             "guard_mode": r["guard_mode"],
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")},
             "score": r["score"], "triggers_5y": r["triggers_5y"]}
            for r in results
        ],
    }
    (OUT / "allocation_optimization_b0207_bearish_guard.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
