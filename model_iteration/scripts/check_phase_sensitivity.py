# -*- coding: utf-8 -*-
"""B0207 决策周期相位敏感性检验。

回测决策周 = 窗口起点 + offset + k×4 周；生产周期 = 固定名义日（2026-08-22 起）。
本脚本对比 offset=0/1/2/3 下 B0207 原版与改版的收益/回撤差异，
并列出各相位在 2026 年 6-8 月的实际决策日，判断“相位差”的影响量级。
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
    OUT_DIR,
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

    def run(deploy: bool, offset: int) -> dict:
        rec = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], B0207_ORDER,
                                     B0207_PARAMS, weekly_deploy=deploy, phase_offset=offset)
               for w in ("Y1", "Y2", "Y3", "Y5")}
        rec["score"] = robust_score(rec)
        return rec

    def decision_dates(offset: int) -> list[str]:
        s, e = idx["Y5"]
        return [master_dates[i] for i in range(s + offset, e + 1, 4)]

    rows = []
    for deploy, label in ((False, "B0207 原版"), (True, "B0207 改版")):
        for offset in (0, 1, 2, 3):
            rec = run(deploy, offset)
            dates = decision_dates(offset)
            last_dates = [d for d in dates if d >= "2026-06-01"]
            rows.append({
                "label": label, "offset": offset, **rec,
                "last_dates": last_dates,
            })
            print(label, "offset", offset,
                  "5y", round(rec["Y5"]["cum_return"], 4),
                  "score", round(rec["score"], 4),
                  "MDD", round(rec["Y5"]["max_drawdown"], 4),
                  "dates", last_dates)

    lines = [
        "# B0207 决策周期相位敏感性检验",
        "",
        "回测决策周 = 5年窗口起点 + offset + k×4 周；offset=0 即现有回测（2026年7/8月落在 7/3、7/31）。",
        "生产名义周期固定为 2026-08-22 → 09-19 → 10-17（每28天），可视为另一种相位。",
        "",
        "| 方案 | 相位 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 2026年6月后决策日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        y5 = r["Y5"]
        lines.append(
            f"| {r['label']} | offset={r['offset']} | {pct(r['Y1']['cum_return'])} | "
            f"{pct(r['Y2']['cum_return'])} | {pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | "
            f"{r['score']:.3f} | {pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | "
            f"{pct(y5['max_drawdown'])} | {'、'.join(r['last_dates'])} |"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "b0207_phase_sensitivity.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "b0207_phase_sensitivity.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "相位", "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "MDD", "期末市值"])
        for r in rows:
            y5 = r["Y5"]
            writer.writerow([r["label"], r["offset"], pct(r["Y1"]["cum_return"]),
                             pct(r["Y2"]["cum_return"]), pct(r["Y3"]["cum_return"]),
                             pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), f"{y5['final_value']:.0f}"])
    payload = {
        "schema_version": "b0207-phase-sensitivity-v1",
        "rows": [
            {"label": r["label"], "offset": r["offset"],
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")},
             "score": r["score"], "last_dates": r["last_dates"]}
            for r in rows
        ],
    }
    (OUT_DIR / "b0207_phase_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
