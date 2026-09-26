# -*- coding: utf-8 -*-
"""4 个相位冠军模型的全周期（2012-09-14~2026-08-07）模拟验证。

两种相位锚定：
- A：相位 p 的决策周 = 全周期起点 + p + k×4（相对全周期锚定）；
- B：决策周与 5 年窗口寻优时一致（offset_full = (start_5y + p) % 4）。
统一使用 weekly_deploy=False（与原版 B0207 口径一致）。
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
    simulate_realistic,
)


def pct(x) -> str:
    return f"{float(x):.1%}"


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
    full_idx = (0, len(master_dates) - 1)
    start_5y = range_idx(master_dates, *WINDOW_5Y)[0]
    print("full anchors:", len(master_dates), "start_5y anchor:", start_5y,
          master_dates[start_5y])

    data = json.loads((OUT_DIR / "b0207_phase_champions_v2_standard.json").read_text(encoding="utf-8"))
    champions = data["champions"]

    lines = [
        "# 4 相位冠军模型：全周期模拟验证（2012-09-14 ~ 2026-08-07）",
        "",
        "模型 = 各相位在 v2 同一协议下寻优出的冠军（相位0 = B0207）。",
        "锚定A：决策周 = 全周期起点 + 相位 + k×4；锚定B：决策周与5年寻优时完全一致。",
        "口径：weekly_deploy=False、真实费率、100份整手、每周入金2000×n（未开市ETF不计）。",
        "",
        "| 锚定 | 相位 | 模型 | 频率 | 全周期累计收益率 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for anchor_mode, label in (("A", "A(全周期锚定)"), ("B", "B(与5年窗口一致)")):
        for p, ch in enumerate(champions):
            m = ch["model"]
            if anchor_mode == "A":
                offset = p
            else:
                offset = (start_5y + p) % 4
            r = simulate_realistic(etfs, master_dates, master_execs, *full_idx,
                                   m["order"], m["params"], weekly_deploy=False,
                                   phase_offset=offset)
            rows.append({
                "anchor": label, "phase": p, "id": m["id"],
                "freq": m["params"]["rebalance"],
                "cum": r["cum_return"], "irr": r["annual_return_irr"],
                "sharpe": r["sharpe"], "mdd": r["max_drawdown"],
                "avg_pos": r["avg_position"], "turnover": r["turnover"],
                "final": r["final_value"],
            })
            print(label, "phase", p, m["id"], "freq", m["params"]["rebalance"],
                  "cum", round(r["cum_return"], 4), "MDD", round(r["max_drawdown"], 4))
            lines.append(
                f"| {label} | {p} | {m['id']} | {m['params']['rebalance']}周 | "
                f"{pct(r['cum_return'])} | {pct(r['annual_return_irr'])} | {r['sharpe']:.2f} | "
                f"{pct(r['max_drawdown'])} | {pct(r['avg_position'])} | {r['turnover']:.2f} | "
                f"{r['final_value']:.0f} |"
            )
    for anchor_label in ("A(全周期锚定)", "B(与5年窗口一致)"):
        sub = [r for r in rows if r["anchor"] == anchor_label]
        desc = "、".join(f"相位{p}={r['cum']:.1%}" for p, r in enumerate(sub))
        mono = all(sub[i]["cum"] > sub[i + 1]["cum"] for i in range(len(sub) - 1))
        lines.append("")
        lines.append(f"{anchor_label}：{desc}；严格递减：{'是' if mono else '否'}")

    md = OUT_DIR / "b0207_phase_champions_full_period.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "b0207_phase_champions_full_period.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["锚定", "相位", "模型", "频率", "全周期累计收益率", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "期末市值"])
        for r in rows:
            writer.writerow([r["anchor"], r["phase"], r["id"], r["freq"], pct(r["cum"]),
                             pct(r["irr"]), f"{r['sharpe']:.2f}", pct(r["mdd"]),
                             pct(r["avg_pos"]), f"{r['turnover']:.3f}", f"{r['final']:.0f}"])
    payload = {"schema_version": "phase-champions-full-period-v1", "rows": rows}
    (OUT_DIR / "b0207_phase_champions_full_period.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
