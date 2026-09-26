# -*- coding: utf-8 -*-
"""从 v2 模型表 + 每周补充搜索中提取 每周/2周/3周 再平衡的 Top 5。

统一按近5年累计收益率排序，表格包含 Y1/Y2/Y3/近5年。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "champion_vs_dca"
SRC = OUT / "allocation_optimization_v2_models.csv"

# 列索引：0模型ID 1顺序 2核心 3剩余 4下限 5档位 6上限 7频率
# 8Y1 9Y2 10Y3 11近5年 12综合分 13门槛 14IRR 15Sharpe 16MDD 17持仓 18换手 19市值


def load_rows():
    with SRC.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:]


def top5(body, freq: str):
    pool = [r for r in body if r[7] == freq and r[13] == "是"]
    if not pool:
        pool = [r for r in body if r[7] == freq]
    pool.sort(key=lambda r: float(r[11].rstrip("%")) / 100.0, reverse=True)
    return pool[:5]


def pct(x) -> str:
    value = float(x.rstrip("%")) / 100.0 if isinstance(x, str) else float(x)
    return f"{value:.1%}"


def main() -> int:
    body = load_rows()
    weekly_json = json.loads((OUT / "allocation_optimization_weekly_top10.json").read_text(encoding="utf-8"))
    weekly = sorted(weekly_json["top10"], key=lambda r: r["Y5"]["cum_return"], reverse=True)[:5]
    lines = ["# 再平衡频率 Top 5 对比（按近5年累计收益率排序）", ""]

    def table(title: str, items: list) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| 排名 | 模型 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | 最大回撤 | 核心/剩余 | 看空下限 | 优先级顺序 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    def row(i: int, rid: str, params: dict, order: list[str], y: dict, score: float) -> None:
        lines.append(
            f"| {i} | {rid} | {pct(y['Y1']['cum_return'])} | {pct(y['Y2']['cum_return'])} | "
            f"{pct(y['Y3']['cum_return'])} | {pct(y['Y5']['cum_return'])} | "
            f"{score:.3f} | {pct(y['Y5']['annual_return_irr'])} | "
            f"{y['Y5']['sharpe']:.2f} | {pct(y['Y5']['max_drawdown'])} | "
            f"{params['core_budget']}%/{params['remainder_budget']}% | "
            f"{params['floor']:.0%} | {'/'.join(order)} |"
        )

    table("每周调仓 Top 5", weekly)
    for i, r in enumerate(weekly, 1):
        row(i, r["id"], r["params"], r["order"], {w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, r["score"])
    lines.append("")

    for freq, label in (("2", "2周调仓"), ("3", "3周调仓")):
        table(f"{label} Top 5", [])
        for i, r in enumerate(top5(body, freq), 1):
            params = {
                "core_budget": r[2], "remainder_budget": r[3], "floor": float(r[4].rstrip("%")) / 100.0,
            }
            y = {
                "Y1": {"cum_return": float(r[8].rstrip("%")) / 100.0},
                "Y2": {"cum_return": float(r[9].rstrip("%")) / 100.0},
                "Y3": {"cum_return": float(r[10].rstrip("%")) / 100.0},
                "Y5": {
                    "cum_return": float(r[11].rstrip("%")) / 100.0,
                    "annual_return_irr": float(r[14].rstrip("%")) / 100.0,
                    "sharpe": float(r[15]),
                    "max_drawdown": float(r[16].rstrip("%")) / 100.0,
                    "score": float(r[12]),
                },
            }
            row(i, r[0], params, r[1].split("/"), y, float(r[12]))
        lines.append("")

    md = OUT / "allocation_optimization_rebalance_top5.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
