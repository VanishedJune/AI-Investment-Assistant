# -*- coding: utf-8 -*-
"""提取 1周/2周/3周/4周 再平衡 TOP1 的详细对比。"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "champion_vs_dca"


def pct(x) -> str:
    return f"{float(x):.1%}"


def main() -> int:
    v2 = json.loads((OUT / "allocation_optimization_v2.json").read_text(encoding="utf-8"))
    weekly = json.loads((OUT / "allocation_optimization_weekly_top10.json").read_text(encoding="utf-8"))
    by_id = {m["id"]: m for m in v2["all_models"]}

    top = {
        "1周": next(r for r in weekly["top10"] if r["id"] == "W1799"),
        "2周": by_id["B2590"],
        "3周": by_id["B1406"],
        "4周": v2["champion"],
    }

    lines = ["# 1/2/3/4 周再平衡 TOP1 详细对比", ""]
    lines.append("## 一、收益与风险")
    lines.append("")
    lines.append("| 频率 | 模型 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | 年化IRR | Sharpe | 最大回撤 | 平均持仓 | 换手 | 累计投入 | 累计盈利 | 期末市值 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for freq, r in top.items():
        y5 = r["Y5"]
        lines.append(
            f"| {freq} | {r['id']} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['invested']:.0f} | "
            f"{y5['profit']:.0f} | {y5['final_value']:.0f} |"
        )
    lines.append("")
    lines.append("## 二、参数与优先级顺序")
    lines.append("")
    lines.append("| 频率 | 模型 | 核心预算 | 剩余预算 | 看空下限 | 档位变体 | 单只上限 | 再平衡 | 优先级顺序 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for freq, r in top.items():
        p = r["params"]
        lines.append(
            f"| {freq} | {r['id']} | {p['core_budget']}% | {p['remainder_budget']}% | "
            f"{p['floor']:.0%} | {p['ladder']} | {p['max_single']}% | "
            f"{p['rebalance']}周 | {'/'.join(r['order'])} |"
        )

    md = OUT / "allocation_optimization_top1_by_freq.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print("saved", md)
    for freq, r in top.items():
        print(freq, r["id"], "5y", round(r["Y5"]["cum_return"], 4), "score", round(r["score"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
