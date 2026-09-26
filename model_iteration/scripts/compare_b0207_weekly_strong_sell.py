# -*- coding: utf-8 -*-
"""B0207 + 周度强看空减仓：优化减仓份额 X。

周度强看空定义（仅用于每周保护，4 周决策仍用原 B0207 冠军信号）：
日K DIF 一阶导<0 且 二阶导<0，且 周K DIF 一阶导<0 且 二阶导<0。

动作：非决策周，若当前持仓的 ETF 满足强看空，则卖出其 X% 份额（一次/段，
信号恢复后重置）；X 在 5%~100%（步长5%）共 20 档逐一模拟对比。
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
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    robust_score,
)
from scripts.compare_b0207_risk_guards import (  # noqa: E402
    B0207_ORDER,
    B0207_PARAMS,
    simulate_risk,
)

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

    base = simulate_risk(etfs, master_dates, master_execs, *idx["Y5"],
                         B0207_ORDER, B0207_PARAMS, guard=None)
    print("baseline 5y:", round(base["cum_return"], 4))

    fractions = [round(0.05 * k, 2) for k in range(1, 21)]
    rows = []
    for frac in fractions:
        guard = {"type": "strong_bearish_sell", "sell_frac": frac}
        rec = {"frac": frac}
        for w in ("Y1", "Y2", "Y3", "Y5"):
            rec[w] = simulate_risk(etfs, master_dates, master_execs, *idx[w],
                                   B0207_ORDER, B0207_PARAMS, guard=guard)
        rec["score"] = robust_score(rec)
        rows.append(rec)
        print(f"frac={frac:.0%} 5y={rec['Y5']['cum_return']:.4f} score={rec['score']:.4f} "
              f"MDD={rec['Y5']['max_drawdown']:.4f} triggers={rec['Y5']['triggers']}")

    by_5y = sorted(rows, key=lambda r: -r["Y5"]["cum_return"])
    by_score = sorted(rows, key=lambda r: -r["score"])
    by_mdd = sorted(rows, key=lambda r: -r["Y5"]["max_drawdown"])
    best_5y, best_score, best_mdd = by_5y[0], by_score[0], by_mdd[0]

    lines = [
        "# B0207 + 周度强看空减仓：20 档减仓份额对比",
        "",
        "周度强看空定义：日K DIF 一阶导<0 且 二阶导<0，且 周K DIF 一阶导<0 且 二阶导<0；",
        "4 周决策仍使用原 B0207 冠军信号（不改变）。",
        "动作：非决策周持仓 ETF 满足强看空时，卖出其 X% 份额（同一段只减一次，信号恢复后重置）。",
        f"基准（无保护）近5年：{pct(base['cum_return'])}，MDD {pct(base['max_drawdown'])}。",
        "",
        "| 减仓份额 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 | 触发次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["frac"]):
        y5 = r["Y5"]
        lines.append(
            f"| {pct(r['frac'])} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['final_value']:.0f} | "
            f"{y5['triggers']} |"
        )
    lines += [
        "",
        f"按近5年最优：减仓 {pct(best_5y['frac'])} → {pct(best_5y['Y5']['cum_return'])} "
        f"（MDD {pct(best_5y['Y5']['max_drawdown'])}）",
        f"按综合分最优：减仓 {pct(best_score['frac'])} → 评分 {best_score['score']:.3f} "
        f"（近5年 {pct(best_score['Y5']['cum_return'])}）",
        f"按回撤最优：减仓 {pct(best_mdd['frac'])} → MDD {pct(best_mdd['Y5']['max_drawdown'])} "
        f"（近5年 {pct(best_mdd['Y5']['cum_return'])}）",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "allocation_optimization_b0207_strong_sell.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "allocation_optimization_b0207_strong_sell.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["减仓份额", "Y1", "Y2", "Y3", "近5年", "综合分", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "期末市值", "触发次数"])
        for r in sorted(rows, key=lambda x: x["frac"]):
            y5 = r["Y5"]
            writer.writerow([pct(r["frac"]), pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(y5["cum_return"]), f"{r['score']:.4f}",
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), pct(y5["avg_position"]),
                             f"{y5['turnover']:.3f}", f"{y5['final_value']:.0f}", y5["triggers"]])
    payload = {
        "schema_version": "b0207-strong-sell-v1",
        "baseline": base,
        "best_5y": {"frac": best_5y["frac"], **best_5y["Y5"], "score": best_5y["score"]},
        "best_score": {"frac": best_score["frac"], **best_score["Y5"], "score": best_score["score"]},
        "best_mdd": {"frac": best_mdd["frac"], **best_mdd["Y5"], "score": best_mdd["score"]},
        "results": [
            {"frac": r["frac"], **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]}
            for r in rows
        ],
    }
    (OUT / "allocation_optimization_b0207_strong_sell.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
