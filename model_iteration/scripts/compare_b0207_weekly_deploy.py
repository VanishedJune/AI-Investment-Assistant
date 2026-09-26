# -*- coding: utf-8 -*-
"""B0207 原版 vs 修改版对比：每周入金立即按现有持仓加仓，仍每 4 周决策。

原版：非决策周入金存入现金池，决策周统一按目标权重调仓。
修改版：非决策周入金立即按“现有持仓市值比例”买入加仓；决策周照常按目标权重调仓。
"""

from __future__ import annotations

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
                "ladder": 0, "max_single": 100, "rebalance": 4}
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

    def run(deploy: bool) -> dict:
        rec = {}
        for w in ("Y1", "Y2", "Y3", "Y5"):
            rec[w] = simulate_realistic(etfs, master_dates, master_execs, *idx[w],
                                        B0207_ORDER, B0207_PARAMS, weekly_deploy=deploy)
        rec["score"] = robust_score(rec)
        return rec

    original = run(False)
    modified = run(True)

    lines = [
        "# B0207 规则修改对比：每周入金立即加仓 vs 原版攒现金",
        "",
        "相同点：优先级顺序、核心85%/剩余15%、看空下限0%、档位0、单只上限100%、每 4 周决策一次；"
        "真实账户口径（现金1.75%、100份整手、佣金万2.5最低5元、免印花税/过户费）。",
        "不同点：原版非决策周入金存现金；修改版非决策周入金立即按现有持仓市值比例加仓。",
        "",
        "## 收益与风险对比",
        "",
        "| 方案 | Y1(23-24) | Y2(24-25) | Y3(25-26) | 近5年 | 综合分 | IRR | Sharpe | 最大回撤 | 平均持仓 | 换手 | 累计投入 | 累计盈利 | 期末市值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in (("B0207 原版", original), ("B0207 修改版(每周加仓)", modified)):
        y5 = r["Y5"]
        lines.append(
            f"| {label} | {pct(r['Y1']['cum_return'])} | {pct(r['Y2']['cum_return'])} | "
            f"{pct(r['Y3']['cum_return'])} | {pct(y5['cum_return'])} | {r['score']:.3f} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{pct(y5['avg_position'])} | {y5['turnover']:.2f} | {y5['invested']:.0f} | "
            f"{y5['profit']:.0f} | {y5['final_value']:.0f} |"
        )

    md = OUT / "allocation_optimization_b0207_weekly_deploy.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": "b0207-weekly-deploy-v1",
        "order": B0207_ORDER,
        "params": B0207_PARAMS,
        "original": original,
        "modified": modified,
    }
    (OUT / "allocation_optimization_b0207_weekly_deploy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    for label, r in (("original", original), ("modified", modified)):
        print(label, "5y", round(r["Y5"]["cum_return"], 4), "score", round(r["score"], 4),
              "turnover", round(r["Y5"]["turnover"], 3), "final", round(r["Y5"]["final_value"], 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
