"""当前冠军 B0418 vs 之前冠军 B0207：近五年/近三年/近一年 真实账户收益率。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    simulate_realistic,
)

# B0418（当前生产口径已部署：core80/rem20/ladder3/max60/4周）
B0418_ORDER = ["159915", "518600", "515220", "516150", "159941", "512690", "159622", "512800"]
B0418_PARAMS = {"core_budget": 80, "remainder_budget": 20, "floor": 0.0,
                "ladder": 3, "max_single": 60, "rebalance": 4}

# 之前冠军 B0207（旧顺序：煤炭第 6；生产参数 85/15/ladder0/max100）
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "515220", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4}

WINDOWS = [
    ("近五年", WINDOW_5Y),
    ("近三年", ("2023-08-10", "2026-08-07")),
    ("近一年", WINDOW_Y3),
]


def main() -> int:
    etfs = {}
    for code in CODES:
        etfs[code] = load_etf(code)
    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    idx = {label: range_idx(master_dates, *bounds) for label, bounds in WINDOWS}

    def run(order, params):
        out = {}
        for label in ("近五年", "近三年", "近一年"):
            r = simulate_realistic(
                etfs, master_dates, master_execs, *idx[label], order, params,
                weekly_deploy=True, bearish_guard=True,
            )
            out[label] = r
        return out

    b0418 = run(B0418_ORDER, B0418_PARAMS)
    b0207 = run(B0207_ORDER, B0207_PARAMS)

    lines = [
        "# 当前冠军 B0418 vs 之前冠军 B0207：真实账户收益率对比",
        "",
        "口径：每周加仓 + 周度看空保护；现金年化 1.75% 计息；100 份整手；佣金万2.5最低5元；"
        "raw_open 成交；显式 CA 账本。收益率为真实资金口径（期末市值/累计投入-1，含现金与成本）。",
        "",
        f"- B0418 顺序：{' → '.join(B0418_ORDER)}；参数 core80/rem20/ladder3/max60/4周",
        f"- B0207 顺序：{' → '.join(B0207_ORDER)}；参数 core85/rem15/ladder0/max100/4周",
        "",
        "| 窗口 | 方案 | 真实收益率 | 累计投入 | 累计盈利 | 期末市值 | 年化IRR | Sharpe | 最大回撤 | 平均持仓 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("近五年", "近三年", "近一年"):
        for name, r in (("B0418（当前）", b0418[label]), ("B0207（之前）", b0207[label])):
            lines.append(
                f"| {label} | {name} | {r['cum_return']:.2%} | {r['invested']:.0f} | "
                f"{r['profit']:.0f} | {r['final_value']:.0f} | {r['annual_return_irr']:.2%} | "
                f"{r['sharpe']:.2f} | {r['max_drawdown']:.2%} | {r['avg_position']:.1%} |"
            )
    out = ROOT / "logs" / "对比_B0418_vs_B0207_真实收益_20260829.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
