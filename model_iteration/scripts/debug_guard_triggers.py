# -*- coding: utf-8 -*-
"""打印各窗口下回撤保护的触发次数（诊断用）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
)
from scripts.compare_b0207_risk_guards import B0207_ORDER, B0207_PARAMS, simulate_risk  # noqa: E402


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
    idx = {w: range_idx(master_dates, *b) for w, b in {
        "Y1": WINDOW_Y1, "Y2": WINDOW_Y2, "Y3": WINDOW_Y3, "Y5": WINDOW_5Y,
    }.items()}
    guard = {"type": "drawdown", "threshold": 0.08, "risk_level": 0.5}
    for w in ("Y1", "Y2", "Y3", "Y5"):
        r = simulate_risk(etfs, master_dates, master_execs, *idx[w], B0207_ORDER, B0207_PARAMS, guard=guard)
        print(w, "triggers", r["triggers"], "dates", r.get("trigger_dates", []),
              "cum", round(r["cum_return"], 4), "mdd", round(r["max_drawdown"], 4))
    # 基线在 2026-06-01~2026-08-07 子区间的回撤
    s, e = range_idx(master_dates, "2026-06-01", "2026-08-07")
    base = simulate_risk(etfs, master_dates, master_execs, s, e, B0207_ORDER, B0207_PARAMS, guard=None)
    print("July subwindow baseline cum", round(base["cum_return"], 4), "mdd", round(base["max_drawdown"], 4))
    guard_july = simulate_risk(etfs, master_dates, master_execs, s, e, B0207_ORDER, B0207_PARAMS, guard=guard)
    print("July subwindow guard cum", round(guard_july["cum_return"], 4),
          "mdd", round(guard_july["max_drawdown"], 4), "triggers", guard_july["triggers"],
          "dates", guard_july.get("trigger_dates", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
