"""回退后验证：M0207（investment_priority.json 顺序/参数）+ 旧冠军 + 未平滑特征 的真实收益率。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.calendar import build_anchors, load_daily  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.features import ALL_FEATURES, analysis_frames, build_extended_features  # noqa: E402
from rolling.ledger import load_state, set_workspace  # noqa: E402
from scripts.champion_chain_vs_dca import _ca_events  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import WINDOW_5Y, WINDOW_Y3, simulate_realistic  # noqa: E402


def main() -> int:
    cfg = json.loads((ROOT / "configs" / "investment_priority.json").read_text(encoding="utf-8"))
    order = cfg["priority"]
    params = {
        "core_budget": cfg["core_budget_pct"],
        "remainder_budget": cfg["remainder_budget_pct"],
        "floor": 0.0,
        "ladder": 0,
        "max_single": cfg["max_single_pct"],
        "rebalance": 4,
    }
    etfs = {}
    for code in order:
        set_workspace(code)
        state = load_state()
        spec = state["champion"]["spec"]
        ca = load_ca(code)
        daily_a, weekly_a = analysis_frames(code, ca)
        daily = load_daily(code)
        anchors = build_anchors(code, daily_a, weekly_a)
        feat_raw = build_extended_features(anchors, code, ca)
        mask = ~feat_raw[ALL_FEATURES].isna().any(axis=1)
        usable = anchors[mask].reset_index(drop=True)
        features = feat_raw[mask].reset_index(drop=True)
        pos = position_path(spec, features, len(usable) - 1)
        by_date = {usable.iloc[i]["signal"].date().isoformat(): float(pos[i]) for i in range(len(usable))}
        opens = daily_a.set_index("date")["raw_open"]
        price_map = {}
        for i in range(len(usable)):
            ex = usable.iloc[i]["exec"]
            if ex is not None:
                exs = str(np.datetime64(ex).astype("datetime64[D]"))
                if exs in opens.index:
                    price_map[exs] = float(opens.loc[exs])
        etfs[code] = {
            "by_date": by_date,
            "opens": opens,
            "closes": daily_a.set_index("date")["analysis_close"],
            "ca_events": _ca_events(ca),
            "start_date": min(by_date.keys()),
            "price_map": price_map,
        }
    sw = set_workspace
    sw("159915")
    _, _, _, _, usable_m, _ = load_aligned_data("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    idx = {w: range_idx(master_dates, *b) for w, b in {
        "5y": WINDOW_5Y, "3y": ("2023-08-10", "2026-08-07"), "1y": WINDOW_Y3,
    }.items()}
    print("M0207 顺序：", " → ".join(order), "| 参数：", params)
    for w in ("5y", "3y", "1y"):
        r = simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params,
                               weekly_deploy=True, bearish_guard=True)
        print(f"  {w}: 真实收益率={r['cum_return']:.2%} 平均持仓={r['avg_position']:.1%} "
              f"IRR={r['annual_return_irr']:.2%} Sharpe={r['sharpe']:.2f} MDD={r['max_drawdown']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
