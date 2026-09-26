"""B0207 收益差距归因：219.9%(08-11) vs 53.83%(今日)。

在相同窗口/参数/顺序/执行下，只替换冠军规则与特征平滑，量化各因素。
"""

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
from rolling.ledger import set_workspace  # noqa: E402
from scripts.champion_chain_vs_dca import _ca_events  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import WINDOW_5Y, WINDOW_Y3, simulate_realistic  # noqa: E402
from scripts.frozen_state import load_oldrules_state  # noqa: E402

ORDER = ["159915", "518600", "516150", "159622", "159941", "515220", "512800", "512690"]
PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
          "ladder": 0, "max_single": 100, "rebalance": 4}


def load_frames(code: str, smooth: bool):
    set_workspace(code)
    if smooth:
        return load_aligned_data(code)
    ca = load_ca(code)
    daily_a, weekly_a = analysis_frames(code, ca)
    daily = load_daily(code)
    anchors = build_anchors(code, daily_a, weekly_a)
    feat_raw = build_extended_features(anchors, code, ca)
    mask = ~feat_raw[ALL_FEATURES].isna().any(axis=1)
    usable = anchors[mask].reset_index(drop=True)
    features = feat_raw[mask].reset_index(drop=True)
    return daily_a, weekly_a, daily, anchors, usable, features


def old_spec(code: str) -> dict:
    return load_oldrules_state(code).get("champion", {}).get("spec") or {}


def new_spec(code: str) -> dict:
    set_workspace(code)
    from rolling.ledger import load_state
    return load_state().get("champion", {}).get("spec") or {}


def make_etf(code: str, spec: dict, smooth: bool) -> dict:
    daily_a, weekly_a, daily, anchors, usable, features = load_frames(code, smooth)
    ca = load_ca(code)
    pos = position_path(spec, features, len(usable) - 1)
    by_date = {usable.iloc[i]["signal"].date().isoformat(): float(pos[i]) for i in range(len(usable))}
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    price_map = {}
    for i in range(len(usable)):
        ex = usable.iloc[i]["exec"]
        if ex is not None:
            exs = str(np.datetime64(ex).astype("datetime64[D]"))
            if exs in opens.index:
                price_map[exs] = float(opens.loc[exs])
    return {
        "by_date": by_date,
        "opens": opens,
        "closes": closes,
        "ca_events": _ca_events(ca),
        "start_date": min(by_date.keys()),
        "price_map": price_map,
    }


def run_variant(specs: dict, smooth: bool, label: str, etfs_cache: dict) -> None:
    etfs = {}
    for code in ORDER:
        key = (label, code, smooth)
        if key not in etfs_cache:
            etfs_cache[key] = make_etf(code, specs[code], smooth)
        etfs[code] = etfs_cache[key]
    from rolling.ledger import set_workspace as sw
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
    print(f"## {label}")
    for w in ("5y", "3y", "1y"):
        r = simulate_realistic(etfs, master_dates, master_execs, *idx[w], ORDER, PARAMS,
                               weekly_deploy=True, bearish_guard=True)
        print(f"  {w}: 收益率={r['cum_return']:.2%} 平均持仓={r['avg_position']:.1%} IRR={r['annual_return_irr']:.2%} "
              f"Sharpe={r['sharpe']:.2f} MDD={r['max_drawdown']:.2%}")


def main() -> int:
    old_specs = {c: old_spec(c) for c in CODES}
    new_specs = {c: new_spec(c) for c in CODES}
    print("旧冠军:", {c: (old_specs[c].get("score", {}).get("terms") or [{}])[0].get("feature") for c in CODES})
    cache = {}
    run_variant(new_specs, True, "V_new_smooth（最新冠军+平滑，即今日53.83%口径）", cache)
    run_variant(new_specs, False, "V_new_raw（最新冠军+未平滑）", cache)
    run_variant(old_specs, False, "V_old_raw（旧冠军+未平滑）", cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
