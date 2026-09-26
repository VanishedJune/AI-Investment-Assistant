"""CH_P166_5 新版本：决策周用 5 日均值；非决策周按 avg5(d_dif1) 双向调仓
（>0 → 现金 50% 加仓；<0 → 持仓 50% 卖出）。

对比：原版 / 5日版(旧非决策规则) / 新分级调仓版 / 智能定投。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.execution import is_decision_anchor  # noqa: E402
from rolling.features import weekly_indicator_frame  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts.features import indicator_frame  # noqa: E402
from scripts.pk_vs_smart_dca import (  # noqa: E402
    _apply_ca,
    _ca_events,
    _first_exec,
    _metrics,
    _trade_cost,
    champion_b0207_same_cash,
    dca_same_cash,
)

WEEKLY = 2000.0
COST_BPS = 10

SPEC = {
    "score": {"terms": [{"feature": "d_slope", "weight": 15.0}, {"feature": "d_dif1", "weight": 1500.0}]},
    "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 1.0}], "min": 0.0, "max": 1.0},
    "gates": [{"feature": "d_slope", "op": ">", "value": 0.0}],
}


def smoothed_features(usable, daily_a, weekly_a, features, window: int = 5):
    """按最近 window 个交易日平均平滑 d_slope/d_dif1/w_dif1 的特征副本。"""
    import pandas as pd

    out = features.copy()
    di = indicator_frame(daily_a).set_index("date")
    wk = weekly_indicator_frame(weekly_a).set_index("date")["w_dif1"]
    daily_idx = daily_a.set_index("date")
    w_daily = wk.reindex(daily_idx.index, method="ffill")
    rows = []
    for s in usable["signal"]:
        vis = di.loc[:s]
        dslope = vis["d_slope"].tail(window).mean() if len(vis) else np.nan
        ddif1 = vis["d_dif1"].tail(window).mean() if len(vis) else np.nan
        wv = w_daily.loc[:s].tail(window).mean() if len(w_daily.loc[:s]) else np.nan
        rows.append({"d_slope": float(dslope), "d_dif1": float(ddif1), "w_dif1": float(wv)})
    sm = pd.DataFrame(rows, index=out.index)
    out["d_slope"] = sm["d_slope"]
    out["d_dif1"] = sm["d_dif1"]
    out["w_dif1"] = sm["w_dif1"]
    return out


def modified_policy(policy: dict) -> dict:
    """旧 5 日版非决策周规则：avg5(d_dif1)<0 且 avg5(w_dif1)<0 时清仓。"""
    import copy

    pol = copy.deepcopy(policy)
    guard = pol.setdefault("bearish_guard", {})
    guard["enabled"] = True
    guard["sell_frac"] = 1.0
    guard["per_episode"] = True
    guard["reset_on_recovery"] = True
    guard["conditions"] = [
        {"feature": "d_dif1", "op": "<", "value": 0.0},
        {"feature": "w_dif1", "op": "<", "value": 0.0},
    ]
    return pol

WINDOWS = [
    ("全周期", None),
    ("2026年3月以来", "2026-03-01"),
    ("最近10年", "2016-08-10"),
    ("最近5年", "2021-08-10"),
    ("最近3年", "2023-08-10"),
    ("最近1年", "2025-08-10"),
]


def graded_same_cash(usable, opens, ca, feat, positions, policy, start_idx, end_idx) -> dict:
    """决策周按 avg5(d_slope) 目标调仓；非决策周 avg5(d_dif1)>0 现金50%加仓，
    avg5(d_dif1)<0 持仓50%卖出。"""
    cycle = int((policy or {}).get("decision_cycle_weeks", 4))
    cash = 0.0
    shares = 0.0
    nav_after: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    events = _ca_events(ca)
    applied = {key: False for key, _ in events}
    first_ex = _first_exec(usable, start_idx, end_idx)
    if first_ex is not None:
        for key, event in events:
            event_date = str(event.get("pay_date") or event.get("ex_date"))
            applied[key] = event_date < first_ex
    for i in range(start_idx, end_idx + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        ex = np.datetime64(ex)
        ex_date = str(ex.astype("datetime64[D]"))
        price = float(opens.loc[ex])
        shares, cash = _apply_ca(shares, cash, events, ex_date, applied)
        nav_before = shares * price + cash
        if nav_after:
            twr_rets.append(nav_before / nav_after[-1] - 1.0)
        cash += WEEKLY
        invested += WEEKLY
        flows.append((i, -WEEKLY))
        value = shares * price + cash
        if is_decision_anchor(i, cycle):
            target = float(positions[i])
            target_shares = target * value / price
            cost = _trade_cost(abs(target_shares - shares) * price)
            cash = value - target_shares * price - cost
            shares = target_shares
        else:
            d5 = float(feat["d_dif1"].iloc[i])
            if d5 > 0 and cash > 0:
                add_cash = cash * 0.5
                fee = _trade_cost(add_cash)
                shares_new = shares + (add_cash - fee) / price
                cash_new = cash - add_cash
                shares, cash = shares_new, cash_new
            elif d5 < 0 and shares > 0:
                sell_gross = shares * price * 0.5
                fee = _trade_cost(sell_gross)
                shares_new = shares * 0.5
                cash_new = cash + sell_gross - fee
                shares, cash = shares_new, cash_new
        value_after = shares * price + cash
        nav_after.append(value_after)
        exposures.append(shares * price / value_after if value_after > 0 else 0.0)
    final = nav_after[-1] if nav_after else 0.0
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    return _metrics(twr_rets, flows, final, invested, avg_pos)


def main() -> int:
    set_workspace("159915")
    cfg = load_etf_config("159915")
    policy = cfg.get("execution_policy") or {}
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data("159915")
    ca = load_ca("159915")
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    signals = usable["signal"].dt.date.astype(str)

    pos_orig = position_path(SPEC, features, len(usable) - 1)
    feat5 = smoothed_features(usable, daily_a, weekly_a, features, window=5)
    pos5 = position_path(SPEC, feat5, len(usable) - 1)
    pol_mod = modified_policy(policy)

    lines = [
        "# CH_P166_5 新分级调仓版对比（决策周 5 日均值；非决策周 avg5(d_dif1) 双向 50% 调仓）",
        "",
        "新规则：非决策周 avg5(d_dif1)>0 → 现金 50% 加仓；avg5(d_dif1)<0 → 持仓 50% 卖出；"
        "决策周仍按 avg5(d_slope)>0 → 满仓/空仓。",
        "",
    ]
    for label, start_date in WINDOWS:
        start = 0 if start_date is None else next(i for i, d in enumerate(signals) if d >= start_date)
        end = len(usable) - 1
        orig = champion_b0207_same_cash(usable, opens, ca, pos_orig, features, policy, start, end)
        v5 = champion_b0207_same_cash(usable, opens, ca, pos5, feat5, pol_mod, start, end)
        graded = graded_same_cash(usable, opens, ca, feat5, pos5, policy, start, end)
        dc = dca_same_cash(usable, opens, closes, ca, start, end, smart=True)
        lines.append(f"## {label}（{signals.iloc[start]}~{signals.iloc[end]}）")
        lines.append("")
        lines.append("| 方案 | TWR | 年化IRR | Sharpe | 最大回撤 | 胜率 | 平均仓位 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, r in (
            ("原版 CH_P166_5", orig),
            ("5日版（旧非决策规则）", v5),
            ("新分级调仓版（非决策周50%双向）", graded),
            ("智能定投", dc),
        ):
            lines.append(
                f"| {name} | {r['twr']:.2%} | {r['irr_annual']:.2%} | {r['sharpe']:.2f} | "
                f"{r['mdd']:.2%} | {r['win_rate']:.1%} | {r['avg_position']:.1%} | {r['final_nav']:.0f} |"
            )
        lines.append("")
    out = ROOT / "logs" / "对比_CH_P166_5_分级调仓版_20260829.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
