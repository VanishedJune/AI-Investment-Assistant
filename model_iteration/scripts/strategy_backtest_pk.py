# -*- coding: utf-8 -*-
"""投资策略 vs 全仓创业板智能定投 PK 回测。

按投资策略.md：每周用各 ETF「当时生效冠军模型」的目标仓位作为多空信号与仓位缩放，
调用 investment_priority.allocate 得到组合权重，周频再平衡（raw_open 成交、10bp 双边成本、
CA-aware 份额/分红），与 159915 智能定投做同现金流 PK。

用法:
    python -m scripts.strategy_backtest_pk --start 2023-08-07 [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CODES = ["159915", "512010", "159622", "516150", "159941", "512690", "512800", "518600"]
COST_BPS = 10


def _champion_timeline(etf: str, usable: pd.DataFrame):
    """从 state_history 快照重建 PIT 冠军时间线：返回 [(eff_start, eff_end_exclusive, spec)]。"""
    sh_dir = ROOT / f"etf_{etf}" / "weekly_rolling" / "state_history"
    snap_by_anchor: dict[int, dict] = {}
    for p in sorted(sh_dir.glob("*.json")):
        snap = json.loads(p.read_text(encoding="utf-8"))
        anchor = int(snap.get("latest_anchor_index", -1))
        if anchor >= 0:
            snap_by_anchor[anchor] = snap  # 后写覆盖：取该锚点最终快照
    segments: list[tuple[int, int, dict]] = []
    current: tuple[int, dict] | None = None  # (eff, spec)
    for anchor in sorted(snap_by_anchor):
        ch = snap_by_anchor[anchor].get("champion") or {}
        eff = int(ch.get("effective_anchor", 0))
        spec = ch.get("spec") or {}
        if current is None:
            current = (eff, spec)
            continue
        cur_eff, cur_spec = current
        if spec != cur_spec:
            segments.append((cur_eff, eff - 1, cur_spec))
            current = (eff, spec)
    if current is not None:
        segments.append((current[0], len(usable) - 1, current[1]))
    return segments, usable


def _load_etf(etf: str):
    from rolling.ca import load_ca
    from rolling.challenge import position_path
    from rolling.data import load_aligned_data
    from rolling.ledger import set_workspace

    set_workspace(etf)
    ca = load_ca(etf)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    fwd = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    segments, _ = _champion_timeline(etf, usable)
    pos = np.zeros(len(usable))
    for s, e, spec in segments:
        if e < s:
            continue
        seg = position_path(spec, features, e, start_idx=s, fwd_arrays=fwd)
        pos[s : e + 1] = seg[s : e + 1]
    return {
        "etf": etf,
        "ca": ca,
        "usable": usable,
        "daily": daily,
        "pos": pos,
        "date_to_idx": {pd.Timestamp(d): i for i, d in enumerate(usable["signal"])},
    }


def _apply_ca(etf_info, last_date, cur_date, cash):
    """在 [last_date, cur_date) 内应用拆分/分红（拆分调份额，分红进现金）。"""
    events = etf_info["ca"].get("events", [])
    for e in events:
        if e["type"] == "split":
            ex = pd.Timestamp(e["ex_date"])
            if last_date < ex <= cur_date:
                etf_info["shares"] *= float(e["ratio"])
        elif e["type"] == "cash_dividend":
            ent = pd.Timestamp(e.get("entitlement_date") or e["ex_date"])
            pay = pd.Timestamp(e.get("pay_date") or e["ex_date"])
            if last_date < ent <= cur_date:
                etf_info["eligible"] = etf_info["shares"] * float(e["dps"])
            if last_date < pay <= cur_date and etf_info.get("eligible"):
                cash += etf_info.pop("eligible", 0.0)
    return cash


def _simulate(etfs: dict, dates: list, weights_by_date: dict, start_anchor: int, deposit: float):
    """周频再平衡组合模拟。返回指标 dict。"""
    for info in etfs.values():
        info["shares"] = 0.0
        info["cash"] = 0.0
        info["eligible"] = 0.0
    cash = 0.0
    navs: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    prev_total = None
    prev_weights: dict[str, float] = {}
    last_date: dict[str, pd.Timestamp] = {c: None for c in etfs}
    for t, d in enumerate(dates):
        w = weights_by_date.get(d, {})
        # 1) 入金
        if deposit > 0 and t >= 0:
            cash += deposit
            invested += deposit
            flows.append((t, -deposit))
        # 2) CA 事件
        for c, info in etfs.items():
            i = info["date_to_idx"].get(d)
            if i is None:
                continue
            ex = info["usable"].iloc[i]["exec"]
            if ex is None:
                continue
            exd = pd.Timestamp(ex)
            cash = _apply_ca(info, last_date[c] if last_date[c] is not None else pd.Timestamp("1970-01-01"), exd, cash)
            last_date[c] = exd
        # 3) 按 exec raw_open 再平衡
        total = cash
        for c, info in etfs.items():
            i = info["date_to_idx"].get(d)
            if i is None:
                continue
            price = info["usable"].iloc[i]["exec_raw_open"]
            if np.isnan(price):
                continue
            total += info["shares"] * price
        if total > 0:
            for c, info in etfs.items():
                i = info["date_to_idx"].get(d)
                if i is None:
                    continue
                price = info["usable"].iloc[i]["exec_raw_open"]
                if np.isnan(price) or float(w.get(c, 0.0)) <= 0:
                    target = 0.0
                else:
                    target = total * float(w[c]) / 100.0
                cur = info["shares"] * price
                delta_val = target - cur
                if abs(delta_val) > 1e-9:
                    shares_delta = delta_val / price
                    cost = abs(delta_val) * COST_BPS / 10000.0
                    cash -= delta_val + cost
                    info["shares"] += shares_delta
        nav = cash
        for c, info in etfs.items():
            i = info["date_to_idx"].get(d)
            if i is None:
                continue
            price = info["usable"].iloc[i]["exec_raw_open"]
            if not np.isnan(price):
                nav += info["shares"] * price
        navs.append(nav)
        base = prev_total + deposit if prev_total is not None else None
        if base is not None and base > 0:
            twr_rets.append(nav / base - 1.0)
        prev_total = nav
        prev_weights = {c: float(w.get(c, 0.0)) / 100.0 for c in etfs}
    navs = np.asarray(navs)
    twr_rets = np.asarray(twr_rets)
    twr_index = np.cumprod(1.0 + twr_rets)
    peak = np.maximum.accumulate(twr_index)
    mdd = float(np.nanmin(twr_index / peak - 1.0)) if len(twr_index) else 0.0
    mean = float(twr_rets.mean()) if len(twr_rets) else 0.0
    std = float(twr_rets.std(ddof=1)) if len(twr_rets) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    final_nav = float(navs[-1]) if len(navs) else 0.0

    def irr_f(r_week: float) -> float:
        n = flows[-1][0] if flows else 0
        total = final_nav
        for week, cf in flows:
            total += cf * (1.0 + r_week) ** (n - week)
        return total

    lo, hi = -0.5, 1.0
    flo = irr_f(lo)
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if abs(irr_f(mid)) < 1e-12:
            lo = mid
            break
        if np.sign(irr_f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    irr = (1.0 + lo) ** 52.0 - 1.0
    return {
        "twr": float(twr_index[-1] - 1.0) if len(twr_index) else 0.0,
        "irr_annual": irr,
        "mdd": mdd,
        "sharpe": sharpe,
        "invested": invested,
        "final_nav": final_nav,
        "cum_money": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "avg_exposure": float(np.mean([sum(float(w.get(c, 0.0)) for c in etfs) / 100.0 for w in weights_by_date.values()])) if weights_by_date else 0.0,
    }


def main(start: str) -> dict:
    from scripts.investment_priority import allocate
    from rolling.benchmark import smart_dca
    from rolling.data import load_aligned_data
    from rolling.ledger import set_workspace

    etfs = {c: _load_etf(c) for c in CODES}
    master = etfs["159915"]["usable"]
    master_idx = {pd.Timestamp(d): i for i, d in enumerate(master["signal"])}
    start_ts = pd.Timestamp(start)
    dates = [pd.Timestamp(d) for d in master["signal"] if pd.Timestamp(d) >= start_ts]
    # 权重表
    weights_by_date: dict[pd.Timestamp, dict] = {}
    for d in dates:
        signals: dict[str, bool] = {}
        positions: dict[str, float] = {}
        for c, info in etfs.items():
            i = info["date_to_idx"].get(d)
            if i is None:
                continue
            p = float(info["pos"][i])
            signals[c] = p > 0
            positions[c] = p
        weights_by_date[d] = allocate(signals, positions)["weights"]
    strat = _simulate(etfs, dates, weights_by_date, 0, deposit=2000.0)
    # 基线：全仓创业板智能定投（同现金流）
    set_workspace("159915")
    base_start = int(master_idx[dates[0]])
    dca = smart_dca("159915", end_idx=len(master) - 1, start_anchor=base_start)
    return {
        "window": {"start": str(dates[0].date()), "end": str(dates[-1].date()), "weeks": len(dates)},
        "strategy": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in strat.items()},
        "dca_159915": {
            "twr": round(float(dca["twr"]), 4),
            "irr_annual": round(float(dca["irr_annual"]), 4),
            "mdd": round(float(dca["mdd"]), 4),
            "sharpe": round(float(dca["sharpe"]), 4),
            "invested": round(float(dca["invested"]), 2),
            "final_nav": round(float(dca["final_nav"]), 2),
            "cum_money": round(float(dca["cum_money"]), 4),
        },
    }


if __name__ == "__main__":
    raise SystemExit(
        "LEGACY_BLOCKED: 本脚本固定使用518600/512010旧ETF池，只保留历史PK复现；"
        "当前项目不得将其结果作为正式策略或月报输入。"
    )
