# -*- coding: utf-8 -*-
"""CA-aware 账户模拟：成交 raw_open，NAV=raw×shares+cash，拆分/分红按显式事件。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .execution import guard_triggered, is_decision_anchor, next_executed_weight


def graded_adjust(
    cfg: dict,
    features: pd.DataFrame,
    i: int,
    shares: float,
    cash: float,
    price: float,
    cost_bps: int = 10,
) -> tuple[float, float]:
    """非决策周分级调仓：feature>0 → 现金 add_cash_frac 加仓；
    feature<0 → 持仓 sell_holdings_frac 卖出；==0 保持。"""
    feature = str(cfg.get("feature", "d_dif1"))
    add_frac = float(cfg.get("add_cash_frac", 0.5))
    sell_frac = float(cfg.get("sell_holdings_frac", 0.5))
    value = float(features.iloc[i][feature])
    if value > 0 and cash > 0:
        add_cash = cash * add_frac
        fee = add_cash * cost_bps / 10000.0
        shares += (add_cash - fee) / price
        cash -= add_cash
    elif value < 0 and shares > 0:
        sell_gross = shares * price * sell_frac
        fee = sell_gross * cost_bps / 10000.0
        cash += sell_gross - fee
        shares *= 1.0 - sell_frac
    return shares, cash


def run_account(
    positions: np.ndarray,
    anchors: pd.DataFrame,
    daily: pd.DataFrame,
    ca_payload: dict,
    cost_bps: int = 10,
    start_snapshot: dict | None = None,
    start_anchor: int = 0,
    exec_prices: np.ndarray | None = None,
    execution_policy: dict | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    """positions[i] 为 anchor i 的目标仓位；返回每周 NAV/收益（从 start_anchor 起）。

    execution_policy 提供时（B0207 式执行策略），目标仓位先转换为实际执行仓位：
    决策锚点按目标完整调仓；非决策周只加仓；验证看空触发时按 sell_frac 减持。
    """
    if execution_policy is not None and features is None:
        raise ValueError("run_account 使用执行策略时必须提供 features")
    events = ca_payload.get("events", [])
    if not events:
        return _run_account_fast(
            positions, anchors, daily, cost_bps, start_snapshot, start_anchor,
            exec_prices, execution_policy, features,
        )
    splits = [
        (pd.Timestamp(e["ex_date"]), float(e["ratio"]))
        for e in events
        if e["type"] == "split"
    ]
    dividends = [
        (
            pd.Timestamp(e["ex_date"]),
            pd.Timestamp(e["entitlement_date"]),
            pd.Timestamp(e["pay_date"]),
            float(e["dps"]),
        )
        for e in events
        if e["type"] == "cash_dividend"
    ]

    cash = float(start_snapshot.get("cash", 0.0)) if start_snapshot else 100000.0
    shares = float(start_snapshot.get("shares", 0.0)) if start_snapshot else 0.0
    eligible: dict[tuple, float] = {}
    executed_positions: list[float] = []
    episode = False
    cycle = int((execution_policy or {}).get("decision_cycle_weeks", 4))

    exec_by_date: dict[pd.Timestamp, int] = {}
    sim_start: pd.Timestamp | None = None
    for i in range(start_anchor, len(positions)):
        ex = anchors.iloc[i]["exec"]
        if ex is not None:
            exec_by_date[pd.Timestamp(ex)] = i
            if sim_start is None:
                sim_start = pd.Timestamp(ex)

    raw_open_by_date = daily.set_index("date")["raw_open"]
    navs: list[float] = []
    for _, day_row in daily.iterrows():
        day = day_row["date"]
        if sim_start is not None and day < sim_start:
            continue
        for ex_date, ratio in splits:
            if day == ex_date:
                shares *= ratio
        for div in dividends:
            ex_date, ent_date, pay_date, dps = div
            if day == ent_date:
                eligible[(ent_date, pay_date)] = shares
            if day == pay_date:
                cash += eligible.get((ent_date, pay_date), 0.0) * dps
        if day in exec_by_date:
            i = exec_by_date[day]
            price = float(raw_open_by_date.loc[day])
            value = shares * price + cash
            if execution_policy is not None:
                graded_cfg = (execution_policy.get("non_decision_graded") or {})
                dec = is_decision_anchor(i, cycle)
                if graded_cfg.get("enabled") and not dec:
                    shares, cash = graded_adjust(
                        graded_cfg, features, i, shares, cash, price, cost_bps
                    )
                    value_after = shares * price + cash
                    executed = float(shares * price / value_after) if value_after > 0 else 0.0
                    executed_positions.append(executed)
                    navs.append(value_after)
                    continue
                current_weight = float(shares * price / value) if value > 0 else 0.0
                executed, episode = next_executed_weight(
                    current_weight,
                    float(positions[i]),
                    dec,
                    guard_triggered(features, i, execution_policy),
                    episode,
                    execution_policy,
                )
                executed_positions.append(executed)
            else:
                executed = float(positions[i])
                executed_positions.append(executed)
            target_shares = executed * value / price
            cost = abs(target_shares - shares) * price * cost_bps / 10000.0
            cash = value - target_shares * price - cost
            shares = target_shares
            navs.append(shares * price + cash)

    navs = np.asarray(navs, dtype=float)
    rets = navs[1:] / navs[:-1] - 1.0 if len(navs) > 1 else np.array([])
    return {
        "nav": navs,
        "returns": rets,
        "final_cash": cash,
        "final_shares": shares,
        "positions": np.asarray(executed_positions, dtype=float),
    }


def _run_account_fast(
    positions: np.ndarray,
    anchors: pd.DataFrame,
    daily: pd.DataFrame,
    cost_bps: int,
    start_snapshot: dict | None,
    start_anchor: int,
    exec_prices: np.ndarray | None = None,
    execution_policy: dict | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    """无 CA 事件时的周频快路径：与通用路径在成交/成本/NAV 口径上完全一致。"""
    cash = float(start_snapshot.get("cash", 0.0)) if start_snapshot else 100000.0
    shares = float(start_snapshot.get("shares", 0.0)) if start_snapshot else 0.0
    if exec_prices is None:
        date_idx = {np.datetime64(d): i for i, d in enumerate(daily["date"].to_numpy())}
        opens = daily["raw_open"].to_numpy()
        exec_prices = np.full(len(anchors), np.nan)
        for i, ex in enumerate(anchors["exec"]):
            if ex is not None:
                exec_prices[i] = float(opens[date_idx[np.datetime64(ex)]])
    navs: list[float] = []
    executed_positions: list[float] = []
    episode = False
    cycle = int((execution_policy or {}).get("decision_cycle_weeks", 4))
    for i in range(start_anchor, len(positions)):
        price = float(exec_prices[i]) if i < len(exec_prices) else np.nan
        if np.isnan(price):
            continue
        value = shares * price + cash
        if execution_policy is not None:
            graded_cfg = (execution_policy.get("non_decision_graded") or {})
            dec = is_decision_anchor(i, cycle)
            if graded_cfg.get("enabled") and not dec:
                shares, cash = graded_adjust(
                    graded_cfg, features, i, shares, cash, price, cost_bps
                )
                value_after = shares * price + cash
                executed = float(shares * price / value_after) if value_after > 0 else 0.0
                executed_positions.append(executed)
                navs.append(value_after)
                continue
            current_weight = float(shares * price / value) if value > 0 else 0.0
            executed, episode = next_executed_weight(
                current_weight,
                float(positions[i]),
                dec,
                guard_triggered(features, i, execution_policy),
                episode,
                execution_policy,
            )
            executed_positions.append(executed)
        else:
            executed = float(positions[i])
            executed_positions.append(executed)
        target_shares = executed * value / price
        cost = abs(target_shares - shares) * price * cost_bps / 10000.0
        cash = value - target_shares * price - cost
        shares = target_shares
        navs.append(shares * price + cash)
    navs = np.asarray(navs, dtype=float)
    rets = navs[1:] / navs[:-1] - 1.0 if len(navs) > 1 else np.array([])
    return {
        "nav": navs,
        "returns": rets,
        "final_cash": cash,
        "final_shares": shares,
        "positions": np.asarray(executed_positions, dtype=float),
    }


def anchor_exec_prices(anchors: pd.DataFrame, daily: pd.DataFrame) -> np.ndarray:
    """预计算每个锚点执行日的 raw_open（供 fast 账户路径复用）。"""
    date_idx = {np.datetime64(d): i for i, d in enumerate(daily["date"].to_numpy())}
    opens = daily["raw_open"].to_numpy()
    out = np.full(len(anchors), np.nan)
    for i, ex in enumerate(anchors["exec"]):
        if ex is not None:
            out[i] = float(opens[date_idx[np.datetime64(ex)]])
    return out


def weekly_metrics(
    navs: np.ndarray,
    positions: np.ndarray,
    returns: np.ndarray,
    window_start: int = 0,
    window_end: int | None = None,
) -> dict:
    nav = navs[window_start:window_end]
    ret = returns[window_start : (window_end - 1 if window_end is not None else None)]
    pos = positions[window_start:window_end]
    if len(nav) == 0:
        return {}
    cum = float(nav[-1] / nav[0] - 1.0)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.nanmin(nav / peak - 1.0))
    mean = float(ret.mean()) if len(ret) else 0.0
    std = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    wins = ret[ret > 0] if len(ret) else np.array([])
    win_rate = float(len(wins) / len(ret)) if len(ret) else 0.0
    return {
        "cumulative_return": round(cum, 6),
        "max_drawdown": round(mdd, 6),
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "average_position": round(float(pos.mean()), 4),
        "median_weekly_return": round(float(np.median(ret)), 6) if len(ret) else 0.0,
        "turnover": round(float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0, 4),
    }
