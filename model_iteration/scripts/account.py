# -*- coding: utf-8 -*-
"""单标的账户模拟与指标：0–100% 仓位、周度复利、10bp 往返成本。"""

from __future__ import annotations

import numpy as np


def simulate(position: np.ndarray, fwd1: np.ndarray, cost_bps: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    pos = np.asarray(position, dtype=float)
    fwd = np.asarray(fwd1, dtype=float)
    cost_rate = cost_bps / 10000.0
    nets = np.zeros(len(pos))
    prev = 0.0
    turnover = 0.0
    for i in range(len(pos)):
        if np.isnan(fwd[i]):
            nets[i] = 0.0
            continue
        change = abs(pos[i] - prev)
        turnover += change
        nets[i] = pos[i] * fwd[i] - change * cost_rate
        prev = pos[i]
    equity = np.cumprod(1.0 + nets)
    return nets, equity, turnover


def metrics(
    nets: np.ndarray,
    pos: np.ndarray,
    fwd1: np.ndarray,
    label8: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    prob: np.ndarray | None = None,
) -> dict:
    valid = ~np.isnan(nets)
    eq = np.cumprod(1.0 + nets[valid])
    cum = eq[-1] - 1.0
    mean = nets[valid].mean()
    std = nets[valid].std(ddof=1)
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    up = fwd1[valid] > 0
    participation = float((pos[valid][up] > 0).mean()) if up.any() else 0.0
    missed = float(np.nansum(fwd1[(pos == 0) & (fwd1 > 0)]))
    false_pos = float(np.nansum(fwd1[(pos > 0) & (fwd1 < 0)]))
    out = {
        "cumulative_return": round(float(cum), 6),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(mdd, 6),
        "win_rate": round(float((nets[valid] > 0).mean()), 4),
        "median_weekly_return": round(float(np.median(nets[valid])), 6),
        "average_exposure": round(float(pos[valid].mean()), 4),
        "participation_up_weeks": round(participation, 4),
        "missed_up_sum": round(missed, 4),
        "false_positive_sum": round(false_pos, 4),
    }
    if label8 is not None:
        base = valid & ~np.isnan(label8)
        if forecast is not None:
            labeled = base & ~np.isnan(forecast)
            if labeled.any():
                out["mae"] = round(float(np.mean(np.abs(forecast[labeled] - label8[labeled]))), 6)
        if prob is not None:
            labeled = base & ~np.isnan(prob)
            if labeled.any():
                outcome = (label8[labeled] > 0).astype(float)
                out["brier"] = round(float(np.mean((prob[labeled] - outcome) ** 2)), 6)
    return out


def passive_same_exposure(fwd1: np.ndarray, avg_exposure: float) -> tuple[np.ndarray, np.ndarray]:
    nets = np.full(len(fwd1), np.nan)
    valid = ~np.isnan(fwd1)
    nets[valid] = avg_exposure * fwd1[valid]
    return nets, np.full(len(fwd1), avg_exposure)
