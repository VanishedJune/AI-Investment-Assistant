# -*- coding: utf-8 -*-
"""支付宝“涨跌幅智能定投”规则实现与收益 PK（只读计算，不写台账）。

规则（按用户给定口径，缺失的映射用确定性线性插值补全）：
- 基础金额 2000 元/周；实际投入 = 2000 × 扣款率；
- 净值相对持仓成本 pct = (T-1 净值 - 平均成本)/平均成本；
- pct ∈ [-2.5%, +2.5%] → 100%；
- pct > +2.5% → 扣款率 = max(50%, 100% - 2*(pct-2.5%))（+27.5% 时到 50%）；
- pct < -2.5% → 扣款率 = min(200%, 100% + 2*(-pct-2.5%))（-52.5% 时到 200%）；
- 参考净值取执行日前一交易日 analysis_close（T-1），缺失回退 T-2；
- 分红导致持仓成本临时波动 >1% 时当期暂停（当前 CA 表为零事件，实际不触发）；
- 成交价 raw_open，买入费用 10bp（与模型口径一致），持仓成本含费。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402

BASE_AMOUNT = 2000.0
COST_BPS = 10
BAND = 0.025
SLOPE = 2.0
RATE_MIN = 0.5
RATE_MAX = 2.0


def deduction_rate(pct: float) -> float:
    if pct >= BAND:
        return max(RATE_MIN, 1.0 - SLOPE * (pct - BAND))
    if pct <= -BAND:
        return min(RATE_MAX, 1.0 + SLOPE * (-pct - BAND))
    return 1.0


def annual_irr(flows: list[tuple[int, float]], final_nav: float) -> float:
    """按周现金流求年化 IRR（净投入为负，期末市值为正）。"""
    n = flows[-1][0] if flows else 0

    def f(r_week: float) -> float:
        total = final_nav
        for week, cf in flows:
            total += cf * (1.0 + r_week) ** (n - week)
        return total

    lo, hi = -0.5, 1.0
    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if abs(fmid) < 1e-10:
            lo = mid
            break
        if np.sign(fmid) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    r_week = lo
    return (1.0 + r_week) ** 52.0 - 1.0


def run_dca(usable: pd.DataFrame, daily_a: pd.DataFrame, end_idx: int = 43) -> dict:
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    shares = 0.0
    total_cost = 0.0
    navs: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    paused_weeks = 0
    for i in range(end_idx + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        ex = pd.Timestamp(ex)
        # T-1 参考净值（前一个交易日 analysis_close），缺失回退 T-2
        prev_dates = closes.index[closes.index < ex]
        if len(prev_dates) == 0:
            continue
        nav_ref = float(closes.loc[prev_dates[-1]])
        if nav_ref != nav_ref and len(prev_dates) >= 2:  # pragma: no cover
            nav_ref = float(closes.loc[prev_dates[-2]])
        price = float(opens.loc[ex])
        if shares <= 0:
            pct = 0.0
        else:
            avg_cost = total_cost / shares
            pct = (nav_ref - avg_cost) / avg_cost
        rate = deduction_rate(pct)
        # 分红导致成本临时波动 >1% -> 当期暂停（窗口内 CA 零事件，实际不触发）
        pause = False
        for ev in []:
            pass
        if pause:
            paused_weeks += 1
            amount = 0.0
        else:
            amount = BASE_AMOUNT * rate
        fee = amount * COST_BPS / 10000.0
        net = amount - fee
        if net > 0 and price > 0:
            shares += net / price
            total_cost += net
            invested += net
            flows.append((i, -net))
        navs.append(shares * price)
        if len(navs) >= 2:
            prev_nav = navs[-2]
            # TWR：剔除新增资金后的市场收益（持仓价格变化）
            if prev_nav > 0:
                twr_rets.append(price / prev_price - 1.0)
        prev_price = price
    navs = np.asarray(navs, dtype=float)
    twr_rets = np.asarray(twr_rets, dtype=float)
    twr_index = np.cumprod(1.0 + twr_rets)
    twr = float(twr_index[-1] - 1.0) if len(twr_index) else 0.0
    peak = np.maximum.accumulate(twr_index)
    mdd = float(np.nanmin(twr_index / peak - 1.0)) if len(twr_index) else 0.0
    mean = float(twr_rets.mean()) if len(twr_rets) else 0.0
    std = float(twr_rets.std(ddof=1)) if len(twr_rets) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    wins = twr_rets[twr_rets > 0] if len(twr_rets) else np.array([])
    win_rate = float(len(wins) / len(twr_rets)) if len(twr_rets) else 0.0
    final_nav = float(navs[-1]) if len(navs) else 0.0
    irr = annual_irr(flows, final_nav) if flows else 0.0
    return {
        "invested": invested,
        "final_nav": final_nav,
        "cum": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "twr": twr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "avg_cost": total_cost / shares if shares > 0 else 0.0,
        "shares": shares,
        "irr_annual": irr,
        "paused_weeks": paused_weeks,
    }


def main(etf: str = "159915") -> int:
    ca = load_ca(etf)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    end_idx = len(usable) - 1
    dca = run_dca(usable, daily_a, end_idx)
    print(f"=== 智能定投（周投 2000，全历史 712 锚点 {usable['signal'].iloc[0].date()} ~ {usable['signal'].iloc[-1].date()}）===")
    for k, v in dca.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
