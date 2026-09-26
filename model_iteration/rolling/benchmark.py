# -*- coding: utf-8 -*-
"""常驻基准轨（NON_DECISIONAL reference）：买入持有 + 智能定投。

每次迭代输出“规则 vs 基准”对比；基准不参与晋级，只回答
“规则是否值得存在（风险调整后跑赢买入持有）”。
"""

from __future__ import annotations

import numpy as np

from .account import graded_adjust, run_account, weekly_metrics
from .challenge import position_path
from .config import load_etf_config
from .data import load_aligned_data
from .execution import guard_triggered, is_decision_anchor


MOMENTUM_DD_SPEC = {
    "score": {"terms": [{"feature": "mom20", "weight": 1000.0}]},
    "gates": [{"feature": "mom20", "op": ">", "value": 0.0}],
    "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.8}], "min": 0.0, "max": 0.8},
    "drawdown_control": {"window_weeks": 8, "threshold": 0.10, "factor": 0.5},
}


def compute_reference_benchmarks(etf: str, daily_a, weekly_a, daily, anchors, usable, features, end_idx: int) -> dict:
    """常驻基准（NON_DECISIONAL）：买入持有 / 智能定投 / 动量+回撤，复用已装配数据。"""
    from .ca import load_ca

    ca = load_ca(etf)
    execution_policy = load_etf_config(etf).get("execution_policy") or None
    frames = (ca, daily_a, weekly_a, daily, anchors, usable, features)
    bh = buy_and_hold(end_idx=end_idx, _frames=frames)
    dca = smart_dca(end_idx=end_idx, _frames=frames)
    pos = position_path(MOMENTUM_DD_SPEC, features, end_idx)
    acct = run_account(
        pos, usable, daily, ca, cost_bps=10,
        execution_policy=execution_policy, features=features,
    )
    m = weekly_metrics(acct["nav"], acct["positions"], acct["returns"])
    mm = {"name": "动量+回撤", "cum": m["cumulative_return"], "sharpe": m["sharpe"],
          "mdd": m["max_drawdown"], "win_rate": m["win_rate"]}
    return {"buy_hold": bh, "smart_dca": dca, "momentum_dd": mm}


def buy_and_hold(etf: str = "159915", end_idx: int | None = None, _frames=None) -> dict:
    from .ca import load_ca

    if _frames is not None:
        ca, daily_a, weekly_a, daily, anchors, usable, features = _frames
    else:
        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    end = len(usable) - 1 if end_idx is None else end_idx
    pos = np.ones(end + 1)
    # 必须传显式 CA：拆分/分红事件下走 CA-aware 慢路径，否则净值会在除权日失真
    acct = run_account(pos, usable, daily, ca, cost_bps=10)
    m = weekly_metrics(acct["nav"], pos, acct["returns"])
    return {
        "name": "买入持有",
        "cum": m["cumulative_return"],
        "sharpe": m["sharpe"],
        "mdd": m["max_drawdown"],
        "win_rate": m["win_rate"],
        "avg_pos": 1.0,
    }


def smart_dca(
    etf: str = "159915",
    end_idx: int | None = None,
    base_amount: float = 2000.0,
    cost_bps: int = 10,
    _frames=None,
    start_anchor: int = 0,
) -> dict:
    """支付宝“涨跌幅智能定投”口径：±2.5% 内 100%，涨多降（最低 50%），跌多升（最高 200%）。"""
    from .ca import load_ca

    if _frames is not None:
        ca, daily_a, weekly_a, daily, anchors, usable, features = _frames
    else:
        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    end = len(usable) - 1 if end_idx is None else end_idx
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    # 拆分/分红感知的成交价：TWR 用复权后价格，避免除权日出现虚假跳变；
    # 实际买入仍按 raw_open（真实现金流）。
    sf = daily_a.set_index("date")["sf"]

    def rate(pct: float) -> float:
        if pct >= 0.025:
            return max(0.5, 1.0 - 2.0 * (pct - 0.025))
        if pct <= -0.025:
            return min(2.0, 1.0 + 2.0 * (-pct - 0.025))
        return 1.0

    shares = 0.0
    total_cost = 0.0
    cost_adj = 0.0  # 复权基准累计成本（跨拆分归一化，与 T-1 参考净值同口径）
    navs: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    prev_price = None
    twr_rets: list[float] = []
    for i in range(start_anchor, end + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        ex = np.datetime64(ex)
        prev_dates = closes.index[closes.index < ex]
        if len(prev_dates) == 0:
            continue
        nav_ref = float(closes.loc[prev_dates[-1]])
        price = float(opens.loc[ex])
        adj_price = float(opens.loc[ex]) * float(sf.loc[ex])
        avg_cost = cost_adj / shares if shares > 0 else nav_ref
        pct = (nav_ref - avg_cost) / avg_cost if shares > 0 else 0.0
        amount = base_amount * rate(pct)
        fee = amount * cost_bps / 10000.0
        net = amount - fee
        if net > 0 and price > 0:
            shares += net / price
            total_cost += net
            cost_adj += net * float(sf.loc[ex])
            invested += net
            flows.append((i, -net))
        navs.append(shares * price)
        if prev_price is not None:
            twr_rets.append(adj_price / prev_price - 1.0)
        prev_price = adj_price
    navs = np.asarray(navs)
    twr_rets = np.asarray(twr_rets)
    twr_index = np.cumprod(1.0 + twr_rets)
    peak = np.maximum.accumulate(twr_index)
    mdd = float(np.nanmin(twr_index / peak - 1.0)) if len(twr_index) else 0.0
    mean = float(twr_rets.mean()) if len(twr_rets) else 0.0
    std = float(twr_rets.std(ddof=1)) if len(twr_rets) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    wins = twr_rets[twr_rets > 0] if len(twr_rets) else np.array([])
    win_rate = float(len(wins) / len(twr_rets)) if len(twr_rets) else 0.0
    final_nav = float(navs[-1]) if len(navs) else 0.0

    def irr_f(r_week: float) -> float:
        n = flows[-1][0] if flows else 0
        total = final_nav
        for week, cf in flows:
            total += cf * (1.0 + r_week) ** (n - week)
        return total

    lo, hi = -0.5, 1.0
    flo = irr_f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if abs(irr_f(mid)) < 1e-10:
            lo = mid
            break
        if np.sign(irr_f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    irr = (1.0 + lo) ** 52.0 - 1.0
    return {
        "name": "智能定投",
        "twr": float(twr_index[-1] - 1.0) if len(twr_index) else 0.0,
        "cum_money": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "irr_annual": irr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "invested": invested,
        "final_nav": final_nav,
    }


def fixed_dca(
    etf: str = "159915",
    end_idx: int | None = None,
    base_amount: float = 2000.0,
    cost_bps: int = 10,
    _frames=None,
    start_anchor: int = 0,
) -> dict:
    """固定定投（普通定投）：每周固定 base_amount，扣款率恒为 100%。"""
    from .ca import load_ca

    if _frames is not None:
        ca, daily_a, weekly_a, daily, anchors, usable, features = _frames
    else:
        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    end = len(usable) - 1 if end_idx is None else end_idx
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    shares = 0.0
    total_cost = 0.0
    navs: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    prev_price = None
    twr_rets: list[float] = []
    for i in range(start_anchor, end + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        ex = np.datetime64(ex)
        prev_dates = closes.index[closes.index < ex]
        if len(prev_dates) == 0:
            continue
        price = float(opens.loc[ex])
        amount = base_amount
        fee = amount * cost_bps / 10000.0
        net = amount - fee
        if net > 0 and price > 0:
            shares += net / price
            total_cost += net
            invested += net
            flows.append((i, -net))
        navs.append(shares * price)
        if prev_price is not None:
            twr_rets.append(price / prev_price - 1.0)
        prev_price = price
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
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if abs(irr_f(mid)) < 1e-10:
            lo = mid
            break
        if np.sign(irr_f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    irr = (1.0 + lo) ** 52.0 - 1.0
    return {
        "name": "固定定投",
        "twr": float(twr_index[-1] - 1.0) if len(twr_index) else 0.0,
        "cum_money": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "irr_annual": irr,
        "sharpe": sharpe,
        "mdd": mdd,
        "invested": invested,
        "final_nav": final_nav,
    }


def run_reference_benchmarks(etf: str = "159915") -> dict:
    return {"buy_and_hold": buy_and_hold(etf), "smart_dca": smart_dca(etf)}


def rule_with_deposits(
    etf: str,
    spec: dict,
    base_amount: float = 2000.0,
    cost_bps: int = 10,
    end_idx: int | None = None,
    start_anchor: int = 0,
    _frames=None,
    fwd_arrays: dict | None = None,
    positions: np.ndarray | None = None,
    execution_policy: dict | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    """规则 + 每周定投现金流：每周入金 base_amount，规则决定目标仓位，现金留存可累积。

    口径与智能定投一致：
    - TWR：剔除外部现金流的规则收益（本期入金前市值 / 上期入金后市值 - 1）；
    - IRR：资金加权年化（周现金流折现）；
    - 市值/累计投入：期末市值 / 累计入金 - 1。
    """
    if _frames is not None:
        ca, daily_a, weekly_a, daily, anchors, usable, features = _frames
    else:
        from .ca import load_ca
        from .data import load_aligned_data as _lad

        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = _lad(etf)
    if execution_policy is not None and features is None:
        raise ValueError("rule_with_deposits 使用执行策略时必须提供 features")
    end = len(usable) - 1 if end_idx is None else end_idx
    if positions is None:
        if fwd_arrays is None:
            fwd_arrays = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
        positions = position_path(spec, features, end, fwd_arrays=fwd_arrays)
    opens = daily_a.set_index("date")["raw_open"]
    cash = 0.0
    shares = 0.0
    nav_after: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    episode = False
    cycle = int((execution_policy or {}).get("decision_cycle_weeks", 4))
    guard_cfg = (execution_policy or {}).get("bearish_guard") or {}
    sell_frac = float(guard_cfg.get("sell_frac", 0.7))
    per_episode = bool(guard_cfg.get("per_episode", True))
    reset_on_recovery = bool(guard_cfg.get("reset_on_recovery", True))
    for i in range(start_anchor, end + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        price = float(opens.loc[ex])
        nav_before = shares * price + cash
        if nav_after:
            twr_rets.append(nav_before / nav_after[-1] - 1.0)
        cash += base_amount
        invested += base_amount
        flows.append((i, -base_amount))
        value = shares * price + cash
        if execution_policy is None:
            target_shares = float(positions[i]) * value / price
        elif is_decision_anchor(i, cycle):
            # 决策周：按目标权重完整调仓，并重置保护期
            target_shares = float(positions[i]) * value / price
            episode = False
        elif (execution_policy.get("non_decision_graded") or {}).get("enabled"):
            # 非决策周分级调仓：feature>0 现金50%加仓；feature<0 持仓50%卖出
            shares, cash = graded_adjust(
                execution_policy.get("non_decision_graded") or {},
                features, i, shares, cash, price, cost_bps,
            )
            nav_after.append(shares * price + cash)
            continue
        elif guard_triggered(features, i, execution_policy):
            # 验证看空：单次触发减持 sell_frac，条件持续期间不再重复卖出
            if per_episode and not episode:
                if shares > 0:
                    target_shares = shares * (1.0 - sell_frac)
                    episode = True
                else:
                    target_shares = 0.0
            elif per_episode:
                target_shares = shares
            else:
                # per_episode=False：每个看空周都减持 sell_frac
                target_shares = shares * (1.0 - sell_frac) if shares > 0 else 0.0
        else:
            if episode and reset_on_recovery:
                episode = False
            # 非决策周：入金只用于加仓（当前持有份额时），不新开仓、不卖出
            target_shares = shares + base_amount / price if shares > 0 else 0.0
        cost = abs(target_shares - shares) * price * cost_bps / 10000.0
        cash = value - target_shares * price - cost
        shares = target_shares
        nav_after.append(shares * price + cash)
    navs = np.asarray(nav_after)
    twr_rets = np.asarray(twr_rets)
    twr_index = np.cumprod(1.0 + twr_rets)
    peak = np.maximum.accumulate(twr_index)
    mdd = float(np.nanmin(twr_index / peak - 1.0)) if len(twr_index) else 0.0
    mean = float(twr_rets.mean()) if len(twr_rets) else 0.0
    std = float(twr_rets.std(ddof=1)) if len(twr_rets) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    wins = twr_rets[twr_rets > 0] if len(twr_rets) else np.array([])
    win_rate = float(len(wins) / len(twr_rets)) if len(twr_rets) else 0.0
    final_nav = float(navs[-1]) if len(navs) else 0.0

    def irr_f(r_week: float) -> float:
        n = flows[-1][0] if flows else 0
        total = final_nav
        for week, cf in flows:
            total += cf * (1.0 + r_week) ** (n - week)
        return total

    lo, hi = -0.5, 1.0
    flo = irr_f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if abs(irr_f(mid)) < 1e-10:
            lo = mid
            break
        if np.sign(irr_f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    irr = (1.0 + lo) ** 52.0 - 1.0
    return {
        "name": "规则+周投2000",
        "twr": float(twr_index[-1] - 1.0) if len(twr_index) else 0.0,
        "cum_money": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "irr_annual": irr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "invested": invested,
        "final_nav": final_nav,
    }


def cashflow_gate(
    etf: str,
    challenger_spec: dict,
    champion_spec: dict,
    start_anchor: int,
    end_anchor: int,
    base_amount: float = 2000.0,
    _frames=None,
    penalize_dd: bool = True,
) -> dict:
    """同现金流晋级门禁：每周 base_amount 从 start_anchor 累积到 end_anchor。

    挑战者必须同时跑赢：现任 Champion 规则、智能定投（TWR 主判，IRR/MDD 辅助）。
    """
    if _frames is not None:
        ca, daily_a, weekly_a, daily, anchors, usable, features = _frames
    else:
        from .ca import load_ca
        from .data import load_aligned_data

        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    frames = (ca, daily_a, weekly_a, daily, anchors, usable, features)
    fwd_arrays = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    execution_policy = load_etf_config(etf).get("execution_policy") or None
    ch = rule_with_deposits(etf, challenger_spec, base_amount=base_amount,
                            end_idx=end_anchor, start_anchor=start_anchor,
                            _frames=frames, fwd_arrays=fwd_arrays,
                            execution_policy=execution_policy, features=features)
    champ = rule_with_deposits(etf, champion_spec, base_amount=base_amount,
                               end_idx=end_anchor, start_anchor=start_anchor,
                               _frames=frames, fwd_arrays=fwd_arrays,
                               execution_policy=execution_policy, features=features)
    dca = smart_dca(etf, end_idx=end_anchor, base_amount=base_amount,
                    _frames=frames, start_anchor=start_anchor)
    fixed = fixed_dca(etf, end_idx=end_anchor, base_amount=base_amount,
                      _frames=frames, start_anchor=start_anchor)
    ref_twr_max = max(champ["twr"], dca["twr"], fixed["twr"])
    ref_irr_max = max(champ["irr_annual"], dca["irr_annual"], fixed["irr_annual"])
    ref_mdd_min = min(champ["mdd"], dca["mdd"], fixed["mdd"])
    advantage = max(ref_twr_max * 0.05, 0.02)
    if penalize_dd:
        win = bool(
            ch["twr"] >= ref_twr_max + advantage
            and ch["irr_annual"] >= ref_irr_max - 1e-9
            and ch["mdd"] >= ref_mdd_min - 0.05
        )
    else:
        # 回撤不扣分：同现金流门禁只看 TWR（主判）与 IRR，不看 MDD
        win = bool(
            ch["twr"] >= ref_twr_max + advantage
            and ch["irr_annual"] >= ref_irr_max - 1e-9
        )
    return {
        "pass": win,
        "challenger_twr": ch["twr"],
        "champion_twr": champ["twr"],
        "dca_twr": dca["twr"],
        "fixed_dca_twr": fixed["twr"],
        "challenger_irr": ch["irr_annual"],
        "champion_irr": champ["irr_annual"],
        "dca_irr": dca["irr_annual"],
        "fixed_dca_irr": fixed["irr_annual"],
        "challenger_mdd": ch["mdd"],
        "champion_mdd": champ["mdd"],
        "dca_mdd": dca["mdd"],
        "fixed_dca_mdd": fixed["mdd"],
        "advantage": advantage,
    }
