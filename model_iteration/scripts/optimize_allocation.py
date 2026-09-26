# -*- coding: utf-8 -*-
"""8 ETF 投资策略持仓分配优化（近 5 年批量回测，不逐周迭代）。

目标：在投资策略.md 的确定性规则框架（5% 倍数、合计 100%）内参数化
持仓占比/分配/优先级顺序/再平衡频率，随机采样约 400 个候选模型，
用“近一年 / 近三年 / 近五年”加权综合评分选出最优模型
（权重 70% / 20% / 10%，近一年优先）。

硬约束（用户指定）：
- 159915 创业板必须排在优先级前二（rank 1 或 2）；
- 159622 创新药必须排在前四（rank 1~4）；
- 516150 稀土必须排在前四（rank 1~4）。

口径（与 PK 完全一致）：
- 区间 2021-08-10 ~ 2026-08-07，周频；
- 每周总入金 = 2000 × n（n = 当周已开市 ETF 数量）；
- raw_open 成交、10bp 成本、显式 CA 事件账本；
- 信号 = 各 ETF 冠军链（逐代切换）；
- 看空不清零：floor 参数对所有 ETF 生效（冠军仓位 < floor 时按 floor 计）；
- 优先级顺序：随机排列抽样；再平衡频率：周/双周/月。

评选：
- 每个模型分别跑 近一年 / 近三年 / 近五年 三个窗口的独立回测；
- 综合评分 = 0.7×近一年累计收益率 + 0.2×近三年累计收益率 + 0.1×近五年累计收益率；
- 按综合评分取冠军与 Top 20；另输出近五年最高收益模型作为 in-sample 参照。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _irr  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import (  # noqa: E402
    CODES,
    NAMES,
    WEEKLY,
    gem_dca_account,
    load_etf,
    trade_cost,
)

CURRENT_ORDER = ["159915", "512010", "159622", "516150", "159941", "512690", "512800", "518600"]
DEFAULT_PARAMS = {
    "core_budget": 70,
    "remainder_budget": 30,
    "floor": 0.0,
    "ladder": 0,
    "max_single": 100,
    "rebalance": 1,
}

MODEL_COUNT = 400
SEED = 42

WINDOW_1Y = ("2025-08-10", "2026-08-07")
WINDOW_3Y = ("2023-08-10", "2026-08-07")
WINDOW_5Y = ("2021-08-10", "2026-08-07")
COMPOSITE_WEIGHTS = {"1y": 0.7, "3y": 0.2, "5y": 0.1}

OUT_DIR = ROOT.parent / "champion_vs_dca"

LADDERS = {
    0: {"1": [70], "2": [70], "3": [70], "1,2": [40, 30], "1,3": [50, 20], "2,3": [40, 30], "1,2,3": [40, 20, 10]},
    1: {"1": [70], "2": [70], "3": [70], "1,2": [35, 35], "1,3": [45, 25], "2,3": [35, 35], "1,2,3": [35, 20, 15]},
    2: {"1": [70], "2": [70], "3": [70], "1,2": [40, 30], "1,3": [40, 30], "2,3": [40, 30], "1,2,3": [30, 25, 15]},
    3: {"1": [70], "2": [70], "3": [70], "1,2": [45, 25], "1,3": [55, 15], "2,3": [45, 25], "1,2,3": [45, 15, 10]},
}

# 核心组=前四 的档位阶梯（基准权重相对比例，scale_group 会按预算缩放）
LADDERS4 = {
    0: {
        "1": [100], "2": [100], "3": [100], "4": [100],
        "1,2": [60, 40], "1,3": [65, 35], "1,4": [70, 30],
        "2,3": [55, 45], "2,4": [60, 40], "3,4": [55, 45],
        "1,2,3": [45, 30, 25], "1,2,4": [50, 30, 20],
        "1,3,4": [55, 25, 20], "2,3,4": [45, 35, 20],
        "1,2,3,4": [40, 25, 20, 15],
    },
    1: {
        "1": [100], "2": [100], "3": [100], "4": [100],
        "1,2": [55, 45], "1,3": [55, 45], "1,4": [55, 45],
        "2,3": [50, 50], "2,4": [55, 45], "3,4": [50, 50],
        "1,2,3": [40, 30, 30], "1,2,4": [40, 30, 30],
        "1,3,4": [40, 30, 30], "2,3,4": [40, 30, 30],
        "1,2,3,4": [30, 25, 25, 20],
    },
    2: {
        "1": [100], "2": [100], "3": [100], "4": [100],
        "1,2": [50, 50], "1,3": [50, 50], "1,4": [50, 50],
        "2,3": [50, 50], "2,4": [50, 50], "3,4": [50, 50],
        "1,2,3": [35, 35, 30], "1,2,4": [35, 35, 30],
        "1,3,4": [35, 35, 30], "2,3,4": [35, 35, 30],
        "1,2,3,4": [25, 25, 25, 25],
    },
    3: {
        "1": [100], "2": [100], "3": [100], "4": [100],
        "1,2": [65, 35], "1,3": [70, 30], "1,4": [75, 25],
        "2,3": [60, 40], "2,4": [65, 35], "3,4": [60, 40],
        "1,2,3": [50, 25, 25], "1,2,4": [55, 25, 20],
        "1,3,4": [60, 20, 20], "2,3,4": [50, 30, 20],
        "1,2,3,4": [45, 20, 20, 15],
    },
}

# 核心组=前二/单只 的档位阶梯（基准相对权重，scale_group 会按预算缩放）
LADDERS2 = {
    0: {"1": [100], "2": [100], "1,2": [60, 40]},
    1: {"1": [100], "2": [100], "1,2": [55, 45]},
    2: {"1": [100], "2": [100], "1,2": [50, 50]},
    3: {"1": [100], "2": [100], "1,2": [65, 35]},
}


def allocate_cfg(signals: dict, positions: dict, order: list[str], params: dict) -> dict:
    """参数化优先级分配：5% 倍数、合计 100%（权重 + 现金）。

    参数：
    - core_budget：核心组预算（默认 70）
    - remainder_budget：剩余组预算（默认 30）
    - floor：看空下限，所有 ETF 生效（仓位低于 floor 按 floor 计）
    - ladder：档位阶梯变体 0~3
    - max_single：单只上限（超出部分转现金）
    - rebalance：周/双周/月（账户层控制，分配函数本身不关心）
    - core_size：核心组规模，默认 3（前 3 名）；设为 4 时核心组为前 4 名，
      兜底组为 5-6 名、7-8 名；设为 2 时核心组为前 2 名（兜底 3-4/5-6/7-8）；
      设为 1 时核心组为第 1 名（兜底 2-3/4-5/6-7/8）。
    """
    core_budget = int(params["core_budget"])
    rem_budget = int(params["remainder_budget"])
    core_size = int(params.get("core_size", 3))
    if core_size >= 4:
        ladder = LADDERS4[int(params["ladder"])]
    elif core_size <= 2:
        ladder = LADDERS2[int(params["ladder"])]
    else:
        ladder = LADDERS[int(params["ladder"])]
    max_single = int(params["max_single"])
    if core_size >= 4:
        core = order[:4]
        next2 = order[4:6]
        last2 = order[6:8]
        weights: dict[str, int] = {c: 0 for c in order}

        def bullish(codes: list[str]) -> list[str]:
            return [c for c in codes if bool(signals.get(c, False))]

        def group_ladder(bull_codes: list[str], group: list[str]) -> list[int]:
            pos = [group.index(c) + 1 for c in bull_codes]
            key = ",".join(str(p) for p in pos)
            return [int(w) for w in ladder[key]]

        def scale_group(bull_codes: list[str], group: list[str], budget: int, base: list[int]) -> dict[str, int]:
            if len(bull_codes) == 1:
                return {bull_codes[0]: budget}
            raw = [
                base_w * max(0.0, float(positions.get(c, 1.0) or 1.0))
                for c, base_w in zip(bull_codes, base)
            ]
            total = sum(raw)
            if total <= 0:
                return {bull_codes[0]: budget}
            eff = [int(budget * v / total / 5.0) * 5 for v in raw]
            for idx in range(1, len(eff)):
                if eff[idx] > eff[idx - 1]:
                    eff[idx] = eff[idx - 1]
            leftover = budget - sum(eff)
            idx = 0
            while leftover > 0 and idx < len(eff):
                cap = budget if idx == 0 else eff[idx - 1]
                add = min(5, leftover, cap - eff[idx])
                if add <= 0:
                    idx += 1
                    continue
                eff[idx] += add
                leftover -= add
            return dict(zip(bull_codes, eff))

        core_bull = bullish(core)
        next_bull = bullish(next2)
        last_bull = bullish(last2)
        if core_bull:
            weights.update(scale_group(core_bull, core, core_budget, group_ladder(core_bull, core)))
            rem_codes = bullish(order[4:])[:2]
            if rem_codes:
                base = [15] if len(rem_codes) == 1 else [10, 5]
                budget = rem_budget if len(rem_codes) == 2 else min(20, rem_budget)
                weights.update(scale_group(rem_codes, order[4:], budget, base))
        elif next_bull:
            weights.update(scale_group(next_bull, next2, core_budget, group_ladder(next_bull, next2)))
        elif last_bull:
            weights.update(scale_group(last_bull, last2, core_budget, group_ladder(last_bull, last2)))
        for c in order:
            if weights[c] > max_single:
                weights[c] = max_single
        cash = 100 - sum(weights.values())
        if cash < 0:
            excess = -cash
            for c in sorted(weights, key=lambda x: -weights[x]):
                if excess <= 0:
                    break
                cut = min(weights[c], excess)
                weights[c] -= cut
                excess -= cut
            cash = 0
        return {"weights": weights, "cash_pct": cash}

    if core_size <= 2:
        core = order[:core_size]
        fallback_groups = [
            order[core_size : core_size + 2],
            order[core_size + 2 : core_size + 4],
            order[core_size + 4 : core_size + 6],
            order[core_size + 6 : core_size + 8],
        ]
        weights: dict[str, int] = {c: 0 for c in order}

        def bullish(codes: list[str]) -> list[str]:
            return [c for c in codes if bool(signals.get(c, False))]

        def group_ladder(bull_codes: list[str], group: list[str]) -> list[int]:
            pos = [group.index(c) + 1 for c in bull_codes]
            key = ",".join(str(p) for p in pos)
            return [int(w) for w in ladder[key]]

        def scale_group(bull_codes: list[str], group: list[str], budget: int, base: list[int]) -> dict[str, int]:
            if len(bull_codes) == 1:
                return {bull_codes[0]: budget}
            raw = [
                base_w * max(0.0, float(positions.get(c, 1.0) or 1.0))
                for c, base_w in zip(bull_codes, base)
            ]
            total = sum(raw)
            if total <= 0:
                return {bull_codes[0]: budget}
            eff = [int(budget * v / total / 5.0) * 5 for v in raw]
            for idx in range(1, len(eff)):
                if eff[idx] > eff[idx - 1]:
                    eff[idx] = eff[idx - 1]
            leftover = budget - sum(eff)
            idx = 0
            while leftover > 0 and idx < len(eff):
                cap = budget if idx == 0 else eff[idx - 1]
                add = min(5, leftover, cap - eff[idx])
                if add <= 0:
                    idx += 1
                    continue
                eff[idx] += add
                leftover -= add
            return dict(zip(bull_codes, eff))

        core_bull = bullish(core)
        if core_bull:
            weights.update(scale_group(core_bull, core, core_budget, group_ladder(core_bull, core)))
            rem_codes = bullish(order[core_size:])[:2]
            if rem_codes:
                base = [15] if len(rem_codes) == 1 else [10, 5]
                budget = rem_budget if len(rem_codes) == 2 else min(20, rem_budget)
                weights.update(scale_group(rem_codes, order[core_size:], budget, base))
        else:
            for group in fallback_groups:
                gb = bullish(group)
                if gb:
                    weights.update(scale_group(gb, group, core_budget, group_ladder(gb, group)))
                    break
        for c in order:
            if weights[c] > max_single:
                weights[c] = max_single
        cash = 100 - sum(weights.values())
        if cash < 0:
            excess = -cash
            for c in sorted(weights, key=lambda x: -weights[x]):
                if excess <= 0:
                    break
                cut = min(weights[c], excess)
                weights[c] -= cut
                excess -= cut
            cash = 0
        return {"weights": weights, "cash_pct": cash}

    top3 = order[:3]
    core45 = order[3:5]
    tail = order[5:8]
    weights: dict[str, int] = {c: 0 for c in order}

    def bullish(codes: list[str]) -> list[str]:
        return [c for c in codes if bool(signals.get(c, False))]

    def group_ladder(bull_codes: list[str], group: list[str]) -> list[int]:
        pos = [group.index(c) + 1 for c in bull_codes]
        key = ",".join(str(p) for p in pos)
        return [int(w) for w in ladder[key]]

    def scale_group(bull_codes: list[str], group: list[str], budget: int, base: list[int]) -> dict[str, int]:
        if len(bull_codes) == 1:
            return {bull_codes[0]: budget}
        raw = [
            base_w * max(0.0, float(positions.get(c, 1.0) or 1.0))
            for c, base_w in zip(bull_codes, base)
        ]
        total = sum(raw)
        if total <= 0:
            return {bull_codes[0]: budget}
        eff = [int(budget * v / total / 5.0) * 5 for v in raw]
        for idx in range(1, len(eff)):
            if eff[idx] > eff[idx - 1]:
                eff[idx] = eff[idx - 1]
        leftover = budget - sum(eff)
        idx = 0
        while leftover > 0 and idx < len(eff):
            cap = budget if idx == 0 else eff[idx - 1]
            add = min(5, leftover, cap - eff[idx])
            if add <= 0:
                idx += 1
                continue
            eff[idx] += add
            leftover -= add
        return dict(zip(bull_codes, eff))

    top3_bull = bullish(top3)
    other_bull = bullish(order[3:])
    core45_bull = bullish(core45)
    tail_bull = bullish(tail)

    if top3_bull:
        weights.update(scale_group(top3_bull, top3, core_budget, group_ladder(top3_bull, top3)))
        if other_bull:
            rem_codes = other_bull[:2]
            base = [20] if len(rem_codes) == 1 else [20, 10]
            budget = rem_budget if len(rem_codes) == 2 else min(20, rem_budget)
            weights.update(scale_group(rem_codes, order[3:], budget, base))
    elif core45_bull:
        weights.update(scale_group(core45_bull, core45, core_budget, group_ladder(core45_bull, core45)))
        if tail_bull:
            rem_codes = tail_bull[:2]
            base = [20] if len(rem_codes) == 1 else [20, 10]
            budget = rem_budget if len(rem_codes) == 2 else min(20, rem_budget)
            weights.update(scale_group(rem_codes, tail, budget, base))
    elif tail_bull:
        weights.update(scale_group(tail_bull, tail, core_budget, group_ladder(tail_bull, tail)))

    for c in order:
        if weights[c] > max_single:
            weights[c] = max_single
    cash = 100 - sum(weights.values())
    if cash < 0:
        # 兜底：按权重从大到小削减到合计 100%
        excess = -cash
        for c in sorted(weights, key=lambda x: -weights[x]):
            if excess <= 0:
                break
            cut = min(weights[c], ((weights[c] + 4) // 5) * 5 - max(0, weights[c] - excess))
            cut = min(cut, weights[c])
            weights[c] -= cut
            excess -= cut
        cash = 0
    return {"weights": weights, "cash_pct": cash}


def _metrics_full(twr_rets: list[float], flows: list[tuple[int, float]], final_nav: float,
                  invested: float, avg_pos: float, turnover: float) -> dict:
    arr = np.asarray(twr_rets, dtype=float)
    index = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(index)
    mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    return {
        "cum_return": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "annual_return_irr": _irr(flows, final_nav),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "invested": invested,
        "profit": final_nav - invested,
        "avg_position": avg_pos,
        "final_value": final_nav,
        "twr": float(index[-1] - 1.0) if len(index) else 0.0,
        "turnover": turnover,
    }


def simulate(etfs: dict, master_dates: list[str], master_execs: list[str],
             start_idx: int, end_idx: int, order: list[str], params: dict) -> dict:
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = None
    for i in range(start_idx, end_idx + 1):
        if master_execs[i] is not None:
            first_ex = master_execs[i]
            break
    for c, e in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
            for key, event in e["ca_events"]
        }
    freq = max(1, int(params["rebalance"]))
    floor = float(params["floor"])
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    traded_total = 0.0
    nav_after_list: list[float] = []
    prev_after = None
    for i in range(start_idx, end_idx + 1):
        date = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        n_avail = sum(1 for e in etfs.values() if e["start_date"] <= date)
        if n_avail == 0:
            continue
        prices = {c: e["price_map"].get(ex, 0.0) for c, e in etfs.items()}
        for c, e in etfs.items():
            if prices[c] > 0 or shares[c] != 0:
                shares[c], cash = _apply_ca(shares[c], cash, e["ca_events"], ex, applied[c])
        nav_before = sum(shares[c] * prices[c] for c in etfs) + cash
        if prev_after is not None:
            twr_rets.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        flows.append((i - start_idx, -inflow))
        if (i - start_idx) % freq == 0:
            value = sum(shares[c] * prices[c] for c in etfs) + cash
            eff_pos = {}
            signals = {}
            for c, e in etfs.items():
                raw = e["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0
                eff = min(1.0, max(floor, raw))
                eff_pos[c] = eff
                signals[c] = eff > 0.0
            target = allocate_cfg(signals, eff_pos, order, params)
            for c, e in etfs.items():
                if prices[c] <= 0:
                    continue
                weight = float(target["weights"].get(c, 0)) / 100.0
                tgt_share = weight * value / prices[c]
                delta = tgt_share - shares[c]
                traded_total += abs(delta) * prices[c]
                cost = trade_cost(abs(delta) * prices[c])
                cash -= delta * prices[c] + cost
                shares[c] = tgt_share
        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        nav_after_list.append(nav_after)
        prev_after = nav_after
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    return _metrics_full(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover)


def range_idx(dates: list[str], start: str, end: str) -> tuple[int, int]:
    s = next((i for i, d in enumerate(dates) if d >= start), None)
    e_pos = next((i for i, d in enumerate(dates) if d > end), None)
    e = len(dates) - 1 if e_pos is None else e_pos - 1
    if s is None or s > e:
        raise ValueError(f"empty range {start}~{end}")
    return s, e


def make_models() -> list[dict]:
    rng = np.random.default_rng(SEED)

    def random_order() -> list[str]:
        while True:
            order = [CODES[i] for i in rng.permutation(len(CODES))]
            if (
                order.index("159915") <= 1
                and order.index("159622") <= 3
                and order.index("516150") <= 3
            ):
                return order

    models: list[dict] = [{
        "id": "BASE_current",
        "order": list(CURRENT_ORDER),
        "params": dict(DEFAULT_PARAMS),
        "baseline": True,
    }]
    core_choices = [50, 60, 70, 80]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    rebal_choices = [1, 2, 4]
    for idx in range(1, MODEL_COUNT + 1):
        core_budget = int(rng.choice(core_choices))
        rem_options = [r for r in (20, 30) if r <= 100 - core_budget]
        params = {
            "core_budget": core_budget,
            "remainder_budget": int(rng.choice(rem_options)),
            "floor": float(rng.choice(floor_choices)),
            "ladder": int(rng.choice(ladder_choices)),
            "max_single": int(rng.choice(max_choices)),
            "rebalance": int(rng.choice(rebal_choices)),
        }
        models.append({"id": f"M{idx:04d}", "order": random_order(), "params": params, "baseline": False})
    return models


def fmt_model(m: dict) -> str:
    p = m["params"]
    return (f"{m['id']} core={p['core_budget']} rem={p['remainder_budget']} floor={p['floor']:.0%} "
            f"ladder={p['ladder']} max={p['max_single']} freq={p['rebalance']}w "
            f"order={'/'.join(m['order'])}")


def main() -> int:
    etfs = {}
    for code in CODES:
        etfs[code] = load_etf(code)
        print("loaded", code, NAMES[code], "start", etfs[code]["start_date"])

    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]

    idx = {name: range_idx(master_dates, *bounds) for name, bounds in {
        "1y": WINDOW_1Y, "3y": WINDOW_3Y, "5y": WINDOW_5Y,
    }.items()}

    models = make_models()
    results: dict[str, dict] = {}
    for m in models:
        r1 = simulate(etfs, master_dates, master_execs, *idx["1y"], m["order"], m["params"])
        r3 = simulate(etfs, master_dates, master_execs, *idx["3y"], m["order"], m["params"])
        r5 = simulate(etfs, master_dates, master_execs, *idx["5y"], m["order"], m["params"])
        score = (
            COMPOSITE_WEIGHTS["1y"] * r1["cum_return"]
            + COMPOSITE_WEIGHTS["3y"] * r3["cum_return"]
            + COMPOSITE_WEIGHTS["5y"] * r5["cum_return"]
        )
        results[m["id"]] = {"model": m, "1y": r1, "3y": r3, "5y": r5, "score": score}

    base = results["BASE_current"]
    champion = max(results.values(), key=lambda r: r["score"])
    top20 = sorted(results.values(), key=lambda r: r["score"], reverse=True)[:20]
    in_sample_best = max(results.values(), key=lambda r: r["5y"]["cum_return"])
    dca = {w: gem_dca_account(etfs, master_dates, master_execs, *idx[w]) for w in ("1y", "3y", "5y")}

    # ---------- 落盘 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "allocation_optimization_models.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        headers = [
            "模型ID", "是否基线", "顺序", "核心预算", "剩余预算", "看空下限", "档位变体", "单只上限",
            "再平衡(周)", "近1年累计收益率", "近3年累计收益率", "近5年累计收益率", "综合评分",
            "5年IRR", "5年Sharpe", "5年最大回撤", "5年平均持仓", "5年换手",
            "5年期末市值", "5年累计投入",
        ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rid, r in results.items():
            m = r["model"]
            p = m["params"]
            writer.writerow({
                "模型ID": rid,
                "是否基线": "是" if m["baseline"] else "否",
                "顺序": "/".join(m["order"]),
                "核心预算": p["core_budget"],
                "剩余预算": p["remainder_budget"],
                "看空下限": f"{p['floor']:.0%}",
                "档位变体": p["ladder"],
                "单只上限": p["max_single"],
                "再平衡(周)": p["rebalance"],
                "近1年累计收益率": f"{r['1y']['cum_return']:.2%}",
                "近3年累计收益率": f"{r['3y']['cum_return']:.2%}",
                "近5年累计收益率": f"{r['5y']['cum_return']:.2%}",
                "综合评分": f"{r['score']:.4f}",
                "5年IRR": f"{r['5y']['annual_return_irr']:.2%}",
                "5年Sharpe": f"{r['5y']['sharpe']:.2f}",
                "5年最大回撤": f"{r['5y']['max_drawdown']:.2%}",
                "5年平均持仓": f"{r['5y']['avg_position']:.2%}",
                "5年换手": f"{r['5y']['turnover']:.3f}",
                "5年期末市值": f"{r['5y']['final_value']:.0f}",
                "5年累计投入": f"{r['5y']['invested']:.0f}",
            })

    def row_line(r: dict) -> str:
        m = r["model"]
        return (f"| {m['id']} | {m['params']['core_budget']}% | {m['params']['remainder_budget']}% | "
                f"{m['params']['floor']:.0%} | {m['params']['ladder']} | {m['params']['max_single']}% | "
                f"{m['params']['rebalance']}周 | {r['1y']['cum_return']:.1%} | "
                f"{r['3y']['cum_return']:.1%} | {r['5y']['cum_return']:.1%} | {r['score']:.3f} | "
                f"{r['5y']['annual_return_irr']:.1%} | {r['5y']['sharpe']:.2f} | "
                f"{r['5y']['max_drawdown']:.1%} | {r['5y']['avg_position']:.1%} | "
                f"{r['5y']['final_value']:.0f} |")

    lines = [
        "# 8 ETF 投资策略持仓分配优化报告",
        "",
        f"模型数：{len(models)}（含基线 1 个 + 随机采样 {MODEL_COUNT} 个）· 种子 {SEED}",
        "口径：2021-08-10~2026-08-07 周频；每周入金 2000×n；raw_open+10bp+CA；冠军链信号。",
        "硬约束：159915 创业板必须排在前二，159622 创新药与 516150 稀土必须排在前四。",
        "评选：综合评分 = 70%×近一年累计收益率 + 20%×近三年累计收益率 + 10%×近五年累计收益率；"
        "近五年最高收益模型为 in-sample 参照。",
        "",
        f"## 最终冠军：{champion['model']['id']}",
        "",
        f"- 参数：核心预算 {champion['model']['params']['core_budget']}% / 剩余预算 "
        f"{champion['model']['params']['remainder_budget']}% / 看空下限 {champion['model']['params']['floor']:.0%} / "
        f"档位变体 {champion['model']['params']['ladder']} / 单只上限 {champion['model']['params']['max_single']}% / "
        f"再平衡 {champion['model']['params']['rebalance']} 周",
        f"- 优先级顺序：{' → '.join(champion['model']['order'])}",
        f"- 近一年累计收益率：{champion['1y']['cum_return']:.2%}",
        f"- 近三年累计收益率：{champion['3y']['cum_return']:.2%}",
        f"- 近五年累计收益率：{champion['5y']['cum_return']:.2%}",
        f"- 综合评分：{champion['score']:.4f}",
        "",
        "## 冠军 vs 基线 vs 智能定投（三窗口）",
        "",
        "| 方案 | 近1年 | 近3年 | 近5年 | 综合评分 | 5年IRR | 5年Sharpe | 5年MDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in (
        ("综合冠军", champion),
        ("当前策略基线", base),
    ):
        lines.append(
            f"| {label} | {r['1y']['cum_return']:.1%} | {r['3y']['cum_return']:.1%} | "
            f"{r['5y']['cum_return']:.1%} | {r['score']:.3f} | "
            f"{r['5y']['annual_return_irr']:.1%} | {r['5y']['sharpe']:.2f} | "
            f"{r['5y']['max_drawdown']:.1%} |"
        )
    lines.append(
        f"| 近5年最高(in-sample) | {in_sample_best['1y']['cum_return']:.1%} | "
        f"{in_sample_best['3y']['cum_return']:.1%} | {in_sample_best['5y']['cum_return']:.1%} | "
        f"{in_sample_best['score']:.3f} | {in_sample_best['5y']['annual_return_irr']:.1%} | "
        f"{in_sample_best['5y']['sharpe']:.2f} | {in_sample_best['5y']['max_drawdown']:.1%} |"
    )
    for w, wname in (("1y", "近1年"), ("3y", "近3年"), ("5y", "近5年")):
        m = dca[w]
        lines.append(
            f"| 创业板智能定投({wname}) | {m['cum_return']:.1%} | - | - | - | - | "
            f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} |"
        )
    lines += ["", "## Top 20（按综合评分排序）", "",
              "| 模型 | 核心 | 剩余 | 下限 | 档位 | 单只上限 | 频率 | 近1年 | 近3年 | 近5年 | 综合分 | 5年IRR | Sharpe | MDD | 持仓 | 期末市值 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in top20:
        lines.append(row_line(r))
    lines += ["", f"## 近5年最高收益模型（in-sample 参照）：{in_sample_best['model']['id']}", ""]
    lines.append(row_line(in_sample_best))
    lines += ["", f"完整 {len(results)} 个模型指标见 allocation_optimization_models.csv / .json"]
    md_path = OUT_DIR / "allocation_optimization_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_payload = {
        "schema_version": "allocation-optimization-v2",
        "seed": SEED,
        "model_count": len(models),
        "windows": {k: {"start": v[0], "end": v[1]} for k, v in {
            "1y": WINDOW_1Y, "3y": WINDOW_3Y, "5y": WINDOW_5Y,
        }.items()},
        "composite_weights": COMPOSITE_WEIGHTS,
        "champion": {
            "id": champion["model"]["id"],
            "order": champion["model"]["order"],
            "params": champion["model"]["params"],
            "1y": champion["1y"],
            "3y": champion["3y"],
            "5y": champion["5y"],
            "score": champion["score"],
        },
        "in_sample_best": {
            "id": in_sample_best["model"]["id"],
            "order": in_sample_best["model"]["order"],
            "params": in_sample_best["model"]["params"],
            "1y": in_sample_best["1y"],
            "3y": in_sample_best["3y"],
            "5y": in_sample_best["5y"],
            "score": in_sample_best["score"],
        },
        "baseline": {
            "id": "BASE_current",
            "1y": base["1y"],
            "3y": base["3y"],
            "5y": base["5y"],
            "score": base["score"],
        },
        "gem_dca": dca,
        "top20": [
            {"id": r["model"]["id"], "order": r["model"]["order"], "params": r["model"]["params"],
             "1y": r["1y"], "3y": r["3y"], "5y": r["5y"], "score": r["score"]}
            for r in top20
        ],
        "all_models": [
            {"id": rid, "order": r["model"]["order"], "params": r["model"]["params"],
             "1y": r["1y"], "3y": r["3y"], "5y": r["5y"], "score": r["score"]}
            for rid, r in results.items()
        ],
    }
    (OUT_DIR / "allocation_optimization.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    champion_cfg = {
        "champion_id": champion["model"]["id"],
        "order": champion["model"]["order"],
        "params": champion["model"]["params"],
        "note": "由 allocation_optimization.py 综合评价评选产出（1y/3y/5y 权重 0.7/0.2/0.1）",
    }
    cfg_path = ROOT / "configs" / "optimized_allocation_champion.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(champion_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved", md_path)
    print("champion:", fmt_model(champion["model"]))
    for w in ("1y", "3y", "5y"):
        print(w, "champion", round(champion[w]["cum_return"], 6), "| baseline", round(base[w]["cum_return"], 6))
    print("score champion", round(champion["score"], 4), "| baseline", round(base["score"], 4))
    print("in_sample best:", in_sample_best["model"]["id"], round(in_sample_best["5y"]["cum_return"], 6))
    print("gem dca:", {w: round(dca[w]["cum_return"], 6) for w in ("1y", "3y", "5y")})
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "LEGACY_BLOCKED: 本脚本固定使用518600/512010旧ETF池，只保留历史研究复现；"
        "当前项目不得运行或覆盖optimized_allocation_champion.json。"
    )
