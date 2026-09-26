# -*- coding: utf-8 -*-
"""8 ETF 投资策略持仓分配优化 v2（稳健评价 + 全枚举顺序 + 真实账户）。

相比 v1 的三处升级：
1. 评价稳健化：使用三个不重叠年度窗口
   - Y1 2023-08-10~2024-08-09（早期年）
   - Y2 2024-08-10~2025-08-09（中间年）
   - Y3 2025-08-10~2026-08-07（最近年）
   综合评分 = 70%×Y3 + 20%×Y2 + 10%×Y1（近一年优先，避免嵌套窗口重复加权）。
   稳健门槛：三个年度至少两年为正，且近五年累计收益率 ≥ 基线。
2. 搜索空间：全枚举满足硬约束的 1440 种优先级顺序
   （159915 前二、159622 前四、516150 前四）× 基础参数 → Top 顺序；
   再对 Top 顺序做局部参数细化随机搜索（约 4000 个模型）。
3. 账户真实化：
   - 现金按年化 1.75% 计息；
   - ETF 份额按 100 份整手成交；
   - 成交成本按东方财富实收：佣金 万2.5（0.025%），单笔最低 5 元，
     深市 ETF 免印花税、免过户费（单边计算）。

其余口径不变：周入金 2000×n、raw_open 成交、显式 CA 账本、冠军链信号、
5% 倍数、合计 100%。
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _irr  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402

WEEKLY = 2000.0
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0
CASH_APR = 0.0175
LOT = 100

SEED = 42
PHASE_B_SEED = 2026
PHASE_B_PER_ORDER = 80
TOP_ORDERS = 50

WINDOW_Y1 = ("2023-08-10", "2024-08-09")
WINDOW_Y2 = ("2024-08-10", "2025-08-09")
WINDOW_Y3 = ("2025-08-10", "2026-08-07")
WINDOW_5Y = ("2021-08-10", "2026-08-07")
SCORE_WEIGHTS = {"Y1": 0.1, "Y2": 0.2, "Y3": 0.7}

CURRENT_ORDER = ["159915", "512010", "159622", "516150", "159941", "512690", "512800", "518600"]
BASE_PARAMS = {"core_budget": 80, "remainder_budget": 20, "floor": 0.0,
               "ladder": 3, "max_single": 100, "rebalance": 4}
DEFAULT_PARAMS = {"core_budget": 70, "remainder_budget": 30, "floor": 0.0,
                  "ladder": 0, "max_single": 100, "rebalance": 1}

OUT_DIR = ROOT.parent / "champion_vs_dca"
CASH_FACTOR = (1.0 + CASH_APR) ** (1.0 / 52.0)


def trade_cost(value: float) -> float:
    """单边成交成本：佣金 max(成交金额×0.025%, 5元)；深市 ETF 无印花税/过户费。"""
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def all_valid_orders() -> list[list[str]]:
    """全枚举满足硬约束的优先级顺序：159915 前二、159622/516150 前四。"""
    fixed = {"159915", "159622", "516150"}
    others = [c for c in CODES if c not in fixed]
    orders: set[tuple[str, ...]] = set()
    for pos915 in (0, 1):
        slots = [p for p in range(4) if p != pos915]
        for s1, s2 in itertools.permutations(slots, 2):
            base: list[str | None] = [None] * 8
            base[pos915] = "159915"
            base[s1] = "159622"
            base[s2] = "516150"
            free = [i for i in range(8) if base[i] is None]
            for rest in itertools.permutations(others):
                order = list(base)
                for idx, code in zip(free, rest):
                    order[idx] = code
                orders.add(tuple(order))
    return [list(o) for o in sorted(orders)]


def _metrics(twr_rets: list[float], flows: list[tuple[int, float]], final_nav: float,
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


def simulate_realistic(etfs: dict, master_dates: list[str], master_execs: list[str],
                       start_idx: int, end_idx: int, order: list[str], params: dict,
                       weekly_deploy: bool = False, bearish_guard: bool = False,
                       guard_stats: dict | None = None,
                       guard_mode: str = "any", phase_offset: int = 0) -> dict:
    """真实账户模拟。

    weekly_deploy=False（默认）：非决策周入金存入现金池，决策周统一按目标权重调仓。
    weekly_deploy=True：非决策周入金立即按“现有持仓市值比例”加仓，
    决策周仍按目标权重调仓（每 4 周决策一次的方案保持不变）。
    bearish_guard=True：非决策周额外做“周度看空分析”——只要当前持仓中任一 ETF
    冠军仓位归零（看空），立即按当时 B0207 目标权重调仓；否则按 weekly_deploy
    决定加仓或持有，不因普通信号变化而交易。
    guard_mode：any=任一持仓看空即触发；confirm2=任一持仓连续2周看空才触发；
    majority=持仓中超一半看空才触发；all=全部持仓看空才触发。
    phase_offset：决策周起点偏移（0~freq-1），用于相位敏感性测试。
    """
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[start_idx : end_idx + 1] if ex is not None), None)
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
    consecutive_bearish = {c: 0 for c in etfs}
    for i in range(start_idx, end_idx + 1):
        date = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        cash *= CASH_FACTOR
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
        if (i - start_idx - int(phase_offset)) % freq == 0:
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
                tgt_share = int(weight * value / prices[c] // LOT) * LOT
                delta = tgt_share - shares[c]
                traded = abs(delta) * prices[c]
                traded_total += traded
                cost = trade_cost(traded)
                cash -= delta * prices[c] + cost
                shares[c] = tgt_share
        elif bearish_guard or weekly_deploy:
            value = sum(shares[c] * prices[c] for c in etfs) + cash
            eff_pos = {}
            signals = {}
            raw_pos = {}
            for c, e in etfs.items():
                raw = e["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0
                raw_pos[c] = raw
                eff = min(1.0, max(floor, raw))
                eff_pos[c] = eff
                signals[c] = eff > 0.0
            if bearish_guard:
                held = [c for c in etfs if shares[c] > 0 and prices[c] > 0]
                for c in etfs:
                    consecutive_bearish[c] = (
                        consecutive_bearish[c] + 1
                        if shares[c] > 0 and prices[c] > 0 and raw_pos[c] <= 0
                        else 0
                    )
                if guard_mode == "confirm2":
                    trigger = any(consecutive_bearish[c] >= 2 for c in held)
                elif guard_mode == "majority":
                    bearish_count = sum(1 for c in held if raw_pos[c] <= 0)
                    trigger = bool(held) and bearish_count * 2 > len(held)
                elif guard_mode == "all":
                    trigger = bool(held) and all(raw_pos[c] <= 0 for c in held)
                else:
                    trigger = any(raw_pos[c] <= 0 for c in held)
                if trigger:
                    if guard_stats is not None:
                        guard_stats["triggers"] = guard_stats.get("triggers", 0) + 1
                    target = allocate_cfg(signals, eff_pos, order, params)
                    for c, e in etfs.items():
                        if prices[c] <= 0:
                            continue
                        weight = float(target["weights"].get(c, 0)) / 100.0
                        tgt_share = int(weight * value / prices[c] // LOT) * LOT
                        delta = tgt_share - shares[c]
                        traded = abs(delta) * prices[c]
                        traded_total += traded
                        cost = trade_cost(traded)
                        cash -= delta * prices[c] + cost
                        shares[c] = tgt_share
                elif weekly_deploy:
                    holdings_value = sum(shares[c] * prices[c] for c in etfs if prices[c] > 0)
                    if holdings_value > 0:
                        for c, e in etfs.items():
                            if prices[c] <= 0 or shares[c] <= 0:
                                continue
                            alloc = inflow * (shares[c] * prices[c] / holdings_value)
                            add_share = int(alloc / prices[c] // LOT) * LOT
                            if add_share <= 0:
                                continue
                            traded = add_share * prices[c]
                            traded_total += traded
                            cost = trade_cost(traded)
                            cash -= traded + cost
                            shares[c] += add_share
            elif weekly_deploy:
                holdings_value = sum(shares[c] * prices[c] for c in etfs if prices[c] > 0)
                if holdings_value > 0:
                    for c, e in etfs.items():
                        if prices[c] <= 0 or shares[c] <= 0:
                            continue
                        alloc = inflow * (shares[c] * prices[c] / holdings_value)
                        add_share = int(alloc / prices[c] // LOT) * LOT
                        if add_share <= 0:
                            continue
                        traded = add_share * prices[c]
                        traded_total += traded
                        cost = trade_cost(traded)
                        cash -= traded + cost
                        shares[c] += add_share
        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        nav_after_list.append(nav_after)
        prev_after = nav_after
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover)


def gem_dca_realistic(etfs: dict, master_dates: list[str], master_execs: list[str],
                      start_idx: int, end_idx: int) -> dict:
    e = etfs["159915"]
    shares = 0.0
    cash = 0.0
    cost_adj = 0.0
    first_ex = next((ex for ex in master_execs[start_idx : end_idx + 1] if ex is not None), None)
    applied = {
        key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
        for key, event in e["ca_events"]
    }
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    prev_after = None
    for i in range(start_idx, end_idx + 1):
        date = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        cash *= CASH_FACTOR
        n_avail = sum(1 for ee in etfs.values() if ee["start_date"] <= date)
        if n_avail == 0:
            continue
        price = float(e["price_map"][ex])
        shares, cash = _apply_ca(shares, cash, e["ca_events"], ex, applied)
        nav_before = shares * price + cash
        if prev_after is not None:
            twr_rets.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        flows.append((i - start_idx, -inflow))
        prev_dates = e["closes"].index[e["closes"].index < ex]
        nav_ref = float(e["closes"].loc[prev_dates[-1]]) if len(prev_dates) else price
        avg_cost = cost_adj / shares if shares > 0 else nav_ref
        pct = (nav_ref - avg_cost) / avg_cost if shares > 0 else 0.0
        if pct >= 0.025:
            rate = max(0.5, 1.0 - 2.0 * (pct - 0.025))
        elif pct <= -0.025:
            rate = min(2.0, 1.0 + 2.0 * (-pct - 0.025))
        else:
            rate = 1.0
        amount = min(inflow * rate, cash)
        if amount > 0:
            fee = trade_cost(amount)
            net = amount - fee
            lot = int(net / price // LOT) * LOT
            if lot > 0:
                shares += lot
                cost_adj += lot * price
                cash -= lot * price + fee
            else:
                cash -= fee
        nav_after = shares * price + cash
        exposures.append(shares * price / nav_after if nav_after > 0 else 0.0)
        prev_after = nav_after
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, 0.0)


def robust_score(r: dict) -> float:
    return sum(SCORE_WEIGHTS[w] * r[w]["cum_return"] for w in ("Y1", "Y2", "Y3"))


def passes_gates(r: dict, baseline_5y: float) -> bool:
    annual = [r["Y1"]["cum_return"], r["Y2"]["cum_return"], r["Y3"]["cum_return"]]
    pos_years = sum(1 for x in annual if x > 0)
    return pos_years >= 2 and r["Y5"]["cum_return"] >= baseline_5y


def run_windows(etfs, master_dates, master_execs, idx, order, params) -> dict:
    return {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params)
            for w in ("Y1", "Y2", "Y3", "Y5")}


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
    idx = {w: range_idx(master_dates, *bounds) for w, bounds in {
        "Y1": WINDOW_Y1, "Y2": WINDOW_Y2, "Y3": WINDOW_Y3, "Y5": WINDOW_5Y,
    }.items()}

    # 基线（真实账户口径）
    base = {"model": {"id": "BASE_current", "order": CURRENT_ORDER, "params": DEFAULT_PARAMS}}
    base.update(run_windows(etfs, master_dates, master_execs, idx, CURRENT_ORDER, DEFAULT_PARAMS))
    base["score"] = robust_score(base)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline:", {w: round(base[w]["cum_return"], 6) for w in ("Y1", "Y2", "Y3", "Y5")},
          "score", round(base["score"], 4))

    # 阶段 A：全枚举顺序 × 基础参数
    orders = all_valid_orders()
    print("valid orders:", len(orders))
    phase_a: list[dict] = []
    for oi, order in enumerate(orders):
        r = run_windows(etfs, master_dates, master_execs, idx, order, BASE_PARAMS)
        rec = {"model": {"id": f"A{oi:04d}", "order": order, "params": dict(BASE_PARAMS), "baseline": False}}
        rec.update(r)
        rec["score"] = robust_score(rec)
        phase_a.append(rec)
    top_orders = sorted(
        (r for r in phase_a if passes_gates(r, baseline_5y)),
        key=lambda r: r["score"], reverse=True,
    )[:TOP_ORDERS]
    if not top_orders:
        top_orders = sorted(phase_a, key=lambda r: r["score"], reverse=True)[:TOP_ORDERS]
    print("phase A done; top order score:", round(top_orders[0]["score"], 4) if top_orders else None)

    # 阶段 B：Top 顺序 × 细化参数随机搜索
    rng = np.random.default_rng(PHASE_B_SEED)
    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    rebal_choices = [2, 3, 4, 6]
    phase_b: list[dict] = []
    bid = 0
    for tr in top_orders:
        order = tr["model"]["order"]
        for _ in range(PHASE_B_PER_ORDER):
            core = int(rng.choice(core_choices))
            rem_options = [r for r in rem_choices if r <= 100 - core]
            params = {
                "core_budget": core,
                "remainder_budget": int(rng.choice(rem_options)),
                "floor": float(rng.choice(floor_choices)),
                "ladder": int(rng.choice(ladder_choices)),
                "max_single": int(rng.choice(max_choices)),
                "rebalance": int(rng.choice(rebal_choices)),
            }
            rec = {"model": {"id": f"B{bid:04d}", "order": order, "params": params, "baseline": False}}
            rec.update(run_windows(etfs, master_dates, master_execs, idx, order, params))
            rec["score"] = robust_score(rec)
            phase_b.append(rec)
            bid += 1
    print("phase B done:", len(phase_b))

    candidates = [r for r in phase_b if passes_gates(r, baseline_5y)]
    if not candidates:
        candidates = phase_b
    champion = max(candidates, key=lambda r: r["score"])
    top20 = sorted(candidates, key=lambda r: r["score"], reverse=True)[:20]
    in_sample_best = max(phase_b, key=lambda r: r["Y5"]["cum_return"])
    dca = {w: gem_dca_realistic(etfs, master_dates, master_execs, *idx[w]) for w in ("Y1", "Y2", "Y3", "Y5")}

    # ---------- 落盘 ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_models = phase_a + phase_b
    csv_path = OUT_DIR / "allocation_optimization_v2_models.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        headers = [
            "模型ID", "顺序", "核心预算", "剩余预算", "看空下限", "档位变体", "单只上限", "再平衡(周)",
            "Y1累计收益率", "Y2累计收益率", "Y3累计收益率", "5年累计收益率", "综合评分",
            "通过门槛", "5年IRR", "5年Sharpe", "5年最大回撤", "5年平均持仓", "5年换手", "5年期末市值",
        ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in all_models:
            m = r["model"]
            p = m["params"]
            writer.writerow({
                "模型ID": m["id"],
                "顺序": "/".join(m["order"]),
                "核心预算": p["core_budget"],
                "剩余预算": p["remainder_budget"],
                "看空下限": f"{p['floor']:.0%}",
                "档位变体": p["ladder"],
                "单只上限": p["max_single"],
                "再平衡(周)": p["rebalance"],
                "Y1累计收益率": f"{r['Y1']['cum_return']:.2%}",
                "Y2累计收益率": f"{r['Y2']['cum_return']:.2%}",
                "Y3累计收益率": f"{r['Y3']['cum_return']:.2%}",
                "5年累计收益率": f"{r['Y5']['cum_return']:.2%}",
                "综合评分": f"{r['score']:.4f}",
                "通过门槛": "是" if passes_gates(r, baseline_5y) else "否",
                "5年IRR": f"{r['Y5']['annual_return_irr']:.2%}",
                "5年Sharpe": f"{r['Y5']['sharpe']:.2f}",
                "5年最大回撤": f"{r['Y5']['max_drawdown']:.2%}",
                "5年平均持仓": f"{r['Y5']['avg_position']:.2%}",
                "5年换手": f"{r['Y5']['turnover']:.3f}",
                "5年期末市值": f"{r['Y5']['final_value']:.0f}",
            })

    def row_line(r: dict) -> str:
        m = r["model"]
        p = m["params"]
        gate = "是" if passes_gates(r, baseline_5y) else "否"
        return (f"| {m['id']} | {p['core_budget']}% | {p['remainder_budget']}% | {p['floor']:.0%} | "
                f"{p['ladder']} | {p['max_single']}% | {p['rebalance']}周 | "
                f"{r['Y1']['cum_return']:.1%} | {r['Y2']['cum_return']:.1%} | {r['Y3']['cum_return']:.1%} | "
                f"{r['Y5']['cum_return']:.1%} | {r['score']:.3f} | {gate} | "
                f"{r['Y5']['annual_return_irr']:.1%} | {r['Y5']['sharpe']:.2f} | "
                f"{r['Y5']['max_drawdown']:.1%} | {r['Y5']['avg_position']:.1%} | "
                f"{r['Y5']['final_value']:.0f} |")

    lines = [
        "# 8 ETF 投资策略持仓分配优化报告 v2（稳健评价 + 全枚举顺序 + 真实账户）",
        "",
        f"模型数：阶段A 全枚举顺序 {len(orders)} × 基础参数 + 阶段B Top{TOP_ORDERS} 顺序 × "
        f"每顺序 {PHASE_B_PER_ORDER} 个细化参数 = {len(phase_a) + len(phase_b)} 个模型",
        "硬约束：159915 创业板前二；159622 创新药与 516150 稀土前四。",
        "评价：综合评分 = 70%×最近年(Y3) + 20%×中间年(Y2) + 10%×早期年(Y1)，"
        "三个年度窗口不重叠；门槛 = 至少两年为正 且 近五年 ≥ 基线。",
        "账户真实化：现金年化 1.75% 计息；100 份整手；佣金万2.5（0.025%）单笔最低 5 元，"
        "深市 ETF 免印花税/过户费。",
        "",
        f"## 最终冠军：{champion['model']['id']}",
        "",
        f"- 优先级顺序：{' → '.join(champion['model']['order'])}",
        f"- 参数：核心预算 {champion['model']['params']['core_budget']}% / 剩余预算 "
        f"{champion['model']['params']['remainder_budget']}% / 看空下限 {champion['model']['params']['floor']:.0%} / "
        f"档位变体 {champion['model']['params']['ladder']} / 单只上限 {champion['model']['params']['max_single']}% / "
        f"再平衡 {champion['model']['params']['rebalance']} 周",
        f"- 早期年(Y1 2023-08~2024-08)：{champion['Y1']['cum_return']:.2%}",
        f"- 中间年(Y2 2024-08~2025-08)：{champion['Y2']['cum_return']:.2%}",
        f"- 最近年(Y3 2025-08~2026-08)：{champion['Y3']['cum_return']:.2%}",
        f"- 近五年：{champion['Y5']['cum_return']:.2%}（IRR {champion['Y5']['annual_return_irr']:.2%}）",
        f"- 综合评分：{champion['score']:.4f}",
        "",
        "## 冠军 vs 基线 vs 智能定投（真实账户口径）",
        "",
        "| 方案 | Y1 | Y2 | Y3 | 近5年 | 综合分 | 5年IRR | Sharpe | MDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in (("综合冠军", champion), ("当前策略基线", base)):
        lines.append(
            f"| {label} | {r['Y1']['cum_return']:.1%} | {r['Y2']['cum_return']:.1%} | "
            f"{r['Y3']['cum_return']:.1%} | {r['Y5']['cum_return']:.1%} | {r['score']:.3f} | "
            f"{r['Y5']['annual_return_irr']:.1%} | {r['Y5']['sharpe']:.2f} | "
            f"{r['Y5']['max_drawdown']:.1%} |"
        )
    lines.append(
        f"| 近5年最高(in-sample) | {in_sample_best['Y1']['cum_return']:.1%} | "
        f"{in_sample_best['Y2']['cum_return']:.1%} | {in_sample_best['Y3']['cum_return']:.1%} | "
        f"{in_sample_best['Y5']['cum_return']:.1%} | {in_sample_best['score']:.3f} | "
        f"{in_sample_best['Y5']['annual_return_irr']:.1%} | {in_sample_best['Y5']['sharpe']:.2f} | "
        f"{in_sample_best['Y5']['max_drawdown']:.1%} |"
    )
    for w, wname in (("Y1", "早期年"), ("Y2", "中间年"), ("Y3", "最近年"), ("Y5", "近5年")):
        m = dca[w]
        lines.append(
            f"| 创业板智能定投({wname}) | {m['cum_return']:.1%} | - | - | - | - | - | "
            f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} |"
        )
    lines += ["", "## Top 20（按综合评分，通过门槛优先）", "",
              "| 模型 | 核心 | 剩余 | 下限 | 档位 | 单只上限 | 频率 | Y1 | Y2 | Y3 | 近5年 | 综合分 | 门槛 | IRR | Sharpe | MDD | 持仓 | 市值 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in top20:
        lines.append(row_line(r))
    lines += ["", f"## 近5年最高收益模型（in-sample 参照）：{in_sample_best['model']['id']}", ""]
    lines.append(row_line(in_sample_best))
    lines += ["", f"完整 {len(all_models)} 个模型指标见 allocation_optimization_v2_models.csv / .json"]
    md_path = OUT_DIR / "allocation_optimization_v2_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_payload = {
        "schema_version": "allocation-optimization-v2",
        "method": {
            "score_weights": SCORE_WEIGHTS,
            "windows": {w: {"start": b[0], "end": b[1]} for w, b in {
                "Y1": WINDOW_Y1, "Y2": WINDOW_Y2, "Y3": WINDOW_Y3, "Y5": WINDOW_5Y,
            }.items()},
            "constraints": {"159915_top2": True, "159622_top4": True, "516150_top4": True},
            "account": {"cash_apr": CASH_APR, "lot": LOT,
                        "commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION},
            "phase_a_orders": len(orders),
            "phase_b_per_order": PHASE_B_PER_ORDER,
        },
        "baseline_5y_cum": baseline_5y,
        "champion": {
            "id": champion["model"]["id"], "order": champion["model"]["order"],
            "params": champion["model"]["params"],
            **{w: champion[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": champion["score"],
        },
        "baseline": {
            "id": "BASE_current", "order": CURRENT_ORDER, "params": DEFAULT_PARAMS,
            **{w: base[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": base["score"],
        },
        "in_sample_best": {
            "id": in_sample_best["model"]["id"], "order": in_sample_best["model"]["order"],
            "params": in_sample_best["model"]["params"],
            **{w: in_sample_best[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": in_sample_best["score"],
        },
        "gem_dca": dca,
        "top20": [
            {"id": r["model"]["id"], "order": r["model"]["order"], "params": r["model"]["params"],
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]}
            for r in top20
        ],
        "all_models": [
            {"id": r["model"]["id"], "order": r["model"]["order"], "params": r["model"]["params"],
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]}
            for r in all_models
        ],
    }
    (OUT_DIR / "allocation_optimization_v2.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    champion_cfg = {
        "champion_id": champion["model"]["id"],
        "order": champion["model"]["order"],
        "params": champion["model"]["params"],
        "note": "由 optimize_allocation_v2.py 产出（稳健年度评分 70/20/10 + 全枚举顺序 + 真实账户）",
    }
    cfg_path = ROOT / "configs" / "optimized_allocation_champion.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(champion_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved", md_path)
    print("champion:", champion["model"]["id"], "score", round(champion["score"], 4))
    print("champion Y1/Y2/Y3/Y5:", [round(champion[w]["cum_return"], 4) for w in ("Y1", "Y2", "Y3", "Y5")])
    print("baseline score:", round(base["score"], 4))
    print("dca:", {w: round(dca[w]["cum_return"], 4) for w in ("Y1", "Y2", "Y3", "Y5")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
