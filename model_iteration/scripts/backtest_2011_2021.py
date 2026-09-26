# -*- coding: utf-8 -*-
"""按当前策略（B0207 + 周度强看空保护）回测 2011-2021（数据自 2012-09-14 起）。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _irr  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CASH_FACTOR,
    gem_dca_realistic,
    LOT,
    MIN_COMMISSION,
    COMMISSION_RATE,
    OUT_DIR,
    WINDOW_5Y,
)

WEEKLY = 2000.0
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
MOM20_TH = -0.05


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def _metrics(twr_rets, flows, final_nav, invested, avg_pos, turnover, triggers):
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
        "triggers": triggers,
    }


def pct(x) -> str:
    return f"{float(x):.1%}"


def main() -> int:
    etfs = {c: load_etf(c) for c in CODES}
    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    start5y = range_idx(master_dates, *WINDOW_5Y)[0]
    s = 0
    e = next((k for k in range(len(master_dates) - 1, -1, -1)
              if master_dates[k] <= "2021-12-31"), 0)
    print("window:", master_dates[s], "~", master_dates[e], "anchors", e - s + 1)

    def simulate(use_guard: bool) -> dict:
        cash = 0.0
        shares = {c: 0.0 for c in etfs}
        applied = {}
        first_ex = next((ex for ex in master_execs[s : e + 1] if ex is not None), None)
        for c, et in etfs.items():
            applied[c] = {
                key: (str(ev.get("pay_date") or ev.get("ex_date")) < first_ex)
                for key, ev in et["ca_events"]
            }
        twr_rets: list[float] = []
        flows: list[tuple[int, float]] = []
        invested = 0.0
        traded_total = 0.0
        nav_after_list: list[float] = []
        exposures: list[float] = []
        prev_after = None
        reduced = {c: False for c in etfs}
        triggers = 0

        def apply_weights(target: dict, value: float) -> None:
            nonlocal cash, traded_total
            for c, et in etfs.items():
                if prices[c] <= 0:
                    continue
                weight = float(target.get(c, 0)) / 100.0
                tgt_share = int(weight * value / prices[c] // LOT) * LOT
                delta = tgt_share - shares[c]
                traded = abs(delta) * prices[c]
                traded_total += traded
                cost = trade_cost(traded)
                cash -= delta * prices[c] + cost
                shares[c] = tgt_share

        def deploy(inflow: float) -> None:
            nonlocal cash, traded_total
            holdings_value = sum(shares[c] * prices[c] for c in etfs if prices[c] > 0)
            if holdings_value <= 0:
                return
            for c, et in etfs.items():
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

        for i in range(s, e + 1):
            date = master_dates[i]
            ex = master_execs[i]
            if ex is None:
                continue
            cash *= CASH_FACTOR
            n_avail = sum(1 for et in etfs.values() if et["start_date"] <= date)
            if n_avail == 0:
                continue
            prices = {c: et["price_map"].get(ex, 0.0) for c, et in etfs.items()}
            for c, et in etfs.items():
                if prices[c] > 0 or shares[c] != 0:
                    shares[c], cash = _apply_ca(shares[c], cash, et["ca_events"], ex, applied[c])
            nav_before = sum(shares[c] * prices[c] for c in etfs) + cash
            if prev_after is not None:
                twr_rets.append(nav_before / prev_after - 1.0)
            inflow = WEEKLY * n_avail
            cash += inflow
            invested += inflow
            flows.append((i - s, -inflow))
            value = sum(shares[c] * prices[c] for c in etfs) + cash

            # 周度强看空保护
            if use_guard:
                for c, et in etfs.items():
                    feat = et.get("feat_by_date", {}).get(date, {})
                    trig = (
                        shares[c] > 0 and prices[c] > 0
                        and float(feat.get("d_dif1", 1.0)) < 0
                        and float(feat.get("d_dif2", 1.0)) < 0
                        and float(feat.get("w_dif1", 1.0)) < 0
                        and float(feat.get("w_dif2", 1.0)) < 0
                        and float(feat.get("mom20", 0.0)) < MOM20_TH
                    )
                    if trig:
                        if not reduced[c]:
                            sell_lots = int(shares[c] * 0.7 // LOT) * LOT
                            if sell_lots > 0:
                                traded = sell_lots * prices[c]
                                traded_total += traded
                                cost = trade_cost(traded)
                                cash += sell_lots * prices[c] - cost
                                shares[c] -= sell_lots
                                triggers += 1
                            reduced[c] = True
                    else:
                        reduced[c] = False

            if (i - start5y) % int(B0207_PARAMS["rebalance"]) == 0:
                raw = {c: (et["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0)
                       for c, et in etfs.items()}
                floor = float(B0207_PARAMS["floor"])
                eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
                sig = {c: eff[c] > 0.0 for c in etfs}
                target = allocate_cfg(sig, eff, B0207_ORDER, B0207_PARAMS)
                apply_weights(target["weights"], value)
            else:
                deploy(inflow)

            nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
            exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
            nav_after_list.append(nav_after)
            prev_after = nav_after

        avg_pos = float(np.mean(exposures)) if exposures else 0.0
        total_nav = sum(nav_after_list)
        turnover = traded_total / total_nav if total_nav > 0 else 0.0
        return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover, triggers)

    base = simulate(False)
    guarded = simulate(True)
    dca = gem_dca_realistic(etfs, master_dates, master_execs, s, e)
    print("base   :", {k: round(v, 4) for k, v in base.items() if k in ("cum_return", "twr", "mdd", "sharpe", "triggers")})
    print("guarded:", {k: round(v, 4) for k, v in guarded.items() if k in ("cum_return", "twr", "mdd", "sharpe", "triggers")})
    print("dca    :", {k: round(v, 4) for k, v in dca.items() if k in ("cum_return", "twr", "mdd", "sharpe")})

    lines = [
        "# 当前策略 2011-2021 回测（数据自 2012-09-14 起，至 2021-12-31）",
        "",
        "策略 = B0207（核心85/15、28天决策、每周入金2000×n、每周加仓）",
        "周度强看空保护 = 日K一阶导<0、日K二阶导<0、周K一阶导<0、周K二阶导<0 且 mom20<-5% → 卖出该ETF 70%（同段一次，恢复重置）",
        "",
        "| 方案 | 累计收益率 | TWR | IRR | Sharpe | MDD | 平均持仓 | 换手 | 期末市值 | 保护触发次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in (("当前策略（含保护）", guarded), ("当前策略（无保护）", base),
                     ("智能定投（159915）", dca)):
        lines.append(
            f"| {label} | {pct(r['cum_return'])} | {pct(r['twr'])} | {pct(r['annual_return_irr'])} | "
            f"{r['sharpe']:.2f} | {pct(r['max_drawdown'])} | {pct(r['avg_position'])} | "
            f"{r['turnover']:.2f} | {r['final_value']:.0f} | {r.get('triggers', '-')} |"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "backtest_2011_2021.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "backtest_2011_2021.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方案", "累计收益率", "TWR", "IRR", "Sharpe", "MDD", "平均持仓", "换手", "期末市值", "保护触发次数"])
        for label, r in (("当前策略（含保护）", guarded), ("当前策略（无保护）", base),
                         ("智能定投（159915）", dca)):
            writer.writerow([label, pct(r["cum_return"]), pct(r["twr"]), pct(r["annual_return_irr"]),
                             f"{r['sharpe']:.2f}", pct(r["max_drawdown"]), pct(r["avg_position"]),
                             f"{r['turnover']:.3f}", f"{r['final_value']:.0f}", r.get("triggers", "-")])
    payload = {"schema_version": "backtest-2011-2021-v1", "base": base, "guarded": guarded, "dca": dca}
    (OUT_DIR / "backtest_2011_2021.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
