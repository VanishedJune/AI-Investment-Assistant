# -*- coding: utf-8 -*-
"""按 B0207 锚点（2021-08-13）计算最近5年/3年/1年收益，对比智能定投。

收益率口径：期末总价值 / 累计总投入 - 1。
窗口：
- 5年：2021-08-13 起 ~ 2026-08-07；
- 3年：2023-08 起 ~ 2026-08-07；
- 1年：2025-08 起 ~ 2026-08-07。
策略：B0207（28天决策、每周入金2000×n、每周加仓）+ 周度强看空保护（可选）。
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
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
    end = len(master_dates) - 1
    starts = {
        "5y": start5y,
        "3y": range_idx(master_dates, "2023-08-10", "2026-08-07")[0],
        "1y": range_idx(master_dates, "2025-08-10", "2026-08-07")[0],
    }

    def simulate(s: int, e: int, use_guard: bool) -> dict:
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

    results = {}
    for label, s in starts.items():
        results[label] = {
            "window": f"{master_dates[s]} ~ {master_dates[end]}",
            "guarded": simulate(s, end, True),
            "base": simulate(s, end, False),
            "dca": gem_dca_realistic(etfs, master_dates, master_execs, s, end),
        }
        print(label, results[label]["window"],
              "guarded", round(results[label]["guarded"]["cum_return"], 4),
              "base", round(results[label]["base"]["cum_return"], 4),
              "dca", round(results[label]["dca"]["cum_return"], 4))

    lines = [
        "# B0207 锚点窗口收益 vs 智能定投（收益率=总价值/总投入-1）",
        "",
        "锚点：2021-08-13 起每28天决策；窗口统一截至 2026-08-07。",
        "口径：每周入金2000×n、raw_open、万2.5最低5元、100份整手、CA账本。",
        "",
    ]
    for label, cn in (("5y", "最近5年"), ("3y", "最近3年"), ("1y", "最近1年")):
        r = results[label]
        lines.append(f"## {cn}（{r['window']}）")
        lines.append("")
        lines.append("| 方案 | 累计收益率(总价值/总投入-1) | TWR | IRR | Sharpe | MDD | 平均持仓 | 累计投入 | 期末总价值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, m in (("当前策略（含保护）", r["guarded"]),
                        ("当前策略（无保护）", r["base"]),
                        ("智能定投（159915）", r["dca"])):
            lines.append(
                f"| {name} | {pct(m['cum_return'])} | {pct(m['twr'])} | {pct(m['annual_return_irr'])} | "
                f"{m['sharpe']:.2f} | {pct(m['max_drawdown'])} | {pct(m['avg_position'])} | "
                f"{m['invested']:.0f} | {m['final_value']:.0f} |"
            )
        lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / "backtest_anchor_windows.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "backtest_anchor_windows.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["窗口", "区间", "方案", "累计收益率", "TWR", "IRR", "Sharpe", "MDD",
                         "平均持仓", "累计投入", "期末总价值"])
        for label, cn in (("5y", "最近5年"), ("3y", "最近3年"), ("1y", "最近1年")):
            r = results[label]
            for name, m in (("当前策略（含保护）", r["guarded"]),
                            ("当前策略（无保护）", r["base"]),
                            ("智能定投（159915）", r["dca"])):
                writer.writerow([cn, r["window"], name, pct(m["cum_return"]), pct(m["twr"]),
                                 pct(m["annual_return_irr"]), f"{m['sharpe']:.2f}",
                                 pct(m["max_drawdown"]), pct(m["avg_position"]),
                                 f"{m['invested']:.0f}", f"{m['final_value']:.0f}"])
    payload = {
        "schema_version": "anchor-windows-v1",
        "anchor_start": master_dates[start5y],
        "source_script": "scripts/backtest_anchor_windows.py",
        "command": "python -B -m scripts.backtest_anchor_windows",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }
    (OUT_DIR / "backtest_anchor_windows.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
