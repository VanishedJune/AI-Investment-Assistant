# -*- coding: utf-8 -*-
"""Null Model + 近5年分年度测试。

Null：P0~P3 起点各自“每周全额买入持有 159915”（无择时/轮动），统一截至 2026-08-07；
年度：把 4 个相位对角模型（P0=B0207@P0，P1_TOP1@P1，P2_TOP1@P2，P3_TOP1@P3）
的 5 年周收益按自然年拆开，横向比较各年度收益率。
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
from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CASH_FACTOR,
    LOT,
    MIN_COMMISSION,
    COMMISSION_RATE,
    OUT_DIR,
    WINDOW_5Y,
)

WEEKLY = 2000.0


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def pct(x) -> str:
    return f"{float(x):.1%}"


def series_metrics(weeks: list[dict], s_idx: int) -> dict:
    twrs = [w["twr"] for w in weeks if w["twr"] is not None]
    index = np.cumprod([1.0 + t for t in twrs])
    peak = np.maximum.accumulate(index)
    mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
    twr_total = float(index[-1] - 1.0) if len(index) else 0.0
    flows = [(k, -w["inflow"]) for k, w in enumerate(weeks)]
    final = weeks[-1]["nav_after"]
    invested = weeks[-1]["invested"]
    arr = np.asarray(twrs, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    avg_pos = float(np.mean([w["exposure"] for w in weeks])) if weeks else 0.0
    return {
        "cum_return": final / invested - 1.0 if invested > 0 else 0.0,
        "twr": twr_total,
        "irr": _irr(flows, final),
        "sharpe": sharpe,
        "mdd": mdd,
        "avg_position": avg_pos,
        "invested": invested,
        "final_value": final,
    }


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
    idx5 = range_idx(master_dates, *WINDOW_5Y)
    s0, e = idx5

    matrix = json.loads((OUT_DIR / "b0207_4phase_top1_matrix.json").read_text(encoding="utf-8"))
    champions = [
        {"id": matrix["champions"][str(i)]["id"],
         "order": matrix["champions"][str(i)]["order"],
         "params": matrix["champions"][str(i)]["params"]}
        for i in range(4)
    ]

    def simulate_model(order, params, offset: int) -> list[dict]:
        cash = 0.0
        shares = {c: 0.0 for c in etfs}
        applied = {}
        first_ex = next(ex for ex in master_execs[s0 : e + 1] if ex is not None)
        for c, et in etfs.items():
            applied[c] = {
                key: (str(ev.get("pay_date") or ev.get("ex_date")) < first_ex)
                for key, ev in et["ca_events"]
            }
        weeks: list[dict] = []
        invested = 0.0
        prev_after = None
        for i in range(s0, e + 1):
            date = master_dates[i]
            ex = master_execs[i]
            if ex is None:
                continue
            cash *= CASH_FACTOR
            n_avail = sum(1 for et in etfs.values() if et["start_date"] <= date)
            prices = {c: et["price_map"].get(ex, 0.0) for c, et in etfs.items()}
            for c, et in etfs.items():
                if prices[c] > 0 or shares[c] != 0:
                    shares[c], cash = _apply_ca(shares[c], cash, et["ca_events"], ex, applied[c])
            nav_before = sum(shares[c] * prices[c] for c in etfs) + cash
            twr = nav_before / prev_after - 1.0 if prev_after is not None else None
            inflow = WEEKLY * n_avail
            cash += inflow
            invested += inflow
            value = sum(shares[c] * prices[c] for c in etfs) + cash
            if (i - s0 - offset) % int(params["rebalance"]) == 0:
                raw = {c: (et["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0)
                       for c, et in etfs.items()}
                floor = float(params["floor"])
                eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
                sig = {c: eff[c] > 0.0 for c in etfs}
                target = allocate_cfg(sig, eff, order, params)
                for c, et in etfs.items():
                    if prices[c] <= 0:
                        continue
                    weight = float(target["weights"].get(c, 0)) / 100.0
                    tgt_share = int(weight * value / prices[c] // LOT) * LOT
                    delta = tgt_share - shares[c]
                    traded = abs(delta) * prices[c]
                    cost = trade_cost(traded)
                    cash -= delta * prices[c] + cost
                    shares[c] = tgt_share
            nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
            exposure = sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0
            weeks.append({"date": date, "inflow": inflow, "nav_after": nav_after,
                          "twr": twr, "invested": invested, "exposure": exposure})
            prev_after = nav_after
        return weeks

    def simulate_null(offset: int) -> list[dict]:
        et = etfs["159915"]
        shares = 0.0
        cash = 0.0
        cost_adj = 0.0
        applied = {}
        first_ex = next(ex for ex in master_execs[s0 + offset : e + 1] if ex is not None)
        for key, ev in et["ca_events"]:
            applied[key] = str(ev.get("pay_date") or ev.get("ex_date")) < first_ex
        weeks: list[dict] = []
        invested = 0.0
        prev_after = None
        for i in range(s0 + offset, e + 1):
            date = master_dates[i]
            ex = master_execs[i]
            if ex is None:
                continue
            cash *= CASH_FACTOR
            n_avail = sum(1 for e in etfs.values() if e["start_date"] <= date)
            price = float(et["price_map"][ex])
            shares, cash = _apply_ca(shares, cash, et["ca_events"], ex, applied)
            nav_before = shares * price + cash
            twr = nav_before / prev_after - 1.0 if prev_after is not None else None
            inflow = WEEKLY * n_avail
            cash += inflow
            invested += inflow
            amount = inflow
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
            exposure = shares * price / nav_after if nav_after > 0 else 0.0
            weeks.append({"date": date, "inflow": inflow, "nav_after": nav_after,
                          "twr": twr, "invested": invested, "exposure": exposure})
            prev_after = nav_after
        return weeks

    # 四个相位对角模型 + 四个 Null
    model_weeks = [simulate_model(champions[i]["order"], champions[i]["params"], i) for i in range(4)]
    null_weeks = [simulate_null(i) for i in range(4)]

    # Null 汇总
    null_rows = []
    for i in range(4):
        m = series_metrics(null_weeks[i], s0)
        null_rows.append({
            "phase": f"P{i}",
            "start_signal": master_dates[s0 + i],
            "start_exec": master_execs[s0 + i],
            **m,
        })
        print("null P", i, "start", master_dates[s0 + i], "cum", round(m["cum_return"], 4),
              "twr", round(m["twr"], 4), "mdd", round(m["mdd"], 4))

    # 年度拆分（TWR）
    years = sorted({w["date"][:4] for ws in model_weeks for w in ws})
    annual = {y: {} for y in years}
    for i in range(4):
        by_year: dict[str, list[float]] = {y: [] for y in years}
        for w in model_weeks[i]:
            if w["twr"] is not None:
                by_year[w["date"][:4]].append(w["twr"])
        for y in years:
            annual[y][f"P{i}"] = float(np.prod([1.0 + t for t in by_year[y]]) - 1.0) if by_year[y] else 0.0
    annual_null = {y: {} for y in years}
    for i in range(4):
        by_year: dict[str, list[float]] = {y: [] for y in years}
        for w in null_weeks[i]:
            if w["twr"] is not None:
                by_year[w["date"][:4]].append(w["twr"])
        for y in years:
            annual_null[y][f"P{i}"] = float(np.prod([1.0 + t for t in by_year[y]]) - 1.0) if by_year[y] else 0.0
    print("years:", years)
    for y in years:
        print(y, {p: round(v, 4) for p, v in annual[y].items()})

    lines = [
        "# Null Model + 近5年分年度测试",
        "",
        "## 1. Null Model（无择时/轮动：每周全额买入持有 159915）",
        "",
        "| 相位 | 起始信号日 | 起始执行日 | 累计收益率 | TWR | IRR | Sharpe | MDD | 累计投入 | 期末资产 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in null_rows:
        lines.append(
            f"| {r['phase']} | {r['start_signal']} | {r['start_exec']} | {pct(r['cum_return'])} | "
            f"{pct(r['twr'])} | {pct(r['irr'])} | {r['sharpe']:.2f} | {pct(r['mdd'])} | "
            f"{r['invested']:.0f} | {r['final_value']:.0f} |"
        )
    lines += ["", "## 2. 近5年分年度收益率（四个相位对角模型，TWR）", "",
              "| 年度 | P0 | P1 | P2 | P3 | 当年最优 |", "|---|---:|---:|---:|---:|---|"]
    for y in years:
        best = max(range(4), key=lambda i: annual[y][f"P{i}"])
        lines.append(f"| {y} | " + " | ".join(pct(annual[y][f"P{i}"]) for i in range(4)) +
                     f" | P{best} |")
    lines += ["", "## 3. Null Model 分年度收益率（对照）", "",
              "| 年度 | P0 | P1 | P2 | P3 |", "|---|---:|---:|---:|---:|"]
    for y in years:
        lines.append(f"| {y} | " + " | ".join(pct(annual_null[y][f"P{i}"]) for i in range(4)) + " |")

    md = OUT_DIR / "null_and_annual_tests.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "null_and_annual_tests.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["测试", "相位", "年份", "累计收益率", "TWR", "IRR", "Sharpe", "MDD", "期末资产"])
        for r in null_rows:
            writer.writerow(["Null", r["phase"], "近5年", pct(r["cum_return"]), pct(r["twr"]),
                             pct(r["irr"]), f"{r['sharpe']:.2f}", pct(r["mdd"]), f"{r['final_value']:.0f}"])
        for y in years:
            for i in range(4):
                writer.writerow(["模型", f"P{i}", y, "", pct(annual[y][f"P{i}"]), "", "", "", ""])
                writer.writerow(["Null", f"P{i}", y, "", pct(annual_null[y][f"P{i}"]), "", "", "", ""])
    payload = {
        "schema_version": "null-and-annual-v1",
        "null": null_rows,
        "annual_model": annual,
        "annual_null": annual_null,
        "years": years,
    }
    (OUT_DIR / "null_and_annual_tests.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
