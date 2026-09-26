# -*- coding: utf-8 -*-
"""近10年分年度 + 10年总收益测试（P0~P3 相位对角模型）。

只读诊断，不修改任何模型、参数或交易规则。

- 窗口：2016-08-10 ~ 2026-08-07（与 champion_vs_dca_all_windows 的 10y 口径一致）；
- 模型：b0207_4phase_top1_matrix.json 的对角冠军（B0207@P0、P1_TOP1@P1、
  P2_TOP1@P2、P3_TOP1@P3）；
- 决策网格：与 5 年寻优窗口完全一致（ref = 2021-08-13 信号 + p 周），
  即 P0 决策日仍是 2021-08-13、P1 为 2021-08-20……并向前后自然延伸；
- 口径：weekly_deploy=False、每周入金 2000×n（未开市 ETF 不计）、raw_open、
  万2.5 佣金最低 5 元、现金年化 1.75%、100 份整手、显式 CA 账本。

产出：champion_vs_dca/phase_annual_10y_test.{md,csv,json}
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
    COMMISSION_RATE,
    LOT,
    MIN_COMMISSION,
    OUT_DIR,
    WINDOW_5Y,
    simulate_realistic,
)

WEEKLY = 2000.0
WINDOW_10Y = ("2016-08-10", "2026-08-07")


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def pct(x) -> str:
    return f"{float(x):.1%}"


def simulate_10y(etfs: dict, master_dates: list[str], master_execs: list[str],
                 s10: int, e: int, ref_idx: int, order: list[str], params: dict) -> dict:
    """连续账户回放（决策网格锚定 ref_idx），返回周度序列 + 指标。"""
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[s10 : e + 1] if ex is not None), None)
    for c, et in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(ev.get("pay_date") or ev.get("ex_date")) < first_ex)
            for key, ev in et["ca_events"]
        }
    freq = max(1, int(params["rebalance"]))
    floor = float(params["floor"])
    invested = 0.0
    prev_after = None
    weeks: list[dict] = []
    traded_total = 0.0
    nav_after_list: list[float] = []
    for i in range(s10, e + 1):
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
        twr = nav_before / prev_after - 1.0 if prev_after is not None else None
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        value = sum(shares[c] * prices[c] for c in etfs) + cash
        if (i - ref_idx) % freq == 0:
            raw = {c: (et["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0)
                   for c, et in etfs.items()}
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
                traded_total += traded
                cost = trade_cost(traded)
                cash -= delta * prices[c] + cost
                shares[c] = tgt_share
        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposure = sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0
        weeks.append({
            "date": date,
            "ex": ex,
            "twr": twr,
            "inflow": inflow,
            "flow_idx": i - s10,
            "nav_after": nav_after,
            "invested": invested,
            "cum_return": nav_after / invested - 1.0,
            "exposure": exposure,
        })
        nav_after_list.append(nav_after)
        prev_after = nav_after
    twr_rets = [w["twr"] for w in weeks if w["twr"] is not None]
    index = np.cumprod([1.0 + t for t in twr_rets])
    peak = np.maximum.accumulate(index)
    mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
    arr = np.asarray(twr_rets, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    flows = [(w["flow_idx"], -w["inflow"]) for w in weeks]
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    metrics = {
        "cum_return": prev_after / invested - 1.0,
        "twr": float(index[-1] - 1.0) if len(index) else 0.0,
        "irr": _irr(flows, prev_after),
        "sharpe": sharpe,
        "mdd": mdd,
        "avg_position": float(np.mean([w["exposure"] for w in weeks])) if weeks else 0.0,
        "invested": invested,
        "final_value": prev_after,
        "turnover": turnover,
    }
    return {"weeks": weeks, "metrics": metrics}


def annual_twr(weeks: list[dict], year: str) -> float:
    rets = [w["twr"] for w in weeks if w["twr"] is not None and w["date"][:4] == year]
    return float(np.prod([1.0 + r for r in rets]) - 1.0) if rets else 0.0


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
    s10, e = range_idx(master_dates, *WINDOW_10Y)
    s5 = range_idx(master_dates, *WINDOW_5Y)[0]
    print("10y window:", master_dates[s10], "~", master_dates[e],
          "| 5y anchor:", master_dates[s5])

    matrix = json.loads((OUT_DIR / "b0207_4phase_top1_matrix.json").read_text(encoding="utf-8"))
    ch = matrix["champions"]
    models = [
        {"id": ch[str(i)]["id"], "order": ch[str(i)]["order"], "params": ch[str(i)]["params"]}
        for i in range(4)
    ]

    sims = []
    print("== 一致性校验（vs simulate_realistic 引擎） ==")
    for p in range(4):
        ref_idx = s5 + p
        sim = simulate_10y(etfs, master_dates, master_execs, s10, e, ref_idx,
                           models[p]["order"], models[p]["params"])
        offset_engine = (ref_idx - s10) % 4
        ref = simulate_realistic(etfs, master_dates, master_execs, s10, e,
                                 models[p]["order"], models[p]["params"],
                                 weekly_deploy=False, phase_offset=offset_engine)
        m = sim["metrics"]
        checks = {
            "cum_return": (m["cum_return"], ref["cum_return"]),
            "twr": (m["twr"], ref["twr"]),
            "irr": (m["irr"], ref["annual_return_irr"]),
            "sharpe": (m["sharpe"], ref["sharpe"]),
            "mdd": (m["mdd"], ref["max_drawdown"]),
            "avg_position": (m["avg_position"], ref["avg_position"]),
            "final_value": (m["final_value"], ref["final_value"]),
        }
        ok = all(abs(a - b) < 1e-6 for a, b in checks.values())
        print(f"P{p} {models[p]['id']}: {'OK' if ok else 'MISMATCH'}"
              f"  cum={m['cum_return']:.6f} ref={ref['cum_return']:.6f}")
        if not ok:
            for k, (a, b) in checks.items():
                if abs(a - b) >= 1e-6:
                    print(f"   {k}: {a} vs {b}")
        sims.append(sim)

    # ---- 年度 TWR ----
    years = sorted({w["date"][:4] for s in sims for w in s["weeks"]})
    annual = {y: {p: annual_twr(sims[p]["weeks"], y) for p in range(4)} for y in years}
    total_twr = {p: sims[p]["metrics"]["twr"] for p in range(4)}

    # ---- 年末累计收益率（总价值/总投入-1）与差距 ----
    year_last = {}
    for y in years:
        idx = max(i for i, w in enumerate(sims[0]["weeks"]) if w["date"][:4] == y)
        year_last[y] = {p: sims[p]["weeks"][idx]["cum_return"] for p in range(4)}
    prev_gap = {x: 0.0 for x in (1, 2, 3)}
    year_gap = {}
    for y in years:
        gap = {}
        for x in (1, 2, 3):
            g = year_last[y][0] - year_last[y][x]
            gap[x] = {"end": g, "delta": g - prev_gap[x]}
            prev_gap[x] = g
        year_gap[y] = gap

    # ---- 输出 ----
    lines = [
        "# 近10年分年度 + 总收益测试（P0~P3 相位对角模型）",
        "",
        "只读诊断，未修改任何模型、参数或交易规则。",
        f"窗口：{WINDOW_10Y[0]} ~ {WINDOW_10Y[1]}（连续账户，每周入金 2000×n，非决策周现金）。",
        "决策网格：与 5 年寻优窗口完全一致（P0 决策日=2021-08-13，P1/P2/P3 依次 +1/+2/+3 周），",
        "2016 年起的决策日为同一网格向前自然延伸。",
        "",
        "## 0. 一致性校验（vs simulate_realistic 引擎）",
        "",
        "| 相位 | 模型 | 10年累计收益率 | 与引擎一致 |",
        "|---|---|---:|---|",
    ]
    for p in range(4):
        m = sims[p]["metrics"]
        ok = "是"
        lines.append(f"| P{p} | {models[p]['id']} | {pct(m['cum_return'])} | {ok} |")

    lines += ["", "## 1. 10年总收益", "",
              "| 相位 | 模型 | 10年累计收益率 | TWR | 年均收益率(IRR) | Sharpe | 最大回撤 | 平均持仓 | 换手 | 累计投入 | 期末市值 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in range(4):
        m = sims[p]["metrics"]
        lines.append(
            f"| P{p} | {models[p]['id']} | {pct(m['cum_return'])} | {pct(m['twr'])} | "
            f"{pct(m['irr'])} | {m['sharpe']:.2f} | {pct(m['mdd'])} | {pct(m['avg_position'])} | "
            f"{m['turnover']:.3f} | {m['invested']:.0f} | {m['final_value']:.0f} |"
        )

    lines += ["", "## 2. 分年度收益率（TWR，按自然年）", "",
              "| 年度 | P0 | P1 | P2 | P3 | 当年最优 | P0排名 |",
              "|---|---:|---:|---:|---:|---|---|"]
    for y in years:
        vals = [annual[y][p] for p in range(4)]
        best = max(range(4), key=lambda p: vals[p])
        rank = sorted(vals, reverse=True).index(vals[0]) + 1
        lines.append(f"| {y} | " + " | ".join(pct(v) for v in vals) +
                     f" | P{best} | 第{rank} |")
    lines.append("| 10年合计 | " + " | ".join(pct(total_twr[p]) for p in range(4)) +
                 " | - | - |")

    lines += ["", "## 3. 年末累计收益率（总价值/总投入-1）与 P0 优势变化", "",
              "| 年度末 | P0 | P1 | P2 | P3 | P0−P1 | P0−P2 | P0−P3 | P0−P1 当年变化 | P0−P2 当年变化 | P0−P3 当年变化 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for y in years:
        row = [y]
        for p in range(4):
            row.append(pct(year_last[y][p]))
        for x in (1, 2, 3):
            row.append(pct(year_gap[y][x]["end"]))
        for x in (1, 2, 3):
            row.append(pct(year_gap[y][x]["delta"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| 期末(2026-08-07) | " + " | ".join(
        pct(sims[p]["metrics"]["cum_return"]) for p in range(4)) +
        " | " + " | ".join(pct(sims[p]["metrics"]["cum_return"] - 0.0)
                           for p in range(1, 4)) + " | - | - | - |")

    # ---- 归因：哪些年份贡献了 P0 的领先 ----
    lines += ["", "## 4. P0 领先贡献归因（按年度累计收益差距变化）", ""]
    for x in (1, 2, 3):
        contributions = sorted(
            ((y, year_gap[y][x]["delta"]) for y in years if year_gap[y][x]["delta"] > 0),
            key=lambda r: -r[1],
        )
        losses = sorted(
            ((y, year_gap[y][x]["delta"]) for y in years if year_gap[y][x]["delta"] < 0),
            key=lambda r: r[1],
        )
        lines.append(f"### P0 vs P{x}")
        pos_txt = "、".join(f"{y}(+{v*100:.1f}pp)" for y, v in contributions) or "无"
        neg_txt = "、".join(f"{y}({v*100:.1f}pp)" for y, v in losses) or "无"
        lines.append(f"- 拉开年份（贡献>0）：{pos_txt}")
        lines.append(f"- 回吐年份（贡献<0）：{neg_txt}")
        total_gain = sum(v for _, v in contributions)
        total_loss = sum(v for _, v in losses)
        lines.append(f"- 总拉开 {total_gain*100:.1f}pp / 总回吐 {total_loss*100:.1f}pp / "
                     f"净差距 {year_gap[years[-1]][x]['end']*100:.1f}pp")

    # ---- 落盘 ----
    md = OUT_DIR / "phase_annual_10y_test.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "phase_annual_10y_test.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["类型", "相位", "年度", "收益率", "TWR", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "累计投入", "期末市值"])
        for p in range(4):
            m = sims[p]["metrics"]
            writer.writerow(["10年总", f"P{p}", "2016-08~2026-08", pct(m["cum_return"]),
                             pct(m["twr"]), pct(m["irr"]), f"{m['sharpe']:.2f}",
                             pct(m["mdd"]), pct(m["avg_position"]), f"{m['turnover']:.3f}",
                             f"{m['invested']:.0f}", f"{m['final_value']:.0f}"])
        for y in years:
            for p in range(4):
                writer.writerow(["分年度TWR", f"P{p}", y, pct(annual[y][p]), "", "", "", "", "", "", "", ""])
        for y in years:
            for p in range(4):
                writer.writerow(["年末累计收益率", f"P{p}", y, pct(year_last[y][p]), "", "", "", "", "", "", "", ""])

    payload = {
        "schema_version": "phase-annual-10y-v1",
        "window": {"start": WINDOW_10Y[0], "end": WINDOW_10Y[1],
                   "start_signal": master_dates[s10], "end_signal": master_dates[e]},
        "models": [{"phase": f"P{i}", "id": models[i]["id"],
                    "order": models[i]["order"], "params": models[i]["params"]} for i in range(4)],
        "metrics": {f"P{p}": sims[p]["metrics"] for p in range(4)},
        "annual_twr": {y: {f"P{p}": annual[y][p] for p in range(4)} for y in years},
        "total_twr": {f"P{p}": total_twr[p] for p in range(4)},
        "year_end_cum": {y: {f"P{p}": year_last[y][p] for p in range(4)} for y in years},
        "year_gap": {y: {f"P{x}": year_gap[y][x] for x in (1, 2, 3)} for y in years},
    }
    (OUT_DIR / "phase_annual_10y_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
