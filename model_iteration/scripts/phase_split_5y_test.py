# -*- coding: utf-8 -*-
"""前5年 / 后5年 独立验证：P0~P3 排名。

只读验证，不重新搜索模型、不调参数。
- 前5年：2016-08-10 ~ 2021-08-07（独立起账户）
- 后5年：2021-08-10 ~ 2026-08-07（独立起账户，即原 5 年寻优窗口）
- 决策网格：两段均锚定 2021-08-13（P0），P1/P2/P3 依次 +1/+2/+3 周，
  与前 10 年测试、5 年寻优完全一致。
- 口径：weekly_deploy=False、每周入金 2000×n、raw_open、万2.5 佣金最低 5 元、
  现金年化 1.75%、100 份整手、显式 CA 账本。

产出：champion_vs_dca/phase_split_5y_test.{md,csv,json}
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
FRONT = ("2016-08-10", "2021-08-07")
BACK = WINDOW_5Y


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def pct(x) -> str:
    return f"{float(x):.1%}"


def simulate_win(etfs: dict, master_dates: list[str], master_execs: list[str],
                 s: int, e: int, ref_idx: int, order: list[str], params: dict) -> dict:
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[s : e + 1] if ex is not None), None)
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
            "date": date, "ex": ex, "twr": twr, "inflow": inflow,
            "flow_idx": i - s, "nav_after": nav_after, "invested": invested,
            "cum_return": nav_after / invested - 1.0, "exposure": exposure,
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
    s5 = range_idx(master_dates, *WINDOW_5Y)[0]
    wins = {"front": range_idx(master_dates, *FRONT), "back": range_idx(master_dates, *BACK)}
    print("front:", master_dates[wins["front"][0]], "~", master_dates[wins["front"][1]])
    print("back:", master_dates[wins["back"][0]], "~", master_dates[wins["back"][1]])

    matrix = json.loads((OUT_DIR / "b0207_4phase_top1_matrix.json").read_text(encoding="utf-8"))
    ch = matrix["champions"]
    models = [
        {"id": ch[str(i)]["id"], "order": ch[str(i)]["order"], "params": ch[str(i)]["params"]}
        for i in range(4)
    ]

    results = {}
    for wname, (s, e) in wins.items():
        results[wname] = {}
        print(f"== {wname} ==")
        for p in range(4):
            ref_idx = s5 + p
            sim = simulate_win(etfs, master_dates, master_execs, s, e, ref_idx,
                               models[p]["order"], models[p]["params"])
            offset_engine = (ref_idx - s) % 4
            ref = simulate_realistic(etfs, master_dates, master_execs, s, e,
                                     models[p]["order"], models[p]["params"],
                                     weekly_deploy=False, phase_offset=offset_engine)
            m = sim["metrics"]
            ok = abs(m["cum_return"] - ref["cum_return"]) < 1e-6
            print(f"P{p} {models[p]['id']}: cum={m['cum_return']:.4f} ref={ref['cum_return']:.4f}"
                  f" {'OK' if ok else 'MISMATCH'}")
            results[wname][p] = m

    lines = [
        "# 前5年 / 后5年 独立验证（P0~P3 排名）",
        "",
        "只读验证：不重新搜索模型、不调参数；两阶段各自独立起账户。",
        f"- 前5年：{FRONT[0]} ~ {FRONT[1]}",
        f"- 后5年：{BACK[0]} ~ {BACK[1]}（即原 5 年寻优窗口，in-sample）",
        "- 决策网格：两段均锚定 2021-08-13（P0），P1/P2/P3 依次 +1/+2/+3 周。",
        "",
        "## 排名（按累计收益率）",
        "",
        "| 阶段 | 第1名 | 第2名 | 第3名 | 第4名 |",
        "|---|---|---|---|---|",
    ]
    for wname, label in (("front", "前5年"), ("back", "后5年")):
        rank = sorted(range(4), key=lambda p: -results[wname][p]["cum_return"])
        lines.append(f"| {label} | " + " | ".join(f"P{p}（{pct(results[wname][p]['cum_return'])}）"
                                                   for p in rank) + " |")

    lines += ["", "## 各阶段明细", "",
              "| 阶段 | 相位 | 模型 | 累计收益率 | TWR | IRR | Sharpe | 最大回撤 | 平均持仓 | 换手 | 期末市值 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for wname, label in (("front", "前5年"), ("back", "后5年")):
        for p in range(4):
            m = results[wname][p]
            lines.append(
                f"| {label} | P{p} | {models[p]['id']} | {pct(m['cum_return'])} | {pct(m['twr'])} | "
                f"{pct(m['irr'])} | {m['sharpe']:.2f} | {pct(m['mdd'])} | {pct(m['avg_position'])} | "
                f"{m['turnover']:.3f} | {m['final_value']:.0f} |"
            )

    md = OUT_DIR / "phase_split_5y_test.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "phase_split_5y_test.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["阶段", "相位", "模型", "累计收益率", "TWR", "IRR", "Sharpe", "MDD",
                         "平均持仓", "换手", "期末市值"])
        for wname, label in (("front", "前5年"), ("back", "后5年")):
            for p in range(4):
                m = results[wname][p]
                writer.writerow([label, f"P{p}", models[p]["id"], pct(m["cum_return"]),
                                 pct(m["twr"]), pct(m["irr"]), f"{m['sharpe']:.2f}",
                                 pct(m["mdd"]), pct(m["avg_position"]),
                                 f"{m['turnover']:.3f}", f"{m['final_value']:.0f}"])
    payload = {
        "schema_version": "phase-split-5y-v1",
        "windows": {"front": FRONT, "back": BACK,
                    "front_signal": [master_dates[wins["front"][0]], master_dates[wins["front"][1]]],
                    "back_signal": [master_dates[wins["back"][0]], master_dates[wins["back"][1]]]},
        "models": [{"phase": f"P{i}", "id": models[i]["id"],
                    "order": models[i]["order"], "params": models[i]["params"]} for i in range(4)],
        "results": {wname: {f"P{p}": results[wname][p] for p in range(4)} for wname in wins},
        "rank": {wname: [f"P{p}" for p in
                         sorted(range(4), key=lambda p: -results[wname][p]["cum_return"])]
                 for wname in wins},
    }
    (OUT_DIR / "phase_split_5y_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
