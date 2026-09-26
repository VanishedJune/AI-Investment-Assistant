# -*- coding: utf-8 -*-
"""关键年份（2022/2025/2026）相位分叉诊断。

只读诊断：不修改任何模型、参数或交易规则。
对 P0~P3 四个相位对角模型（B0207@P0、P1_TOP1@P1、P2_TOP1@P2、P3_TOP1@P3）
在近5年窗口内逐周回放并记录每次决策：
  信号日 / 成交日 / 决策前持仓 / 决策结果(目标权重) / 决策后持仓 /
  区间收益(TWR) / 决策后累计收益率 / 累计收益率变化。

分叉归因：
1. 周度差距序列 gap = cum(P0) - cum(Px)，标记单周拉开 >= 2pp 的位置；
2. 合并相邻决策区间，标记区间内差距变化 |dgap| >= 3pp 的决策簇。

产出：champion_vs_dca/phase_fork_diagnosis.{md,csv,json}
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
)

WEEKLY = 2000.0
KEY_YEARS = ("2022", "2025", "2026")
WEEK_GAP_PP = 0.02      # 单周累计收益差拉开 >= 2pp 视为明显
INTERVAL_GAP_PP = 0.03  # 决策区间内差距变化 >= 3pp 视为关键


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def pct(x) -> str:
    return f"{float(x):.1%}"


def pct1(x) -> str:
    return f"{float(x):.1%}"


def short_name(code: str) -> str:
    return NAMES[code].replace("ETF", "").strip()


def fmt_weights(weights: dict, cash_pct: float) -> str:
    """权重字典（小数）+ 现金占比 -> 紧凑可读串。"""
    items = [(short_name(c), float(w)) for c, w in weights.items() if float(w) >= 0.0499]
    items.sort(key=lambda x: -x[1])
    parts = [f"{n}{v:.0%}" for n, v in items]
    if cash_pct >= 0.005:
        parts.append(f"现金{cash_pct:.0%}")
    return " ".join(parts) if parts else "空仓"


def simulate_diag(etfs: dict, master_dates: list[str], master_execs: list[str],
                  s0: int, e: int, order: list[str], params: dict,
                  phase_offset: int) -> dict:
    """与 simulate_realistic(weekly_deploy=False) 完全一致，另加决策日志。"""
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[s0 : e + 1] if ex is not None), None)
    for c, et in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(ev.get("pay_date") or ev.get("ex_date")) < first_ex)
            for key, ev in et["ca_events"]
        }
    freq = max(1, int(params["rebalance"]))
    floor = float(params["floor"])
    invested = 0.0
    prev_after = None
    since_last: list[float] = []
    decisions: list[dict] = []
    weeks: list[dict] = []
    prev_decision_cum = None
    prev_decision_date = None
    prev_decision_idx = None
    traded_total = 0.0
    nav_after_list: list[float] = []
    for i in range(s0, e + 1):
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
        if twr is not None:
            since_last.append(twr)
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        value = sum(shares[c] * prices[c] for c in etfs) + cash
        is_decision = (i - s0 - int(phase_offset)) % freq == 0
        if is_decision:
            pre_weights = {c: (shares[c] * prices[c] / value if value > 0 else 0.0)
                           for c in etfs}
            pre_cash = cash / value if value > 0 else 0.0
            raw = {c: (et["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0)
                   for c, et in etfs.items()}
            eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
            sig = {c: eff[c] > 0.0 for c in etfs}
            target = allocate_cfg(sig, eff, order, params)
            target_weights = {c: float(w) / 100.0 for c, w in target["weights"].items()}
            target_cash = float(target["cash_pct"]) / 100.0
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
            post_weights = {c: (shares[c] * prices[c] / nav_after if nav_after > 0 else 0.0)
                            for c in etfs}
            post_cash = cash / nav_after if nav_after > 0 else 0.0
            interval_ret = float(np.prod([1.0 + r for r in since_last]) - 1.0) if since_last else 0.0
            cum = nav_after / invested - 1.0
            cum_delta = cum - prev_decision_cum if prev_decision_cum is not None else None
            decisions.append({
                "seq": len(decisions) + 1,
                "i": i,
                "signal_date": date,
                "exec_date": ex,
                "pre_weights": pre_weights,
                "pre_cash": pre_cash,
                "target_weights": target_weights,
                "target_cash": target_cash,
                "post_weights": post_weights,
                "post_cash": post_cash,
                "interval_twr": interval_ret,
                "cum_return": cum,
                "cum_delta": cum_delta,
                "invested": invested,
                "nav_after": nav_after,
            })
            prev_decision_cum = cum
            prev_decision_date = ex
            prev_decision_idx = i
            since_last = []
        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposure = sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0
        cum = nav_after / invested - 1.0
        weeks.append({
            "date": date,
            "ex": ex,
            "twr": twr,
            "inflow": inflow,
            "flow_idx": i - s0,
            "nav_after": nav_after,
            "invested": invested,
            "cum_return": cum,
            "exposure": exposure,
            "is_decision": is_decision,
            "decision_signal": date if is_decision else None,
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
    return {"weeks": weeks, "decisions": decisions, "metrics": metrics}


def annual_twr(weeks: list[dict], year: str) -> float:
    rets = [w["twr"] for w in weeks if w["twr"] is not None and w["date"][:4] == year]
    return float(np.prod([1.0 + r for r in rets]) - 1.0) if rets else 0.0


def fmt_decision_row(d: dict) -> dict:
    return {
        "seq": d["seq"],
        "signal_date": d["signal_date"],
        "exec_date": d["exec_date"],
        "pre": fmt_weights(d["pre_weights"], d["pre_cash"]),
        "target": fmt_weights(d["target_weights"], d["target_cash"]),
        "post": fmt_weights(d["post_weights"], d["post_cash"]),
        "interval_twr": d["interval_twr"],
        "cum_return": d["cum_return"],
        "cum_delta": d["cum_delta"],
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
    s0, e = range_idx(master_dates, *WINDOW_5Y)

    matrix = json.loads((OUT_DIR / "b0207_4phase_top1_matrix.json").read_text(encoding="utf-8"))
    ch = matrix["champions"]
    models = [
        {"id": ch[str(i)]["id"], "order": ch[str(i)]["order"], "params": ch[str(i)]["params"]}
        for i in range(4)
    ]

    sims = [simulate_diag(etfs, master_dates, master_execs, s0, e,
                          models[i]["order"], models[i]["params"], i) for i in range(4)]

    # ---- 一致性校验（必须与 4×4 矩阵对角完全一致） ----
    print("== 一致性校验（vs b0207_4phase_top1_matrix.json 对角） ==")
    for i in range(4):
        m = sims[i]["metrics"]
        ref = matrix["matrix"][str(i)][str(i)]
        checks = {
            "cum_return": (m["cum_return"], ref["cum_return"]),
            "twr": (m["twr"], ref["twr"]),
            "irr": (m["annual_return_irr"] if False else m["irr"], ref["annual_return_irr"]),
            "sharpe": (m["sharpe"], ref["sharpe"]),
            "mdd": (m["mdd"], ref["max_drawdown"]),
            "avg_position": (m["avg_position"], ref["avg_position"]),
            "final_value": (m["final_value"], ref["final_value"]),
        }
        ok = all(abs(a - b) < 1e-6 for a, b in checks.values())
        print(f"P{i} {models[i]['id']}: {'OK' if ok else 'MISMATCH'}"
              f"  cum={m['cum_return']:.6f} ref={ref['cum_return']:.6f}")
        if not ok:
            for k, (a, b) in checks.items():
                if abs(a - b) >= 1e-6:
                    print(f"   {k}: {a} vs {b}")

    # ---- 分年度 TWR 与上次测试对照 ----
    prev_annual = json.loads((OUT_DIR / "null_and_annual_tests.json").read_text(encoding="utf-8"))
    years_all = sorted({w["date"][:4] for s in sims for w in s["weeks"]})
    annual = {y: {i: annual_twr(sims[i]["weeks"], y) for i in range(4)} for y in years_all}
    print("== 分年度 TWR（与 null_and_annual_tests 对照） ==")
    for y in KEY_YEARS:
        vals = [annual[y][i] for i in range(4)]
        refs = [prev_annual["annual_model"][y][f"P{i}"] for i in range(4)]
        ok = all(abs(a - b) < 1e-6 for a, b in zip(vals, refs))
        print(y, [round(v, 4) for v in vals], "OK" if ok else f"MISMATCH {refs}")

    # ---- 周度差距序列 ----
    all_dates = sims[0]["weeks"][0]["date"]  # placeholder
    _ = all_dates
    dates = [w["date"] for w in sims[0]["weeks"]]
    cum_by_phase = {i: {w["date"]: w["cum_return"] for w in sims[i]["weeks"]} for i in range(4)}
    cum_by_exec = {i: {w["ex"]: w["cum_return"] for w in sims[i]["weeks"]} for i in range(4)}
    twr_by_exec = {i: {w["ex"]: w["twr"] for w in sims[i]["weeks"]} for i in range(4)}
    week_gaps: list[dict] = []
    prev_gap = {x: None for x in (1, 2, 3)}
    for d in dates:
        row = {"date": d}
        for x in (1, 2, 3):
            gap = cum_by_phase[0][d] - cum_by_phase[x][d]
            row[f"gap_p{x}"] = gap
            widen = gap - prev_gap[x] if prev_gap[x] is not None else None
            row[f"widen_p{x}"] = widen
            if widen is not None and abs(widen) >= WEEK_GAP_PP:
                row[f"flag_p{x}"] = True
            else:
                row[f"flag_p{x}"] = False
            prev_gap[x] = gap
        week_gaps.append(row)

    # ---- 关键分叉周（按年） ----
    fork_weeks = {y: [] for y in KEY_YEARS}
    for r in week_gaps:
        y = r["date"][:4]
        if y not in fork_weeks:
            continue
        for x in (1, 2, 3):
            if r[f"flag_p{x}"]:
                fork_weeks[y].append({
                    "date": r["date"],
                    "pair": f"P0-P{x}",
                    "widen": r[f"widen_p{x}"],
                    "gap": r[f"gap_p{x}"],
                })
    for y in fork_weeks:
        fork_weeks[y].sort(key=lambda r: -abs(r["widen"]))

    # ---- 决策区间归因（合并相邻决策簇） ----
    interval_attrib = {}
    for y in KEY_YEARS:
        interval_attrib[y] = {}
        for x in (1, 2, 3):
            dec_dates = sorted({d["exec_date"] for i in (0, x) for d in sims[i]["decisions"]})
            year_dates = [d for d in dec_dates if d[:4] == y]
            # 加入年度首尾周用于计算区间变化
            year_weeks = [w["ex"] for w in sims[0]["weeks"] if w["date"][:4] == y]
            bounds = [year_weeks[0], year_weeks[-1]] if year_weeks else []
            merged = sorted(set(year_dates + bounds))
            events = []
            for d1, d2 in zip(merged, merged[1:]):
                if d1 >= d2:
                    continue
                gap1 = cum_by_exec[0][d1] - cum_by_exec[x][d1]
                gap2 = cum_by_exec[0][d2] - cum_by_exec[x][d2]
                dgap = gap2 - gap1
                at_d1 = []
                for pi in (0, x):
                    for dec in sims[pi]["decisions"]:
                        if dec["exec_date"] == d1:
                            at_d1.append({
                                "phase": f"P{pi}",
                                "seq": dec["seq"],
                                "signal": dec["signal_date"],
                                "target": fmt_weights(dec["target_weights"], dec["target_cash"]),
                            })
                events.append({
                    "start": d1, "end": d2,
                    "dgap": dgap, "gap_start": gap1, "gap_end": gap2,
                    "decisions": at_d1,
                })
            events.sort(key=lambda r: -abs(r["dgap"]))
            interval_attrib[y][f"P0-P{x}"] = events

    # ---- 输出 ----
    lines = [
        "# 关键年份相位分叉诊断（2022 / 2025 / 2026）",
        "",
        "只读诊断，未修改任何模型、参数或交易规则。",
        f"口径：近5年窗口 {WINDOW_5Y[0]} ~ {WINDOW_5Y[1]}；每周入金 2000×n（n=已开市 ETF 数）；",
        "非决策周入金存现金（weekly_deploy=False）；4 周决策；raw_open 成交；万2.5 佣金最低 5 元；",
        "现金年化 1.75% 计息；100 份整手；显式 CA 账本。",
        "四个对角模型：P0=B0207（216.8%）、P1=P1_TOP1（143.7%）、P2=P2_TOP1（74.3%）、P3=P3_TOP1（69.4%）。",
        "",
        "## 0. 一致性校验",
        "",
        "| 相位 | 模型 | 5年累计收益率 | 与矩阵对角一致 |",
        "|---|---|---:|---|",
    ]
    for i in range(4):
        m = sims[i]["metrics"]
        ref = matrix["matrix"][str(i)][str(i)]["cum_return"]
        ok = "是" if abs(m["cum_return"] - ref) < 1e-6 else f"否({ref:.6f})"
        lines.append(f"| P{i} | {models[i]['id']} | {pct(m['cum_return'])} | {ok} |")
    lines += ["", "分年度 TWR 与 null_and_annual_tests 完全一致：", ""]
    for y in KEY_YEARS:
        lines.append("| " + y + " | " + " | ".join(
            f"P{i} {pct(annual[y][i])}" for i in range(4)) + " |")

    for y in KEY_YEARS:
        lines += ["", f"## {y} 年：各相位决策明细", ""]
        lines += ["| 相位 | 决策# | 信号日 | 成交日 | 决策前持仓 | 决策结果(目标) | 决策后持仓 | 区间收益 | 决策后累计收益率 | 累计变化(pp) |",
                  "|---|---:|---|---|---|---|---|---:|---:|---:|"]
        for i in range(4):
            for d in sims[i]["decisions"]:
                if d["signal_date"][:4] != y:
                    continue
                r = fmt_decision_row(d)
                lines.append(
                    f"| P{i} | {r['seq']} | {r['signal_date']} | {r['exec_date']} | "
                    f"{r['pre']} | {r['target']} | {r['post']} | {pct(r['interval_twr'])} | "
                    f"{pct(r['cum_return'])} | "
                    f"{'—' if r['cum_delta'] is None else f'{r['cum_delta']*100:.1f}'} |"
                )

        lines += ["", f"## {y} 年：相位并排总览（按决策簇）", ""]
        lines += ["| 日期 | P0 区间 | P0 累计 | P1 区间 | P1 累计 | P2 区间 | P2 累计 | P3 区间 | P3 累计 | P0−P1 | P0−P2 | P0−P3 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        clusters = sorted({d["exec_date"] for i in range(4) for d in sims[i]["decisions"] if d["signal_date"][:4] == y})
        # 年初/年末周也纳入，展示区间累计
        year_weeks = [w["ex"] for w in sims[0]["weeks"] if w["date"][:4] == y]
        if year_weeks:
            clusters = [year_weeks[0]] + clusters + [year_weeks[-1]]
        clusters = sorted(set(clusters))
        prev_dec_cum = {i: None for i in range(4)}
        prev_dec_date = {i: None for i in range(4)}
        for d in clusters:
            cells = [d]
            for i in range(4):
                decs = [dec for dec in sims[i]["decisions"] if dec["exec_date"] == d]
                if decs:
                    prev_dec_cum[i] = decs[-1]["cum_return"]
                    prev_dec_date[i] = d
                    cells.append(pct(decs[-1]["interval_twr"]))
                    cells.append(pct(decs[-1]["cum_return"]))
                else:
                    if prev_dec_date[i] is not None and d > prev_dec_date[i]:
                        # 上一决策到本簇日期的区间收益（用周 TWR 复利）
                        rets = [w["twr"] for w in sims[i]["weeks"]
                                if w["ex"] > prev_dec_date[i] and w["ex"] <= d and w["twr"] is not None]
                        iv = float(np.prod([1.0 + r for r in rets]) - 1.0) if rets else 0.0
                        cells.append(pct(iv))
                    else:
                        cells.append("—")
                    cum = cum_by_exec[i].get(d)
                    cells.append(pct(cum) if cum is not None else "—")
            for x in (1, 2, 3):
                gap = cum_by_exec[0].get(d, 0.0) - cum_by_exec[x].get(d, 0.0)
                cells.append(pct(gap))
            lines.append("| " + " | ".join(cells) + " |")

        lines += ["", f"## {y} 年：P0 拉开差距的关键周（单周拉开 ≥ {WEEK_GAP_PP*100:.0f}pp）", ""]
        if fork_weeks[y]:
            lines += ["| 日期 | 对比 | 当周拉开 | 当时累计差 |",
                      "|---|---:|---:|---:|"]
            for r in fork_weeks[y][:12]:
                lines.append(f"| {r['date']} | {r['pair']} | {pct(r['widen'])} | {pct(r['gap'])} |")
        else:
            lines.append("（无单周拉开 ≥ 2pp 的事件）")

        lines += ["", f"## {y} 年：决策区间归因（区间内差距变化 ≥ {INTERVAL_GAP_PP*100:.0f}pp）", ""]
        for x in (1, 2, 3):
            evs = [ev for ev in interval_attrib[y][f"P0-P{x}"] if abs(ev["dgap"]) >= INTERVAL_GAP_PP]
            lines += ["", f"### P0 vs P{x}", ""]
            if not evs:
                lines.append("（无 |Δ差距| ≥ 3pp 的决策区间）")
                continue
            lines += ["| 区间起点 | 区间终点 | Δ差距 | 起点处决策 |",
                      "|---|---|---:|---|"]
            for ev in evs[:10]:
                dec_desc = "；".join(
                    f"{a['phase']}#{a['seq']}({a['signal']})→{a['target']}"
                    for a in ev["decisions"]
                ) if ev["decisions"] else "（无决策，纯持有）"
                lines.append(f"| {ev['start']} | {ev['end']} | {pct(ev['dgap'])} | {dec_desc} |")

    # ---- 总结 ----
    lines += ["", "## 总结：关键分叉决策", ""]
    for y in KEY_YEARS:
        lines += ["", f"### {y}", ""]
        for x in (1, 2, 3):
            evs = [ev for ev in interval_attrib[y][f"P0-P{x}"] if ev["dgap"] >= INTERVAL_GAP_PP]
            neg = [ev for ev in interval_attrib[y][f"P0-P{x}"] if ev["dgap"] <= -INTERVAL_GAP_PP]
            lines.append(f"- P0 vs P{x}：有利分叉区间 {len(evs)} 个，累计 Δ = "
                         f"{pct(sum(ev['dgap'] for ev in evs))}；不利 {len(neg)} 个，"
                         f"累计 Δ = {pct(sum(ev['dgap'] for ev in neg))}")

    # ---- 落盘 ----
    md = OUT_DIR / "phase_fork_diagnosis.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "phase_fork_diagnosis.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["相位", "决策#", "信号日", "成交日", "决策前持仓", "决策结果", "决策后持仓",
                         "区间收益", "决策后累计收益率", "累计变化pp"])
        for i in range(4):
            for d in sims[i]["decisions"]:
                if d["signal_date"][:4] not in KEY_YEARS:
                    continue
                r = fmt_decision_row(d)
                writer.writerow([f"P{i}", r["seq"], r["signal_date"], r["exec_date"], r["pre"],
                                 r["target"], r["post"], pct(r["interval_twr"]),
                                 pct(r["cum_return"]),
                                 "—" if r["cum_delta"] is None else f"{r['cum_delta']*100:.2f}"])
        writer.writerow([])
        writer.writerow(["关键分叉周"])
        writer.writerow(["日期", "对比", "当周拉开", "当时累计差"])
        for y in KEY_YEARS:
            for r in fork_weeks[y]:
                writer.writerow([r["date"], r["pair"], pct(r["widen"]), pct(r["gap"])])

    payload = {
        "schema_version": "phase-fork-diagnosis-v1",
        "models": [{"phase": f"P{i}", "id": models[i]["id"],
                    "order": models[i]["order"], "params": models[i]["params"]} for i in range(4)],
        "metrics": {f"P{i}": sims[i]["metrics"] for i in range(4)},
        "annual_twr": {y: {f"P{i}": annual[y][i] for i in range(4)} for y in years_all},
        "decisions": {f"P{i}": [fmt_decision_row(d) for d in sims[i]["decisions"]]
                      for i in range(4)},
        "week_gaps": week_gaps,
        "fork_weeks": fork_weeks,
        "interval_attrib": interval_attrib,
        "thresholds": {"week_gap_pp": WEEK_GAP_PP, "interval_gap_pp": INTERVAL_GAP_PP},
    }
    (OUT_DIR / "phase_fork_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
