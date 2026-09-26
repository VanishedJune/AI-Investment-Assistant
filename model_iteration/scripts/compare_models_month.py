# -*- coding: utf-8 -*-
"""以指定信号日为起点，对比 B0207 与 Meta(5pp) 未来 1 个月的交易。"""

from __future__ import annotations

import csv
import json
import sys
from datetime import timedelta
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
THRESHOLD = 5.0


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


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

    meta = json.loads((OUT_DIR / "meta_weighted_model.json").read_text(encoding="utf-8"))
    weights = meta["weights"]
    data = json.loads((OUT_DIR / "b0207_phase_champions_v2_standard.json").read_text(encoding="utf-8"))
    phase_configs = [
        {"id": ch["model"]["id"], "order": ch["model"]["order"], "params": ch["model"]["params"]}
        for ch in data["champions"]
    ]
    b0207_cfg = phase_configs[0]

    def target_at(cfg: dict, i: int) -> dict:
        date = master_dates[i]
        ex = master_execs[i]
        raw = {c: (et["by_date"].get(date, 0.0) if ex is not None and et["price_map"].get(ex, 0) > 0 else 0.0)
               for c, et in etfs.items()}
        floor = float(cfg["params"]["floor"])
        eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
        sig = {c: eff[c] > 0.0 for c in etfs}
        return dict(allocate_cfg(sig, eff, cfg["order"], cfg["params"])["weights"])

    def run(kind: str, s: int, e: int) -> dict:
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
        trades: list[dict] = []
        weekly: list[dict] = []
        meta_targets: dict[int, dict[str, float]] = {}
        b_tgt: dict[str, float] | None = None

        def apply_weights(target: dict, value: float, date: str) -> None:
            nonlocal cash, traded_total
            for c, et in etfs.items():
                if prices[c] <= 0:
                    continue
                weight = float(target.get(c, 0)) / 100.0
                tgt_share = int(weight * value / prices[c] // LOT) * LOT
                delta = tgt_share - shares[c]
                if abs(delta) > 0:
                    traded = abs(delta) * prices[c]
                    traded_total += traded
                    cost = trade_cost(traded)
                    cash -= delta * prices[c] + cost
                    shares[c] = tgt_share
                    trades.append({"date": date, "code": c, "action": "买" if delta > 0 else "卖",
                                   "value": round(traded, 0)})

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

            if kind == "b0207":
                if b_tgt is None or (i - start5y) % int(b0207_cfg["params"]["rebalance"]) == 0:
                    b_tgt = target_at(b0207_cfg, i)
                apply_weights(b_tgt, value, date)
                current = b_tgt
            else:
                if not meta_targets:
                    for p, cfg in enumerate(phase_configs):
                        freq = int(cfg["params"]["rebalance"])
                        last = s - ((s - (start5y + p)) % freq)
                        meta_targets[p] = target_at(cfg, last) if last >= 0 else {c: 0.0 for c in etfs}
                for p, cfg in enumerate(phase_configs):
                    freq = int(cfg["params"]["rebalance"])
                    if (i - (start5y + p)) % freq == 0:
                        meta_targets[p] = target_at(cfg, i)
                m_tgt = {c: sum(weights[p] * meta_targets[p].get(c, 0.0) for p in range(4))
                         for c in etfs}
                current_pct = {
                    c: (shares[c] * prices[c] / value * 100.0 if value > 0 and prices[c] > 0 else 0.0)
                    for c in etfs
                }
                moved = sum(abs(m_tgt[c] - current_pct[c]) for c in etfs)
                if moved >= THRESHOLD:
                    apply_weights(m_tgt, value, date)
                else:
                    deploy(inflow)
                current = m_tgt

            nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
            exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
            nav_after_list.append(nav_after)
            prev_after = nav_after
            weekly.append({
                "date": date, "exec": ex, "nav": nav_after,
                "target": {c: v for c, v in current.items() if v > 0},
            })

        arr = np.asarray(twr_rets, dtype=float)
        index = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(index)
        mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
        return {
            "cum": prev_after / invested - 1.0 if invested > 0 else 0.0,
            "twr": float(index[-1] - 1.0) if len(index) else 0.0,
            "mdd": mdd, "sharpe": sharpe,
            "final": prev_after or 0.0, "invested": invested,
            "trades": trades, "weekly": weekly,
            "turnover": traded_total / sum(nav_after_list) if nav_after_list else 0.0,
            "avg_pos": float(np.mean(exposures)) if exposures else 0.0,
        }

    starts = ("2021-12-10", "2026-06-18")
    lines = [
        "# B0207 vs Meta(5pp)：未来 1 个月交易对比",
        "",
        "起点为信号日，账户从零开始、每周入金2000×n；B0207按原口径（非决策周入金存现金），",
        "Meta按每周评估+5pp阈值（保持周按持仓比例加仓）。",
        "",
    ]
    all_payload = []
    for start in starts:
        s = next(k for k in range(len(master_dates) - 1, -1, -1) if master_dates[k] <= start)
        from datetime import date as ddate
        sd = ddate.fromisoformat(master_dates[s])
        ed = (sd + timedelta(days=28)).isoformat()
        e = next((k for k in range(len(master_dates) - 1, -1, -1) if master_dates[k] <= ed), s)
        print(start, "-> window", master_dates[s], "~", master_dates[e])
        rb = run("b0207", s, e)
        rm = run("meta", s, e)
        lines.append(f"## 起点 {master_dates[s]}（执行 {master_execs[s]}）~ {master_dates[e]}（约1个月）")
        lines.append("")
        lines.append("| 信号日 | B0207 NAV | B0207 周收益 | B0207 目标 | Meta NAV | Meta 周收益 | Meta 目标 |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for wb, wm in zip(rb["weekly"], rm["weekly"]):
            tb = "、".join(f"{NAMES[c]}{v:.0f}%" for c, v in wb["target"].items())
            tm = "、".join(f"{NAMES[c]}{v:.0f}%" for c, v in wm["target"].items())
            lines.append(f"| {wb['date']} | {wb['nav']:.0f} | - | {tb} | {wm['nav']:.0f} | - | {tm} |")
        lines.append("")
        lines.append("### B0207 交易明细")
        lines.append("")
        lines.append("| 执行日 | ETF | 方向 | 金额 |")
        lines.append("|---|---:|---:|---:|")
        for t in rb["trades"]:
            lines.append(f"| {t['date']} | {NAMES[t['code']]} | {t['action']} | {t['value']:.0f} |")
        lines.append("")
        lines.append("### Meta(5pp) 交易明细")
        lines.append("")
        lines.append("| 执行日 | ETF | 方向 | 金额 |")
        lines.append("|---|---:|---:|---:|")
        for t in rm["trades"]:
            lines.append(f"| {t['date']} | {NAMES[t['code']]} | {t['action']} | {t['value']:.0f} |")
        lines.append("")
        lines.append(f"**汇总：B0207 累计 {pct(rb['cum'])}（TWR {pct(rb['twr'])}，期末 {rb['final']:.0f}，"
                     f"交易 {len(rb['trades'])} 笔）；Meta 累计 {pct(rm['cum'])}（TWR {pct(rm['twr'])}，"
                     f"期末 {rm['final']:.0f}，交易 {len(rm['trades'])} 笔）**")
        lines.append("")
        all_payload.append({
            "start": master_dates[s], "end": master_dates[e],
            "b0207": {k: v for k, v in rb.items() if k not in ("weekly", "trades")},
            "meta": {k: v for k, v in rm.items() if k not in ("weekly", "trades")},
            "b0207_trades": rb["trades"], "meta_trades": rm["trades"],
        })

    md = OUT_DIR / "compare_b0207_vs_meta_month.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "compare_b0207_vs_meta_month.json").write_text(
        json.dumps(all_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
