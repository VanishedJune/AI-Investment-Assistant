# -*- coding: utf-8 -*-
"""投资策略（8 ETF 优先级分配）vs 智能定投（逐 ETF 袖套）最近五年 PK。

口径（两边资金完全一致）：
- 每周每只“已开市（有可用数据）”ETF 固定入金 2000 元；未开市 ETF 该周不入金、
  不产生资金积累；
- 投资策略：把当周总入金（2000×N_available）并入组合账户，按冠军信号与优先级
  目标权重（含现金）在下一交易日 raw_open 再平衡，10bp 成本；
- 智能定投：每只 ETF 袖套每周同样入金 2000 元；扣款率只决定把多少投入份额、
  多少留在袖套现金（rate 最高 200%，超出当周入金的部分从袖套既有现金缓冲中
  支取；不足则按可用现金封顶），raw_open 成交，10bp 成本。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.benchmark import smart_dca  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts.investment_priority import allocate  # noqa: E402

CODES = ["159915", "512010", "159622", "516150", "159941", "512690", "512800", "518600"]
START = "2021-08-10"
WEEKLY = 2000.0
COST_BPS = 10


def champion_spec(code: str) -> tuple[str, dict]:
    root = ROOT / f"etf_{code}"
    best = None
    best_anchor = -1
    for ws in sorted(p for p in root.iterdir() if p.is_dir()):
        st = ws / "state.json"
        if not st.is_file():
            continue
        try:
            payload = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        anchor = int(payload.get("latest_anchor_index", -1) or -1)
        if anchor > best_anchor:
            best_anchor = anchor
            best = payload
    champion = (best or {}).get("champion") or {}
    return str(champion.get("id", "")), champion.get("spec") or {}


def _irr(flows: list[tuple[int, float]], final_nav: float) -> float:
    if not flows:
        return 0.0
    n = flows[-1][0]

    def f(r_week: float) -> float:
        total = final_nav
        for week, cf in flows:
            total += cf * (1.0 + r_week) ** (n - week)
        return total

    lo, hi = -0.5, 1.0
    flo = f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if abs(fmid) < 1e-10:
            lo = mid
            break
        if np.sign(fmid) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    return (1.0 + lo) ** 52.0 - 1.0


def _metrics(twr_rets: list[float], flows: list[tuple[int, float]], final_nav: float, invested: float) -> dict:
    arr = np.asarray(twr_rets, dtype=float)
    index = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(index)
    mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    return {
        "twr": float(index[-1] - 1.0) if len(index) else 0.0,
        "cum_money": final_nav / invested - 1.0 if invested > 0 else 0.0,
        "irr": _irr(flows, final_nav),
        "sharpe": sharpe,
        "mdd": mdd,
        "invested": invested,
        "final_nav": final_nav,
    }


def load_frames(code: str):
    set_workspace(code)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(code)
    _, spec = champion_spec(code)
    pos = position_path(spec, features, len(usable) - 1)
    by_date = {
        usable.iloc[i]["signal"].date().isoformat(): float(pos[i])
        for i in range(len(usable))
    }
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    return by_date, opens, closes


def main(start: str = START) -> int:
    frames = {}
    for code in CODES:
        frames[code] = load_frames(code)
        usable_start = min(frames[code][0].keys())
        frames[code] = (frames[code][0], frames[code][1], frames[code][2], usable_start)

    master_dates, master_exec = [], []
    set_workspace("159915")
    _, _, _, _, usable_m, _ = load_aligned_data("159915")
    for i in range(len(usable_m)):
        d = usable_m.iloc[i]["signal"].date().isoformat()
        ex = usable_m.iloc[i]["exec"]
        if d >= start:
            master_dates.append(d)
            master_exec.append(None if ex is None else str(np.datetime64(ex).astype("datetime64[D]")))

    # ---------- 投资策略账户 ----------
    cash = 0.0
    shares = {c: 0.0 for c in CODES}
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    prev_after = None
    nav_after = 0.0
    for week, date in enumerate(master_dates):
        ex = master_exec[week]
        if ex is None:
            continue
        n_available = sum(1 for c in CODES if frames[c][3] <= date)
        if n_available == 0:
            continue
        prices = {}
        for c in CODES:
            opens = frames[c][1]
            prices[c] = float(opens.loc[ex]) if ex in opens.index else 0.0
        nav_before = sum(shares[c] * prices[c] for c in CODES) + cash
        if prev_after is not None:
            twr_rets.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_available
        cash += inflow
        invested += inflow
        flows.append((week, -inflow))
        signals = {c: (frames[c][0].get(date, 0.0) > 0.0) for c in CODES}
        positions = {c: max(0.0, min(1.0, frames[c][0].get(date, 0.0))) for c in CODES}
        target = allocate(signals, positions)
        value = sum(shares[c] * prices[c] for c in CODES) + cash
        for c in CODES:
            tgt_share = float(target["weights"][c]) / 100.0 * value / prices[c] if prices[c] > 0 else 0.0
            delta = tgt_share - shares[c]
            cost = abs(delta) * prices[c] * COST_BPS / 10000.0
            cash -= delta * prices[c] + cost
            shares[c] = tgt_share
        nav_after = sum(shares[c] * prices[c] for c in CODES) + cash
        prev_after = nav_after
    strategy = _metrics(twr_rets, flows, nav_after, invested)

    # ---------- 智能定投（逐 ETF 袖套，每周每只 2000 固定入金） ----------
    sleeve = {c: {"shares": 0.0, "cost_adj": 0.0, "cash": 0.0, "final": 0.0} for c in CODES}
    twr_rets2: list[float] = []
    flows2: list[tuple[int, float]] = []
    invested2 = 0.0
    prev_after2 = None
    for week, date in enumerate(master_dates):
        ex = master_exec[week]
        if ex is None:
            continue
        n_available = sum(1 for c in CODES if frames[c][3] <= date)
        if n_available == 0:
            continue
        value_before = 0.0
        for c in CODES:
            opens = frames[c][1]
            if ex not in opens.index:
                continue
            s = sleeve[c]
            price = float(opens.loc[ex])
            value_before += s["shares"] * price + s["cash"]
        if prev_after2 is not None:
            twr_rets2.append(value_before / prev_after2 - 1.0)
        inflow_total = 0.0
        for c in CODES:
            opens = frames[c][1]
            closes = frames[c][2]
            if frames[c][3] > date or ex not in opens.index:
                continue
            s = sleeve[c]
            s["cash"] += WEEKLY
            inflow_total += WEEKLY
            price = float(opens.loc[ex])
            prev_dates = closes.index[closes.index < ex]
            nav_ref = float(closes.loc[prev_dates[-1]]) if len(prev_dates) else price
            avg_cost = s["cost_adj"] / s["shares"] if s["shares"] > 0 else nav_ref
            pct = (nav_ref - avg_cost) / avg_cost if s["shares"] > 0 else 0.0
            rate = 1.0
            if pct >= 0.025:
                rate = max(0.5, 1.0 - 2.0 * (pct - 0.025))
            elif pct <= -0.025:
                rate = min(2.0, 1.0 + 2.0 * (-pct - 0.025))
            amount = min(WEEKLY * rate, s["cash"])
            if amount <= 0:
                continue
            fee = amount * COST_BPS / 10000.0
            net = amount - fee
            s["shares"] += net / price
            s["cost_adj"] += net
            s["cash"] -= amount
        flows2.append((week, -inflow_total))
        invested2 += inflow_total
        value_after = 0.0
        for c in CODES:
            opens = frames[c][1]
            if ex in opens.index:
                s = sleeve[c]
                value_after += s["shares"] * float(opens.loc[ex]) + s["cash"]
        prev_after2 = value_after
        for c in CODES:
            if ex in frames[c][1].index:
                s = sleeve[c]
                sleeve[c]["final"] = s["shares"] * float(frames[c][1].loc[ex]) + s["cash"]
    dca = _metrics(twr_rets2, flows2, prev_after2 or 0.0, invested2)

    def fmt(r: dict) -> str:
        return (
            f"cum={r['cum_money']:>8.2%}  twr={r['twr']:>8.2%}  irr={r['irr']:>8.2%}  "
            f"sharpe={r['sharpe']:>6.2f}  mdd={r['mdd']:>8.2%}  invested={r['invested']:>9.0f}  final={r['final_nav']:>10.0f}"
        )
    print("window:", master_dates[0], "~", master_dates[-1], f"({len(master_dates)} weeks)")
    print("strategy:", fmt(strategy))
    print("dca_basket:", fmt(dca))
    print("sleeve finals (incl cash):", {c: round(sleeve[c]["final"]) for c in CODES})
    print("available starts:", {c: frames[c][3] for c in CODES})
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START, help="YYYY-MM-DD")
    args = parser.parse_args()
    raise SystemExit(main(args.start))
