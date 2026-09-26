# -*- coding: utf-8 -*-
"""8 ETF 冠军模型 vs 智能定投：全周期/10年/5年/3年/1年 PK 并落盘。

口径（两边资金完全一致）：
- 每周每只 ETF 固定入金 2000 元；未开市（无可用数据）周不入金；
- 冠军侧：按冠军规则目标仓位再平衡，现金留存，raw_open 成交，10bp；
- 智能定投侧：每周入金 2000 进入袖套账户，按支付宝“涨跌幅智能定投”扣款率
  决定投多少（最高 200%，超出部分从袖套现金缓冲支取，不足则按可用现金封顶），
  未投部分留在袖套现金，raw_open 成交，10bp。

指标：累计收益率、年均收益率(IRR)、Sharpe、最大回撤、累计投入、累计盈利、
平均持仓(平均暴露)、期末市值。
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402

CODES = ["159941", "159915", "518600", "512800", "512690", "512010", "159622", "516150"]
NAMES = {
    "159941": "纳指ETF",
    "159915": "创业板ETF",
    "518600": "黄金ETF",
    "512800": "银行ETF",
    "512690": "酒ETF",
    "512010": "医药ETF",
    "159622": "创新药ETF",
    "516150": "稀土ETF",
}
WINDOWS = {
    "full": None,
    "10y": "2016-08-10",
    "5y": "2021-08-10",
    "3y": "2023-08-10",
    "1y": "2025-08-10",
}
WEEKLY = 2000.0
COST_BPS = 10
OUT_DIR = ROOT.parent / "champion_vs_dca"
WINDOW_CN = {"full": "全周期", "10y": "最近10年", "5y": "最近5年", "3y": "最近3年", "1y": "最近1年"}
COLUMN_CN = {
    "etf": "ETF代码",
    "name": "ETF名称",
    "champion": "冠军模型",
    "window": "周期",
    "window_label": "区间",
    "champion_cum_return": "冠军_累计收益率",
    "champion_annual_return_irr": "冠军_年均收益率",
    "champion_sharpe": "冠军_Sharpe",
    "champion_max_drawdown": "冠军_最大回撤",
    "champion_invested": "冠军_累计投入",
    "champion_profit": "冠军_累计盈利",
    "champion_avg_position": "冠军_平均持仓",
    "champion_final_value": "冠军_期末市值",
    "champion_twr": "冠军_TWR",
    "dca_cum_return": "智能定投_累计收益率",
    "dca_annual_return_irr": "智能定投_年均收益率",
    "dca_sharpe": "智能定投_Sharpe",
    "dca_max_drawdown": "智能定投_最大回撤",
    "dca_invested": "智能定投_累计投入",
    "dca_profit": "智能定投_累计盈利",
    "dca_avg_position": "智能定投_平均持仓",
    "dca_final_value": "智能定投_期末市值",
    "dca_twr": "智能定投_TWR",
}


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


def _metrics(twr_rets: list[float], flows: list[tuple[int, float]], final_nav: float,
             invested: float, avg_pos: float) -> dict:
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
    }


def _ca_events(ca: dict) -> list[tuple[str, dict]]:
    return [(f"{e['type']}-{e['ex_date']}", e) for e in ca.get("events", [])]


def _ca_applied(events: list, first_ex: str | None) -> dict:
    applied = {}
    for key, event in events:
        event_date = str(event.get("pay_date") or event.get("ex_date"))
        applied[key] = first_ex is not None and event_date < first_ex
    return applied


def _apply_ca(shares: float, cash: float, events: list, ex_date: str, applied: dict) -> tuple[float, float]:
    for key, event in events:
        if applied.get(key):
            continue
        event_date = str(event.get("pay_date") or event.get("ex_date"))
        if ex_date >= event_date:
            if event["type"] == "split":
                shares *= float(event["ratio"])
            elif event["type"] == "cash_dividend":
                cash += shares * float(event.get("dps") or 0.0)
            applied[key] = True
    return shares, cash


def _first_exec(usable, start_idx: int, end_idx: int) -> str | None:
    for i in range(start_idx, end_idx + 1):
        ex = usable.iloc[i]["exec"]
        if ex is not None:
            return str(np.datetime64(ex).astype("datetime64[D]"))
    return None


def smart_dca_same_cash(usable, opens, closes, ca, start_idx: int, end_idx: int) -> dict:
    shares = 0.0
    cash = 0.0
    cost_adj = 0.0
    nav_after: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    events = _ca_events(ca)
    applied = _ca_applied(events, _first_exec(usable, start_idx, end_idx))
    for i in range(start_idx, end_idx + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        ex = np.datetime64(ex)
        ex_date = str(ex.astype("datetime64[D]"))
        price = float(opens.loc[ex])
        shares, cash = _apply_ca(shares, cash, events, ex_date, applied)
        value_before = shares * price + cash
        if nav_after:
            twr_rets.append(value_before / nav_after[-1] - 1.0)
        cash += WEEKLY
        invested += WEEKLY
        flows.append((i, -WEEKLY))
        prev_dates = closes.index[closes.index < ex]
        nav_ref = float(closes.loc[prev_dates[-1]]) if len(prev_dates) else price
        avg_cost = cost_adj / shares if shares > 0 else nav_ref
        pct = (nav_ref - avg_cost) / avg_cost if shares > 0 else 0.0
        if pct >= 0.025:
            rate = max(0.5, 1.0 - 2.0 * (pct - 0.025))
        elif pct <= -0.025:
            rate = min(2.0, 1.0 + 2.0 * (-pct - 0.025))
        else:
            rate = 1.0
        amount = min(WEEKLY * rate, cash)
        if amount > 0:
            fee = amount * COST_BPS / 10000.0
            net = amount - fee
            shares += net / price
            cost_adj += net
            cash -= amount
        value_after = shares * price + cash
        nav_after.append(value_after)
        exposures.append(shares * price / value_after if value_after > 0 else 0.0)
    final = nav_after[-1] if nav_after else 0.0
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    return _metrics(twr_rets, flows, final, invested, avg_pos)


def champion_same_cash(usable, opens, ca, positions: np.ndarray, start_idx: int, end_idx: int) -> dict:
    cash = 0.0
    shares = 0.0
    nav_after: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    events = _ca_events(ca)
    applied = _ca_applied(events, _first_exec(usable, start_idx, end_idx))
    for i in range(start_idx, end_idx + 1):
        ex = usable.iloc[i]["exec"]
        if ex is None:
            continue
        ex = np.datetime64(ex)
        ex_date = str(ex.astype("datetime64[D]"))
        price = float(opens.loc[ex])
        shares, cash = _apply_ca(shares, cash, events, ex_date, applied)
        nav_before = shares * price + cash
        if nav_after:
            twr_rets.append(nav_before / nav_after[-1] - 1.0)
        cash += WEEKLY
        invested += WEEKLY
        flows.append((i, -WEEKLY))
        value = shares * price + cash
        target_shares = float(positions[i]) * value / price
        delta = target_shares - shares
        cost = abs(delta) * price * COST_BPS / 10000.0
        cash -= delta * price + cost
        shares = target_shares
        value_after = shares * price + cash
        nav_after.append(value_after)
        exposures.append(shares * price / value_after if value_after > 0 else 0.0)
    final = nav_after[-1] if nav_after else 0.0
    avg_pos = float(np.mean(exposures)) if exposures else float(np.mean(positions[start_idx : end_idx + 1]))
    return _metrics(twr_rets, flows, final, invested, avg_pos)


def main() -> int:
    report: dict = {"schema_version": "champion-vs-dca-v1", "windows": WINDOWS, "results": {}}
    rows: list[dict] = []
    for code in CODES:
        set_workspace(code)
        ca = load_ca(code)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(code)
        champion_id, spec = champion_spec(code)
        positions = position_path(spec, features, len(usable) - 1)
        opens = daily_a.set_index("date")["raw_open"]
        closes = daily_a.set_index("date")["analysis_close"]
        dates = [d.date().isoformat() for d in usable["signal"]]
        report.setdefault("results", {})[code] = {"champion_id": champion_id, "windows": {}}
        for window, start_date in WINDOWS.items():
            start_idx = 0 if start_date is None else next(
                (i for i, d in enumerate(dates) if d >= start_date), len(dates) - 1
            )
            end_idx = len(usable) - 1
            if start_idx > end_idx:
                continue
            champ_m = champion_same_cash(usable, opens, ca, positions, start_idx, end_idx)
            dca_m = smart_dca_same_cash(usable, opens, closes, ca, start_idx, end_idx)
            window_label = f"{dates[start_idx]}~{dates[end_idx]}"
            entry = {
                "window": window,
                "window_label": window_label,
                "champion": champ_m,
                "dca": dca_m,
            }
            report["results"][code]["windows"][window] = entry
            rows.append({
                "etf": code, "name": NAMES[code], "champion": champion_id, "window": window,
                "window_label": window_label,
                **{f"champion_{k}": v for k, v in champ_m.items()},
                **{f"dca_{k}": v for k, v in dca_m.items()},
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT_DIR / "report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        headers = [COLUMN_CN[key] for key in rows[0]]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            out = {}
            for key, value in r.items():
                if key == "window":
                    out[COLUMN_CN[key]] = WINDOW_CN[value]
                elif key in {"etf", "name", "champion", "window_label"}:
                    out[COLUMN_CN[key]] = value
                elif key.endswith(("_cum_return", "_annual_return_irr", "_max_drawdown", "_twr")):
                    out[COLUMN_CN[key]] = f"{float(value):.1%}"
                elif key.endswith(("_sharpe", "_avg_position")):
                    out[COLUMN_CN[key]] = f"{float(value):.2f}"
                else:
                    out[COLUMN_CN[key]] = f"{float(value):.0f}"
            writer.writerow(out)

    lines = [
        "# 冠军模型 vs 智能定投 PK（同资金流：每周每只已开市 ETF 入金 2000）",
        "",
        "口径：两边累计投入完全一致（每周每只 ETF 2000，未开市周不入金；10bp 成本，raw_open 成交，"
        "拆分/分红按显式 CA 事件调整）。",
        "指标：累计收益率=期末市值/累计投入−1；年均收益率=资金加权 IRR；Sharpe/MDD 基于周收益。",
        "完整指标（累计投入、累计盈利、平均持仓等）见 report.csv / report.json。",
        "",
    ]
    labels = {"full": "全周期", "10y": "最近10年", "5y": "最近5年", "3y": "最近3年", "1y": "最近1年"}
    lines.append("## 汇总（冠军胜出的 ETF 数量，共 8 只）")
    lines.append("")
    lines.append("| 周期 | 累计收益率胜 | 年均收益率胜 | Sharpe胜 | 回撤更小 | 冠军胜出名单（按累计收益率） |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for window in WINDOWS:
        wr = [r for r in rows if r["window"] == window]
        cum_win = [r["etf"] for r in wr if float(r["champion_cum_return"]) > float(r["dca_cum_return"])]
        irr_win = sum(1 for r in wr if float(r["champion_annual_return_irr"]) > float(r["dca_annual_return_irr"]))
        sh_win = sum(1 for r in wr if float(r["champion_sharpe"]) > float(r["dca_sharpe"]))
        mdd_win = sum(1 for r in wr if float(r["champion_max_drawdown"]) > float(r["dca_max_drawdown"]))
        names = "、".join(cum_win) if cum_win else "无"
        lines.append(f"| {labels[window]} | {len(cum_win)} | {irr_win} | {sh_win} | {mdd_win} | {names} |")
    lines.append("")
    for window in WINDOWS:
        wr = [r for r in rows if r["window"] == window]
        if not wr:
            continue
        invested_note = f"{wr[0]['champion_invested']:.0f}"
        if window == "full":
            window_head = "全周期（各 ETF 自可用数据起，截至2026-08-07）"
        else:
            window_head = f"{labels[window]}（{wr[0]['window_label']}）"
        lines.append(f"\n## {window_head}（累计投入 {invested_note}，两边相同）")
        lines.append("")
        lines.append("| ETF | 策略 | 累计收益率 | 年均收益率 | Sharpe | 最大回撤 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in wr:
            for side, side_name in (("champion", "冠军"), ("dca", "智能定投")):
                lines.append(
                    f"| {r['etf']} {NAMES[r['etf']]} | {side_name} | "
                    f"{r[side + '_cum_return']:.1%} | {r[side + '_annual_return_irr']:.1%} | "
                    f"{r[side + '_sharpe']:.2f} | {r[side + '_max_drawdown']:.1%} | "
                    f"{r[side + '_final_value']:.0f} |"
                )
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved to", OUT_DIR)
    print("rows:", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
