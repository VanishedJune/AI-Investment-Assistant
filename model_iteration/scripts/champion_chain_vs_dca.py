# -*- coding: utf-8 -*-
"""159915 创业板冠军链（按生效锚点逐代切换）vs 智能定投 PK。

口径（两边资金完全一致）：
- 每周固定入金 2000 元；未开市（无可用数据）周不入金；
- 冠军链侧：按“届时生效的 Champion spec”逐周给出目标仓位再平衡，
  现金留存，raw_open 成交，10bp 成本，拆分/分红按显式 CA 事件调整；
- 智能定投侧：每周入金 2000 进入袖套现金，按支付宝“涨跌幅智能定投”扣款率
  决定投多少（最高 200%，超出从袖套现金支取，不足按可用现金封顶），
  未投部分留存，raw_open 成交，10bp 成本。

冠军链从 state.json 的 promotion_history 提取（effective_anchor 生效），
不把当前 Champion 回填到历史。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402

CODE = "159915"
NAME = "创业板ETF"
WEEKLY = 2000.0
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0


def trade_cost(value: float) -> float:
    """单边佣金：max(成交金额×0.025%, 5元)；深市 ETF 免印花税/过户费。"""
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


WINDOWS = {
    "full": None,
    "10y": "2016-08-10",
    "5y": "2021-08-10",
    "3y": "2023-08-10",
    "1y": "2025-08-10",
}
WINDOW_CN = {"full": "全周期", "10y": "最近10年", "5y": "最近5年", "3y": "最近3年", "1y": "最近1年"}
OUT_DIR = ROOT.parent / "champion_vs_dca"

COLUMN_CN = {
    "side": "方案",
    "window": "周期",
    "window_label": "区间",
    "cum_return": "累计收益率",
    "annual_return_irr": "年均收益率",
    "sharpe": "Sharpe",
    "max_drawdown": "最大回撤",
    "invested": "累计投入",
    "profit": "累计盈利",
    "avg_position": "平均持仓",
    "final_value": "期末市值",
    "twr": "TWR",
}


def load_champion_chain(state: dict) -> list[dict]:
    """返回 [(champion_id, spec, start_anchor, end_anchor)]，按生效锚点排序。"""
    by_id: dict[str, dict] = {}
    for ch in state.get("active_challengers", []):
        by_id.setdefault(ch.get("id"), ch)
    current = state.get("champion") or {}
    by_id.setdefault(current.get("id"), current)

    chain: list[dict] = [{
        "id": "champion_zero_axis",
        "spec": {"kind": "champion_zero_axis"},
        "effective_anchor": 0,
    }]
    for entry in state.get("promotion_history", []):
        eff = int(entry.get("effective_anchor", entry.get("anchor", 0)))
        chain.append({
            "id": entry.get("new_champion_id"),
            "spec": None,
            "effective_anchor": eff,
        })

    # 补齐每代 spec：优先由 active_challengers/当前 champion 提供
    for item in chain:
        if item["spec"] is not None:
            continue
        rec = by_id.get(item["id"]) or {}
        spec = rec.get("spec")
        if not spec:
            raise ValueError(f"找不到冠军 {item['id']} 的 spec")
        item["spec"] = spec

    chain.sort(key=lambda x: x["effective_anchor"])
    total = int(state.get("latest_anchor_index", 711))
    for idx, item in enumerate(chain):
        item["start_anchor"] = item["effective_anchor"]
        item["end_anchor"] = chain[idx + 1]["effective_anchor"] - 1 if idx + 1 < len(chain) else total
    return chain


def chain_positions(chain: list[dict], features, n_total: int) -> tuple[np.ndarray, dict]:
    """逐段用该代 Champion spec 计算全长仓位路径，再按生效区间拼接。"""
    positions = np.full(n_total, np.nan)
    segments: dict[str, dict] = {}
    for item in chain:
        spec = item["spec"]
        path = position_path(spec, features, n_total - 1)
        start = item["start_anchor"]
        end = item["end_anchor"]
        positions[start : end + 1] = path[start : end + 1]
        segments[item["id"]] = {
            "start": start,
            "end": end,
            "avg_position": float(np.nanmean(path[start : end + 1])) if end >= start else 0.0,
        }
    if np.isnan(positions).any():
        bad = int(np.where(np.isnan(positions))[0][0])
        raise ValueError(f"冠军链仓位路径存在空档: anchor={bad}")
    return positions, segments


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


def _first_exec(usable, start_idx: int, end_idx: int) -> str | None:
    for i in range(start_idx, end_idx + 1):
        ex = usable.iloc[i]["exec"]
        if ex is not None:
            return str(np.datetime64(ex).astype("datetime64[D]"))
    return None


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


def champion_chain_same_cash(usable, opens, ca, positions: np.ndarray, start_idx: int, end_idx: int) -> dict:
    cash = 0.0
    shares = 0.0
    nav_after: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    events = _ca_events(ca)
    applied = {key: False for key, _ in events}
    first_ex = _first_exec(usable, start_idx, end_idx)
    if first_ex is not None:
        for key, event in events:
            event_date = str(event.get("pay_date") or event.get("ex_date"))
            applied[key] = event_date < first_ex
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
        cost = trade_cost(abs(delta) * price)
        cash -= delta * price + cost
        shares = target_shares
        value_after = shares * price + cash
        nav_after.append(value_after)
        exposures.append(shares * price / value_after if value_after > 0 else 0.0)
    final = nav_after[-1] if nav_after else 0.0
    avg_pos = float(np.mean(exposures)) if exposures else float(np.nanmean(positions[start_idx : end_idx + 1]))
    return _metrics(twr_rets, flows, final, invested, avg_pos)


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
    applied = {key: False for key, _ in events}
    first_ex = _first_exec(usable, start_idx, end_idx)
    if first_ex is not None:
        for key, event in events:
            event_date = str(event.get("pay_date") or event.get("ex_date"))
            applied[key] = event_date < first_ex
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
            fee = trade_cost(amount)
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


def main() -> int:
    set_workspace(CODE)
    ca = load_ca(CODE)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(CODE)
    state = json.loads((ROOT / f"etf_{CODE}" / "weekly_rolling" / "state.json").read_text(encoding="utf-8"))
    chain = load_champion_chain(state)
    positions, segments = chain_positions(chain, features, len(usable))
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    dates = [d.date().isoformat() for d in usable["signal"]]

    report: dict = {
        "schema_version": "champion-chain-vs-dca-v1",
        "etf": CODE,
        "name": NAME,
        "chain": [
            {"id": c["id"], "start_anchor": c["start_anchor"], "end_anchor": c["end_anchor"],
             "avg_position": segments[c["id"]]["avg_position"]}
            for c in chain
        ],
        "windows": {},
    }
    rows: list[dict] = []
    for window, start_date in WINDOWS.items():
        start_idx = 0 if start_date is None else next(
            (i for i, d in enumerate(dates) if d >= start_date), len(dates) - 1
        )
        end_idx = len(usable) - 1
        if start_idx > end_idx:
            continue
        champ_m = champion_chain_same_cash(usable, opens, ca, positions, start_idx, end_idx)
        dca_m = smart_dca_same_cash(usable, opens, closes, ca, start_idx, end_idx)
        window_label = f"{dates[start_idx]}~{dates[end_idx]}"
        report["windows"][window] = {"window_label": window_label, "champion_chain": champ_m, "smart_dca": dca_m}
        rows.append({"window": window, "window_label": window_label, "champion_chain": champ_m, "smart_dca": dca_m})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "champion_chain_vs_dca.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    flat: list[dict] = []
    for r in rows:
        for side, side_name in (("champion_chain", "冠军链"), ("smart_dca", "智能定投")):
            flat.append({"side": side_name, "window": r["window"], "window_label": r["window_label"],
                         **{k: v for k, v in r[side].items()}})
    with (OUT_DIR / "champion_chain_vs_dca.csv").open("w", encoding="utf-8-sig", newline="") as f:
        headers = [COLUMN_CN[k] for k in flat[0]]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in flat:
            out = {}
            for key, value in r.items():
                if key == "window":
                    out[COLUMN_CN[key]] = WINDOW_CN[value]
                elif key in {"side", "window_label"}:
                    out[COLUMN_CN[key]] = value
                elif key in {"cum_return", "annual_return_irr", "max_drawdown", "twr"}:
                    out[COLUMN_CN[key]] = f"{float(value):.1%}"
                elif key in {"sharpe", "avg_position"}:
                    out[COLUMN_CN[key]] = f"{float(value):.2f}"
                else:
                    out[COLUMN_CN[key]] = f"{float(value):.0f}"
            writer.writerow(out)

    lines = [
        "# 159915 创业板冠军链 vs 智能定投 PK",
        "",
        "口径：两边资金完全一致，每周入金 2000 元，未开市周不入金；raw_open 成交、10bp 成本、"
        "拆分/分红按显式 CA 事件调整。冠军链按晋级历史逐代切换（不把当前冠军回填历史）。",
        "",
        "## 冠军链各代生效区间",
        "",
        "| 冠军 | 生效锚点区间 | 规则摘要 |",
        "|---|---|---|",
    ]
    brief = {
        "champion_zero_axis": "日周双正 → 满仓，否则空仓",
        "CH_S025_3": "mom20>0 → 80%；8周回撤>10%减半",
        "CH_S030_3": "mom20>0.06 → 90%；8周回撤>10%减半",
        "CH_S035_3": "mom20>0 → 80%；8周回撤>10%减半",
        "CH_P132_3": "mom20>0.06 → 满仓",
    }
    for c in chain:
        lines.append(f"| {c['id']} | {c['start_anchor']}~{c['end_anchor']} | {brief.get(c['id'], '')} |")

    lines.append("")
    lines.append("## PK 结果（同一张表）")
    lines.append("")
    for r in rows:
        lines.append(f"### {WINDOW_CN[r['window']]}（{r['window_label']}）")
        lines.append("")
        lines.append("| 方案 | 累计收益率 | 年均收益率 | Sharpe | 最大回撤 | 累计投入 | 累计盈利 | 平均持仓 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for side, side_name in (("champion_chain", "冠军链"), ("smart_dca", "智能定投")):
            m = r[side]
            lines.append(
                f"| {side_name} | {m['cum_return']:.1%} | {m['annual_return_irr']:.1%} | "
                f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} | {m['invested']:.0f} | "
                f"{m['profit']:.0f} | {m['avg_position']:.1%} | {m['final_value']:.0f} |"
            )
        lines.append("")

    (OUT_DIR / "champion_chain_vs_dca.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved to", OUT_DIR / "champion_chain_vs_dca.md")
    print(rows[0]["champion_chain"])
    print(rows[0]["smart_dca"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
