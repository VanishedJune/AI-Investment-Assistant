# -*- coding: utf-8 -*-
"""8 只 ETF 投资策略 vs 全部资金创业板(159915)智能定投 PK。

口径（两边资金完全一致）：
- 每周总入金 = 2000 × n，n = 该周已开市（有可用数据）ETF 数量；未开市 ETF
  该周不入金、不产生资金积累；
- 投资策略侧：总入金进入组合现金，按投资策略.md 的冠军信号 + 优先级仓位
  分配（5% 倍数、冠军仓位缩放）在下一交易日 raw_open 再平衡，10bp 成本，
  拆分/分红按显式 CA 事件调整；
- 智能定投侧：全部入金进入创业板(159915)单一袖套，按支付宝“涨跌幅智能定投”
  扣款率决定投多少（最高 200%，超出部分从袖套现金缓冲支取，不足按可用现金
  封顶），未投部分留存，raw_open 成交，10bp 成本，CA 事件调整。

冠军信号使用各 ETF 的冠军链（按 promotion_history 的 effective_anchor 逐代
切换，不把当前冠军回填历史）。
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

from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _metrics  # noqa: E402
from scripts.investment_priority import allocate  # noqa: E402

CODES = ["159915", "515220", "159622", "516150", "159941", "512690", "512800", "518600"]
NAMES = {
    "159915": "创业板ETF",
    "515220": "煤炭ETF国泰",
    "159622": "创新药ETF",
    "516150": "稀土ETF",
    "159941": "纳指ETF",
    "512690": "酒ETF",
    "512800": "银行ETF",
    "518600": "黄金ETF",
}
WEEKLY = 2000.0
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0


def trade_cost(value: float) -> float:
    """单边佣金：max(成交金额×0.025%, 5元)；深市 ETF 免印花税/过户费。"""
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


WINDOWS = {
    "5y": "2021-08-10",
    "3y": "2023-08-10",
}
WINDOW_CN = {"5y": "最近5年", "3y": "最近3年"}
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


def build_chain(state: dict) -> list[dict]:
    champ = state.get("champion") or {}
    ph = state.get("promotion_history") or []
    if not ph:
        chain = [{
            "id": champ.get("id", "champion_zero_axis"),
            "spec": champ.get("spec") or {"kind": "champion_zero_axis"},
            "effective_anchor": int(champ.get("effective_anchor", 0) or 0),
        }]
        total = int(state.get("latest_anchor_index", 0))
        chain[0]["start_anchor"] = chain[0]["effective_anchor"]
        chain[0]["end_anchor"] = total
        return chain
    by_id = {ch.get("id"): ch for ch in state.get("active_challengers", [])}
    by_id.setdefault(champ.get("id"), champ)
    chain = [{"id": "champion_zero_axis", "spec": {"kind": "champion_zero_axis"}, "effective_anchor": 0}]
    for p in ph:
        chain.append({
            "id": p.get("new_champion_id"),
            "spec": None,
            "effective_anchor": int(p.get("effective_anchor", p.get("anchor", 0))),
        })
    for item in chain:
        if item["spec"] is None:
            rec = by_id.get(item["id"]) or {}
            spec = rec.get("spec")
            if not spec:
                raise ValueError(f"{state.get('etf')}: 找不到冠军 {item['id']} 的 spec")
            item["spec"] = spec
    chain.sort(key=lambda x: x["effective_anchor"])
    total = int(state.get("latest_anchor_index", 0))
    for idx, item in enumerate(chain):
        item["start_anchor"] = item["effective_anchor"]
        item["end_anchor"] = chain[idx + 1]["effective_anchor"] - 1 if idx + 1 < len(chain) else total
    return chain


def chain_positions(state: dict, features, n_total: int) -> np.ndarray:
    chain = build_chain(state)
    pos = np.full(n_total, np.nan)
    for item in chain:
        path = position_path(item["spec"], features, n_total - 1)
        s, e = item["start_anchor"], item["end_anchor"]
        pos[s : e + 1] = path[s : e + 1]
    if np.isnan(pos).any():
        raise ValueError(f"{state.get('etf')}: 冠军链仓位存在空档")
    return pos


def newest_state(code: str) -> dict:
    root = ROOT / f"etf_{code}"
    best = None
    best_anchor = -1
    for ws in sorted(p for p in root.iterdir() if p.is_dir()):
        st = ws / "state.json"
        if not st.is_file():
            continue
        try:
            s = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        a = int(s.get("latest_anchor_index", -1) or -1)
        if a > best_anchor:
            best_anchor = a
            best = s
    if best is None:
        raise FileNotFoundError(f"no state for {code}")
    return best


def load_etf(code: str) -> dict:
    set_workspace(code)
    state = newest_state(code)
    ca = load_ca(code)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(code)
    pos = chain_positions(state, features, len(usable))
    by_date = {usable.iloc[i]["signal"].date().isoformat(): float(pos[i]) for i in range(len(usable))}
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    price_map = {}
    for i in range(len(usable)):
        ex = usable.iloc[i]["exec"]
        if ex is not None:
            exs = str(np.datetime64(ex).astype("datetime64[D]"))
            if exs in opens.index:
                price_map[exs] = float(opens.loc[exs])
    feat_by_date = {
        usable.iloc[i]["signal"].date().isoformat(): {
            col: float(features.iloc[i][col])
            for col in ("d_dif1", "d_dif2", "w_dif1", "w_dif2", "mom20")
        }
        for i in range(len(usable))
    }
    return {
        "state": state,
        "by_date": by_date,
        "opens": opens,
        "closes": closes,
        "ca_events": _ca_events(ca),
        "start_date": min(by_date.keys()),
        "price_map": price_map,
        "feat_by_date": feat_by_date,
    }


def strategy_account(etfs: dict, master_dates: list[str], master_execs: list[str],
                     start_idx: int, end_idx: int) -> dict:
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    for c, e in etfs.items():
        first_ex = None
        for i in range(start_idx, end_idx + 1):
            ex = master_execs[i]
            if ex is not None:
                first_ex = ex
                break
        applied[c] = {key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
                       for key, event in e["ca_events"]}
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    prev_after = None
    for i in range(start_idx, end_idx + 1):
        date = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        n_avail = sum(1 for e in etfs.values() if e["start_date"] <= date)
        if n_avail == 0:
            continue
        prices = {}
        for c, e in etfs.items():
            prices[c] = float(e["opens"].loc[ex]) if ex in e["opens"].index else 0.0
        for c, e in etfs.items():
            if prices[c] > 0 or shares[c] != 0:
                shares[c], cash = _apply_ca(shares[c], cash, e["ca_events"], ex, applied[c])
        nav_before = sum(shares[c] * prices[c] for c in etfs) + cash
        if prev_after is not None:
            twr_rets.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        flows.append((i - start_idx, -inflow))
        value = sum(shares[c] * prices[c] for c in etfs) + cash
        signals = {
            c: (e["by_date"].get(date, 0.0) > 0.0 and prices[c] > 0)
            for c, e in etfs.items()
        }
        positions = {
            c: (max(0.0, min(1.0, e["by_date"].get(date, 0.0))) if prices[c] > 0 else 0.0)
            for c, e in etfs.items()
        }
        target = allocate(signals, positions)
        for c, e in etfs.items():
            if prices[c] <= 0:
                continue
            weight = float(target["weights"].get(c, 0)) / 100.0
            tgt_share = weight * value / prices[c]
            delta = tgt_share - shares[c]
            cost = trade_cost(abs(delta) * prices[c])
            cash -= delta * prices[c] + cost
            shares[c] = tgt_share
        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        prev_after = nav_after
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos)


def gem_dca_account(etfs: dict, master_dates: list[str], master_execs: list[str],
                    start_idx: int, end_idx: int) -> dict:
    e = etfs["159915"]
    shares = 0.0
    cash = 0.0
    cost_adj = 0.0
    first_ex = None
    for i in range(start_idx, end_idx + 1):
        if master_execs[i] is not None:
            first_ex = master_execs[i]
            break
    applied = {key: (first_ex is not None and str(event.get("pay_date") or event.get("ex_date")) < first_ex)
               for key, event in e["ca_events"]}
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    prev_after = None
    for i in range(start_idx, end_idx + 1):
        date = master_dates[i]
        ex = master_execs[i]
        if ex is None:
            continue
        n_avail = sum(1 for ee in etfs.values() if ee["start_date"] <= date)
        if n_avail == 0:
            continue
        price = float(e["opens"].loc[ex])
        shares, cash = _apply_ca(shares, cash, e["ca_events"], ex, applied)
        nav_before = shares * price + cash
        if prev_after is not None:
            twr_rets.append(nav_before / prev_after - 1.0)
        inflow = WEEKLY * n_avail
        cash += inflow
        invested += inflow
        flows.append((i - start_idx, -inflow))
        prev_dates = e["closes"].index[e["closes"].index < ex]
        nav_ref = float(e["closes"].loc[prev_dates[-1]]) if len(prev_dates) else price
        avg_cost = cost_adj / shares if shares > 0 else nav_ref
        pct = (nav_ref - avg_cost) / avg_cost if shares > 0 else 0.0
        if pct >= 0.025:
            rate = max(0.5, 1.0 - 2.0 * (pct - 0.025))
        elif pct <= -0.025:
            rate = min(2.0, 1.0 + 2.0 * (-pct - 0.025))
        else:
            rate = 1.0
        amount = min(inflow * rate, cash)
        if amount > 0:
            fee = trade_cost(amount)
            net = amount - fee
            shares += net / price
            cost_adj += net
            cash -= amount
        nav_after = shares * price + cash
        exposures.append(shares * price / nav_after if nav_after > 0 else 0.0)
        prev_after = nav_after
    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    return _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos)


def main() -> int:
    etfs = {}
    for code in CODES:
        etfs[code] = load_etf(code)
        print("loaded", code, NAMES[code], "start", etfs[code]["start_date"],
              "chain", [c["id"] for c in build_chain(etfs[code]["state"])])

    set_workspace("159915")
    _, _, _, _, usable_m, _ = load_aligned_data("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]

    report: dict = {
        "schema_version": "investment-strategy-vs-gem-dca-v1",
        "rule": "投资策略.md（冠军信号 + 优先级仓位分配，5% 倍数）",
        "windows": {},
    }
    rows: list[dict] = []
    for window, start_date in WINDOWS.items():
        start_idx = next((i for i, d in enumerate(master_dates) if d >= start_date), len(master_dates) - 1)
        end_idx = len(master_dates) - 1
        strategy = strategy_account(etfs, master_dates, master_execs, start_idx, end_idx)
        dca = gem_dca_account(etfs, master_dates, master_execs, start_idx, end_idx)
        window_label = f"{master_dates[start_idx]}~{master_dates[end_idx]}"
        report["windows"][window] = {"window_label": window_label, "strategy": strategy, "gem_dca": dca}
        rows.append({"window": window, "window_label": window_label, "strategy": strategy, "gem_dca": dca})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "investment_strategy_vs_gem_dca.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flat: list[dict] = []
    for r in rows:
        for side, side_name in (("strategy", "投资策略(8ETF)"), ("gem_dca", "创业板智能定投")):
            flat.append({"side": side_name, "window": r["window"], "window_label": r["window_label"],
                         **{k: v for k, v in r[side].items()}})
    with (OUT_DIR / "investment_strategy_vs_gem_dca.csv").open("w", encoding="utf-8-sig", newline="") as f:
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
        "# 8 ETF 投资策略 vs 全部资金创业板智能定投 PK",
        "",
        "口径：每周总入金 = 2000 × n（n = 当周已开市 ETF 数量），两边资金完全一致；"
        "raw_open 成交、10bp 成本、拆分/分红按显式 CA 事件调整。",
        "投资策略：按投资策略.md 的冠军信号 + 优先级仓位分配（5% 倍数、冠军仓位缩放）；"
        "冠军信号使用各 ETF 冠军链（逐代切换）。",
        "智能定投：全部入金进入 159915 创业板单一袖套，按支付宝涨跌幅扣款率定投，未投部分留存现金。",
        "",
    ]
    for r in rows:
        lines.append(f"### {WINDOW_CN[r['window']]}（{r['window_label']}）")
        lines.append("")
        lines.append("| 方案 | 累计收益率 | 年均收益率 | Sharpe | 最大回撤 | 累计投入 | 累计盈利 | 平均持仓 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for side, side_name in (("strategy", "投资策略(8ETF)"), ("gem_dca", "创业板智能定投")):
            m = r[side]
            lines.append(
                f"| {side_name} | {m['cum_return']:.1%} | {m['annual_return_irr']:.1%} | "
                f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} | {m['invested']:.0f} | "
                f"{m['profit']:.0f} | {m['avg_position']:.1%} | {m['final_value']:.0f} |"
            )
        lines.append("")

    (OUT_DIR / "investment_strategy_vs_gem_dca.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved to", OUT_DIR / "investment_strategy_vs_gem_dca.md")
    for w in WINDOWS:
        print(w, report["windows"][w]["strategy"])
        print(w, report["windows"][w]["gem_dca"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
