"""M0207 看空保护对比：五条件(现状) vs avg5(d_dif1)<0（最近五日平均日K一阶导）。

其余口径保持 08-11 忠实复现：旧冠军链、5年窗口相位、每周加仓、85/15/ladder0/max100、
煤炭第6位。卖出均为 70%（单次保护，条件恢复后重置；否则继续持有）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.calendar import build_anchors, load_daily  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.features import ALL_FEATURES, analysis_frames, build_extended_features  # noqa: E402
from rolling.ledger import load_state, set_workspace  # noqa: E402
from scripts.champion_chain_vs_dca import _apply_ca, _ca_events, _irr  # noqa: E402
from scripts.features import indicator_frame  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import build_chain  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import CASH_FACTOR, LOT, MIN_COMMISSION, COMMISSION_RATE  # noqa: E402
from scripts.frozen_state import load_oldrules_state  # noqa: E402

ORDER = ["159915", "518600", "516150", "159622", "159941", "515220", "512800", "512690"]
PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
          "ladder": 0, "max_single": 100, "rebalance": 4}
WEEKLY = 2000.0
MOM20_TH = -0.05


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def load_old_state(code: str) -> dict:
    return load_oldrules_state(code)


def chain_positions_old(state: dict, features, n: int) -> np.ndarray:
    chain = build_chain(state)
    pos = np.full(n, np.nan)
    for idx, item in enumerate(chain):
        path = position_path(item["spec"], features, n - 1)
        s = int(item["start_anchor"])
        e = int(item["end_anchor"]) if idx < len(chain) - 1 else n - 1
        s = max(0, min(s, n - 1))
        e = max(0, min(e, n - 1))
        pos[s : e + 1] = path[s : e + 1]
    if np.isnan(pos).any():
        first = int(np.where(~np.isnan(pos))[0][0])
        pos[:first] = 0.0
    if np.isnan(pos).any():
        raise ValueError(f"{state.get('etf')}: 冠军链存在空档")
    return pos


def load_frames(code: str):
    set_workspace(code)
    ca = load_ca(code)
    daily_a, weekly_a = analysis_frames(code, ca)
    daily = load_daily(code)
    anchors = build_anchors(code, daily_a, weekly_a)
    feat_raw = build_extended_features(anchors, code, ca)
    mask = ~feat_raw[ALL_FEATURES].isna().any(axis=1)
    usable = anchors[mask].reset_index(drop=True)
    features = feat_raw[mask].reset_index(drop=True)
    return ca, daily_a, usable, features


def main() -> int:
    etfs = {}
    for code in ORDER:
        ca, daily_a, usable, features = load_frames(code)
        state = load_old_state(code)
        pos = chain_positions_old(state, features, len(usable))
        by_date = {usable.iloc[i]["signal"].date().isoformat(): float(pos[i]) for i in range(len(usable))}
        opens = daily_a.set_index("date")["raw_open"]
        price_map = {}
        for i in range(len(usable)):
            ex = usable.iloc[i]["exec"]
            if ex is not None:
                exs = str(np.datetime64(ex).astype("datetime64[D]"))
                if exs in opens.index:
                    price_map[exs] = float(opens.loc[exs])
        di = indicator_frame(daily_a).set_index("date")
        feat_by_date = {}
        for i in range(len(usable)):
            sig = usable.iloc[i]["signal"].date().isoformat()
            vis = di.loc[: usable.iloc[i]["signal"]]
            avg5 = float(vis["d_dif1"].tail(5).mean()) if len(vis) else float("nan")
            row = features.iloc[i]
            feat_by_date[sig] = {
                "d_dif1": float(row["d_dif1"]),
                "d_dif2": float(row["d_dif2"]),
                "w_dif1": float(row["w_dif1"]),
                "w_dif2": float(row["w_dif2"]),
                "mom20": float(row["mom20"]),
                "avg5_dif1": avg5,
            }
        etfs[code] = {
            "by_date": by_date, "opens": opens,
            "ca_events": _ca_events(ca),
            "start_date": min(by_date.keys()),
            "price_map": price_map,
            "feat_by_date": feat_by_date,
        }

    set_workspace("159915")
    _, _, _, _, usable_m, _ = load_aligned_data("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    start5y = range_idx(master_dates, "2021-08-10", "2026-08-07")[0]
    end = len(master_dates) - 1
    starts = {
        "5y": start5y,
        "3y": range_idx(master_dates, "2023-08-10", "2026-08-07")[0],
        "1y": range_idx(master_dates, "2025-08-10", "2026-08-07")[0],
    }

    def simulate(s: int, e: int, guard_mode: str) -> dict:
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

            for c, et in etfs.items():
                feat = et.get("feat_by_date", {}).get(date, {})
                if guard_mode == "avg5":
                    trig = (
                        shares[c] > 0 and prices[c] > 0
                        and float(feat.get("avg5_dif1", 1.0)) < 0
                    )
                else:
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

            if (i - start5y) % int(PARAMS["rebalance"]) == 0:
                raw = {c: (et["by_date"].get(date, 0.0) if prices[c] > 0 else 0.0)
                       for c, et in etfs.items()}
                eff = {c: min(1.0, max(PARAMS["floor"], raw[c])) for c in etfs}
                sig = {c: eff[c] > 0.0 for c in etfs}
                target = allocate_cfg(sig, eff, ORDER, PARAMS)
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
        arr = np.asarray(twr_rets, dtype=float)
        index = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(index)
        mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
        return {
            "cum_return": prev_after / invested - 1.0 if invested > 0 else 0.0,
            "annual_return_irr": _irr(flows, prev_after or 0.0),
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "invested": invested,
            "profit": (prev_after or 0.0) - invested,
            "avg_position": avg_pos,
            "final_value": prev_after or 0.0,
            "turnover": turnover,
            "triggers": triggers,
        }

    lines = [
        "# M0207 看空保护对比：五条件(现状) vs avg5(d_dif1)<0（最近五日平均日K一阶导）",
        "",
        "其余口径一致：旧冠军链、5年窗口相位、每周加仓、85/15/ladder0/max100、煤炭第6位；"
        "卖出 70%、单次保护、条件恢复后重置（否则继续持有）。",
        "",
        "| 窗口 | 方案 | 真实收益率 | IRR | Sharpe | MDD | 平均持仓 | 保护触发 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, s in starts.items():
        r5 = simulate(s, end, "five")
        ra = simulate(s, end, "avg5")
        lines.append(
            f"| {label} | 五条件（现状） | {r5['cum_return']:.2%} | {r5['annual_return_irr']:.2%} | "
            f"{r5['sharpe']:.2f} | {r5['max_drawdown']:.2%} | {r5['avg_position']:.1%} | {r5['triggers']} |"
        )
        lines.append(
            f"| {label} | avg5(d_dif1)<0（新） | {ra['cum_return']:.2%} | {ra['annual_return_irr']:.2%} | "
            f"{ra['sharpe']:.2f} | {ra['max_drawdown']:.2%} | {ra['avg_position']:.1%} | {ra['triggers']} |"
        )
        print(label, "five", f"{r5['cum_return']:.4f}", "trig", r5["triggers"],
              "| avg5", f"{ra['cum_return']:.4f}", "trig", ra["triggers"])
    out = ROOT / "logs" / "对比_看空保护_五条件_vs_avg5_20260829.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
