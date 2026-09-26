# -*- coding: utf-8 -*-
"""加权 Meta 模型：按全周期累计收益率（锚定B）给 4 个相位冠军定权，每周决策。

- 权重：w_p = 全周期累计收益率_p / Σ(全周期累计收益率)，收益率越高权重越高；
- 每周：每个相位在自己的决策日（锚定B，与训练一致）更新目标，其余时间保持；
  Meta 目标 = Σ w_p × 相位目标；
- 决策：当前持仓与 Meta 目标的绝对差异总和 ≥ 阈值时才调仓到 Meta 目标，
  否则保持当前仓位（每周入金按现有持仓比例加仓）；
- 阈值对比：0 / 5pp / 10pp / 15pp；窗口：全周期 + 近5年。
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
    simulate_realistic,
)

WEEKLY = 2000.0


def trade_cost(value: float) -> float:
    return max(value * COMMISSION_RATE, MIN_COMMISSION) if value > 0 else 0.0


def _metrics(twr_rets, flows, final_nav, invested, avg_pos, turnover):
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
        "turnover": turnover,
    }


def pct(x) -> str:
    return f"{float(x):.1%}"


def simulate_meta(etfs, master_dates, master_execs, s, e, phase_configs, weights,
                  threshold_pp: float, start5y: int) -> dict:
    cash = 0.0
    shares = {c: 0.0 for c in etfs}
    applied = {}
    first_ex = next((ex for ex in master_execs[s : e + 1] if ex is not None), None)
    for c, et in etfs.items():
        applied[c] = {
            key: (first_ex is not None and str(ev.get("pay_date") or ev.get("ex_date")) < first_ex)
            for key, ev in et["ca_events"]
        }
    phase_targets: dict[int, dict[str, float]] = {}
    initialized = False
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    traded_total = 0.0
    nav_after_list: list[float] = []
    prev_after = None
    changes = 0

    def phase_target_at(cfg: dict, i: int) -> dict:
        date = master_dates[i]
        raw = {c: (et["by_date"].get(date, 0.0) if et["price_map"].get(master_execs[i], 0) > 0 else 0.0)
               for c, et in etfs.items()}
        floor = float(cfg["params"]["floor"])
        eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
        sig = {c: eff[c] > 0.0 for c in etfs}
        return dict(allocate_cfg(sig, eff, cfg["order"], cfg["params"])["weights"])

    def apply_weights(target_weights: dict, value: float) -> None:
        nonlocal cash, traded_total
        for c, et in etfs.items():
            if prices[c] <= 0:
                continue
            weight = float(target_weights.get(c, 0)) / 100.0
            tgt_share = int(weight * value / prices[c] // LOT) * LOT
            delta = tgt_share - shares[c]
            traded = abs(delta) * prices[c]
            traded_total += traded
            cost = trade_cost(traded)
            cash -= delta * prices[c] + cost
            shares[c] = tgt_share

    def deploy_to_holdings(inflow: float) -> None:
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

        if not initialized:
            for p, cfg in enumerate(phase_configs):
                freq = int(cfg["params"]["rebalance"])
                last = s - ((s - (start5y + p)) % freq)
                if last < 0:
                    phase_targets[p] = {c: 0.0 for c in etfs}
                else:
                    phase_targets[p] = phase_target_at(cfg, last)
            initialized = True
        for p, cfg in enumerate(phase_configs):
            freq = int(cfg["params"]["rebalance"])
            if (i - (start5y + p)) % freq == 0:
                phase_targets[p] = phase_target_at(cfg, i)

        meta: dict[str, float] = {c: 0.0 for c in etfs}
        for c in etfs:
            meta[c] = sum(weights[p] * phase_targets[p].get(c, 0.0) for p in range(4))
        current_pct = {
            c: (shares[c] * prices[c] / value * 100.0 if value > 0 and prices[c] > 0 else 0.0)
            for c in etfs
        }
        moved = sum(abs(meta[c] - current_pct[c]) for c in etfs)
        if moved >= threshold_pp:
            apply_weights(meta, value)
            changes += 1
        else:
            deploy_to_holdings(inflow)

        nav_after = sum(shares[c] * prices[c] for c in etfs) + cash
        exposures.append(sum(shares[c] * prices[c] for c in etfs) / nav_after if nav_after > 0 else 0.0)
        nav_after_list.append(nav_after)
        prev_after = nav_after

    avg_pos = float(np.mean(exposures)) if exposures else 0.0
    total_nav = sum(nav_after_list)
    turnover = traded_total / total_nav if total_nav > 0 else 0.0
    metrics = _metrics(twr_rets, flows, prev_after or 0.0, invested, avg_pos, turnover)
    metrics["changes"] = changes
    return metrics


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
    full_idx = (0, len(master_dates) - 1)
    start5y = range_idx(master_dates, *WINDOW_5Y)[0]
    idx5 = range_idx(master_dates, *WINDOW_5Y)

    data = json.loads((OUT_DIR / "b0207_phase_champions_v2_standard.json").read_text(encoding="utf-8"))
    phase_configs = [
        {"id": ch["model"]["id"], "order": ch["model"]["order"], "params": ch["model"]["params"]}
        for ch in data["champions"]
    ]
    # 全周期累计收益率（锚定B）→ 权重
    returns = []
    for p, cfg in enumerate(phase_configs):
        offset_b = (start5y + p) % 4
        r = simulate_realistic(etfs, master_dates, master_execs, *full_idx,
                               cfg["order"], cfg["params"], weekly_deploy=False,
                               phase_offset=offset_b)
        returns.append(r["cum_return"])
    total_r = sum(returns)
    weights = [r / total_r for r in returns]
    print("phase returns (anchor B):", [round(x, 4) for x in returns])
    print("weights:", [round(w, 4) for w in weights])

    thresholds = [0.0, 5.0, 10.0, 15.0]
    results = []
    for thr in thresholds:
        row = {"threshold_pp": thr}
        for label, idx in (("full", full_idx), ("5y", idx5)):
            row[label] = simulate_meta(etfs, master_dates, master_execs, *idx,
                                       phase_configs, weights, thr, start5y)
        results.append(row)
        print(f"thr={thr}pp full={row['full']['cum_return']:.4f} "
              f"5y={row['5y']['cum_return']:.4f} MDD5y={row['5y']['max_drawdown']:.4f} "
              f"changes5y={row['5y']['changes']}")

    ret_str = "、".join(f"相位{p}={x:.1%}" for p, x in enumerate(returns))
    w_str = "、".join(f"相位{p}={w:.1%}" for p, w in enumerate(weights))
    lines = [
        "# 加权 Meta 模型（全周期收益率定权 + 每周决策）",
        "",
        f"相位全周期累计收益率（锚定B）：{ret_str}",
        f"Meta 权重（按收益率占比）：{w_str}",
        "每周：各相位在自己决策日更新目标；Meta目标=Σ权重×相位目标；",
        "当前持仓与Meta目标差异总和≥阈值才调仓，否则保持仓位（入金按持仓比例加仓）。",
        "",
        "| 阈值 | 全周期累计收益率 | 全周期IRR | 全周期Sharpe | 全周期MDD | 近5年累计收益率 | 5年IRR | 5年Sharpe | 5年MDD | 5年调仓次数 | 5年换手 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        f5, y5 = r["full"], r["5y"]
        lines.append(
            f"| {r['threshold_pp']:.0f}pp | {pct(f5['cum_return'])} | {pct(f5['annual_return_irr'])} | "
            f"{f5['sharpe']:.2f} | {pct(f5['max_drawdown'])} | {pct(y5['cum_return'])} | "
            f"{pct(y5['annual_return_irr'])} | {y5['sharpe']:.2f} | {pct(y5['max_drawdown'])} | "
            f"{y5['changes']} | {y5['turnover']:.2f} |"
        )
    md = OUT_DIR / "meta_weighted_model.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT_DIR / "meta_weighted_model.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["阈值", "全周期累计收益率", "全周期IRR", "全周期Sharpe", "全周期MDD",
                         "近5年累计收益率", "5年IRR", "5年Sharpe", "5年MDD", "5年调仓次数", "5年换手"])
        for r in results:
            f5, y5 = r["full"], r["5y"]
            writer.writerow([f"{r['threshold_pp']:.0f}pp", pct(f5["cum_return"]),
                             pct(f5["annual_return_irr"]), f"{f5['sharpe']:.2f}",
                             pct(f5["max_drawdown"]), pct(y5["cum_return"]),
                             pct(y5["annual_return_irr"]), f"{y5['sharpe']:.2f}",
                             pct(y5["max_drawdown"]), y5["changes"], f"{y5['turnover']:.3f}"])
    payload = {
        "schema_version": "meta-weighted-v1",
        "phase_returns_anchorB": returns,
        "weights": weights,
        "results": results,
    }
    (OUT_DIR / "meta_weighted_model.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
