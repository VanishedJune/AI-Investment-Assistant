"""8 只 ETF 最终 Champion（B0207 式执行）vs 智能定投 / 固定定投 PK（CA 感知）。

口径（两侧资金完全一致）：
- 每周固定入金 2000 元；未开市周不入金；raw_open 成交；10bp 双边成本；
- 拆分按显式 CA 事件调整份额、现金分红入现金（两侧同一套 CA 处理）；
- 冠军侧：最终 Champion spec 计算目标仓位 + 配置 execution_policy：
  每 4 周决策锚点完整调仓；非决策周仅加仓（持有份额时最多加至目标权重，
  未持有不新开仓）；验证看空五条件同时成立时减持 70%（单次保护期，恢复后重置）；
- 智能定投：支付宝“涨跌幅智能定投”（±2.5% 内 100%；涨多最低 50%；跌多最高 200%；
  T-1 参考净值）；
- 固定定投：每周固定 2000（扣款率 100%）。

冠军侧为“当前最终规则在全窗口回放”（事后视角，与既有 PK 口径一致）。

用法（model_iteration 目录）：
    python -m scripts.pk_vs_smart_dca
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
from rolling.challenge import position_path  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.execution import guard_triggered, is_decision_anchor  # noqa: E402
from rolling.ledger import load_state, set_workspace  # noqa: E402

CODES = ["159915", "159941", "159622", "512690", "512800", "516150", "518600", "515220"]
WEEKLY = 2000.0
COST_BPS = 10

WINDOWS = [
    ("full", None, "全周期"),
    ("10y", "2016-08-10", "最近10年"),
    ("5y", "2021-08-10", "最近5年"),
    ("3y", "2023-08-10", "最近3年"),
    ("1y", "2025-08-10", "最近1年"),
]


def _ca_events(ca: dict) -> list[tuple[str, dict]]:
    return [(f"{e['type']}-{e['ex_date']}", e) for e in ca.get("events", [])]


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


def _metrics(twr_rets: list[float], flows: list[tuple[int, float]], final_nav: float,
             invested: float, avg_pos: float) -> dict:
    arr = np.asarray(twr_rets, dtype=float)
    index = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(index)
    mdd = float(np.nanmin(index / peak - 1.0)) if len(index) else 0.0
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe = mean / std * np.sqrt(52.0) if std > 0 else 0.0
    wins = arr[arr > 0] if len(arr) else np.array([])
    win_rate = float(len(wins) / len(arr)) if len(arr) else 0.0

    def irr_f(r_week: float) -> float:
        n = flows[-1][0] if flows else 0
        total = final_nav
        for week, cf in flows:
            total += cf * (1.0 + r_week) ** (n - week)
        return total

    lo, hi = -0.5, 1.0
    flo = irr_f(lo)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if abs(irr_f(mid)) < 1e-10:
            lo = mid
            break
        if np.sign(irr_f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    irr = (1.0 + lo) ** 52.0 - 1.0
    return {
        "twr": float(index[-1] - 1.0) if len(index) else 0.0,
        "irr_annual": irr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "invested": invested,
        "final_nav": final_nav,
        "avg_position": avg_pos,
    }


def _trade_cost(value: float) -> float:
    return max(value * COST_BPS / 10000.0, 0.0) if value > 0 else 0.0


def champion_b0207_same_cash(
    usable, opens, ca, positions: np.ndarray, features, policy: dict,
    start_idx: int, end_idx: int,
) -> dict:
    cash = 0.0
    shares = 0.0
    nav_after: list[float] = []
    twr_rets: list[float] = []
    flows: list[tuple[int, float]] = []
    invested = 0.0
    exposures: list[float] = []
    episode = False
    cycle = int((policy or {}).get("decision_cycle_weeks", 4))
    guard_cfg = (policy or {}).get("bearish_guard") or {}
    sell_frac = float(guard_cfg.get("sell_frac", 0.7))
    per_episode = bool(guard_cfg.get("per_episode", True))
    reset_on_recovery = bool(guard_cfg.get("reset_on_recovery", True))
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
        target_w = float(positions[i])
        if is_decision_anchor(i, cycle):
            target_shares = target_w * value / price
            episode = False
        elif guard_triggered(features, i, policy):
            if per_episode:
                if not episode:
                    target_shares = shares * (1.0 - sell_frac)
                    episode = True
                else:
                    target_shares = shares
            else:
                target_shares = shares * (1.0 - sell_frac)
        else:
            if episode and reset_on_recovery:
                episode = False
            if shares <= 0:
                target_shares = 0.0
            else:
                current_w = shares * price / value if value > 0 else 0.0
                desired_w = max(current_w, target_w)
                target_shares = desired_w * value / price
        cost = _trade_cost(abs(target_shares - shares) * price)
        cash = value - target_shares * price - cost
        shares = target_shares
        value_after = shares * price + cash
        nav_after.append(value_after)
        exposures.append(shares * price / value_after if value_after > 0 else 0.0)
    final = nav_after[-1] if nav_after else 0.0
    avg_pos = float(np.mean(exposures)) if exposures else float(np.nanmean(positions[start_idx : end_idx + 1]))
    return _metrics(twr_rets, flows, final, invested, avg_pos)


def dca_same_cash(
    usable, opens, closes, ca, start_idx: int, end_idx: int, *, smart: bool,
) -> dict:
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
        if smart:
            avg_cost = cost_adj / shares if shares > 0 else nav_ref
            pct = (nav_ref - avg_cost) / avg_cost if shares > 0 else 0.0
            if pct >= 0.025:
                rate = max(0.5, 1.0 - 2.0 * (pct - 0.025))
            elif pct <= -0.025:
                rate = min(2.0, 1.0 + 2.0 * (-pct - 0.025))
            else:
                rate = 1.0
            amount = min(WEEKLY * rate, cash)
        else:
            amount = min(WEEKLY, cash)
        if amount > 0:
            fee = _trade_cost(amount)
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


def final_champion(etf: str) -> tuple[str, dict]:
    state = load_state()
    champion = state.get("champion") or {}
    return str(champion.get("id", "")), champion.get("spec") or {}


def main() -> int:
    report: dict = {"schema_version": "pk-8etf-vs-smart-dca-v2-ca-aware", "base_amount": WEEKLY, "cost_bps": COST_BPS, "etfs": {}}
    md: list[str] = []
    md.append("# 8 只 ETF 最终 Champion vs 智能定投 PK（B0207 式执行，CA 感知）")
    md.append("")
    md.append("- 口径：同现金流 = 每周入金 2000 元；raw_open 成交；10bp 双边成本；拆分按显式 CA 事件调整份额、分红入现金。")
    md.append("- 冠军侧：各支迭代后**最终 Champion** + 配置 `execution_policy`（每 4 周决策锚点完整调仓；"
              "非决策周资金仅加仓；验证看空保护卖出 70%）。冠军规则为当前最终规则在全窗口回放（事后视角）。")
    md.append("- 智能定投：支付宝“涨跌幅智能定投”（±2.5% 内 100%；涨多最低 50%；跌多最高 200%；T-1 参考净值）。")
    md.append("- 固定定投：每周固定 2000（扣款率 100%）。")
    md.append("")

    summary_rows: list[dict] = []
    for etf in CODES:
        set_workspace(etf)
        cfg = load_etf_config(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
        ca = load_ca(etf)
        signals = usable["signal"].dt.date.astype(str)
        policy = cfg.get("execution_policy") or None
        champ_id, champ_spec = final_champion(etf)
        positions = position_path(champ_spec, features, len(usable) - 1)
        opens = daily_a.set_index("date")["raw_open"]
        closes = daily_a.set_index("date")["analysis_close"]
        etf_records: dict[str, dict] = {}
        md.append(f"## {etf} {cfg.get('name')}（最终 Champion：{champ_id}）")
        md.append("")
        md.append("| 窗口 | 方案 | TWR | 年化IRR | Sharpe | 最大回撤 | 胜率 | 平均仓位 | 累计投入 | 期末市值 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, start_date, label in WINDOWS:
            if start_date is None:
                start_anchor = 0
            else:
                candidates = [i for i, d in enumerate(signals) if d >= start_date]
                if not candidates:
                    continue
                start_anchor = int(candidates[0])
            end_anchor = len(usable) - 1
            window_label = f"{signals.iloc[start_anchor]}~{signals.iloc[end_anchor]}"
            champ = champion_b0207_same_cash(
                usable, opens, ca, positions, features, policy, start_anchor, end_anchor
            )
            dca = dca_same_cash(usable, opens, closes, ca, start_anchor, end_anchor, smart=True)
            fixed = dca_same_cash(usable, opens, closes, ca, start_anchor, end_anchor, smart=False)
            etf_records[key] = {
                "window_label": window_label,
                "champion": champ,
                "smart_dca": dca,
                "fixed_dca": fixed,
            }
            for side_name, r in (("冠军", champ), ("智能定投", dca), ("固定定投", fixed)):
                md.append(
                    f"| {label}（{window_label}） | {side_name} | {r['twr']:.2%} | {r['irr_annual']:.2%} | "
                    f"{r['sharpe']:.2f} | {r['mdd']:.2%} | {r['win_rate']:.1%} | {r['avg_position']:.1%} | "
                    f"{r['invested']:.0f} | {r['final_nav']:.0f} |"
                )
        md.append("")
        report["etfs"][etf] = {
            "name": cfg.get("name"),
            "final_champion": champ_id,
            "windows": etf_records,
        }
        first_key = next(iter(etf_records))
        rec = etf_records[first_key]
        summary_rows.append({
            "etf": etf,
            "name": cfg.get("name"),
            "champion": champ_id,
            "window": rec["window_label"],
            "champ_twr": rec["champion"]["twr"],
            "champ_irr": rec["champion"]["irr_annual"],
            "champ_mdd": rec["champion"]["mdd"],
            "champ_avgpos": rec["champion"]["avg_position"],
            "dca_twr": rec["smart_dca"]["twr"],
            "dca_irr": rec["smart_dca"]["irr_annual"],
            "dca_mdd": rec["smart_dca"]["mdd"],
            "fixed_twr": rec["fixed_dca"]["twr"],
        })

    md.append("## 汇总（全周期窗口；数据不足的用首个可用窗口）")
    md.append("")
    md.append("| ETF | 窗口 | 冠军 TWR | 智能定投 TWR | 冠军 IRR | 智能定投 IRR | 冠军 MDD | 智能定投 MDD | 冠军平均仓位 | 固定定投 TWR |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        md.append(
            f"| {r['etf']} {r['name']} | {r['window']} | {r['champ_twr']:.2%} | {r['dca_twr']:.2%} | "
            f"{r['champ_irr']:.2%} | {r['dca_irr']:.2%} | {r['champ_mdd']:.2%} | {r['dca_mdd']:.2%} | "
            f"{r['champ_avgpos']:.1%} | {r['fixed_twr']:.2%} |"
        )
    md.append("")
    md.append("> 说明：冠军侧为最终规则的事后回放（hindsight），实际生效的晋级链见各支 `iterations/ITERATION_001_<code>.json`；"
              "本表仅作规则 vs 定投的基准 PK，不构成买卖建议。")

    out_dir = ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PK_8ETF_vs_智能定投_20260829.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "PK_8ETF_vs_智能定投_20260829.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("saved:", out_dir / "PK_8ETF_vs_智能定投_20260829.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
