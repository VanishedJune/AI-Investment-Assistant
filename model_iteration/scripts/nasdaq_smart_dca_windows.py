# -*- coding: utf-8 -*-
"""纳指ETF广发（159941）近1/3/5年智能定投复现。"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts.investment_strategy_vs_gem_dca import load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CASH_APR,
    COMMISSION_RATE,
    LOT,
    MIN_COMMISSION,
    WEEKLY,
    gem_dca_realistic,
)

OUT = ROOT.parent / "champion_vs_dca"
CODE = "159941"
WINDOWS = {
    "近5年": ("2021-08-10", "2026-08-07"),
    "近3年": ("2023-08-10", "2026-08-07"),
    "近1年": ("2025-08-10", "2026-08-07"),
}


def pct(value: float) -> str:
    return f"{float(value):.2%}"


def main() -> int:
    etf = load_etf(CODE)
    set_workspace(CODE)
    _, _, _, _, usable, _ = load_aligned_data(CODE)
    dates = [item.date().isoformat() for item in usable["signal"]]
    execs = [None if item is None else str(np.datetime64(item).astype("datetime64[D]")) for item in usable["exec"]]

    results: dict[str, dict] = {}
    for label, bounds in WINDOWS.items():
        start, end = range_idx(dates, *bounds)
        # gem_dca_realistic内部固定读取键159915；这里仅做键名适配，标的仍是159941。
        metrics = gem_dca_realistic({"159915": etf}, dates, execs, start, end)
        metrics["turnover"] = None
        metrics["turnover_status"] = "UNAVAILABLE_LEGACY_DCA_HELPER"
        metrics["requested_window"] = list(bounds)
        metrics["signal_start"] = dates[start]
        metrics["signal_end"] = dates[end]
        metrics["weeks"] = sum(1 for item in execs[start : end + 1] if item is not None)
        results[label] = metrics

    payload = {
        "schema_version": "single-etf-smart-dca-windows-v1",
        "instrument": {"code": CODE, "name": "纳指ETF广发"},
        "data_as_of": dates[-1],
        "method": {
            "initial_capital": 0,
            "weekly_contribution": WEEKLY,
            "profit_threshold": 0.025,
            "loss_threshold": -0.025,
            "contribution_rate_range": [0.5, 2.0],
            "sell_orders": False,
            "unspent_cash_retained": True,
            "lot": LOT,
            "commission_rate": COMMISSION_RATE,
            "minimum_commission": MIN_COMMISSION,
            "cash_apr": CASH_APR,
            "execution": "weekly signal, next tradeable open",
        },
        "results": results,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "nasdaq_smart_dca_1y_3y_5y"
    json_path = OUT / f"{stem}.json"
    csv_path = OUT / f"{stem}.csv"
    md_path = OUT / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["区间", "信号起点", "信号终点", "有效周数", "累计投入", "期末资产", "累计盈利", "投入收益", "TWR", "IRR", "Sharpe", "最大回撤", "平均持仓"])
        for label in ("近5年", "近3年", "近1年"):
            metrics = results[label]
            writer.writerow([
                label,
                metrics["signal_start"],
                metrics["signal_end"],
                metrics["weeks"],
                round(metrics["invested"], 2),
                round(metrics["final_value"], 2),
                round(metrics["profit"], 2),
                pct(metrics["cum_return"]),
                pct(metrics["twr"]),
                pct(metrics["annual_return_irr"]),
                f"{metrics['sharpe']:.3f}",
                pct(metrics["max_drawdown"]),
                pct(metrics["avg_position"]),
            ])

    lines = [
        "# 纳指ETF广发智能定投：近5年、3年、1年",
        "",
        "> 本地历史研究结果，不代表未来收益或交易指令。",
        "",
        "- 初始资金0元，每个有效周新增2,000元。",
        "- 相对持仓成本盈利超过2.5%时减少当周投入，亏损超过2.5%时增加投入，否则按标准额投入。",
        "- 不卖出；未投入资金保留现金并按1.75%年化计息；使用100份整手及单边万2.5、最低5元佣金。",
        "- 当周完成数据形成投入金额，下一可交易日开盘成交。",
        "",
        "| 区间 | 信号范围 | 有效周数 | 累计投入 | 期末资产 | 累计盈利 | 投入收益 | TWR | IRR | Sharpe | 最大回撤 | 平均持仓 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("近5年", "近3年", "近1年"):
        metrics = results[label]
        lines.append(
            f"| {label} | {metrics['signal_start']}—{metrics['signal_end']} | {metrics['weeks']} | "
            f"{metrics['invested']:,.2f} | {metrics['final_value']:,.2f} | {metrics['profit']:,.2f} | "
            f"{pct(metrics['cum_return'])} | {pct(metrics['twr'])} | {pct(metrics['annual_return_irr'])} | "
            f"{metrics['sharpe']:.3f} | {pct(metrics['max_drawdown'])} | {pct(metrics['avg_position'])} |"
        )
    lines += [
        "",
        "说明：投入收益=累计盈利÷累计外部投入；TWR剔除资金流影响并按期复利；IRR反映不同时点投入资金的年化回报。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"md": str(md_path), "json": str(json_path), "csv": str(csv_path), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
