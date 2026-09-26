# -*- coding: utf-8 -*-
"""159915 创业板“日周 DIF 双正”分层策略 vs 智能定投 PK。

策略规则（可调参数见 BASE_POS / FULL_POS）：
- 日K DIF 一阶导 > 0 且 日K DIF 二阶导 > 0  → 优先买入（基础仓位 60%）；
- 同时满足 周K DIF 一阶导 > 0 且 周K DIF 二阶导 > 0 → 买入更多（满仓 100%）；
- 否则空仓 0%。

口径（两边资金完全一致）：
- 每周固定入金 2000 元；未开市（无可用数据）周不入金；
- raw_open 成交、10bp 成本、拆分/分红按显式 CA 事件调整。
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
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402

from scripts.champion_chain_vs_dca import (
    WINDOWS,
    WINDOW_CN,
    smart_dca_same_cash,
)

CODE = "159915"
NAME = "创业板ETF"
BASE_POS = 0.6   # 仅日线双正：基础仓位
FULL_POS = 1.0   # 日周双正：满仓
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


def rule_positions(features, n_total: int) -> np.ndarray:
    """逐周生成目标仓位：日线双正 60%，日周双正 100%，否则 0%。"""
    pos = np.zeros(n_total, dtype=float)
    for i in range(n_total):
        row = features.iloc[i]
        daily_ok = float(row["d_dif1"]) > 0.0 and float(row["d_dif2"]) > 0.0
        weekly_ok = float(row["w_dif1"]) > 0.0 and float(row["w_dif2"]) > 0.0
        if daily_ok and weekly_ok:
            pos[i] = FULL_POS
        elif daily_ok:
            pos[i] = BASE_POS
        else:
            pos[i] = 0.0
    return pos


def main() -> int:
    set_workspace(CODE)
    ca = load_ca(CODE)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(CODE)
    positions = rule_positions(features, len(usable))
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    dates = [d.date().isoformat() for d in usable["signal"]]

    from scripts.champion_chain_vs_dca import champion_chain_same_cash as _run_rule

    report: dict = {
        "schema_version": "dif-dual-rule-vs-dca-v1",
        "etf": CODE,
        "name": NAME,
        "rule": {
            "daily": "d_dif1>0 and d_dif2>0",
            "weekly": "w_dif1>0 and w_dif2>0",
            "position": f"daily+weekly -> {FULL_POS:.0%}; daily only -> {BASE_POS:.0%}; else 0%",
        },
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
        rule_m = _run_rule(usable, opens, ca, positions, start_idx, end_idx)
        dca_m = smart_dca_same_cash(usable, opens, closes, ca, start_idx, end_idx)
        window_label = f"{dates[start_idx]}~{dates[end_idx]}"
        report["windows"][window] = {"window_label": window_label, "rule": rule_m, "smart_dca": dca_m}
        rows.append({"window": window, "window_label": window_label, "rule": rule_m, "smart_dca": dca_m})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dif_dual_rule_vs_dca.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flat: list[dict] = []
    for r in rows:
        for side, side_name in (("rule", "DIF双正策略"), ("smart_dca", "智能定投")):
            flat.append({"side": side_name, "window": r["window"], "window_label": r["window_label"],
                         **{k: v for k, v in r[side].items()}})
    with (OUT_DIR / "dif_dual_rule_vs_dca.csv").open("w", encoding="utf-8-sig", newline="") as f:
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
        "# 159915 创业板“日周 DIF 双正”策略 vs 智能定投 PK",
        "",
        "策略规则：日K DIF一阶导>0 且 二阶导>0 → 买入基础仓位 60%；"
        "同时满足周K DIF一阶导>0 且 二阶导>0 → 加仓至 100%；否则空仓。",
        "口径：两边资金完全一致，每周入金 2000 元，未开市周不入金；raw_open 成交、10bp 成本、"
        "拆分/分红按显式 CA 事件调整。",
        "",
    ]
    for r in rows:
        lines.append(f"### {WINDOW_CN[r['window']]}（{r['window_label']}）")
        lines.append("")
        lines.append("| 方案 | 累计收益率 | 年均收益率 | Sharpe | 最大回撤 | 累计投入 | 累计盈利 | 平均持仓 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for side, side_name in (("rule", "DIF双正策略"), ("smart_dca", "智能定投")):
            m = r[side]
            lines.append(
                f"| {side_name} | {m['cum_return']:.1%} | {m['annual_return_irr']:.1%} | "
                f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} | {m['invested']:.0f} | "
                f"{m['profit']:.0f} | {m['avg_position']:.1%} | {m['final_value']:.0f} |"
            )
        lines.append("")

    (OUT_DIR / "dif_dual_rule_vs_dca.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved to", OUT_DIR / "dif_dual_rule_vs_dca.md")
    print("full:", report["windows"]["full"]["rule"])
    print("dca :", report["windows"]["full"]["smart_dca"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
