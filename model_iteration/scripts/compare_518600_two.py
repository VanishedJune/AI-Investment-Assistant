"""518600 黄金：CH_S064_1_518600 vs CH_P129_2（159915 策略）收益对比。

两者均在同一执行体制下（5日均值 + 非决策周分级调仓）用同现金流 B0207 评估。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.benchmark import rule_with_deposits, smart_dca  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402

CH_OWN = {
    "score": {"terms": [{"feature": "mom20", "weight": 1250.0}]},
    "gates": [{"feature": "mom20", "op": ">", "value": 0.0}],
    "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.8}], "min": 0.0, "max": 0.8},
    "drawdown_control": {"window_weeks": 8, "threshold": 0.1, "factor": 0.5},
}

CH_P129_2 = {
    "score": {
        "terms": [
            {"feature": "d_dif1", "weight": 1250.0},
            {"feature": "d_dif2", "weight": 625.0},
            {"feature": "w_dif1", "weight": 625.0},
            {"feature": "w_dif2", "weight": 375.0},
        ]
    },
    "gates": [
        {"feature": "w_dif1", "op": ">", "value": 0.0},
        {"feature": "w_ma_align", "op": ">=", "value": 0.5},
    ],
    "position": {
        "type": "step",
        "levels": [{"score_min": 0.0, "weight": 0.5}, {"score_min": 5.0, "weight": 0.8}],
        "min": 0.0,
        "max": 0.8,
    },
}

WINDOWS = [
    ("全周期", None),
    ("最近5年", "2021-08-10"),
    ("最近3年", "2023-08-10"),
    ("最近1年", "2025-08-10"),
]


def main() -> int:
    set_workspace("518600")
    cfg = load_etf_config("518600")
    ep = cfg["execution_policy"]
    _, _, _, _, usable, features = load_aligned_data("518600")
    signals = usable["signal"].dt.date.astype(str)
    n = len(usable)
    pos_own = position_path(CH_OWN, features, n - 1)
    pos_p129 = position_path(CH_P129_2, features, n - 1)

    lines = [
        "# 518600 黄金：CH_S064_1_518600 vs CH_P129_2（159915 策略）收益对比",
        "",
        "执行体制：5日均值信号 + 非决策周分级调仓（现金50%加仓/持仓50%卖出）；同现金流 B0207（周投2000、10bp、CA感知）。",
        "",
    ]
    for label, start_date in WINDOWS:
        start = 0 if start_date is None else next(i for i, d in enumerate(signals) if d >= start_date)
        end = n - 1
        own = rule_with_deposits(
            "518600", CH_OWN, base_amount=2000.0, cost_bps=10,
            end_idx=end, start_anchor=start, execution_policy=ep, features=features,
            positions=pos_own,
        )
        p129 = rule_with_deposits(
            "518600", CH_P129_2, base_amount=2000.0, cost_bps=10,
            end_idx=end, start_anchor=start, execution_policy=ep, features=features,
            positions=pos_p129,
        )
        dca = smart_dca("518600", end_idx=end, base_amount=2000.0, cost_bps=10, start_anchor=start)
        lines.append(f"## {label}（{signals.iloc[start]}~{signals.iloc[end]}）")
        lines.append("")
        lines.append("| 方案 | TWR | 年化IRR | Sharpe | 最大回撤 | 胜率 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, r in (
            ("自有冠军 CH_S064_1_518600", own),
            ("CH_P129_2（159915 策略）", p129),
            ("智能定投（参照）", dca),
        ):
            lines.append(
                f"| {name} | {r['twr']:.2%} | {r['irr_annual']:.2%} | {r['sharpe']:.2f} | "
                f"{r['mdd']:.2%} | {r['win_rate']:.1%} | {r['final_nav']:.0f} |"
            )
        lines.append("")
    out = ROOT / "logs" / "对比_518600_两策略_20260829.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
