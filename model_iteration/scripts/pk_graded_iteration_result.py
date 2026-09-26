"""159915 分级调仓体制下：迭代最终冠军 vs 分级基线 vs 智能定投 PK（引擎口径）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.benchmark import rule_with_deposits, smart_dca  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import load_state, set_workspace  # noqa: E402

GRADED_BASELINE = {
    "score": {"terms": [{"feature": "d_slope", "weight": 15.0}, {"feature": "d_dif1", "weight": 1500.0}]},
    "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 1.0}], "min": 0.0, "max": 1.0},
    "gates": [{"feature": "d_slope", "op": ">", "value": 0.0}],
}

WINDOWS = [
    ("全周期", None),
    ("最近10年", "2016-08-10"),
    ("最近5年", "2021-08-10"),
    ("最近3年", "2023-08-10"),
    ("最近1年", "2025-08-10"),
]


def main() -> int:
    set_workspace("159915")
    cfg = load_etf_config("159915")
    ep = cfg["execution_policy"]
    _, _, _, _, usable, features = load_aligned_data("159915")
    state = load_state()
    final_spec = state["champion"]["spec"]
    final_id = state["champion"]["id"]
    signals = usable["signal"].dt.date.astype(str)
    n = len(usable)

    lines = [
        f"# 159915 分级调仓体制下迭代结果 PK（最终冠军 {final_id} vs 分级基线 vs 智能定投）",
        "",
        "口径：同现金流 B0207（周投 2000、10bp、CA 感知）；执行体制 = 决策周 5 日均值 + 非决策周 50% 双向调仓。",
        "",
    ]
    for label, start_date in WINDOWS:
        start = 0 if start_date is None else next(i for i, d in enumerate(signals) if d >= start_date)
        end = n - 1
        final = rule_with_deposits(
            "159915", final_spec, base_amount=2000.0, cost_bps=10,
            end_idx=end, start_anchor=start, execution_policy=ep, features=features,
        )
        base = rule_with_deposits(
            "159915", GRADED_BASELINE, base_amount=2000.0, cost_bps=10,
            end_idx=end, start_anchor=start, execution_policy=ep, features=features,
        )
        dca = smart_dca("159915", end_idx=end, base_amount=2000.0, cost_bps=10, start_anchor=start)
        lines.append(f"## {label}（{signals.iloc[start]}~{signals.iloc[end]}）")
        lines.append("")
        lines.append("| 方案 | TWR | 年化IRR | Sharpe | 最大回撤 | 胜率 | 期末市值 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, r in (
            (f"迭代最终冠军 {final_id}", final),
            ("分级基线 CH_GRADED_5D", base),
            ("智能定投", dca),
        ):
            lines.append(
                f"| {name} | {r['twr']:.2%} | {r['irr_annual']:.2%} | {r['sharpe']:.2f} | "
                f"{r['mdd']:.2%} | {r['win_rate']:.1%} | {r['final_nav']:.0f} |"
            )
        lines.append("")
    out = ROOT / "logs" / "PK_159915_分级迭代结果_20260829.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
