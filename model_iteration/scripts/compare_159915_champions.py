"""159915 创业板：旧冠军（迭代前 CH_P132_3）vs 新冠军（迭代后 CH_P131_3）。

口径与 PK 报告一致：每周入金 2000、raw_open 成交、10bp 成本、CA 感知、
B0207 执行策略（4 周决策 / 非决策周仅加仓 / 看空保护）。

用法（model_iteration 目录）：
    python -m scripts.compare_159915_champions
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
from rolling.ledger import load_state, set_workspace  # noqa: E402
from scripts.pk_vs_smart_dca import WINDOWS, champion_b0207_same_cash  # noqa: E402

ETF = "159915"


def main() -> int:
    set_workspace(ETF)
    cfg = load_etf_config(ETF)
    state = load_state()
    old_id = (cfg.get("champion") or {}).get("id")
    old_spec = (cfg.get("champion") or {}).get("spec") or {}
    new_id = state["champion"]["id"]
    new_spec = state["champion"]["spec"]
    policy = cfg.get("execution_policy") or None

    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(ETF)
    ca = load_ca(ETF)
    signals = usable["signal"].dt.date.astype(str)
    opens = daily_a.set_index("date")["raw_open"]
    closes = daily_a.set_index("date")["analysis_close"]
    n = len(usable)

    old_pos = position_path(old_spec, features, n - 1)
    new_pos = position_path(new_spec, features, n - 1)

    def pos_stats(pos: np.ndarray) -> dict:
        return {
            "mean_target": float(pos.mean()),
            "in_pos_frac": float((pos > 0).mean()),
            "max_target": float(pos.max()),
        }

    md: list[str] = []
    md.append(f"# 159915 创业板：旧冠军 vs 新冠军（B0207 式同现金流对比）")
    md.append("")
    md.append(f"- 旧冠军（迭代前）：**{old_id}**")
    md.append(f"  - spec：`{json.dumps(old_spec, ensure_ascii=False)}`")
    md.append(f"- 新冠军（迭代后）：**{new_id}**")
    md.append(f"  - spec：`{json.dumps(new_spec, ensure_ascii=False)}`")
    chain = []
    for entry in state.get("promotion_history", []):
        chain.append(f"{entry.get('old_champion_id')} → {entry.get('new_champion_id')}（anchor {entry.get('anchor')}，effective {entry.get('effective_anchor')}）")
    md.append(f"- 本次迭代晋级链：{'；'.join(chain) if chain else '无晋级'}")
    md.append("")

    md.append("## 规则对比（全周期 715 锚点）")
    md.append("")
    md.append("| 项目 | 旧冠军 | 新冠军 |")
    md.append("|---|---|---|")
    md.append(f"| 打分特征 | mom20×175 | d_dif1×1000 − dd20×20 |")
    md.append(f"| 入场门控 | mom20 > 0.06 | 无（score ≥ 0 即持仓） |")
    md.append(f"| 目标仓位 | score≥0 → 100%（满仓） | score≥0 → 50%（半仓封顶） |")
    for label, fn in (("平均目标仓位", "mean_target"), ("在场时间占比", "in_pos_frac"), ("最高目标仓位", "max_target")):
        md.append(f"| {label} | {pos_stats(old_pos)[fn]:.1%} | {pos_stats(new_pos)[fn]:.1%} |")
    md.append("")

    md.append("## 同现金流 PK（每周 2000，B0207 执行）")
    md.append("")
    md.append("| 窗口 | 指标 | 旧冠军 | 新冠军 |")
    md.append("|---|---|---:|---:|")
    for key, start_date, label in WINDOWS:
        if start_date is None:
            start_anchor = 0
        else:
            candidates = [i for i, d in enumerate(signals) if d >= start_date]
            if not candidates:
                continue
            start_anchor = int(candidates[0])
        end_anchor = n - 1
        old_m = champion_b0207_same_cash(usable, opens, ca, old_pos, features, policy, start_anchor, end_anchor)
        new_m = champion_b0207_same_cash(usable, opens, ca, new_pos, features, policy, start_anchor, end_anchor)
        window_label = f"{signals.iloc[start_anchor]}~{signals.iloc[end_anchor]}"
        md.append(f"| **{label}（{window_label}）** | | | |")
        for metric, fmt in (
            ("twr", ".2%"), ("irr_annual", ".2%"), ("sharpe", ".2f"),
            ("mdd", ".2%"), ("win_rate", ".1%"), ("avg_position", ".1%"),
            ("final_nav", ".0f"),
        ):
            md.append(
                f"| | {metric} | {old_m[metric]:{fmt}} | {new_m[metric]:{fmt}} |"
            )
        md.append("")

    md.append("> 口径：同现金流（每周 2000）、raw_open 成交、10bp 成本、CA 感知；两规则均按 B0207 执行策略回放。")
    out = ROOT / "logs" / "对比_159915_新旧冠军_20260829.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
