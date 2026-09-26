"""CH_P129_2 策略摘要 + 近一年实际执行平均仓位。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.account import anchor_exec_prices, run_account, weekly_metrics  # noqa: E402
from rolling.ca import load_ca  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import load_state, set_workspace  # noqa: E402


def main() -> int:
    set_workspace("159915")
    cfg = load_etf_config("159915")
    ep = cfg["execution_policy"]
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data("159915")
    ca = load_ca("159915")
    state = load_state()
    spec = state["champion"]["spec"]
    champ_id = state["champion"]["id"]
    n = len(usable)
    pos = position_path(spec, features, n - 1)
    acct = run_account(
        pos, anchors, daily, ca, cost_bps=10, start_anchor=0,
        exec_prices=anchor_exec_prices(anchors, daily),
        execution_policy=ep, features=features,
    )
    navs = acct["nav"]
    exe = acct["positions"]
    rets = acct["returns"]
    signals = usable["signal"].dt.date.astype(str)
    idx1y = int(np.flatnonzero(signals >= "2025-08-15")[0])
    m_full = weekly_metrics(navs, exe, rets)
    m_1y = weekly_metrics(navs, exe, rets, window_start=idx1y)

    print(f"CHAMPION: {champ_id}")
    print("SPEC:", json.dumps(spec, ensure_ascii=False))
    print("EXECUTION_POLICY:", json.dumps({k: ep.get(k) for k in ("mode", "decision_cycle_weeks", "signal_smooth_days", "non_decision_graded")}, ensure_ascii=False))
    print("目标仓位统计: 全周期 mean=%.1f%% in_pos=%.1f%% max=%.0f%%" % (pos.mean() * 100, (pos > 0).mean() * 100, pos.max() * 100))
    print("实际执行仓位(全周期): %.1f%%" % (m_full["average_position"] * 100))
    print("实际执行仓位(近1年 %s~%s): %.1f%%" % (signals.iloc[idx1y], signals.iloc[-1], m_1y["average_position"] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
