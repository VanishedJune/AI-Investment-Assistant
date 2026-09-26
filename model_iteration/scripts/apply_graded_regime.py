"""把“5日均值 + 非决策周分级调仓”体制写入 8 支 ETF 配置；
159915 Champion 更新为 CH_P129_2，其余 7 支 Champion 更新为各自当前 state 冠军。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling import ledger  # noqa: E402

CONFIG_DIR = ROOT / "configs"

GRADED_EXEC = {
    "mode": "b0207_cycle",
    "decision_cycle_weeks": 4,
    "signal_smooth_days": 5,
    "non_decision_graded": {
        "enabled": True,
        "feature": "d_dif1",
        "add_cash_frac": 0.5,
        "sell_holdings_frac": 0.5,
    },
}


def load_raw(code: str) -> dict:
    return json.loads((CONFIG_DIR / f"etf_{code}.json").read_text(encoding="utf-8"))


def main() -> int:
    codes = ["159915", "159941", "159622", "512690", "512800", "516150", "518600", "515220"]
    for code in codes:
        ledger.set_workspace(code)
        state = ledger.load_state()
        cfg = load_raw(code)
        cfg["champion"] = {
            "id": state["champion"]["id"],
            "spec": state["champion"]["spec"],
        }
        existing = cfg.get("execution_policy") or {}
        cfg["execution_policy"] = {**existing, **GRADED_EXEC}
        (CONFIG_DIR / f"etf_{code}.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{code}: champion={cfg['champion']['id']} regime=graded5d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
