"""为 8 支 ETF 的当前冠军名称加 ETF 编号（如 CH_P129_2 → CH_159915_P129_2）。

同步：configs/etf_<code>.json champion.id、state.json champion.id、
active_challengers 中同名 PROMOTED 记录、promotion_history 末环 new_champion_id。
历史 events.jsonl 与迭代记录保持原名（只读审计）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling import ledger  # noqa: E402

CONFIG_DIR = ROOT / "configs"
CODES = ["159915", "159941", "159622", "512690", "512800", "516150", "518600", "515220"]


def new_id(code: str, old: str) -> str:
    prefix = f"CH_{code}_"
    if old.startswith(prefix) and not old[len(prefix):].startswith("CH_"):
        return old
    tail = old.split("CH_")[-1]
    suffix = f"_{code}"
    if tail.endswith(suffix):
        tail = tail[:-len(suffix)]
    return prefix + tail


def main() -> int:
    for code in CODES:
        cfg_path = CONFIG_DIR / f"etf_{code}.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        old = str(cfg["champion"]["id"])
        new = new_id(code, old)
        if old == new:
            print(f"{code}: 无需改名（{old}）")
            continue
        cfg["champion"]["id"] = new
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        ledger.set_workspace(code)
        state = ledger.load_state()
        state["champion"]["id"] = new
        for ch in state.get("active_challengers", []):
            if ch.get("id") == old:
                ch["id"] = new
        for entry in state.get("promotion_history", []):
            if entry.get("new_champion_id") == old:
                entry["new_champion_id"] = new
        ledger.save_state(state)
        print(f"{code}: {old} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
