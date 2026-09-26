"""读取从大型工作区备份中提取的轻量冠军状态。

冻结目录只保留历史脚本实际需要的 ``state.json``。它不提供完整工作区恢复，
也不得替代当前 ``etf_<code>/weekly_rolling/state.json``。
"""

from __future__ import annotations

import json
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = MODEL_ROOT / "frozen_champion_states" / "2026-08-29"


def frozen_state_path(snapshot_name: str) -> Path:
    if not snapshot_name or Path(snapshot_name).name != snapshot_name:
        raise ValueError(f"冻结快照名称非法：{snapshot_name!r}")
    return FROZEN_ROOT / snapshot_name / "state.json"


def load_frozen_state(snapshot_name: str) -> dict:
    path = frozen_state_path(snapshot_name)
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结冠军状态：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("champion"), dict):
        raise ValueError(f"冻结冠军状态结构无效：{path}")
    return payload


def load_oldrules_state(code: str) -> dict:
    code = str(code)
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"ETF代码非法：{code!r}")
    return load_frozen_state(f"etf_{code}_oldrules_20260828")
