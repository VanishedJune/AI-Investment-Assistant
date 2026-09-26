# -*- coding: utf-8 -*-
"""PIT 盲化诊断包：不含日期/代码/绝对价格/anchor_index。"""

from __future__ import annotations

import json
import re

from .ledger import diagnostic_path


def sanitize(round_id: str, challenge_type: str, stats: dict, failure_cases: list[dict]) -> dict:
    matured = stats.get("matured_1w", 0)
    complete = bool(failure_cases) or matured < 4
    match = re.fullmatch(r"ROUND_(\d+)_(prediction|strategy)", round_id)
    if not match:
        raise ValueError(f"invalid round id: {round_id}")
    type_code = "P" if challenge_type == "prediction" else "S"
    return {
        "opaque_due_id": f"DUE-{type_code}{int(match.group(1)):04d}",
        "asset": "TARGET_A",
        "challenge_type": challenge_type,
        "stats": stats,
        "failure_cases": failure_cases,
        "complete": complete,
    }


def write_diagnostic(round_id: str, payload: dict) -> None:
    diagnostic_path(round_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
