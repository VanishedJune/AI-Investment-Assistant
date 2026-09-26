# -*- coding: utf-8 -*-
"""AGENT_CHALLENGE_PROPOSAL 校验与冻结。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .ledger import proposal_path
from .spec import validate_spec

REQUIRED = [
    "iteration_id", "opaque_due_id", "challenge_round", "challenge_type",
    "diagnosis", "primary_failure_mode", "secondary_failure_mode",
    "evidence", "hypothesis", "challengers",
]
FORBIDDEN_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"159915|创业板|易方达|TARGET_A|anchor_index|start_date", re.IGNORECASE),
]


def _scan(value, errors: list[str], path: str = "", patterns: list = FORBIDDEN_PATTERNS) -> None:
    if path.lstrip(".").startswith("orchestrated"):
        return  # orchestrator 回挂块（anchor/date/asset）豁免盲化扫描
    if isinstance(value, dict):
        for k, v in value.items():
            _scan(v, errors, f"{path}.{k}", patterns)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _scan(v, errors, f"{path}[{i}]", patterns)
    elif isinstance(value, str):
        for pattern in patterns:
            if pattern.search(value):
                errors.append(f"{path} 含禁止内容: {value[:40]}")


def validate_proposal(obj: dict, etf: str | None = None) -> list[str]:
    errors = []
    for field in REQUIRED:
        if field not in obj:
            errors.append(f"缺少字段: {field}")
    patterns = FORBIDDEN_PATTERNS
    if etf:
        from .config import load_etf_config

        terms = [etf]
        name = load_etf_config(etf).get("name")
        if name:
            terms.append(name)
        patterns = FORBIDDEN_PATTERNS + [re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)]
    _scan(obj, errors, patterns=patterns)
    for idx, ch in enumerate(obj.get("challengers", [])):
        spec = ch.get("spec") or {}
        errors.extend(f"challengers[{idx}]: {e}" for e in validate_spec(spec))
        for field in ("id", "target_problem", "change_description", "changed_dimensions",
                      "old_values", "new_values", "expected_effect", "main_risk", "reason"):
            if field not in ch:
                errors.append(f"challengers[{idx}] 缺少字段: {field}")
    return errors


def freeze_final_proposal(obj: dict, etf: str = "159915", usable=None) -> Path:
    errors = validate_proposal(obj, etf=etf)
    if errors:
        raise ValueError("Proposal 校验失败: " + "; ".join(errors))
    round_id = obj["challenge_round"]
    ctype = obj["challenge_type"]
    seq = int(round_id.split("_")[1])
    anchor_index = seq * 4 if ctype == "prediction" else seq * 8
    if usable is None:
        from .data import load_aligned_data

        _, _, _, _, usable, _ = load_aligned_data(etf)
    date = usable.iloc[anchor_index]["signal"].date().isoformat()
    payload = dict(obj)
    payload["status"] = "FINAL"
    payload["orchestrated"] = {
        "anchor_index": anchor_index,
        "date": date,
        "asset": etf,
    }
    path = proposal_path(round_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
