# -*- coding: utf-8 -*-
"""Challenger spec 通用解释器：score/gates/position 与策略修饰（ramp/drawdown）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ALL_SPEC_FEATURES


def score_from_spec(row: pd.Series, spec: dict) -> float:
    terms = (spec.get("score") or {}).get("terms") or []
    total = 0.0
    for term in terms:
        feature = term["feature"]
        total += float(row[feature]) * float(term.get("weight", 1.0))
    return total


def gates_ok(row: pd.Series, gates: list[dict] | None) -> bool:
    for gate in gates or []:
        value = float(row[gate["feature"]])
        threshold = float(gate["value"])
        op = gate["op"]
        if np.isnan(value):
            return False
        if op == ">" and not (value > threshold):
            return False
        if op == ">=" and not (value >= threshold):
            return False
        if op == "<" and not (value < threshold):
            return False
        if op == "<=" and not (value <= threshold):
            return False
    return True


def base_position(spec: dict, row: pd.Series) -> float:
    pos_spec = spec.get("position") or {"type": "linear", "slope": 0.0, "intercept": 0.0}
    score = score_from_spec(row, spec)
    if np.isnan(score):
        return 0.0
    if pos_spec["type"] == "linear":
        pos = float(pos_spec.get("slope", 1.0)) * score + float(pos_spec.get("intercept", 0.0))
    elif pos_spec["type"] == "step":
        pos = 0.0
        for level in pos_spec.get("levels", []):
            if score >= float(level["score_min"]):
                pos = float(level["weight"])
    else:
        pos = 0.0
    return float(np.clip(pos, float(pos_spec.get("min", 0.0)), float(pos_spec.get("max", 1.0))))


def compute_position(spec: dict, row: pd.Series, ctx: dict) -> float:
    if not gates_ok(row, spec.get("gates")):
        return 0.0
    pos = base_position(spec, row)

    ramp = spec.get("ramp")
    if ramp:
        history = ctx.get("position_history", [])
        consecutive = 0
        for h in reversed(history):
            if h > 0:
                consecutive += 1
            else:
                break
        confirm_ok = consecutive >= int(ramp.get("confirm_after_weeks", 2))
        if confirm_ok and gates_ok(row, ramp.get("confirm_gate")):
            pos = float(ramp.get("full_weight", pos))
        elif pos > 0:
            pos = float(ramp.get("initial_weight", pos))

    dd = spec.get("drawdown_control")
    if dd:
        equity = ctx.get("equity_history", [])
        window = int(dd.get("window_weeks", 8))
        threshold = float(dd.get("threshold", 0.10))
        if equity:
            recent = equity[-window:]
            peak = max(recent)
            if peak > 0 and (peak - equity[-1]) / peak > threshold:
                pos *= float(dd.get("factor", 0.5))
    return float(np.clip(pos, 0.0, 1.0))


def validate_spec(spec: dict) -> list[str]:
    errors: list[str] = []
    if spec.get("kind") == "learning_ridge":
        from .learning import validate_learning_spec

        return validate_learning_spec(spec)
    score_terms = (spec.get("score") or {}).get("terms") or []
    if not score_terms:
        errors.append("spec.score.terms 为空")
    for term in score_terms:
        if term.get("feature") not in ALL_SPEC_FEATURES:
            errors.append(f"未知特征: {term.get('feature')}")
    for gate in spec.get("gates") or []:
        if gate.get("feature") not in ALL_SPEC_FEATURES:
            errors.append(f"未知门控特征: {gate.get('feature')}")
    return errors
