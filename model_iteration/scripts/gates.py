# -*- coding: utf-8 -*-
"""Champion 晋级硬门禁：验证期指标 + 路径差异 + 空仓优势隔离。"""

from __future__ import annotations

import numpy as np


def evaluate_gates(
    champion: dict,
    challenger: dict,
    passive_cum: float,
    path_difference_fraction: float,
    gate_cfg: dict,
) -> dict:
    checks = {
        "cumulative_return": challenger["cumulative_return"] >= champion["cumulative_return"] - gate_cfg["cum_tolerance"],
        "sharpe": challenger["sharpe"] >= champion["sharpe"] - gate_cfg["sharpe_tolerance"],
        "max_drawdown": challenger["max_drawdown"] >= champion["max_drawdown"] - gate_cfg["dd_tolerance"],
        "win_rate": challenger["win_rate"] >= champion["win_rate"] - gate_cfg["win_rate_tolerance"],
        "median_weekly_return": challenger["median_weekly_return"] >= champion["median_weekly_return"] - gate_cfg["median_tolerance"],
        "participation": challenger["participation_up_weeks"] >= champion["participation_up_weeks"] - gate_cfg["participation_tolerance"],
        "no_cash_advantage": challenger["cumulative_return"] >= passive_cum - gate_cfg["passive_tolerance"],
        "mae": (challenger.get("mae") or 0.0) <= (champion.get("mae") or 0.0) * gate_cfg["mae_multiple"] + 1e-12,
        "brier": (challenger.get("brier") or 0.0) <= (champion.get("brier") or 0.0) * gate_cfg["brier_multiple"] + 1e-12,
        "path_difference": path_difference_fraction >= gate_cfg["min_path_difference"],
    }
    passed = all(checks.values())
    if passed:
        decision = "PROMOTION_RECOMMENDED"
    elif not checks["cumulative_return"] or not checks["sharpe"] or not checks["participation"]:
        decision = "REJECTED"
    else:
        decision = "SHADOW_EVALUATION"
    return {"decision": decision, "passed": passed, "checks": checks}


def path_difference(champion_pos: np.ndarray, challenger_pos: np.ndarray) -> float:
    valid = ~np.isnan(champion_pos) & ~np.isnan(challenger_pos)
    if not valid.any():
        return 0.0
    return float(np.mean((champion_pos[valid] != challenger_pos[valid])))
