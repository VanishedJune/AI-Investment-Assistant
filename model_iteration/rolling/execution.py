# -*- coding: utf-8 -*-
"""B0207 式执行策略：四周决策 + 非决策周只加仓 + 验证看空才卖出。

执行口径（与 B0207 操作逻辑一致）：
- 决策锚点：每 ``decision_cycle_weeks`` 周一次（默认 4 周，相位锚定首个可用锚点），
  按模型目标权重完整调仓，可买可卖、可新开仓/清仓；
- 非决策周：资金只用于加仓——当前持有份额时最多加至目标权重，不新开仓、不卖出；
- 非决策周卖出的唯一例外是“验证看空”保护：五条件同时成立
  （d_dif1<0、d_dif2<0、w_dif1<0、w_dif2<0、mom20<-0.05）时按
  ``sell_frac`` 减持当前份额；单次触发（``per_episode``），条件持续期间不重复卖出，
  条件恢复后（``reset_on_recovery``）结束保护期，回到只加仓规则。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_decision_anchor(index: int, decision_cycle_weeks: int) -> bool:
    """决策锚点：以 0 号可用锚点为相位起点，每 decision_cycle_weeks 周一次。"""
    return int(index) % max(1, int(decision_cycle_weeks)) == 0


def guard_triggered(features: pd.DataFrame, index: int, policy: dict) -> bool:
    """验证看空：五条件全部成立才算触发；任一条件缺失/NaN 视为未触发（fail-safe）。"""
    guard = policy.get("bearish_guard") or {}
    if not guard.get("enabled", True):
        return False
    conditions = guard.get("conditions") or []
    if not conditions or index < 0 or index >= len(features):
        return False
    row = features.iloc[index]
    for cond in conditions:
        feature = cond.get("feature")
        if feature not in row.index:
            return False
        value = row[feature]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        op = cond.get("op")
        target = cond.get("value")
        if op == "<" and not (float(value) < float(target)):
            return False
        if op == ">" and not (float(value) > float(target)):
            return False
        if op == "<=" and not (float(value) <= float(target)):
            return False
        if op == ">=" and not (float(value) >= float(target)):
            return False
        if op == "==" and not (float(value) == float(target)):
            return False
    return True


def next_executed_weight(
    current: float,
    target: float,
    decision: bool,
    guard: bool,
    episode: bool,
    policy: dict,
) -> tuple[float, bool]:
    """按执行策略把本周目标仓位转换为实际执行仓位，返回 (执行仓位, 保护期是否活动)。"""
    guard_cfg = policy.get("bearish_guard") or {}
    sell_frac = float(guard_cfg.get("sell_frac", 0.7))
    per_episode = bool(guard_cfg.get("per_episode", True))
    reset_on_recovery = bool(guard_cfg.get("reset_on_recovery", True))

    if decision:
        # 决策周：按目标权重完整调仓，并重置保护期
        return float(target), False

    if guard:
        if per_episode:
            if not episode:
                return current * (1.0 - sell_frac), True
            return current, episode
        # per_episode=False：每个看空周都按 sell_frac 减持
        return current * (1.0 - sell_frac), False

    if episode and reset_on_recovery:
        # 看空条件恢复：结束保护期，回到只加仓规则
        episode = False

    if current <= 1e-9:
        # 未持有份额：非决策周不新开仓
        return 0.0, False
    # 持有份额：只加仓（最多加至目标权重），不卖出
    return max(current, float(target)), episode
