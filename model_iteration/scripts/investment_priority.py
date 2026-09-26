# -*- coding: utf-8 -*-
"""B0207 确定性机械基线计算器。

方向和冠军原始仓位由各 ETF 当前生效冠军给出。本模块只按
``configs/investment_priority.json`` 执行 B0207 数学分配，结果只是一组供
Agent比较的机械基线，不是最终建议或用户目标。本模块不产生自然语言交易
结论、不修改真实持仓，也不得写入正式报告。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "investment_priority.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "policy_id",
        "allocation_engine",
        "runtime_role",
        "priority",
        "core_size",
        "secondary_core_size",
        "core_budget_pct",
        "remainder_budget_pct",
        "floor_step_pct",
        "max_single_pct",
        "decision_cycle_days",
        "weekly_deploy",
        "use_champion_position_scale",
        "single_bullish_full_budget",
        "core_ladder_pct",
        "remainder_ladder_pct",
    }
    optional = {"priority_slots"}
    unknown = set(config) - required - optional
    missing = required - set(config)
    if missing or unknown:
        raise ValueError(f"B0207配置字段错误 missing={sorted(missing)} unknown={sorted(unknown)}")
    if config["schema_version"] != "investment-priority-v4":
        raise ValueError("B0207生产配置版本必须为investment-priority-v4")
    if config["policy_id"] != "B0207" or config["allocation_engine"] != "investment_priority_v3":
        raise ValueError("B0207生产配置身份不一致")
    if config["runtime_role"] != "mechanical_baseline_only":
        raise ValueError("B0207只能作为机械基线，不能成为最终决策器")
    priority = list(config["priority"])
    if len(priority) != 8 or len(set(priority)) != 8:
        raise ValueError("B0207优先级必须恰好包含8只不同ETF")
    slots = config.get("priority_slots") or {}
    if set(slots) - set(priority):
        raise ValueError("B0207组合槽位只能挂接在既有8只ETF优先级代码上")
    for allocation_key, slot in slots.items():
        members = list(slot.get("members") or [])
        if not members or members[0] != allocation_key or len(members) != len(set(members)):
            raise ValueError(f"{allocation_key}组合槽位成员定义无效")
    if int(config["core_budget_pct"]) + int(config["remainder_budget_pct"]) != 100:
        raise ValueError("B0207核心与剩余预算必须合计100%")
    if int(config["floor_step_pct"]) != 5:
        raise ValueError("B0207生产步长固定为5%")
    if int(config["decision_cycle_days"]) != 28:
        raise ValueError("B0207生产决策周期固定为28天")
    if not isinstance(config.get("weekly_deploy"), bool) or not config["weekly_deploy"]:
        raise ValueError("B0207生产配置必须开启每周加仓（weekly_deploy=true）")
    return config


CONFIG = load_config()


def _validate_positions(priority: list[str], positions: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for code in priority:
        value = float(positions.get(code, 0.0) or 0.0)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{code}冠军仓位越界: {value}")
        normalized[code] = value
    return normalized


def allocate(
    signals: Mapping[str, bool],
    positions: Mapping[str, float] | None = None,
    *,
    config: dict | None = None,
) -> dict[str, object]:
    """按B0207返回确定性机械基线权重。

    ``signals`` 只能来自当前冠军仓位是否大于0；``positions`` 是包含回撤控制后的
    冠军仓位。返回值顺序与生产配置一致，权重均为5%的整数倍；调用方不得
    把该结果直接标记为Agent建议、用户目标或已成交仓位。
    """

    cfg = CONFIG if config is None else config
    priority = list(cfg["priority"])
    positions_n = _validate_positions(priority, positions or {})
    step = int(cfg["floor_step_pct"])
    core_budget = int(cfg["core_budget_pct"])
    remainder_budget = int(cfg["remainder_budget_pct"])
    max_single = int(cfg["max_single_pct"])
    core_size = int(cfg["core_size"])
    secondary_size = int(cfg["secondary_core_size"])
    weights: dict[str, int] = {code: 0 for code in priority}

    def bullish(codes: list[str]) -> list[str]:
        return [code for code in codes if bool(signals.get(code, False))]

    def ladder(bull_codes: list[str], group: list[str]) -> list[int]:
        ranks = [group.index(code) + 1 for code in bull_codes]
        key = ",".join(str(rank) for rank in ranks)
        try:
            values = cfg["core_ladder_pct"]["positions"][key]
        except KeyError as exc:
            raise ValueError(f"B0207缺少核心档位: group={group} key={key}") from exc
        return [int(value) for value in values]

    def scaled_group(bull_codes: list[str], budget: int, base: list[int]) -> dict[str, int]:
        if not bull_codes:
            return {}
        if len(bull_codes) != len(base):
            raise ValueError("B0207组内代码与档位长度不一致")
        if len(bull_codes) == 1 and bool(cfg["single_bullish_full_budget"]):
            return {bull_codes[0]: min(budget, max_single)}

        if bool(cfg["use_champion_position_scale"]):
            raw = [base_weight * positions_n[code] for code, base_weight in zip(bull_codes, base)]
            total = sum(raw)
            if total <= 0:
                raise ValueError("看多标的的冠军仓位必须大于0")
            effective = [int((budget * value / total) // step) * step for value in raw]
        else:
            effective = [int(value // step) * step for value in base]

        # 优先级非增；因缩放造成的倒挂由前序标的吸收，而不是改写方向。
        for index in range(1, len(effective)):
            effective[index] = min(effective[index], effective[index - 1])
        effective = [min(value, max_single) for value in effective]

        leftover = budget - sum(effective)
        while leftover >= step:
            changed = False
            for index in range(len(effective)):
                rank_cap = max_single if index == 0 else min(max_single, effective[index - 1])
                if effective[index] + step <= rank_cap:
                    effective[index] += step
                    leftover -= step
                    changed = True
                    if leftover < step:
                        break
            if not changed:
                break
        if leftover != 0:
            raise ValueError(f"B0207预算无法按{step}%步长完全分配: leftover={leftover}")
        return dict(zip(bull_codes, effective))

    top = priority[:core_size]
    middle = priority[core_size : core_size + secondary_size]
    tail = priority[core_size + secondary_size :]
    top_bull = bullish(top)
    middle_bull = bullish(middle)
    tail_bull = bullish(tail)

    if top_bull:
        weights.update(scaled_group(top_bull, core_budget, ladder(top_bull, top)))
        remainder_candidates = bullish(priority[core_size:])[:2]
    elif middle_bull:
        weights.update(scaled_group(middle_bull, core_budget, ladder(middle_bull, middle)))
        remainder_candidates = tail_bull[:2]
    elif tail_bull:
        weights.update(scaled_group(tail_bull, core_budget, ladder(tail_bull, tail)))
        remainder_candidates = []
    else:
        remainder_candidates = []

    if remainder_candidates:
        rem_base = [int(value) for value in cfg["remainder_ladder_pct"][str(len(remainder_candidates))]]
        weights.update(scaled_group(remainder_candidates, remainder_budget, rem_base))

    total = sum(weights.values())
    cash = 100 - total
    if cash < 0 or any(value % step for value in weights.values()) or cash % step:
        raise ValueError("B0207政策仓位违反总和或5%步长约束")
    return {
        "policy_id": cfg["policy_id"],
        "allocation_engine": cfg["allocation_engine"],
        "runtime_role": cfg["runtime_role"],
        "weights": weights,
        "cash_pct": cash,
    }


if __name__ == "__main__":
    sample_positions = {code: 0.0 for code in CONFIG["priority"]}
    result = allocate({code: False for code in CONFIG["priority"]}, sample_positions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
