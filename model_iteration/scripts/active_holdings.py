"""Read-only view of actual holdings after an analysis-slot replacement."""

from __future__ import annotations

from copy import deepcopy
import math


class ActiveHoldingsError(ValueError):
    pass


REPLACED_ZERO_WEIGHT_SLOTS = {"159941": "159611"}


def active_portfolio_view(portfolio: dict, order) -> dict:
    """Map a retired zero-weight slot to its active code without editing the ledger.

    A non-zero retired holding is never transferred implicitly because replacing
    an analysis slot is not authorization for an actual trade.
    """
    view = deepcopy(portfolio)
    weights = dict(view.get("weights_pct") or {})
    active = set(order)
    for retired, replacement in REPLACED_ZERO_WEIGHT_SLOTS.items():
        if replacement not in active or replacement in weights or retired not in weights:
            continue
        retired_weight = weights[retired]
        if retired_weight != 0:
            raise ActiveHoldingsError(
                f"退役槽位{retired}仍有非零真实持仓，禁止自动转记到{replacement}"
            )
        weights.pop(retired)
        weights[replacement] = 0.0
    if set(weights) != active:
        raise ActiveHoldingsError("真实持仓不能映射为当前8标的视图")
    view["weights_pct"] = {code: weights[code] for code in order}
    allocation = [*weights.values(), view.get('cash_pct')]
    for value in allocation:
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(value) or not 0 <= value <= 100):
            raise ActiveHoldingsError('真实持仓比例必须是有限的0至100数值或明确缺失')
    if all(value is not None for value in allocation) and abs(sum(allocation) - 100) > 1e-8:
        raise ActiveHoldingsError('真实持仓与现金比例合计不是100%')
    members = view.get("growth_slot_member_weights_pct")
    if members is not None:
        if (set(members) != {"159915", "021528"}
                or any(type(value) not in (int, float) or not 0 <= value <= 100
                       for value in members.values())
                or abs(sum(members.values()) - view["weights_pct"]["159915"]) > 1e-8):
            raise ActiveHoldingsError("第一优先级真实持仓成员与共享槽位总仓位不一致")
    physical = dict(view['weights_pct'])
    if members is not None:
        physical.update(members)
    actual = view.get('actual_instrument_weights_pct')
    if actual is not None and actual != physical:
        raise ActiveHoldingsError('实际证券比例与共享槽位成员比例不一致')
    for code, position in (view.get('positions') or {}).items():
        shares = position.get('shares')
        if shares is not None and (type(shares) not in (int, float)
                                  or not math.isfinite(shares) or shares < 0):
            raise ActiveHoldingsError(code + '持仓份数非法')
        weight = position.get('weight_pct')
        if weight is not None and code in physical and physical[code] is not None and weight != physical[code]:
            raise ActiveHoldingsError(code + '持仓明细比例与账户比例冲突')
    values = view.get('market_values_cny')
    cash, total = view.get('cash_amount_cny'), view.get('total_market_value_cny')
    amounts = [*(values or {}).values(), cash, total]
    if any(value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or value < 0) for value in amounts):
        raise ActiveHoldingsError('持仓市值、现金或总资产金额非法')
    held = set(view.get('held_codes') or []) | {c for c, w in physical.items() if w is not None and w > 0}
    if (values is not None and held.issubset(values) and cash is not None and total is not None
            and all(v is not None for v in values.values())
            and abs(sum(values.values()) + cash - total) > 0.011):
        raise ActiveHoldingsError('持仓市值加现金与总资产不一致')
    view["analysis_slot_view_only"] = True
    view["actual_ledger_unchanged"] = True
    return view
