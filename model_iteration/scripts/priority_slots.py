# -*- coding: utf-8 -*-
"""固定优先级内部的可替代交易标的解析。

第一槽位仍只占B0207的一份预算；021528不会成为第9个并行权重。
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path


class PrioritySlotError(RuntimeError):
    pass


def choose_growth_member(etf_position: float, fund_position: float) -> tuple[str | None, float]:
    """创业板优先；只有创业板为空时，基金自身看多才承接。"""
    if etf_position > 0:
        return "159915", float(etf_position)
    if fund_position > 0:
        return "021528", float(fund_position)
    return None, 0.0


def resolve_growth_slot(project: Path, signal_as_of: str, etf_position: float,
                        *, available_as_of: str | None = None) -> dict:
    config_path = project / "model_iteration" / "configs" / "investment_priority.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slot = ((config.get("priority_slots") or {}).get("159915") or {})
    if not slot:
        return {
            "slot_id": "159915",
            "allocation_key": "159915",
            "selected_member": "159915" if etf_position > 0 else None,
            "slot_position": float(etf_position),
            "member_positions": {"159915": float(etf_position)},
        }

    state_path = project / "model_iteration" / "fund_021528" / "weekly_rolling" / "state.json"
    analysis_path = project / "app" / "features" / "fund_021528_latest.json"
    if not state_path.is_file():
        raise PrioritySlotError("021528冻结冠军状态缺失，无法进行当前净值推理")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    try:
        signal_day = date.fromisoformat(signal_as_of)
        available_day = date.fromisoformat(available_as_of) if available_as_of else None
    except ValueError as exc:
        raise PrioritySlotError("021528信号日或可用数据日格式非法") from exc

    champion = state.get("champion") or {}
    champion_spec = champion.get("spec")
    if not champion.get("id") or not isinstance(champion_spec, dict):
        raise PrioritySlotError("021528冻结冠军缺少id或spec，拒绝推理")
    try:
        from rolling.spec import compute_position
        from scripts.fund_nav_replay import (
            build_frames,
            latest_analysis,
            load_nav,
            round_numbers,
            used_features,
        )

        nav = load_nav()
        if available_day is not None:
            nav = nav[nav["date"] <= str(available_day)].reset_index(drop=True)
        if nav.empty:
            raise PrioritySlotError("021528在报告可用日期内没有净值")
        nav_day = nav.iloc[-1]["date"].date()
        if nav_day < signal_day:
            raise PrioritySlotError("021528最新净值早于本次完整周信号，不能作为当前观察")
        daily_frame, weekly_frame, anchors, features = build_frames(nav)
        eligible = anchors.index[anchors["signal"] <= str(signal_day)].tolist()
        if not eligible:
            raise PrioritySlotError("021528没有不晚于本次信号日的完整周特征")
        feature_index = eligible[-1]
        feature_row = features.iloc[feature_index]
        required = used_features(champion_spec)
        missing = sorted(name for name in required if name not in feature_row.index)
        invalid = sorted(
            name for name in required
            if name in feature_row.index and not math.isfinite(float(feature_row[name]))
        )
        if missing or invalid:
            raise PrioritySlotError(
                f"021528冻结冠军输入不兼容：missing={missing}, invalid={invalid}"
            )
        fund_position = float(compute_position(champion_spec, feature_row, {}))
        inference_week = anchors.iloc[feature_index]["signal"].date().isoformat()
        current_analysis = round_numbers(
            latest_analysis(
                nav,
                daily_frame,
                weekly_frame,
                {"id": champion["id"], "latest_position": fund_position},
            )
        )
    except PrioritySlotError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PrioritySlotError(f"021528冻结冠军当前净值推理失败：{exc}") from exc

    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.is_file() else None
    analysis_week = str((analysis or {}).get("completed_week_as_of") or "")
    replay_week = str(state.get("completed_week_as_of") or "")
    warnings: list[str] = []
    if analysis_week and analysis_week != inference_week:
        warnings.append(
            f"历史派生分析截止{analysis_week}，当前冻结冠军已直接推理至{inference_week}；历史文件不参与当前信号"
        )
    selected, slot_position = choose_growth_member(etf_position, fund_position)
    return {
        "slot_id": slot["slot_id"],
        "display_name": slot["display_name"],
        "allocation_key": "159915",
        "members": list(slot["members"]),
        "selected_member": selected,
        "slot_position": slot_position,
        "member_positions": {"159915": float(etf_position), "021528": fund_position},
        "selection_rule": slot["selection_rule"],
        "allocation_signal_as_of": signal_as_of,
        "fund_model_training_data_end": str((state.get("coverage") or {}).get("end") or ""),
        "fund_latest_replay_week": replay_week,
        "fund_current_nav_as_of": nav_day.isoformat(),
        "fund_current_inference_week": inference_week,
        "fund_report_generated_on": available_as_of,
        "fund_inference_mode": "frozen_champion_current_nav_no_training_no_replay",
        "fund_analysis": current_analysis,
        "fund_historical_analysis_path": analysis_path.relative_to(project).as_posix() if analysis_path.is_file() else None,
        "fund_historical_analysis": analysis,
        "warnings": warnings,
    }


def member_weights(slot: dict, slot_weight_pct: float) -> dict[str, float]:
    selected = slot.get("selected_member")
    return {
        "159915": float(slot_weight_pct) if selected == "159915" else 0.0,
        "021528": float(slot_weight_pct) if selected == "021528" else 0.0,
    }
