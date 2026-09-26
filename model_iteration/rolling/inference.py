"""Current frozen-champion inference, not training-ledger acceptance.

No fitting, promotion, ledger setters, historical champion substitution, or
formal publishing. Optional bounded inputs reuse existing feature mathematics
and the original position_path (including ramp and drawdown context).
"""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calendar import NAMES, build_anchors
from .market_sessions import completed_week
from .ca import build_analysis_price
from .challenge import position_path
from .features import ALL_FEATURES, build_extended_features, smooth_signal_features
from .spec import validate_spec
from scripts.investment_priority import allocate, load_config
from scripts.priority_slots import PrioritySlotError, member_weights, resolve_growth_slot

PROJECT = Path(__file__).resolve().parents[2]


class InferenceError(ValueError):
    pass


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise InferenceError(f"JSON对象无效: {path.name}")
    return value


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InferenceError("输入路径越界")
    return path


def spec_features(value) -> set[str]:
    if isinstance(value, dict):
        result = {value["feature"]} if "feature" in value else set()
        for child in value.values():
            result.update(spec_features(child))
        return result
    if isinstance(value, list):
        return set().union(*(spec_features(v) for v in value))
    return set()


def current_champion(project: Path, code: str) -> tuple[dict, dict, Path]:
    """Read the configured production workspace, never historical/verify copies."""
    model = project / "model_iteration"
    cfg = read_json(model / "configs" / f"etf_{code}.json")
    workspace = inside(model, cfg.get("workspace", f"etf_{code}"))
    state = read_json(workspace / "weekly_rolling" / "state.json")
    if str(state.get("etf")) != code:
        raise InferenceError(f"{code}当前冠军ETF身份不一致")
    champion = copy.deepcopy(state.get("champion") or {})
    spec = champion.get("spec")
    if not champion.get("id") or not isinstance(spec, dict):
        raise InferenceError(f"{code}缺少当前冻结冠军")
    if spec.get("kind") == "learning_ridge":
        raise InferenceError(f"{code}模型需要拟合，禁止在冻结推理中训练")
    errors = [] if spec.get("kind") == "champion_zero_axis" else validate_spec(spec)
    if errors:
        raise InferenceError(f"{code}冻结spec无效: {errors}")
    if "sim8_win" in spec_features(spec):
        raise InferenceError(f"{code}依赖未通过标签成熟检查的sim8_win，禁止推理")
    if cfg.get("training_disabled"):
        raise InferenceError(f"{code}训练状态仍被禁用，不能作为当前冠军推理")
    if "prediction_price_source" in cfg or "frozen_transfer" in cfg:
        raise InferenceError(f"{code}仍配置为旧迁移推理模式，必须改用自身迭代台账")
    completed = cfg.get("full_cycle_iteration")
    if completed is not None:
        if not isinstance(completed, dict) or completed.get("status") != "completed":
            raise InferenceError(f"{code}全周期迭代状态无效")
        record_ref = completed.get("iteration_record")
        if not isinstance(record_ref, str) or not record_ref:
            raise InferenceError(f"{code}缺少全周期迭代记录")
        record = read_json(inside(project, record_ref))
        if record.get("etf") != code:
            raise InferenceError(f"{code}全周期迭代记录ETF身份不一致")
        if state.get("latest_anchor_index") != completed.get("latest_anchor_index"):
            raise InferenceError(f"{code}配置与自身迭代台账锚点不一致")
        if record.get("latest_anchor_index") != state.get("latest_anchor_index"):
            raise InferenceError(f"{code}迭代记录与自身台账锚点不一致")
        if record.get("usable_anchors") != completed.get("usable_anchors"):
            raise InferenceError(f"{code}配置与全周期可用锚点数不一致")
        if champion["id"] != completed.get("final_champion_id"):
            raise InferenceError(f"{code}配置与自身迭代台账冠军不一致")
        if record.get("final_champion") != champion["id"]:
            raise InferenceError(f"{code}迭代记录与自身台账冠军不一致")
    return cfg, {"id": champion["id"], "spec": spec,
                 "trained_anchor_index": state.get("latest_anchor_index")}, workspace


def read_market(path: Path, record: dict, run_id: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"run_id": str})
    if len(frame) != int(record["rows"]) or record.get("run_id") != run_id:
        raise InferenceError(f"清单批次或行数不一致: {path.name}")
    required = ["raw_open", "raw_high", "raw_low", "raw_close",
                "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    if not {"date", "run_id", *required}.issubset(frame.columns):
        raise InferenceError(f"缺少V2日周字段: {path.name}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty or frame["date"].isna().any() or frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise InferenceError(f"日期缺失、重复或逆序: {path.name}")
    if (frame["date"].iloc[0].date().isoformat() != record["start"] or
            frame["date"].iloc[-1].date().isoformat() != record["end"]):
        raise InferenceError(f"清单日期不一致: {path.name}")
    if frame["run_id"].isna().any() or not frame["run_id"].eq(run_id).all():
        raise InferenceError(f"行情批次不一致: {path.name}")
    frame = frame.loc[frame["date"] <= cutoff].copy().reset_index(drop=True)
    if frame.empty:
        raise InferenceError(f"截止日前无行情: {path.name}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise InferenceError(f"关键行情缺失或非有限: {path.name}")
    if (frame[required[:-1]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise InferenceError(f"价格或成交量非法: {path.name}")
    for prefix in ("raw", "adj"):
        tolerance = frame[["raw_high", "adj_high"]].max(axis=1) * 1e-9
        if ((frame[f"{prefix}_high"] < frame[[f"{prefix}_open", f"{prefix}_close", f"{prefix}_low"]].max(axis=1) - tolerance).any()
                or (frame[f"{prefix}_low"] > frame[[f"{prefix}_open", f"{prefix}_close"]].min(axis=1) + tolerance).any()):
            raise InferenceError(f"OHLC关系非法: {path.name}")
    return frame


def visible_ca(path: Path, as_of: str) -> tuple[dict, list[str]]:
    ca = read_json(path)
    coverage = ca.get("coverage") or {}
    if not coverage.get("verified"):
        raise InferenceError("既有公司行动证据未核实")
    events = []
    for event in ca.get("events", []):
        if pd.Timestamp(event["ex_date"]) > pd.Timestamp(as_of):
            continue
        if not event.get("verified"):
            raise InferenceError("存在已生效但未核实的公司行动")
        if event.get("available_at") and pd.Timestamp(str(event["available_at"])[:10]) > pd.Timestamp(as_of):
            raise InferenceError("已生效公司行动在分析日尚不可得")
        events.append(event)
    ca["events"] = events
    warnings = []
    if str(coverage.get("end", "")) < as_of:
        warnings.append(f"公司行动覆盖登记截至{coverage.get('end')}；沿用已核实事件，新增期间未完成独立核实，不冒充已核实")
    return ca, warnings


def assemble(code: str, cfg: dict, ca: dict, daily: pd.DataFrame,
             weekly: pd.DataFrame, all_daily: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    da, wa = build_analysis_price(daily, ca), build_analysis_price(weekly, ca)
    anchors = build_anchors(code, da, wa, daily=daily)
    raw = build_extended_features(anchors, code, ca, frames=(da, wa), all_daily=all_daily)
    window = int((cfg.get("execution_policy") or {}).get("signal_smooth_days") or 0)
    if window > 0:
        raw = smooth_signal_features(anchors, da, wa, raw, window=window)
    mask = ~raw[ALL_FEATURES].isna().any(axis=1)
    usable, features = anchors.loc[mask].reset_index(drop=True), raw.loc[mask].reset_index(drop=True)
    if features.empty or usable["signal"].iloc[-1] != weekly["date"].iloc[-1]:
        raise InferenceError(f"{code}最新信号特征不完整，禁止退回旧信号")
    if not np.isfinite(features[ALL_FEATURES].to_numpy(dtype=float)).all():
        raise InferenceError(f"{code}模型特征含非有限值")
    if not np.allclose(features["close"], usable["anchor_close"], rtol=0, atol=1e-9):
        raise InferenceError(f"{code}特征与锚点价格错位")
    if usable["exec"].iloc[-1] is not None:
        raise InferenceError(f"{code}推理读取了信号后的执行行情")
    return usable, features


def build_reference(as_of: str | None = None, *, project: Path = PROJECT) -> dict:
    project = project.resolve()
    manifest = read_json(project / "data_manifest.json")
    if manifest.get("market_status") != "closed" or manifest.get("quality_status") not in {"passed", "passed_with_warnings"}:
        raise InferenceError("正式行情未收盘或未通过质量检查")
    run_id = manifest.get("run_id")
    if not run_id:
        raise InferenceError("行情缺少run_id")
    requested = date.fromisoformat(as_of or manifest["as_of"]).isoformat()
    if requested > datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat():
        raise InferenceError("分析日不得晚于当前日期")
    required_session = date.fromisoformat(requested)
    while required_session.weekday() >= 5:
        required_session -= timedelta(days=1)
    if required_session.isoformat() > manifest["as_of"]:
        raise InferenceError("本地正式行情未覆盖分析日，请先更新行情")
    cfg_all = load_config(project / "model_iteration" / "configs" / "investment_priority.json")
    codes = cfg_all["priority"]
    records = manifest["files"]
    frames = {}
    for code in codes:
        frames[code] = {}
        for name, chinese in (("daily", "日线"), ("weekly", "周线")):
            relative = f"数据/{code}_{NAMES[code]}_{chinese}.csv"
            matches = [r for r in records if r.get("path", "").replace("\\", "/") == relative]
            if len(matches) != 1:
                raise InferenceError(f"{code}{chinese}正式清单缺失或重复")
            frame = read_market(inside(project, relative), matches[0], run_id, pd.Timestamp(requested))
            if name == "weekly":
                complete = frame["date"].map(lambda value: completed_week(value.date(), required_session))
                frame = frame.loc[complete].reset_index(drop=True)
                if frame.empty:
                    raise InferenceError(f"{code}无完整周线")
            frames[code][name] = frame
    ends = {f["weekly"]["date"].iloc[-1].date().isoformat() for f in frames.values()}
    daily_ends = {f["daily"]["date"].iloc[-1].date().isoformat() for f in frames.values()}
    if len(ends) != 1 or len(daily_ends) != 1:
        raise InferenceError("8只ETF没有一致的最新日/周截止日")
    signal_date, daily_date = ends.pop(), daily_ends.pop()
    if requested == manifest["as_of"] and daily_date != requested:
        raise InferenceError("正式清单截止日与实际日线截止日不一致")
    for code, data in frames.items():
        completed_daily = data["daily"].loc[
            data["daily"]["date"].map(lambda value: completed_week(value.date(), required_session))
        ]
        if completed_daily.empty or completed_daily["date"].iloc[-1].date().isoformat() != signal_date:
            raise InferenceError(f"{code}周线缺少最新完整交易周")
    all_daily = {c: f["daily"].loc[f["daily"]["date"] <= signal_date].reset_index(drop=True) for c, f in frames.items()}
    positions, rows, warnings = {}, [], []
    for code in codes:
        cfg, champion, workspace = current_champion(project, code)
        ca, notes = visible_ca(workspace / "weekly_rolling" / "corporate_actions" / f"{code}_ca_events.json", signal_date)
        daily, weekly = all_daily[code], frames[code]["weekly"]
        if daily["date"].iloc[-1] != weekly["date"].iloc[-1] or not np.isclose(daily["raw_close"].iloc[-1], weekly["raw_close"].iloc[-1], rtol=0, atol=1e-9):
            raise InferenceError(f"{code}日周收盘不一致")
        usable, features = assemble(code, cfg, ca, daily, weekly, all_daily)
        used = spec_features(champion["spec"])
        if used and not np.isfinite(features[sorted(used)].to_numpy(dtype=float)).all():
            raise InferenceError(f"{code}冠军依赖的特征缺失")
        path = position_path(champion["spec"], features, len(features) - 1)
        if not np.isfinite(path).all() or ((path < 0) | (path > 1)).any():
            raise InferenceError(f"{code}推理仓位非法")
        positions[code] = float(path[-1])
        warnings.extend(f"{code}: {note}" for note in notes)
        rows.append({"code": code, "name": NAMES[code], "champion_id": champion["id"],
                     "champion_source": str((workspace / "weekly_rolling" / "state.json").relative_to(project)).replace("\\", "/"),
                     "frozen_spec": champion["spec"], "trained_anchor_index": champion["trained_anchor_index"],
                     "inference_anchor_index": len(features) - 1,
                     "position": positions[code], "bullish": positions[code] > 0,
                     "signal_as_of": signal_date,
                     "reference_raw_close": float(daily["raw_close"].iloc[-1]),
                     "model_features": {k: float(features.iloc[-1][k]) for k in sorted(used)},
                     "warnings": notes})
    try:
        growth_slot = resolve_growth_slot(
            project,
            signal_date,
            positions["159915"],
            available_as_of=requested,
        )
    except PrioritySlotError as exc:
        raise InferenceError(str(exc)) from exc
    allocation_positions = dict(positions)
    allocation_positions["159915"] = float(growth_slot["slot_position"])
    allocation = allocate(
        {c: allocation_positions[c] > 0 for c in codes}, allocation_positions, config=cfg_all
    )
    growth_slot["slot_weight_pct"] = allocation["weights"]["159915"]
    growth_slot["member_weights_pct"] = member_weights(
        growth_slot, allocation["weights"]["159915"]
    )
    return {"schema_version": "monitoring-baseline-v2", "runtime_role": "mechanical_reference_only",
            "requested_as_of": requested, "data_as_of": daily_date, "as_of": signal_date,
            "data_run_id": run_id, "fixed_priority_order": codes,
            "model_mode": "current_frozen_champion_inference", "historical_performance_evidence": False,
            "point_in_time_note": "使用当前冻结模型及当前批次中截止日前行情；不代表历史当时模型或历史可得版本的回测",
            "formal_publication_ready": False,
            "quality_status": "reference_with_warnings" if warnings else "reference_ready",
            "champion_ids": {r["code"]: r["champion_id"] for r in rows},
            "champion_positions": positions, "priority_slot_analysis": {"159915": growth_slot},
            "b0207_weights_pct": allocation["weights"],
            "b0207_cash_pct": allocation["cash_pct"], "instruments": rows, "warnings": warnings}
