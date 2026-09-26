# -*- coding: utf-8 -*-
"""只读日周变化证据；不运行冠军、回测、参数产物生成器或任何发布流程。"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS = {"daily": ("日线", 5, 60, 252), "weekly": ("周线", 4, 52, 52)}
POINT_FIELDS = (
    "raw_close", "adj_close", "adj_low", "adj_high", "dif", "dea", "macd_hist",
    "dif_first_raw", "dif_second_raw", "sma5", "sma10", "sma20", "sma60",
    "rsi14", "volume", "volume_to_20bar_median", "return_1bar", "distance_ma20",
    "prior_20bar_low", "prior_20bar_high",
)


class ForwardEvidenceError(ValueError):
    """无法建立与分析截止日一致的连续观察证据。"""


def _feature_math():
    # 复用现有纯数学函数，避免另造一套MACD/离散导数。绝不调用calculate()。
    scripts_path = str(PROJECT_ROOT / "scripts")
    sys.path.insert(0, scripts_path)
    try:
        return importlib.import_module("calculate_features")._feature_frame
    finally:
        sys.path.remove(scripts_path)


def _points(frame: pd.DataFrame, count: int, mad: int, annualization: int) -> tuple[list[dict], dict]:
    values = _feature_math()(frame, mad, annualization)
    median = values["volume"].rolling(20, min_periods=20).median()
    values["volume_to_20bar_median"] = values["volume"] / median.replace(0, float("nan"))
    values["return_1bar"] = values["adj_close"].pct_change(fill_method=None)
    values["distance_ma20"] = values["adj_close"] / values["sma20"] - 1
    # 是前20根K线的区间极值，不冒充人工识别的摆动前低/前高。
    values["prior_20bar_low"] = values["adj_low"].shift(1).rolling(20).min()
    values["prior_20bar_high"] = values["adj_high"].shift(1).rolling(20).max()
    points = []
    for _, row in values.tail(count).iterrows():
        point = {"date": row["date"].date().isoformat()}
        for key in POINT_FIELDS:
            number = float(row[key])
            if not math.isfinite(number):
                raise ForwardEvidenceError(f"连续观察证据缺少有效值：{point['date']}.{key}")
            point[key] = round(number, 10)
        points.append(point)
    if len(points) != count:
        raise ForwardEvidenceError("连续观察期数不足")
    return points, values.iloc[-1].to_dict()


def load_forward_evidence(
    root: Path, *, data_as_of: str, data_run_id: str, instruments: Mapping[str, dict],
) -> dict[str, dict]:
    """截断后才计算；仅返回最近5日/4完整周，不写任何文件。"""
    try:
        cutoff = pd.Timestamp(date.fromisoformat(data_as_of))
        manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ForwardEvidenceError("无法读取连续观察证据的数据清单或截止日") from exc
    if not data_run_id or manifest.get("run_id") != data_run_id or manifest.get("as_of") != data_as_of:
        raise ForwardEvidenceError("连续观察证据必须使用相同run_id和共同截止日")
    if manifest.get("market_status") != "closed" or manifest.get("quality_status") not in {
        "passed", "passed_with_warnings",
    }:
        raise ForwardEvidenceError("连续观察证据需要已收盘且质量通过的数据")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ForwardEvidenceError("行情清单files必须为数组")
    result = {}
    for code, feature_item in instruments.items():
        evidence = {"data_as_of": data_as_of, "data_run_id": data_run_id,
                    "price_basis": "adjusted_indicators_raw_execution"}
        for timeframe, (suffix, count, mad, annualization) in WINDOWS.items():
            feature = feature_item.get(timeframe) or {}
            if feature.get("is_partial") is not False:
                raise ForwardEvidenceError(f"{code}.{timeframe}未确认为完整K线，不能冒充完整周期证据")
            try:
                last_date = pd.Timestamp(date.fromisoformat(feature["as_of"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ForwardEvidenceError(f"{code}.{timeframe}缺少有效截止日") from exc
            if last_date > cutoff or (timeframe == "daily" and last_date != cutoff):
                raise ForwardEvidenceError(f"{code}.{timeframe}截止日与分析信息边界不一致")
            matches = [record for record in records if isinstance(record, dict)
                       and Path(str(record.get("path", ""))).name.startswith(f"{code}_")
                       and str(record.get("path", "")).endswith(f"_{suffix}.csv")]
            if len(matches) != 1 or matches[0].get("run_id") != data_run_id:
                raise ForwardEvidenceError(f"{code}.{timeframe}正式来源缺失、重复或批次不一致")
            path = (root / matches[0]["path"]).resolve()
            if path.parent != (root / "数据").resolve():
                raise ForwardEvidenceError("连续观察证据路径越界")
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame["date"] = pd.to_datetime(frame["date"], errors="raise")
                # 先排除未来，再校验与计算；禁止将全样本最新截面回填历史。
                frame = frame.loc[frame["date"] <= cutoff].copy()
                if frame.empty or frame["date"].duplicated().any() or frame["date"].max() != last_date:
                    raise ForwardEvidenceError(f"{code}.{timeframe}日期缺失或重复")
                if set(frame["run_id"].astype(str)) != {data_run_id}:
                    raise ForwardEvidenceError(f"{code}.{timeframe}CSV批次不一致")
                for key in [f"{basis}_{field}" for basis in ("raw", "adj")
                            for field in ("open", "high", "low", "close")] + ["volume"]:
                    frame[key] = pd.to_numeric(frame[key], errors="raise")
                    if not frame[key].map(math.isfinite).all() or (frame[key] < 0).any():
                        raise ForwardEvidenceError(f"{code}.{timeframe}.{key}存在无效值")
                for basis in ("raw", "adj"):
                    low, high = frame[f"{basis}_low"], frame[f"{basis}_high"]
                    middle = frame[[f"{basis}_open", f"{basis}_close"]]
                    # 只容忍复权浮点表示的尾差，远小于ETF最小价格单位；不修改原值。
                    tolerance = 1e-10
                    if ((low <= 0).any() or (low - middle.min(axis=1) > tolerance).any()
                            or (middle.max(axis=1) - high > tolerance).any()):
                        raise ForwardEvidenceError(f"{code}.{timeframe}OHLC关系异常")
                points, last = _points(frame.sort_values("date").reset_index(drop=True), count, mad, annualization)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                if isinstance(exc, ForwardEvidenceError):
                    raise
                raise ForwardEvidenceError(f"{code}.{timeframe}无法构建连续观察证据：{exc}") from exc
            for key in ("adj_close", "sma5", "sma10", "sma20", "sma60", "rsi14", "dif", "dea",
                        "macd_hist", "dif_first_raw", "dif_second_raw",
                        "dif_first_normalized", "dif_second_normalized"):
                expected = feature.get(key)
                if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected):
                    raise ForwardEvidenceError(f"{code}.{timeframe}.{key}客观快照缺失")
                if abs(float(last[key]) - expected) > 1e-9:
                    raise ForwardEvidenceError(f"{code}.{timeframe}.{key}连续观察末值与客观快照不一致")
            evidence[timeframe] = points
        result[code] = evidence
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="只读输出8只ETF连续日周证据，不推理或重算B0207")
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    try:
        features = json.loads((PROJECT_ROOT / "app/features/latest.json").read_text(encoding="utf-8"))
        from scripts.investment_priority import CONFIG
        if features.get("as_of") != args.as_of or set(features["instruments"]) != set(CONFIG["priority"]):
            raise ForwardEvidenceError("客观参数快照日期或ETF范围与请求不一致")
        evidence = load_forward_evidence(PROJECT_ROOT, data_as_of=args.as_of,
                                         data_run_id=features["data_run_id"],
                                         instruments={code: features["instruments"][code] for code in CONFIG["priority"]})
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0
    except (ForwardEvidenceError, OSError, KeyError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
