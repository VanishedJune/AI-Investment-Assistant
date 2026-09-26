# -*- coding: utf-8 -*-
"""成熟评价：1/2/4/8W 冻结预测 vs 真实结果。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import STRONG_TH


def window_maxdd(analysis_daily: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    win = analysis_daily[(analysis_daily["date"] > start_date) & (analysis_daily["date"] <= end_date)]
    if win.empty:
        return 0.0
    closes = win["analysis_close"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(closes)
    return float(np.nanmin(closes / peak - 1.0))


def build_evaluation(
    anchor_idx: int,
    horizon: int,
    model_id: str,
    forecast: dict,
    actual: float,
    sub_actual: float | None,
    row: pd.Series,
    analysis_daily: pd.DataFrame,
    anchors: pd.DataFrame,
) -> dict:
    if np.isnan(actual):
        return {"anchor_index": anchor_idx, "horizon": horizon, "model_id": model_id, "pending": True}
    direction = forecast.get(f"direction_{horizon}w", 0)
    expected = forecast.get(f"expected_return_{horizon}w", 0.0)
    p_up = forecast.get(f"p_up_{horizon}w", 0.5)
    p_su = forecast.get(f"p_strong_up_{horizon}w", 0.5)
    p_sd = forecast.get(f"p_strong_down_{horizon}w", 0.5)
    pos = float(forecast.get("position", 0.0))
    strong_th = STRONG_TH[horizon]

    direction_correct = None if direction == 0 else (np.sign(actual) == direction)
    conflict = (row["w_slope"] > 0) != (row["d_slope"] > 0) and row["w_slope"] != 0 and row["d_slope"] != 0
    conflict_correct = None if not conflict else (np.sign(actual) == direction if direction != 0 else None)
    mae = float(abs(expected - actual))
    # 核心 Brier 仅用 p_up；强涨/强跌 Brier 只作诊断输出（v2.0.0）
    brier = float((p_up - (1.0 if actual > 0 else 0.0)) ** 2)
    brier_strong = float(
        ((p_su - (1.0 if actual >= strong_th else 0.0)) ** 2
         + (p_sd - (1.0 if actual <= -strong_th else 0.0)) ** 2)
        / 2.0
    )
    start = anchors.iloc[anchor_idx]["signal"]
    end = anchors.iloc[min(anchor_idx + horizon, len(anchors) - 1)]["signal"]
    mdd = window_maxdd(analysis_daily, start, end)
    advice_valid = (pos > 0 and actual > 0) or (pos == 0 and actual <= 0)
    missed_rally = pos == 0 and actual >= strong_th
    wrong_chase = pos > 0 and actual <= -strong_th
    over_cash = pos == 0 and actual > 0
    early_entry = pos > 0 and sub_actual is not None and sub_actual < 0 and actual > 0
    late_entry = pos == 0 and actual >= strong_th
    return {
        "anchor_index": anchor_idx,
        "horizon": horizon,
        "model_id": model_id,
        "pending": False,
        "actual_return": round(float(actual), 6),
        "direction_correct": None if direction_correct is None else bool(direction_correct),
        "mae": round(mae, 6),
        "brier": round(brier, 6),
        "brier_strong": round(brier_strong, 6),
        "window_maxdd": round(mdd, 6),
        "advice_valid": bool(advice_valid),
        "missed_rally": bool(missed_rally),
        "wrong_chase": bool(wrong_chase),
        "over_cash": bool(over_cash),
        "early_entry": bool(early_entry),
        "late_entry": bool(late_entry),
        "conflict_correct": None if conflict_correct is None else bool(conflict_correct),
        "position": round(pos, 4),
    }
