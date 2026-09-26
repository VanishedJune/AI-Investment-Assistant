# -*- coding: utf-8 -*-
"""每周正式预测生成（PIT，基于 analysis_price 特征）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import compute_position, score_from_spec

STRONG_TH = {1: 0.02, 2: 0.04, 4: 0.06, 8: 0.10}


def state_bucket(row: pd.Series) -> tuple[int, int]:
    return (1 if row["w_slope"] > 0 else 0, 1 if row["d_slope"] > 0 else 0)


def compute_bucket_stats(
    features: pd.DataFrame, fwd_arrays: dict[int, np.ndarray], up_to: int
) -> dict:
    """PIT：只统计已成熟窗口（j + k <= up_to）的同状态桶均值/胜率/强涨跌频率。"""
    out: dict[tuple, dict] = {}
    for j in range(up_to):
        row = features.iloc[j]
        bucket = state_bucket(row)
        for k, fwd in fwd_arrays.items():
            if j + k > up_to:  # 窗口尚未成熟：跳过，禁止未来信息
                continue
            val = fwd[j]
            if np.isnan(val):
                continue
            stats = out.setdefault(
                (bucket, k),
                {"mean": 0.0, "n": 0, "n_up": 0, "strong_up": 0, "strong_down": 0, "sum": 0.0},
            )
            stats["n"] += 1
            stats["sum"] += float(val)
            if val > 0:
                stats["n_up"] += 1
            if val >= STRONG_TH[k]:
                stats["strong_up"] += 1
            if val <= -STRONG_TH[k]:
                stats["strong_down"] += 1
    for stats in out.values():
        stats["mean"] = stats["sum"] / stats["n"] if stats["n"] else 0.0
    return out


def champion_position(row: pd.Series) -> float:
    return 1.0 if (row["w_slope"] > 0 and row["d_slope"] > 0) else 0.0


def build_forecast(
    model_spec: dict,
    row: pd.Series,
    i: int,
    fwd_arrays: dict[int, np.ndarray],
    bucket_stats: dict,
    features: pd.DataFrame | None = None,
) -> dict:
    """生成并冻结一份预测。bucket_stats[(bucket,k)] = (mean, n_up, n, strong_up, strong_down)。"""
    bucket = state_bucket(row)
    if model_spec.get("kind") == "learning_ridge":
        if features is None:
            raise ValueError("learning_ridge 需要 features 帧用于训练")
        from .learning import walkforward_position

        pos, _ = walkforward_position(features, fwd_arrays, i, model_spec)
    elif model_spec.get("kind") == "champion_zero_axis":
        pos = champion_position(row)
    else:
        pos = compute_position(model_spec, row, {"position_history": [], "equity_history": []})
    spec_driven = model_spec.get("kind") not in ("champion_zero_axis", "learning_ridge")
    score = score_from_spec(row, model_spec) if spec_driven else 0.0
    directions: dict[int, int] = {}
    expected: dict[int, float] = {}
    p_up: dict[int, float] = {}
    p_su: dict[int, float] = {}
    p_sd: dict[int, float] = {}
    for k in (1, 2, 4, 8):
        stats = bucket_stats.get((bucket, k))
        if spec_driven:
            # spec 自生成概率：score -> p_up（PIT 确定性映射），期望/方向随之变化，
            # 使 MAE/Brier/准确率不再与 Champion 相同，预测类门禁可被 spec 实质改善。
            scale = {1: 15.0, 2: 20.0, 4: 25.0, 8: 30.0}[k]
            p_up[k] = round(float(np.clip(0.5 + 0.5 * np.tanh(score / scale), 0.05, 0.95)), 4)
            if stats and stats["n"] >= 10:
                prior_abs = abs(float(stats["mean"]))
                p_su[k] = round(float(stats["strong_up"] / stats["n"]), 4)
                p_sd[k] = round(float(stats["strong_down"] / stats["n"]), 4)
            else:
                prior_abs = {1: 0.02, 2: 0.03, 4: 0.05, 8: 0.08}[k]
                p_su[k] = 0.5
                p_sd[k] = 0.5
            expected[k] = round((2.0 * p_up[k] - 1.0) * prior_abs, 6)
            directions[k] = 1 if expected[k] > 0 else -1 if expected[k] < 0 else 0
        else:
            if stats and stats["n"] >= 10:
                mean = stats["mean"]
                expected[k] = round(float(mean), 6)
                p_up[k] = round(float(stats["n_up"] / stats["n"]), 4)
                p_su[k] = round(float(stats["strong_up"] / stats["n"]), 4)
                p_sd[k] = round(float(stats["strong_down"] / stats["n"]), 4)
                directions[k] = 1 if mean > 0 else -1 if mean < 0 else 0
            else:
                expected[k] = 0.0
                p_up[k] = 0.5
                p_su[k] = 0.5
                p_sd[k] = 0.5
                directions[k] = 0
    directions[1] = 1 if pos > 0 else 0

    state = (
        "日强周强" if row["w_slope"] > 0 and row["d_slope"] > 0
        else "日弱周弱" if row["w_slope"] < 0 and row["d_slope"] < 0
        else "日强周弱" if row["d_slope"] > 0
        else "周强日弱" if row["w_slope"] > 0
        else "震荡"
    )
    vol_pct = float(row["vol_pct"]) if not np.isnan(row["vol_pct"]) else 0.5
    risk = "低" if vol_pct < 0.3 else "高" if vol_pct > 0.7 else "中"
    support = max(float(row["d_low20"]), float(row["w_sma20"]) if not np.isnan(row["w_sma20"]) else -1e9)
    pressure = min(float(row["d_high20"]), float(row["w_sma20"]) if not np.isnan(row["w_sma20"]) else 1e9)
    invalid = float(row["w_low4"]) if not np.isnan(row["w_low4"]) else float(row["d_low20"])
    if pos > 0:
        advice = "回踩支撑企稳后建仓/持有"
    else:
        advice = "等待日周同向确认"
    return {
        "model_id": model_spec.get("id", model_spec.get("kind", "model")),
        "direction_1w": directions[1],
        "direction_2w": directions[2],
        "direction_4w": directions[4],
        "direction_8w": directions[8],
        "expected_return_1w": expected[1],
        "expected_return_2w": expected[2],
        "expected_return_4w": expected[4],
        "expected_return_8w": expected[8],
        "p_up_1w": p_up[1],
        "p_up_2w": p_up[2],
        "p_up_4w": p_up[4],
        "p_up_8w": p_up[8],
        "p_strong_up_1w": p_su[1],
        "p_strong_up_2w": p_su[2],
        "p_strong_up_4w": p_su[4],
        "p_strong_up_8w": p_su[8],
        "p_strong_down_1w": p_sd[1],
        "p_strong_down_2w": p_sd[2],
        "p_strong_down_4w": p_sd[4],
        "p_strong_down_8w": p_sd[8],
        "position": round(pos, 4),
        "daily_read": f"日斜率{'正' if row['d_slope']>0 else '负'}(DIF1={row['d_dif1']:.4f},DIF2={row['d_dif2']:.4f},RSI={row['rsi14']:.1f})",
        "weekly_read": f"周斜率{'正' if row['w_slope']>0 else '负'}(W_DIF1={row['w_dif1']:.4f})",
        "state": state,
        "risk": risk,
        "support": round(support, 4),
        "pressure": round(pressure, 4),
        "invalid": round(invalid, 4),
        "advice": advice,
        "basis": "spec score→概率映射" if spec_driven else "状态桶PIT基率+日周双周期协议",
    }
