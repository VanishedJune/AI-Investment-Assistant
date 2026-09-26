# -*- coding: utf-8 -*-
"""Champion / Challenger 信号与 PIT 模型预测（纯 numpy，无新依赖）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def champion_position(feat: pd.DataFrame) -> np.ndarray:
    """周 slope>0 且 日 slope>0 → 100%，否则现金。"""
    return ((feat["w_slope"] > 0) & (feat["d_slope"] > 0)).astype(float).to_numpy()


def ch_a_position(feat: pd.DataFrame) -> np.ndarray:
    """CH_A：双正 100%；日强周弱恢复（周<0 且 日>0 且 日DIF1>0、DIF2≥0）50%；其余现金。"""
    pos = np.zeros(len(feat))
    both = (feat["w_slope"] > 0) & (feat["d_slope"] > 0)
    recovery = (
        (feat["w_slope"] < 0)
        & (feat["d_slope"] > 0)
        & (feat["d_dif1"] > 0)
        & (feat["d_dif2"] >= 0)
    )
    pos[both.to_numpy()] = 1.0
    pos[recovery.to_numpy()] = 0.5
    return pos


def ridge_predict_walkforward(
    feat: pd.DataFrame,
    labels: pd.Series,
    alpha: float = 10.0,
    embargo: int = 8,
    min_train: int = 40,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Expanding-window PIT Ridge：训练样本 s 满足 s + embargo < t（标签窗口不重叠）。"""
    columns = feature_columns or FEATURE_COLUMNS
    x = feat[columns].to_numpy(dtype=float)
    y = labels.to_numpy(dtype=float)
    n = len(feat)
    preds = np.full(n, np.nan)
    indices = np.arange(n)
    for t in range(n):
        train_mask = (indices < t - embargo) & ~np.isnan(y)
        if int(train_mask.sum()) < min_train:
            continue
        x_tr = x[train_mask]
        y_tr = y[train_mask]
        mean_x = x_tr.mean(axis=0)
        std_x = x_tr.std(axis=0)
        std_x[std_x == 0.0] = 1.0
        xs = (x_tr - mean_x) / std_x
        y_mean = y_tr.mean()
        a = xs.T @ xs + alpha * np.eye(xs.shape[1])
        b = xs.T @ (y_tr - y_mean)
        w = np.linalg.solve(a, b)
        xt = (x[t] - mean_x) / std_x
        preds[t] = y_mean + float(xt @ w)
    return preds


def ch_b_position(preds: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(preds))
    pos[preds > 0] = 1.0
    return pos
