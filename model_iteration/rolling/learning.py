# -*- coding: utf-8 -*-
"""学轨模板：Ridge 滚动学习 Challenger（动态权重、冻结超参）。

泄漏口（全部封死）：
1) 训练样本只取 s + embargo <= t（embargo >= horizon，默认 8）的成熟样本；
2) 标准化（mean/std）只在训练集内计算；
3) 超参数（alpha/embargo/min_train/特征白名单/仓位映射）随 Proposal 冻结，
   禁止用 OOS 调参；改进只能作为新一代学习 Challenger 重新注册；
4) 预测在 t 冻结、t+h 成熟评价，复用现有成熟循环。

spec 形态（随 Proposal 冻结）：
{
  "kind": "learning_ridge",
  "alpha": 10.0,
  "horizon": 8,
  "embargo": 8,
  "min_train": 60,
  "standardize": true,
  "features": ["d_slope","w_slope","d_dif1","w_dif1","mom20","vol20"],
  "position": {"type": "step", "weight": 1.0}   // 或 linear: {slope, intercept, min, max}
}
"""

from __future__ import annotations

import numpy as np

from .features import ALL_FEATURES


def validate_learning_spec(spec: dict) -> list[str]:
    errors = []
    if spec.get("kind") != "learning_ridge":
        errors.append("kind 必须为 learning_ridge")
    horizon = int(spec.get("horizon", 8))
    embargo = int(spec.get("embargo", horizon))
    if embargo < horizon:
        errors.append(f"embargo({embargo}) 必须 >= horizon({horizon})")
    if float(spec.get("alpha", 0)) <= 0:
        errors.append("alpha 必须 > 0")
    features = spec.get("features") or []
    if not features:
        errors.append("features 白名单为空")
    for f in features:
        if f not in ALL_FEATURES:
            errors.append(f"未知特征: {f}")
    if int(spec.get("min_train", 60)) < 1:
        errors.append("min_train 必须 >= 1")
    return errors


def _ridge_weights(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """闭式 Ridge（含截距列）：w = (X'X + alpha*I)^{-1} X'y。"""
    p = X.shape[1]
    a = X.T @ X + alpha * np.eye(p)
    b = X.T @ y
    return np.linalg.solve(a, b)


def walkforward_position(
    features,
    fwd_arrays: dict[int, np.ndarray],
    up_to: int,
    spec: dict,
) -> tuple[float, dict]:
    """在 anchor up_to 用 <= up_to-embargo 的成熟样本训练 Ridge，输出目标仓位。"""
    errors = validate_learning_spec(spec)
    if errors:
        raise ValueError("learning spec 非法: " + "; ".join(errors))
    horizon = int(spec["horizon"])
    embargo = int(spec["embargo"])
    alpha = float(spec["alpha"])
    min_train = int(spec.get("min_train", 60))
    standardize = bool(spec.get("standardize", True))
    whitelist = list(spec["features"])
    fwd = fwd_arrays[horizon]

    train_idx = [
        j for j in range(up_to)
        if j + embargo <= up_to and not np.isnan(fwd[j])
    ]
    if len(train_idx) < min_train:
        return 0.0, {"n_train": len(train_idx), "position": 0.0, "reason": "min_train"}
    X_raw = features[whitelist].iloc[train_idx].to_numpy(dtype=float)
    y = fwd[np.asarray(train_idx, dtype=int)]
    keep = ~np.isnan(X_raw).any(axis=1) & ~np.isnan(y)
    X_raw, y = X_raw[keep], y[keep]
    if len(X_raw) < min_train:
        return 0.0, {"n_train": len(X_raw), "position": 0.0, "reason": "min_train_after_nan"}
    if standardize:
        mean = X_raw.mean(axis=0)
        std = X_raw.std(axis=0)
        std[std == 0.0] = 1.0
        X = (X_raw - mean) / std
    else:
        mean, std = None, None
        X = X_raw
    design = np.column_stack([np.ones(len(X)), X])
    w = _ridge_weights(design, y, alpha)

    row = features[whitelist].iloc[up_to].to_numpy(dtype=float)
    if np.isnan(row).any():
        return 0.0, {"n_train": len(X), "position": 0.0, "reason": "feature_nan"}
    if standardize:
        row = (row - mean) / std
    pred = float(w[0] + w[1:] @ row)
    pos_cfg = spec.get("position", {"type": "step", "weight": 1.0})
    if pos_cfg.get("type") == "linear":
        pos = float(pos_cfg.get("slope", 1.0)) * pred + float(pos_cfg.get("intercept", 0.0))
        pos = float(np.clip(pos, pos_cfg.get("min", 0.0), pos_cfg.get("max", 1.0)))
    else:
        pos = float(pos_cfg.get("weight", 1.0)) if pred > 0 else 0.0
    return pos, {"n_train": len(X), "pred": round(pred, 6), "position": round(pos, 4)}
