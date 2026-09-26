# -*- coding: utf-8 -*-
"""数据装配契约（方案修复基线第 0 节）：
features = feat_raw[mask].reset_index(drop=True) 是唯一合法方式；
启动断言：行数==712、末锚 pending、逐行 close 对齐、特征无 NaN，任一不满足 fail-closed。
"""

from __future__ import annotations

import numpy as np

from .ca import load_ca, validate_coverage
from .calendar import build_anchors, first_last, load_daily
from .config import load_etf_config
from .features import (
    ALL_FEATURES,
    analysis_frames,
    build_extended_features,
    smooth_signal_features,
)


class DataAssemblyError(Exception):
    """数据装配 fail-closed：错位/行数不符/末锚非 pending/特征 NaN。"""


def load_aligned_data(etf: str = "159915"):
    """构建并校验 anchors/features 对齐；返回 (daily_a, weekly_a, daily, anchors, usable, features)。"""
    # Fixed-count acceptance belongs to training/formal snapshots. Frozen
    # report-date inference uses rolling.inference and leaves this guard intact.
    cfg = load_etf_config(etf)
    required_start = cfg["coverage"].get("start")
    required_end = cfg["coverage"].get("end")
    expected_anchors = cfg.get("expected_anchors")
    try:
        ca = load_ca(etf)
        if required_start and required_end:
            validate_coverage(ca, required_start, required_end)
    except Exception as exc:  # noqa: BLE001
        raise DataAssemblyError(f"CA 校验失败: {exc}") from exc

    daily_a, weekly_a = analysis_frames(etf, ca)
    daily = load_daily(etf)
    anchors = build_anchors(etf, daily_a, weekly_a)
    feat_raw = build_extended_features(anchors, etf, ca)
    smooth_days = int((cfg.get("execution_policy") or {}).get("signal_smooth_days") or 0)
    if smooth_days > 0:
        feat_raw = smooth_signal_features(
            anchors, daily_a, weekly_a, feat_raw, window=smooth_days
        )
    mask = ~feat_raw[ALL_FEATURES].isna().any(axis=1)
    usable = anchors[mask].reset_index(drop=True)
    features = feat_raw[mask].reset_index(drop=True)

    if len(features) != len(usable):
        raise DataAssemblyError(
            f"行数不符: features={len(features)} usable={len(usable)}"
        )
    if expected_anchors is not None and len(usable) != expected_anchors:
        raise DataAssemblyError(
            f"可用锚点数不符: {len(usable)} 期望={expected_anchors}"
        )
    if usable["exec"].iloc[-1] is not None:
        raise DataAssemblyError("末锚必须 pending（exec 为 None）")
    close_diff = (
        features["close"].to_numpy(dtype=float) - usable["anchor_close"].to_numpy(dtype=float)
    )
    max_diff = float(np.abs(close_diff).max()) if len(close_diff) else 0.0
    if not np.isfinite(max_diff) or max_diff >= 1e-9:
        raise DataAssemblyError(f"特征/锚点错位: max close diff={max_diff}")
    if features[ALL_FEATURES].isna().any().any():
        raise DataAssemblyError("特征含 NaN，拒绝启动")
    return daily_a, weekly_a, daily, anchors, usable, features
