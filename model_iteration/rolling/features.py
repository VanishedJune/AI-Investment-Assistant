# -*- coding: utf-8 -*-
"""PIT 特征：基于 analysis_price（显式 CA）的日/周指标 + 通用扩展列。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.features import (
    indicator_frame,
    slope_frame,
    weekly_indicator_frame,
)

from .ca import build_analysis_price
from .calendar import NAMES, load_daily, load_weekly
from scripts.features import indicator_frame, weekly_indicator_frame  # noqa: E402


def expanding_percentile(series: pd.Series, min_periods: int) -> pd.Series:
    out = np.full(len(series), np.nan)
    values = series.to_numpy(dtype=float)
    for i in range(len(values)):
        if i < min_periods or np.isnan(values[i]):
            continue
        hist = values[:i]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            out[i] = 0.5
            continue
        out[i] = float(np.mean(hist <= values[i]))
    return pd.Series(out, index=series.index)


def smooth_signal_features(
    anchors: pd.DataFrame,
    daily_a: pd.DataFrame,
    weekly_a: pd.DataFrame,
    features: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """按最近 window 个交易日平均平滑 d_slope/d_dif1/w_dif1（周K主导信号同源）。

    用于“决策周 5 日均值”型规则：决策与卖出信号不再依赖单日值。
    """
    out = features.copy()
    di = indicator_frame(daily_a).set_index("date")
    wk = weekly_indicator_frame(weekly_a).set_index("date")["w_dif1"]
    daily_idx = daily_a.set_index("date")
    w_daily = wk.reindex(daily_idx.index, method="ffill")
    rows = []
    for s in anchors["signal"]:
        vis = di.loc[:s]
        dslope = vis["d_slope"].tail(window).mean() if len(vis) else np.nan
        ddif1 = vis["d_dif1"].tail(window).mean() if len(vis) else np.nan
        wv = w_daily.loc[:s].tail(window).mean() if len(w_daily.loc[:s]) else np.nan
        rows.append({"d_slope": float(dslope), "d_dif1": float(ddif1), "w_dif1": float(wv)})
    sm = pd.DataFrame(rows, index=out.index)
    out["d_slope"] = sm["d_slope"]
    out["d_dif1"] = sm["d_dif1"]
    out["w_dif1"] = sm["w_dif1"]
    return out


def analysis_frames(code: str, ca_payload: dict, *, daily=None, weekly=None):
    if daily is None:
        daily = load_daily(code)
    if weekly is None:
        weekly = load_weekly(code)
    daily_a = build_analysis_price(daily, ca_payload)
    weekly_a = build_analysis_price(weekly, ca_payload)
    return daily_a, weekly_a


def relative_strength(signal: pd.Timestamp, code: str, all_daily: dict[str, pd.DataFrame]) -> float:
    mom20: dict[str, float] = {}
    mom60: dict[str, float] = {}
    for c, f in all_daily.items():
        closes = f.loc[f["date"] <= signal, "raw_close"]
        if len(closes) >= 61:
            last = float(closes.iloc[-1])
            mom20[c] = last / float(closes.iloc[-21]) - 1.0
            mom60[c] = last / float(closes.iloc[-61]) - 1.0
    if len(mom20) < 3 or len(mom60) < 3 or code not in mom20:
        return 0.5
    v20 = mom20[code]
    v60 = mom60[code]
    p20 = sum(1 for x in mom20.values() if x <= v20) / len(mom20)
    p60 = sum(1 for x in mom60.values() if x <= v60) / len(mom60)
    return (p20 + p60) / 2.0


def build_extended_features(
    anchors: pd.DataFrame, code: str, ca_payload: dict, min_regime_pct: int = 30,
    *, frames=None, all_daily=None,
) -> pd.DataFrame:
    # Inject date-bounded inputs for inference; legacy callers are unchanged.
    daily_a, weekly_a = analysis_frames(code, ca_payload) if frames is None else frames
    if all_daily is None:
        all_daily = {c: load_daily(c) for c in NAMES}
    di = indicator_frame(daily_a)
    wi = weekly_indicator_frame(weekly_a)
    di_index = di.set_index("date")
    wi_index = wi.set_index("date")
    daily_map = daily_a.set_index("date")

    # ---- 多参数扩展特征（PIT，仅用信号日及以前） ----
    d_close = daily_map["analysis_close"]
    d_vol = daily_map["volume"]
    daily_map["d_ma5"] = d_close.rolling(5).mean()
    daily_map["d_ma10"] = d_close.rolling(10).mean()
    daily_map["d_ma20"] = d_close.rolling(20).mean()
    daily_map["d_ma5_bias"] = d_close / daily_map["d_ma5"] - 1.0
    daily_map["d_ma10_bias"] = d_close / daily_map["d_ma10"] - 1.0
    daily_map["d_ma20_bias"] = d_close / daily_map["d_ma20"] - 1.0
    daily_map["d_ma_align"] = (
        (d_close > daily_map["d_ma5"])
        & (daily_map["d_ma5"] > daily_map["d_ma10"])
        & (daily_map["d_ma10"] > daily_map["d_ma20"])
    ).astype(float)
    daily_map["d_ma20_slope"] = daily_map["d_ma20"].pct_change(5)
    daily_map["d_vol_ma5"] = d_vol / d_vol.rolling(5).mean() - 1.0
    daily_map["d_vol_chg"] = d_vol.pct_change()
    daily_map["d_low20_dist"] = d_close / d_close.rolling(20).min() - 1.0
    daily_map["d_high20_dist"] = d_close / d_close.rolling(20).max() - 1.0

    weekly_map = weekly_a.set_index("date")
    w_close = weekly_map["analysis_close"]
    w_vol = weekly_map["volume"]
    weekly_map["w_ma5"] = w_close.rolling(5).mean()
    weekly_map["w_ma10"] = w_close.rolling(10).mean()
    weekly_map["w_ma20"] = w_close.rolling(20).mean()
    weekly_map["w_ma_align"] = (
        (w_close > weekly_map["w_ma5"])
        & (weekly_map["w_ma5"] > weekly_map["w_ma10"])
        & (weekly_map["w_ma10"] > weekly_map["w_ma20"])
    ).astype(float)
    weekly_map["w_ma20_slope"] = weekly_map["w_ma20"].pct_change(1)
    weekly_map["w_vol_ma5"] = w_vol / w_vol.rolling(5).mean() - 1.0

    rows: list[dict] = []
    for _, row in anchors.iterrows():
        s = row["signal"]
        d_row = di_index.loc[:s].iloc[-1] if not di_index.loc[:s].empty else None
        w_row = wi_index.loc[:s].iloc[-1] if not wi_index.loc[:s].empty else None
        if d_row is None or w_row is None:
            rows.append({})
            continue
        d_vis = daily_map.loc[:s]
        w_vis = weekly_a.loc[weekly_a["date"] <= s]
        d_last = d_vis.iloc[-1]
        w_last = weekly_map.loc[:s].iloc[-1]
        close = float(d_vis["analysis_close"].iloc[-1])
        w_low4_val = float(w_vis["raw_low"].tail(4).min()) if len(w_vis) >= 4 else np.nan
        rows.append(
            {
                "d_slope": float(d_row["d_slope"]),
                "d_dif1": float(d_row["d_dif1"]),
                "d_dif2": float(d_row["d_dif2"]),
                "rsi14": float(d_row["rsi14"]),
                "atr_pct": float(d_row["atr_pct"]),
                "vol20": float(d_row["vol20"]),
                "mom20": float(d_row["mom20"]),
                "mom60": float(d_row["mom60"]),
                "dd20": float(d_row["dd20"]),
                "w_slope": float(w_row["w_slope"]),
                "w_dif1": float(w_row["w_dif1"]),
                "w_dif2": float(w_row["w_dif2"]),
                "rel_strength": relative_strength(s, code, all_daily),
                "sim8_win": np.nan,
                "close": float(d_vis["analysis_close"].iloc[-1]),
                "d_high20": float(d_vis["raw_high"].tail(20).max()),
                "d_low20": float(d_vis["raw_low"].tail(20).min()),
                "w_sma20": float(w_vis["analysis_close"].tail(20).mean()) if len(w_vis) >= 20 else np.nan,
                "w_low4": w_low4_val,
                "mom5": float(close / d_vis["analysis_close"].iloc[-6] - 1.0) if len(d_vis) >= 6 else np.nan,
                "d_ma5": float(d_last["d_ma5"]),
                "d_ma10": float(d_last["d_ma10"]),
                "d_ma20": float(d_last["d_ma20"]),
                "d_ma5_bias": float(d_last["d_ma5_bias"]),
                "d_ma10_bias": float(d_last["d_ma10_bias"]),
                "d_ma20_bias": float(d_last["d_ma20_bias"]),
                "d_ma_align": float(d_last["d_ma_align"]),
                "d_ma20_slope": float(d_last["d_ma20_slope"]),
                "d_vol_ma5": float(d_last["d_vol_ma5"]),
                "d_vol_chg": float(d_last["d_vol_chg"]),
                "d_low20_dist": float(d_last["d_low20_dist"]),
                "d_high20_dist": float(d_last["d_high20_dist"]),
                "w_ma5": float(w_last["w_ma5"]),
                "w_ma10": float(w_last["w_ma10"]),
                "w_ma20": float(w_last["w_ma20"]),
                "w_ma_align": float(w_last["w_ma_align"]),
                "w_ma20_slope": float(w_last["w_ma20_slope"]),
                "w_low4_dist": close / w_low4_val - 1.0 if w_low4_val == w_low4_val else np.nan,
                "w_vol_ma5": float(w_last["w_vol_ma5"]),
            }
        )

    feat = pd.DataFrame(rows, index=anchors.index)
    # vol_ratio
    vol5 = daily_a["analysis_close"].pct_change().rolling(5).std(ddof=1) * np.sqrt(252) * 100.0
    vol20 = daily_a["analysis_close"].pct_change().rolling(20).std(ddof=1) * np.sqrt(252) * 100.0
    vol_map = pd.DataFrame({"date": daily_a["date"], "vol5": vol5, "vol20": vol20}).set_index("date")
    vol_ratios = []
    for s in anchors["signal"]:
        visible = vol_map.loc[:s]
        if visible.empty:
            vol_ratios.append(np.nan)
            continue
        v5 = float(visible.iloc[-1]["vol5"])
        v20 = float(visible.iloc[-1]["vol20"])
        vol_ratios.append(v5 / v20 if v20 and not np.isnan(v20) else np.nan)
    feat["vol_ratio"] = vol_ratios
    # percentiles
    feat["w_slope_pct"] = expanding_percentile(feat["w_slope"], min_regime_pct)
    feat["d_slope_pct"] = expanding_percentile(feat["d_slope"], 5)
    feat["vol_pct"] = expanding_percentile(feat["vol20"], 5)

    # sim8_win：PIT 状态桶 8W 胜率
    buckets: dict[tuple[int, int], list[float]] = {}
    fwd8 = anchors["fwd8"].to_numpy(dtype=float)
    for i in range(len(feat)):
        if pd.isna(feat.loc[i, "w_slope"]) or pd.isna(feat.loc[i, "d_slope"]):
            feat.loc[i, "sim8_win"] = 0.5
            continue
        key = (1 if feat.loc[i, "w_slope"] > 0 else 0, 1 if feat.loc[i, "d_slope"] > 0 else 0)
        hist = buckets.get(key, [])
        feat.loc[i, "sim8_win"] = (sum(hist) / len(hist)) if len(hist) >= 10 else 0.5
        if i + 8 < len(anchors) and not np.isnan(fwd8[i]):
            buckets.setdefault(key, []).append(1.0 if fwd8[i] > 0 else 0.0)
    return feat


ALL_FEATURES = [
    "d_slope", "d_dif1", "d_dif2", "w_slope", "w_dif1", "w_dif2",
    "rsi14", "atr_pct", "vol20", "mom20", "mom60", "dd20",
    "rel_strength", "sim8_win",
    "close", "d_high20", "d_low20", "w_sma20", "w_low4",
    "mom5", "vol_ratio", "w_slope_pct", "d_slope_pct", "vol_pct",
]

# 多参数联合决策扩展特征（对齐 workspace-materials 训练框架）：
# 日/周 MA5/10/20 结构、量价关系、前低/前高距离。只参与 spec 评分/门控，
# 不进入 ALL_FEATURES 可用性掩码（避免改变既有锚点数量）。
MULTI_PARAM_FEATURES = [
    "d_ma5", "d_ma10", "d_ma20",
    "d_ma5_bias", "d_ma10_bias", "d_ma20_bias",
    "d_ma_align", "d_ma20_slope",
    "d_vol_ma5", "d_vol_chg",
    "d_low20_dist", "d_high20_dist",
    "w_ma5", "w_ma10", "w_ma20",
    "w_ma_align", "w_ma20_slope", "w_low4_dist", "w_vol_ma5",
]

ALL_SPEC_FEATURES = ALL_FEATURES + MULTI_PARAM_FEATURES
