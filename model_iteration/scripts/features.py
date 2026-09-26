# -*- coding: utf-8 -*-
"""PIT 确定性特征：仅使用信号日及以前的数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import NAMES, load_daily


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return atr / close * 100.0


def slope_frame(close: pd.Series, mad_window: int = 20) -> pd.DataFrame:
    dif = ema(close, 12) - ema(close, 26)
    dif1 = dif.diff()
    dif2 = dif1.diff()
    center = dif1.rolling(mad_window).mean()
    mad = (dif1 - center).abs().rolling(mad_window).median()
    norm = dif1 / mad.replace(0.0, np.nan)
    return pd.DataFrame({"dif1": dif1, "dif2": dif2, "slope_norm": norm})


def indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["adj_close"]
    sl = slope_frame(close)
    out = pd.DataFrame(
        {
            "date": frame["date"],
            "d_slope": sl["slope_norm"],
            "d_dif1": sl["dif1"],
            "d_dif2": sl["dif2"],
            "rsi14": rsi(close, 14),
            "atr_pct": atr_pct(frame["adj_high"], frame["adj_low"], close, 14),
            "vol20": close.pct_change().rolling(20).std(ddof=1) * np.sqrt(252) * 100.0,
            "mom20": close / close.shift(20) - 1.0,
            "mom60": close / close.shift(60) - 1.0,
            "dd20": close / close.rolling(20).max() - 1.0,
        }
    )
    return out


def weekly_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["adj_close"]
    sl = slope_frame(close)
    return pd.DataFrame(
        {
            "date": frame["date"],
            "w_slope": sl["slope_norm"],
            "w_dif1": sl["dif1"],
            "w_dif2": sl["dif2"],
        }
    )


def relative_strength(signal: pd.Timestamp, mom20: dict[str, float], mom60: dict[str, float]) -> float:
    """159915 的 20/60 日动量在当日可得 ETF 中的百分位均值（不足3只取0.5）。"""
    ranks20 = sorted(mom20.values())
    ranks60 = sorted(mom60.values())
    if len(ranks20) < 3 or len(ranks60) < 3:
        return 0.5
    v20 = mom20.get("159915")
    v60 = mom60.get("159915")
    if v20 is None or v60 is None:
        return 0.5
    p20 = sum(1 for x in ranks20 if x <= v20) / len(ranks20)
    p60 = sum(1 for x in ranks60 if x <= v60) / len(ranks60)
    return (p20 + p60) / 2.0


FEATURE_COLUMNS = [
    "d_slope", "d_dif1", "d_dif2", "w_slope", "w_dif1", "w_dif2",
    "rsi14", "atr_pct", "vol20", "mom20", "mom60", "dd20",
    "rel_strength", "sim8_win",
]


def build_features(signals: pd.DataFrame, code: str = "159915") -> pd.DataFrame:
    daily = load_daily(code)
    from .calendar import load_weekly

    weekly = load_weekly(code)
    di = indicator_frame(daily)
    wi = weekly_indicator_frame(weekly)
    di_index = di.set_index("date")
    wi_index = wi.set_index("date")

    # 其它 ETF 用于相对强弱
    others = {c: load_daily(c) for c in NAMES if c != code}
    others_mom = {c: f.set_index("date")["adj_close"] for c, f in others.items()}
    own_close = daily.set_index("date")["adj_close"]
    all_closes = dict(others_mom)
    all_closes[code] = own_close

    rows: list[dict] = []
    for _, row in signals.iterrows():
        s = row["signal"]
        d_row = di_index.loc[:s].iloc[-1] if not di_index.loc[:s].empty else None
        w_row = wi_index.loc[:s].iloc[-1] if not wi_index.loc[:s].empty else None
        if d_row is None or w_row is None:
            rows.append({c: np.nan for c in FEATURE_COLUMNS})
            continue
        mom20: dict[str, float] = {}
        mom60: dict[str, float] = {}
        for c, close in all_closes.items():
            visible = close.loc[:s]
            if len(visible) >= 61:
                last = float(visible.iloc[-1])
                mom20[c] = last / float(visible.iloc[-21]) - 1.0
                mom60[c] = last / float(visible.iloc[-61]) - 1.0
        rows.append(
            {
                "d_slope": float(d_row["d_slope"]),
                "d_dif1": float(d_row["d_dif1"]),
                "d_dif2": float(d_row["d_dif2"]),
                "w_slope": float(w_row["w_slope"]),
                "w_dif1": float(w_row["w_dif1"]),
                "w_dif2": float(w_row["w_dif2"]),
                "rsi14": float(d_row["rsi14"]),
                "atr_pct": float(d_row["atr_pct"]),
                "vol20": float(d_row["vol20"]),
                "mom20": float(d_row["mom20"]),
                "mom60": float(d_row["mom60"]),
                "dd20": float(d_row["dd20"]),
                "rel_strength": relative_strength(s, mom20, mom60),
                "sim8_win": np.nan,
            }
        )

    feat = pd.DataFrame(rows)
    # sim8_win：PIT 累计同状态后8周胜率（样本<10 取0.5）
    buckets: dict[tuple[int, int], list[float]] = {}
    for i, row in feat.iterrows():
        if pd.isna(row["w_slope"]) or pd.isna(row["d_slope"]):
            feat.loc[i, "sim8_win"] = 0.5
            continue
        key = (1 if row["w_slope"] > 0 else 0, 1 if row["d_slope"] > 0 else 0)
        hist = buckets.get(key, [])
        feat.loc[i, "sim8_win"] = (sum(hist) / len(hist)) if len(hist) >= 10 else 0.5
        label = signals.loc[i, "label8"]
        if i + 8 < len(signals) and pd.notna(label):
            buckets.setdefault(key, []).append(1.0 if label > 0 else 0.0)
    return feat
