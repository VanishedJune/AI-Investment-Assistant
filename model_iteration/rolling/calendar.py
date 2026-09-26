# -*- coding: utf-8 -*-
"""712 锚点日历：信号=已完成周K，执行=下一交易日 raw_open；fwd 用 analysis_price。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "数据"

NAMES = {
    "159941": "纳指ETF广发",
    "159611": "电力ETF广发",
    "159915": "创业板ETF易方达",
    "518600": "广发黄金ETF",
    "517520": "黄金股ETF永赢",
    "512800": "华宝银行ETF",
    "512690": "鹏华酒ETF",
    "515220": "煤炭ETF国泰",
    "159622": "创新药ETF东财",
    "516150": "稀土ETF嘉实",
}


def _resolve_data_dir(code: str) -> Path:
    """所有正式ETF统一读取项目根目录 ``数据/``，不允许工作区私有行情副本。"""
    if code not in NAMES:
        raise KeyError(f"未登记ETF代码：{code}")
    if code == "159941" and not (DATA_DIR / f"{code}_{NAMES[code]}_日线.csv").exists():
        return ROOT.parent / "archive/instrument_replacements/159941_to_159611/before/数据"
    if code == "518600" and not (DATA_DIR / f"{code}_{NAMES[code]}_日线.csv").exists():
        return ROOT.parent / "archive/instrument_replacements/518600_to_517520/数据"
    return DATA_DIR


def load_daily(code: str) -> pd.DataFrame:
    path = _resolve_data_dir(code) / f"{code}_{NAMES[code]}_日线.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def load_weekly(code: str) -> pd.DataFrame:
    path = _resolve_data_dir(code) / f"{code}_{NAMES[code]}_周线.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def build_anchors(code: str, analysis_daily: pd.DataFrame, analysis_weekly: pd.DataFrame, *, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """锚点表：signal、exec（末锚可 None）、exec_raw_open、anchor_analysis_close、fwd1..fwd8。"""
    if daily is None:
        daily = load_daily(code)
    daily_dates = daily["date"].to_numpy()
    raw_open_by_date = daily.set_index("date")["raw_open"]
    analysis_by_date = analysis_daily.set_index("date")["analysis_close"]
    weekly_dates = analysis_weekly["date"].to_numpy()
    weekly_close = analysis_weekly.set_index("date")["analysis_close"]

    signals: list[pd.Timestamp] = []
    execs: list[pd.Timestamp | None] = []
    for week_end in weekly_dates:
        later = daily_dates[daily_dates > week_end]
        execs.append(pd.Timestamp(later[0]) if len(later) else None)
        signals.append(pd.Timestamp(week_end))

    frame = pd.DataFrame({"signal": signals, "exec": execs})
    frame["exec"] = frame["exec"].astype(object)
    frame.loc[frame["exec"].isna(), "exec"] = None
    exec_opens = np.full(len(frame), np.nan)
    for i, ex in enumerate(execs):
        if ex is not None:
            exec_opens[i] = float(raw_open_by_date.loc[ex])
    frame["exec_raw_open"] = exec_opens
    frame["anchor_close"] = weekly_close.reindex(frame["signal"]).to_numpy(dtype=float)

    n = len(frame)
    closes = frame["anchor_close"].to_numpy(dtype=float)
    for k in (1, 2, 4, 8):
        fwd = np.full(n, np.nan)
        for i in range(n - k):
            if not np.isnan(closes[i]) and not np.isnan(closes[i + k]):
                fwd[i] = closes[i + k] / closes[i] - 1.0
        frame[f"fwd{k}"] = fwd
    return frame.reset_index(drop=True)


def first_last(frame: pd.DataFrame) -> tuple[str, str]:
    return frame["signal"].iloc[0].date().isoformat(), frame["signal"].iloc[-1].date().isoformat()
