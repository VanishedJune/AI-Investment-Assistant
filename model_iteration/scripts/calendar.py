# -*- coding: utf-8 -*-
"""单标的周历：信号=已完成周K（ISO周结束日），执行=信号后下一交易日复权开盘。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "数据"

NAMES = {
    "159941": "纳指ETF广发",
    "159915": "创业板ETF易方达",
    "518600": "广发黄金ETF",
    "517520": "黄金股ETF永赢",
    "512800": "华宝银行ETF",
    "512690": "鹏华酒ETF",
    "515220": "煤炭ETF国泰",
    "159622": "创新药ETF东财",
    "516150": "稀土ETF嘉实",
}


def load_daily(code: str) -> pd.DataFrame:
    path = DATA_DIR / f"{code}_{NAMES[code]}_日线.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def load_weekly(code: str) -> pd.DataFrame:
    path = DATA_DIR / f"{code}_{NAMES[code]}_周线.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def build_signal_samples(code: str = "159915") -> pd.DataFrame:
    """返回每个信号周：signal、exec、fwd1、label8（label8 末尾 8 周为 NaN）。"""
    daily = load_daily(code)
    weekly = load_weekly(code)
    daily_dates = daily["date"].to_numpy()

    signals: list[pd.Timestamp] = []
    execs: list[pd.Timestamp] = []
    for week_end in weekly["date"]:
        later = daily_dates[daily_dates > week_end]
        if len(later) == 0:
            break
        signals.append(week_end)
        execs.append(pd.Timestamp(later[0]))

    frame = pd.DataFrame({"signal": signals, "exec": execs})
    exec_opens = daily.set_index("date")["adj_open"].reindex(frame["exec"]).to_numpy()
    n = len(frame)
    fwd1 = np.full(n, np.nan)
    if n > 1:
        fwd1[:-1] = exec_opens[1:] / exec_opens[:-1] - 1.0
    frame["fwd1"] = fwd1
    frame["label8"] = None
    for i in range(n - 8):
        frame.loc[i, "label8"] = exec_opens[i + 8] / exec_opens[i] - 1.0
    return frame.reset_index(drop=True)


def signal_start_end(frame: pd.DataFrame) -> tuple[str, str]:
    return frame["signal"].iloc[0].date().isoformat(), frame["signal"].iloc[-1].date().isoformat()
