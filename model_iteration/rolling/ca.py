# -*- coding: utf-8 -*-
"""显式 PIT Corporate-Action 事件表与 analysis_price 构造。

硬约束：事件参数只能来自显式可核验来源；禁止从价格跳空或 adj_factor 推断。
缺失/未 verified 一律 fail-closed（DATA_CA_INCOMPLETE）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

class DataCAIncomplete(Exception):
    """显式 CA 事件数据缺失、未 verified 或覆盖不足。"""


def load_ca(etf: str = "159915") -> dict:
    from . import ledger as ledger_mod

    path = ledger_mod.CA_DIR / f"{etf}_ca_events.json"
    if not path.exists():
        raise DataCAIncomplete(f"缺少显式CA事件表: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cov = payload.get("coverage") or {}
    if not cov.get("verified"):
        raise DataCAIncomplete(f"CA覆盖未 verified: {path}")
    for event in payload.get("events", []):
        if not event.get("verified"):
            raise DataCAIncomplete(f"存在未 verified 事件: {event}")
    return payload


def validate_coverage(payload: dict, required_start: str, required_end: str) -> None:
    cov = payload.get("coverage") or {}
    if str(cov.get("start", "")) > required_start or str(cov.get("end", "")) < required_end:
        raise DataCAIncomplete(
            f"CA覆盖不足: {cov.get('start')}~{cov.get('end')}，需要 {required_start}~{required_end}"
        )


def ca_events_hash(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_analysis_price(daily: pd.DataFrame, payload: dict) -> pd.DataFrame:
    """analysis_price(t) = raw_close(t)*SF(t) + Σ(dps_e × SF_after(e))，仅用 ex_date ≤ t 的显式事件。

    SF_after(e) = ∏_{split e′: ex_date(e) < ex_date(e′) ≤ t} ratio_{e′}
    （方案封版公式：分红按除息日之后发生的拆分缩放，与权益登记日无关）。
    """
    df = daily.sort_values("date").reset_index(drop=True)
    raw_close = df["raw_close"].to_numpy(dtype=float)
    dates = df["date"].to_numpy()
    events = sorted(payload.get("events", []), key=lambda e: e["ex_date"])

    # SF(t) = 拆分比累积
    sf = np.ones(len(df))
    for e in events:
        if e["type"] != "split":
            continue
        # 从首个 ≥ ex_date 的交易日生效（side="left"）；避免周线缺当日 bar 时提前一期生效
        idx = np.searchsorted(dates, np.datetime64(e["ex_date"]), side="left")
        if idx < len(df):
            sf[idx:] *= float(e["ratio"])

    add = np.zeros(len(df))
    for e in events:
        if e["type"] != "cash_dividend":
            continue
        idx = np.searchsorted(dates, np.datetime64(e["ex_date"]), side="left")
        if idx >= len(df):
            continue
        sf_before = float(sf[idx - 1]) if idx > 0 else 1.0
        add[idx:] += float(e["dps"]) * sf[idx:] / sf_before

    df["analysis_close"] = raw_close * sf + add
    df["sf"] = sf
    return df


def empty_ca_template(etf: str = "159915") -> dict:
    return {
        "schema_version": "ca-events-v1.0.0",
        "etf": etf,
        "coverage": {
            "start": "2011-12-09",
            "end": "2026-08-07",
            "verified": False,
            "source": None,
        },
        "events": [],
    }
