# -*- coding: utf-8 -*-
"""成熟评价索引：O(1) 范围统计，避免全量扫描 events（支撑 265 Due × 千级 Challenger 全历史回放）。"""

from __future__ import annotations

import bisect

import numpy as np


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


class EvalIndex:
    def __init__(self) -> None:
        self._data: dict[tuple[str, int], dict] = {}

    def add(self, ev: dict) -> None:
        if ev.get("pending"):
            return
        key = (ev.get("model_id", ""), ev.get("horizon"))
        d = self._data.setdefault(
            key,
            {
                "anchor": [], "mae": [], "brier": [], "brier_strong": [], "acc": [],
                "actual_return": [], "position": [], "missed_rally": [],
                "wrong_chase": [], "over_cash": [], "late_entry": [],
                "direction": [], "conflict_correct": [],
            },
        )
        d["anchor"].append(ev.get("anchor_index"))
        d["mae"].append(ev["mae"])
        d["brier"].append(ev["brier"])
        d["brier_strong"].append(ev.get("brier_strong"))
        dc = ev.get("direction_correct")
        d["acc"].append(1.0 if dc is True else 0.0 if dc is False else None)
        d["actual_return"].append(ev.get("actual_return"))
        d["position"].append(ev.get("position"))
        d["missed_rally"].append(1 if ev.get("missed_rally") else 0)
        d["wrong_chase"].append(1 if ev.get("wrong_chase") else 0)
        d["over_cash"].append(1 if ev.get("over_cash") else 0)
        d["late_entry"].append(1 if ev.get("late_entry") else 0)
        d["direction"].append(ev.get("direction_correct"))
        d["conflict_correct"].append(ev.get("conflict_correct"))

    def _slice(self, model_id: str, horizon: int, start: int, end: int):
        d = self._data.get((model_id, horizon))
        if not d:
            return [], [], [], [], []
        anchors = d["anchor"]
        lo = bisect.bisect_left(anchors, start)
        hi = bisect.bisect_left(anchors, end)
        return (
            anchors[lo:hi],
            d["mae"][lo:hi],
            d["brier"][lo:hi],
            d["brier_strong"][lo:hi],
            d["acc"][lo:hi],
        )

    def range_cases(self, model_id: str, horizon: int, start: int, end: int) -> list[dict]:
        """逐窗口案例（供诊断 failure_cases 使用，保持盲化相对结构）。"""
        d = self._data.get((model_id, horizon))
        if not d:
            return []
        anchors = d["anchor"]
        lo = bisect.bisect_left(anchors, start)
        hi = bisect.bisect_left(anchors, end)
        out = []
        for idx in range(lo, hi):
            out.append(
                {
                    "anchor": anchors[idx],
                    "actual_return": d["actual_return"][idx],
                    "position": d["position"][idx],
                    "missed_rally": bool(d["missed_rally"][idx]),
                    "wrong_chase": bool(d["wrong_chase"][idx]),
                    "over_cash": bool(d["over_cash"][idx]),
                    "late_entry": bool(d["late_entry"][idx]),
                    "direction_correct": d["direction"][idx],
                    "conflict_correct": d["conflict_correct"][idx],
                }
            )
        return out

    def range_stats(self, model_id: str, horizon: int, start: int, end: int) -> tuple:
        """返回 (MAE 均值, Brier 均值, 方向准确率, Brier_strong 均值)，无样本为 None。"""
        _, mae, brier, strong, acc = self._slice(model_id, horizon, start, end)
        accs = [a for a in acc if a is not None]
        strongs = [a for a in strong if a is not None]
        return (
            _mean(mae),
            _mean(brier),
            _mean(accs) if accs else None,
            _mean(strongs),
        )


def index_from_events(events: list[dict]) -> EvalIndex:
    idx = EvalIndex()
    for ev in events:
        if ev.get("type") == "evaluation":
            idx.add(ev)
    return idx
