# -*- coding: utf-8 -*-
"""B0207 前三名 ETF：预计下跌开始日期 + 完整跌幅 + 月初/月中/月下/月末 分布。

预计下跌开始 = 冠军链仓位由多转空（>0 → 0）的信号日；
完整跌幅 = 从开始日起到“回调”出现前的阶段最大跌幅：
  - 回调 = 收盘 ≥ 阶段低点 × (1 + 5%)，且连续 5 个交易日不创新低；
  - 超过 1 个月也继续计算；数据截止前未出现回调则按截止日截断并标注“未反弹”；
  - 同一段下跌期间的后续多转空不再计入（中间时间不计入）。
确认标准：完整跌幅 ≤ -2% 且 下跌持续 ≥ 15 个自然日。
月份分桶：月初=1~7日；月中=8~14日；月下=15~21日；月末=22日~月底。
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg  # noqa: E402

CODES = ["159915", "518600", "516150"]
ALL_CODES = ["159915", "512010", "159622", "516150", "159941", "512690", "512800", "518600"]
NAMES = {"159915": "创业板ETF", "518600": "黄金ETF", "516150": "稀土ETF"}
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
REBOUND_PCT = 0.05
SUSTAIN_DAYS = 5
MIN_DECLINE = -0.02
MIN_DAYS = 15
BUCKETS = [
    ("月初", lambda d: d.day <= 7),
    ("月中", lambda d: 8 <= d.day <= 14),
    ("月下", lambda d: 15 <= d.day <= 21),
    ("月末", lambda d: d.day >= 22),
]
OUT = ROOT.parent / "champion_vs_dca"


def bucket_of(d: date) -> str:
    for name, pred in BUCKETS:
        if pred(d):
            return name
    return "未知"


def mean_skip(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    etfs = {c: load_etf(c) for c in ALL_CODES}
    date_lists = {c: sorted(etfs[c]["by_date"].keys()) for c in ALL_CODES}

    def prior_weight(code: str, d: date) -> float:
        """该 ETF 多转空前一周 B0207 给它的目标权重（下跌发生前的持仓暴露）。"""
        prev_dates = [x for x in date_lists[code] if x < d.isoformat()]
        if not prev_dates:
            return 0.0
        pd = prev_dates[-1]
        positions = {c: float(etfs[c]["by_date"].get(pd, 0.0)) for c in ALL_CODES}
        eff = {c: min(1.0, max(0.0, positions[c])) for c in ALL_CODES}
        signals = {c: eff[c] > 0.0 for c in ALL_CODES}
        out = allocate_cfg(signals, eff, B0207_ORDER, B0207_PARAMS)
        return float(out["weights"].get(code, 0))

    report = {"schema_version": "decline-full-v1", "codes": CODES, "by_etf": {}}
    lines = [
        "# B0207 前三名 ETF：预计下跌开始日期 + 完整跌幅分布",
        "",
        "预计下跌开始 = 冠军链仓位由多转空（>0→0）的信号日；",
        "完整跌幅 = 开始日到出现回调前的阶段最大跌幅（回调=较阶段低点反弹≥5%且连续5个交易日不创新低）；",
        "超过1个月也继续计算；未出现回调按数据截止日截断并标注；同一段下跌的中间多转空不计入。",
        "确认 = 完整跌幅 ≤ -2% 且持续 ≥ 15 天。",
        "月份分桶：月初=1~7日；月中=8~14日；月下=15~21日；月末=22日~月底。",
        "",
    ]
    all_rows = []
    for code in CODES:
        e = etfs[code]
        closes = e["closes"]
        dates_sorted = sorted(e["by_date"].keys())
        flips = []
        prev_pos = None
        for ds in dates_sorted:
            pos = float(e["by_date"][ds])
            if prev_pos is not None and prev_pos > 0 and pos <= 0:
                flips.append(date.fromisoformat(ds))
            prev_pos = pos

        episodes = []
        active_end = None
        for d in flips:
            if active_end is not None and np.datetime64(d) <= active_end:
                continue  # 中间时间不计入
            c0 = closes.get(d.isoformat())
            if c0 is None or np.isnan(c0):
                continue
            running_low = c0
            low_date = d
            rebound_date = None
            rebound_seen = False
            sustain = 0
            for t in closes.index[closes.index > np.datetime64(d)]:
                c = float(closes.loc[t])
                if c < running_low:
                    running_low = c
                    low_date = t
                    sustain = 0
                elif c >= running_low * (1.0 + REBOUND_PCT):
                    sustain += 1
                    if sustain >= SUSTAIN_DAYS:
                        rebound_date = t
                        rebound_seen = True
                        break
                else:
                    sustain = 0
            if rebound_date is None:
                rebound_date = closes.index[-1]
            rebound_day = date.fromisoformat(str(np.datetime64(rebound_date).astype("datetime64[D]")))
            duration_days = (rebound_day - d).days
            full_decline = running_low / c0 - 1.0
            confirmed = bool(full_decline <= MIN_DECLINE and duration_days >= MIN_DAYS)
            row = {
                "date": d.isoformat(), "bucket": bucket_of(d),
                "full_decline": full_decline,
                "trough_date": str(np.datetime64(low_date).astype("datetime64[D]")),
                "duration_days": duration_days,
                "rebound": rebound_seen,
                "confirmed": confirmed,
            }
            if confirmed:
                row["prior_weight"] = prior_weight(code, d)
            episodes.append(row)
            active_end = rebound_date if rebound_seen else closes.index[-1]

        confirmed_rows = [r for r in episodes if r["confirmed"]]
        counts = {name: 0 for name, _ in BUCKETS}
        for row in confirmed_rows:
            counts[row["bucket"]] += 1
        report["by_etf"][code] = {
            "name": NAMES[code],
            "flips": len(flips),
            "episodes": len(episodes),
            "confirmed": len(confirmed_rows),
            "counts": counts,
            "rows": confirmed_rows,
            "all_episodes": episodes,
        }
        print(code, NAMES[code], "flips", len(flips), "episodes", len(episodes),
              "confirmed", len(confirmed_rows), counts)
        lines.append(f"## {code} {NAMES[code]}（多转空 {len(flips)} 次，确认完整下跌 {len(confirmed_rows)} 次）")
        lines.append("")
        lines.append("| 月份分桶 | 次数 | 日期（完整跌幅/持续天数） | 平均完整跌幅 | 加权完整跌幅 | 平均持续天数 |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for name, _ in BUCKETS:
            rows = [r for r in confirmed_rows if r["bucket"] == name]
            dates = [f"{r['date']}({r['full_decline']:.1%}/{r['duration_days']}天)"
                     for r in rows]
            avg = float(np.mean([r["full_decline"] for r in rows])) if rows else 0.0
            w = sum(r["prior_weight"] for r in rows)
            wavg = (sum(r["full_decline"] * r["prior_weight"] for r in rows) / w
                    if w > 0 else avg)
            avg_days = float(np.mean([r["duration_days"] for r in rows])) if rows else 0.0
            lines.append(f"| {name} | {len(dates)} | {'、'.join(dates) if dates else '-'} | "
                         f"{avg:.1%} | {wavg:.1%} | {avg_days:.0f}天 |")
        lines.append("")
        for row in episodes:
            all_rows.append({"code": code, **row})

    confirmed_rows = [r for r in all_rows if r["confirmed"]]
    total_counts = {name: 0 for name, _ in BUCKETS}
    for row in confirmed_rows:
        total_counts[row["bucket"]] += 1
    report["total"] = {"counts": total_counts, "n": len(confirmed_rows),
                       "all_episodes": len(all_rows)}
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 月份分桶 | 次数 | 占比 | 平均完整跌幅 | 加权完整跌幅 | 平均持续天数 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, _ in BUCKETS:
        n = total_counts[name]
        rows = [r for r in confirmed_rows if r["bucket"] == name]
        avg = float(np.mean([r["full_decline"] for r in rows])) if rows else 0.0
        w = sum(r["prior_weight"] for r in rows)
        wavg = (sum(r["full_decline"] * r["prior_weight"] for r in rows) / w
                if w > 0 else avg)
        avg_days = float(np.mean([r["duration_days"] for r in rows])) if rows else 0.0
        lines.append(f"| {name} | {n} | {n / len(confirmed_rows):.1%} | {avg:.1%} | {wavg:.1%} | "
                     f"{avg_days:.0f}天 |" if confirmed_rows else f"| {name} | 0 | - | - | - | - |")
    lines.append("")
    lines.append(f"确认完整下跌事件合计：{len(confirmed_rows)} 次（全部下跌段 {len(all_rows)} 次）")

    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "b0207_top3_decline_phase.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "b0207_top3_decline_phase.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ETF代码", "ETF名称", "下跌开始日期", "月份分桶", "完整跌幅", "阶段低点日期",
                         "持续天数", "是否回调", "下跌前持仓权重", "是否确认"])
        for row in sorted(all_rows, key=lambda r: (r["code"], r["date"])):
            writer.writerow([row["code"], NAMES[row["code"]], row["date"], row["bucket"],
                             f"{row['full_decline']:.2%}", row["trough_date"],
                             row["duration_days"], "是" if row["rebound"] else "否(截断)",
                             f"{row.get('prior_weight', 0):.0f}",
                             "是" if row["confirmed"] else "否"])
    (OUT / "b0207_top3_decline_phase.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md)
    print("total confirmed:", len(confirmed_rows), total_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
