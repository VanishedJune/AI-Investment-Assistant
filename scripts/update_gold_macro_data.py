# -*- coding: utf-8 -*-
"""Download and normalize gold-specific U.S. macro data.

This updater is deliberately isolated from the eight-ETF market-data pipeline.
It writes only ``辅助数据/黄金宏观/`` and never changes B0207, feature snapshots,
reports, portfolio state, or any ETF price file.

Data definitions
----------------
* 10-year real yield: U.S. Treasury 10-year TIPS par real yield.
* 10-year market-implied inflation compensation: same-date 10-year nominal
  Treasury par yield minus the 10-year TIPS par real yield.

The second series is inflation compensation rather than a pure forecast of CPI;
it can also contain liquidity and risk premia.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import urllib.request
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.publication_io import capture, commit_files
OUTPUT_DIR = ROOT / "辅助数据" / "黄金宏观"
SHANGHAI = ZoneInfo("Asia/Shanghai")
REAL_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)
NOMINAL_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)


class GoldMacroDataError(RuntimeError):
    pass


def _download_csv(url: str, timeout: int = 45) -> list[dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AI-Investment-Assistant/1.0 (official-data updater)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8-sig")
    except Exception as exc:  # pragma: no cover - network failure path
        raise GoldMacroDataError(f"美国财政部数据下载失败：{url}；{exc}") from exc
    rows = list(csv.DictReader(io.StringIO(raw)))
    if not rows:
        raise GoldMacroDataError(f"美国财政部返回空数据：{url}")
    return rows


def _column(row: dict[str, str], wanted: str) -> str:
    normalized = {" ".join(key.upper().split()): key for key in row}
    key = normalized.get(" ".join(wanted.upper().split()))
    if key is None:
        raise GoldMacroDataError(f"官方CSV缺少字段：{wanted}")
    return key


def _parse_series(
    rows: list[dict[str, str]], value_column: str, as_of: date
) -> dict[date, Decimal]:
    date_key = _column(rows[0], "Date")
    value_key = _column(rows[0], value_column)
    result: dict[date, Decimal] = {}
    for row in rows:
        raw_date = (row.get(date_key) or "").strip()
        raw_value = (row.get(value_key) or "").strip()
        if not raw_date or not raw_value:
            continue
        try:
            item_date = datetime.strptime(raw_date, "%m/%d/%Y").date()
            value = Decimal(raw_value)
        except (ValueError, InvalidOperation) as exc:
            raise GoldMacroDataError(
                f"官方CSV存在无法解析的数据：date={raw_date!r}, value={raw_value!r}"
            ) from exc
        if item_date <= as_of:
            result[item_date] = value
    return result


def build_dataset(
    *, as_of: date, start_year: int, downloader=_download_csv
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if start_year < 2003:
        raise GoldMacroDataError("10年TIPS历史起点不得早于2003年")
    if start_year > as_of.year:
        raise GoldMacroDataError("start_year不得晚于as_of年份")

    real: dict[date, Decimal] = {}
    nominal: dict[date, Decimal] = {}
    for year in range(start_year, as_of.year + 1):
        real.update(
            _parse_series(
                downloader(REAL_URL.format(year=year)), "10 YR", as_of
            )
        )
        nominal.update(
            _parse_series(
                downloader(NOMINAL_URL.format(year=year)), "10 YR", as_of
            )
        )

    common_dates = sorted(set(real) & set(nominal))
    if not common_dates:
        raise GoldMacroDataError("名义收益率与TIPS没有共同有效日期")

    tips_rows: list[dict[str, str]] = []
    breakeven_rows: list[dict[str, str]] = []
    for item_date in common_dates:
        real_value = real[item_date]
        nominal_value = nominal[item_date]
        inflation_compensation = nominal_value - real_value
        tips_rows.append(
            {
                "date": item_date.isoformat(),
                "real_yield_10y_pct": f"{real_value:.2f}",
                "source": "U.S. Department of the Treasury",
                "series": "Daily Treasury Par Real Yield Curve Rates - 10 YR",
                "source_url": REAL_URL.format(year=item_date.year),
            }
        )
        breakeven_rows.append(
            {
                "date": item_date.isoformat(),
                "nominal_yield_10y_pct": f"{nominal_value:.2f}",
                "real_yield_10y_pct": f"{real_value:.2f}",
                "inflation_compensation_10y_pct": f"{inflation_compensation:.2f}",
                "method": "10Y nominal Treasury par yield - 10Y TIPS par real yield",
                "nominal_source_url": NOMINAL_URL.format(year=item_date.year),
                "real_yield_source_url": REAL_URL.format(year=item_date.year),
            }
        )
    return tips_rows, breakeven_rows


def _change(rows: list[dict[str, str]], field: str, lag: int) -> float | None:
    if len(rows) <= lag:
        return None
    return round(float(rows[-1][field]) - float(rows[-1 - lag][field]), 4)


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def run(*, as_of: date, start_year: int, check: bool = False) -> dict:
    paths=['辅助数据/黄金宏观/'+name for name in ('美国10年TIPS实际收益率.csv','美国10年市场隐含通胀.csv','latest.json')]
    expected=capture(ROOT,paths)
    tips_rows, breakeven_rows = build_dataset(as_of=as_of, start_year=start_year)
    latest_tips = tips_rows[-1]
    latest_breakeven = breakeven_rows[-1]
    generated_at = datetime.now(SHANGHAI).isoformat(timespec="seconds")
    latest = {
        "schema_version": "gold-macro-v1",
        "instrument_scope": [json.loads((ROOT / 'config/gold_analysis.json').read_text(encoding='utf-8'))['instrument']['code']],
        "generated_at": generated_at,
        "data_as_of": latest_tips["date"],
        "requested_as_of": as_of.isoformat(),
        "history_start": tips_rows[0]["date"],
        "row_count": len(tips_rows),
        "data_status": "official_closed_observations",
        "same_date_cn_session_allowed": False,
        "availability_note": (
            "美国日期为来源观测日；若该值在中国市场收盘后发布，不得用于同日A股盘中或收盘决策，"
            "只能用于发布后进行的分析。"
        ),
        "tips_10y": {
            "value_pct": float(latest_tips["real_yield_10y_pct"]),
            "change_1_observation_pct_point": _change(
                tips_rows, "real_yield_10y_pct", 1
            ),
            "change_5_observations_pct_point": _change(
                tips_rows, "real_yield_10y_pct", 5
            ),
            "change_20_observations_pct_point": _change(
                tips_rows, "real_yield_10y_pct", 20
            ),
            "gold_relation": "通常反向；上升构成压力，下降构成支持",
        },
        "inflation_compensation_10y": {
            "value_pct": float(
                latest_breakeven["inflation_compensation_10y_pct"]
            ),
            "nominal_yield_10y_pct": float(
                latest_breakeven["nominal_yield_10y_pct"]
            ),
            "change_1_observation_pct_point": _change(
                breakeven_rows, "inflation_compensation_10y_pct", 1
            ),
            "change_5_observations_pct_point": _change(
                breakeven_rows, "inflation_compensation_10y_pct", 5
            ),
            "change_20_observations_pct_point": _change(
                breakeven_rows, "inflation_compensation_10y_pct", 20
            ),
            "interpretation_note": (
                "这是市场隐含通胀补偿，不等同于CPI预测；必须与实际利率和名义利率联合解释。"
            ),
        },
        "sources": {
            "real_yield": REAL_URL.format(year=as_of.year),
            "nominal_yield": NOMINAL_URL.format(year=as_of.year),
        },
    }
    if not check:
        def csv_bytes(rows):
            stream=io.StringIO(newline='')
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            return stream.getvalue().encode('utf-8-sig')
        outputs={paths[0]:csv_bytes(tips_rows),paths[1]:csv_bytes(breakeven_rows),
                 paths[2]:(json.dumps(latest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')}
        commit_files(ROOT,outputs,expected)
    return latest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="更新config/gold_analysis.json指定黄金标的的美国宏观数据")
    parser.add_argument("--as-of", help="截止日期 YYYY-MM-DD；默认上海当前日期")
    parser.add_argument(
        "--start-year",
        type=int,
        default=2021,
        help="本地历史起始年份；默认2021，最早允许2003",
    )
    parser.add_argument("--check", action="store_true", help="只下载、解析和校验，不写文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(SHANGHAI).date()
        result = run(as_of=as_of, start_year=args.start_year, check=args.check)
    except (ValueError, GoldMacroDataError) as exc:
        print(f"黄金宏观数据更新失败：{exc}")
        return 1
    mode = "校验通过" if args.check else "更新完成"
    print(
        f"{mode}：{result['history_start']} 至 {result['data_as_of']}，"
        f"共同观测 {result['row_count']} 条；"
        f"10年TIPS {result['tips_10y']['value_pct']:.2f}%，"
        f"10年隐含通胀补偿 {result['inflation_compensation_10y']['value_pct']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
