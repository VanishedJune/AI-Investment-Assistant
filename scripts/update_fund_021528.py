# -*- coding: utf-8 -*-
"""更新021528财通成长优选混合C的公开历史净值。

只保存基金真实公布的单位净值、累计净值和日增长率，不生成OHLCV，
也不把场外基金净值伪装成ETF行情。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "基金数据" / "021528_财通成长优选混合C_净值历史.csv"
API = "https://api.fund.eastmoney.com/f10/lsjz"
FIELDS = [
    "code", "name", "date", "unit_nav", "cum_nav", "daily_growth_pct",
    "purchase_status", "redeem_status", "source", "source_timestamp", "run_id",
]


def fetch_page(page: int, page_size: int = 20) -> dict:
    query = urllib.parse.urlencode(
        {"fundCode": "021528", "pageIndex": page, "pageSize": page_size}
    )
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={
            "Referer": "https://fundf10.eastmoney.com/",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all() -> list[dict]:
    page_size = 20  # 接口固定每页最多20条；更大的pageSize会返回空列表。
    first = fetch_page(1, page_size)
    data = first.get("Data") or {}
    rows = list(data.get("LSJZList") or [])
    total = int(first.get("TotalCount") or len(rows))
    for page in range(2, (total + page_size - 1) // page_size + 1):
        rows.extend((fetch_page(page, page_size).get("Data") or {}).get("LSJZList") or [])
    return rows


def normalize(raw_rows: list[dict], source_timestamp: str, run_id: str) -> list[dict]:
    by_date: dict[str, dict] = {}
    for item in raw_rows:
        date = str(item.get("FSRQ") or "").strip()
        unit_nav = str(item.get("DWJZ") or "").strip()
        if not date or not unit_nav:
            continue
        datetime.strptime(date, "%Y-%m-%d")
        float(unit_nav)
        cum_nav = str(item.get("LJJZ") or "").strip()
        growth = str(item.get("JZZZL") or "").strip()
        if cum_nav:
            float(cum_nav)
        if growth:
            float(growth)
        by_date[date] = {
            "code": "021528",
            "name": "财通成长优选混合C",
            "date": date,
            "unit_nav": unit_nav,
            "cum_nav": cum_nav,
            "daily_growth_pct": growth,
            "purchase_status": str(item.get("SGZT") or "").strip(),
            "redeem_status": str(item.get("SHZT") or "").strip(),
            "source": "天天基金lsjz",
            "source_timestamp": source_timestamp,
            "run_id": run_id,
        }
    rows = [by_date[key] for key in sorted(by_date)]
    if not rows:
        raise RuntimeError("公开接口未返回可用净值")
    if rows[0]["date"] != "2024-05-28":
        raise RuntimeError(f"021528首个净值日异常: {rows[0]['date']}")
    return rows


def atomic_write(rows: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".021528-nav-", suffix=".csv", dir=OUTPUT.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, OUTPUT)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只下载并校验，不写文件")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    rows = normalize(fetch_all(), stamp, run_id)
    if not args.check:
        atomic_write(rows)
    print(
        json.dumps(
            {
                "status": "checked" if args.check else "updated",
                "rows": len(rows),
                "start": rows[0]["date"],
                "end": rows[-1]["date"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
