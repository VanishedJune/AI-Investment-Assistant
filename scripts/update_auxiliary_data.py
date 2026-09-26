# -*- coding: utf-8 -*-
"""Update auxiliary ETF market data without changing the fixed core-8 universe.

Usage:
    python scripts/update_auxiliary_data.py --as-of YYYY-MM-DD
    python scripts/update_auxiliary_data.py --as-of YYYY-MM-DD --force-full

Outputs are isolated under ``辅助数据/``.  The script deliberately does not
write ``data_manifest.json``, ``最新行情快照.csv``, feature snapshots, reports,
portfolio state, or any B0207 configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import update_data as core_updater  # noqa: E402


SHANGHAI = ZoneInfo("Asia/Shanghai")
AUX_DATA_DIR = ROOT / "辅助数据"
AUX_CONFIG_PATH = ROOT / "config" / "auxiliary_instruments.json"
LOCK_PATH = ROOT / ".auxiliary_update.lock"


class AuxiliaryUpdateError(RuntimeError):
    pass


def _load_instruments() -> list[dict]:
    payload = json.loads(AUX_CONFIG_PATH.read_text(encoding="utf-8"))
    instruments = payload.get("instruments")
    if not isinstance(instruments, list):
        raise AuxiliaryUpdateError("辅助标的配置必须是数组")
    core_codes = {item["code"] for item in core_updater.INSTRUMENTS}
    seen: set[str] = set()
    for item in instruments:
        code = str(item.get("code") or "")
        if len(code) != 6 or not code.isdigit():
            raise AuxiliaryUpdateError(f"辅助标的代码非法：{code}")
        if code in core_codes:
            raise AuxiliaryUpdateError(f"辅助标的与固定8只决策池重复：{code}")
        if code in seen:
            raise AuxiliaryUpdateError(f"辅助标的代码重复：{code}")
        seen.add(code)
    return instruments


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_auxiliary(rows: list[dict], instrument: dict, timeframe: str) -> list[str]:
    """Validate structure while allowing legitimate total-return distribution jumps.

    ETF cash distributions can make the adjusted total return exceed the
    exchange's raw-price daily limit, while unit splits can make the raw price
    jump even though the adjusted return remains continuous.  Either one-sided
    discontinuity is retained and disclosed; simultaneous raw and adjusted
    limit breaches remain a hard failure.
    """

    relaxed = deepcopy(instrument)
    relaxed["price_limit_pct"] = 1000
    warnings = core_updater._validate(rows, relaxed, timeframe)
    if timeframe != "daily":
        return warnings
    threshold = (float(instrument["price_limit_pct"]) + 1.0) / 100.0
    previous_raw: float | None = None
    previous_adjusted: float | None = None
    for row in rows:
        raw_close = float(row["raw_close"])
        adjusted_close = float(row["adj_close"])
        if previous_raw is not None and previous_adjusted is not None:
            raw_return = raw_close / previous_raw - 1.0
            adjusted_return = adjusted_close / previous_adjusted - 1.0
            raw_breach = abs(raw_return) > threshold + 1e-9
            adjusted_breach = abs(adjusted_return) > threshold + 1e-9
            if raw_breach and adjusted_breach:
                raise AuxiliaryUpdateError(
                    f"{instrument['code']} {row['date']} 原始收益{raw_return:.2%}、"
                    f"复权收益{adjusted_return:.2%}同时超过价格限制门禁"
                )
            if adjusted_breach:
                warnings.append(
                    f"{instrument['code']} {row['date']}: 复权收益{adjusted_return:.2%}超过原始价格限制，"
                    f"原始收益{raw_return:.2%}仍合规，按分红复权跳变保留"
                )
            elif raw_breach:
                warnings.append(
                    f"{instrument['code']} {row['date']}: 原始收益{raw_return:.2%}超过价格限制，"
                    f"复权收益{adjusted_return:.2%}仍合规，按份额拆分保留"
                )
        previous_raw = raw_close
        previous_adjusted = adjusted_close
    return warnings


def _replace_directory(staging_data: Path, run_id: str) -> None:
    previous = ROOT / f".previous_auxiliary_data_{run_id}"
    if previous.exists():
        raise AuxiliaryUpdateError(f"存在未清理的辅助数据回滚目录：{previous.name}")
    swapped = False
    try:
        if AUX_DATA_DIR.exists():
            os.replace(AUX_DATA_DIR, previous)
        os.replace(staging_data, AUX_DATA_DIR)
        swapped = True
    except Exception:
        if swapped and AUX_DATA_DIR.exists():
            failed = ROOT / f".failed_auxiliary_data_{run_id}"
            os.replace(AUX_DATA_DIR, failed)
            if previous.exists():
                os.replace(previous, AUX_DATA_DIR)
            shutil.rmtree(failed, ignore_errors=True)
        elif previous.exists() and not AUX_DATA_DIR.exists():
            os.replace(previous, AUX_DATA_DIR)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def run(*, as_of: date, force_full: bool) -> dict:
    now = datetime.now(SHANGHAI)
    run_id = now.strftime("AUX%Y%m%dT%H%M%S")
    staging = ROOT / f".auxiliary_staging_{run_id}"
    if staging.exists():
        raise AuxiliaryUpdateError(f"辅助数据暂存目录已存在：{staging.name}")
    lock_fd: int | None = None
    original_data_dir = core_updater.DATA_DIR
    try:
        lock_fd = core_updater._acquire_update_lock(LOCK_PATH, run_id)
        staging_data = staging / "辅助数据"
        staging_data.mkdir(parents=True)
        instruments = _load_instruments()
        if not instruments:
            return {
                "run_id": None,
                "as_of": None,
                "instrument_count": 0,
                "warnings": 0,
                "output": str(AUX_DATA_DIR),
                "status": "no_auxiliary_instruments",
                "message": "当前无独立辅助ETF；正式8只ETF统一由根目录数据通道管理。",
            }
        requests, akshare = core_updater._dependencies()
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
        timestamp = now.isoformat(timespec="seconds")
        warnings: list[str] = []
        records: list[dict] = []
        snapshot_rows: list[dict] = []

        # Reuse the audited price-building and aggregation functions while
        # redirecting their incremental-read path to the auxiliary directory.
        core_updater.DATA_DIR = AUX_DATA_DIR
        for instrument in instruments:
            daily, item_warnings = core_updater._build_daily(
                session,
                instrument,
                as_of,
                run_id,
                timestamp,
                force_full=force_full,
                akshare=akshare,
            )
            weekly = core_updater._aggregate(daily, "weekly")
            monthly = core_updater._aggregate(daily, "monthly")
            warnings.extend(item_warnings)
            for timeframe, label, rows in (
                ("daily", "日线", daily),
                ("weekly", "周线", weekly),
                ("monthly", "月线", monthly),
            ):
                warnings.extend(_validate_auxiliary(rows, instrument, timeframe))
                filename = f"{instrument['code']}_{instrument['name']}_{label}.csv"
                output = staging_data / filename
                _write_csv(output, rows, core_updater.FIELDS)
                records.append(
                    {
                        "path": f"辅助数据/{filename}",
                        "rows": len(rows),
                        "start": rows[0]["date"],
                        "end": rows[-1]["date"],
                        "run_id": run_id,
                    }
                )
            latest = daily[-1]
            snapshot_rows.append(
                {
                    "code": instrument["code"],
                    "name": instrument["name"],
                    "exchange": instrument["exchange"],
                    "date": latest["date"],
                    "raw_close": latest["raw_close"],
                    "adj_close": latest["adj_close"],
                    "amount": latest["amount"],
                    "data_status": "closed",
                    "run_id": run_id,
                }
            )

        latest_dates = {row["code"]: row["date"] for row in snapshot_rows}
        manifest = {
            "schema_version": "auxiliary-data-v1",
            "run_id": run_id,
            "generated_at": timestamp,
            "as_of": min(latest_dates.values()),
            "market_status": "closed",
            "integrity_mode": "not_evaluated",
            "decision_universe": False,
            "latest_dates": latest_dates,
            "warnings": warnings,
            "files": records,
        }
        (staging_data / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_csv(
            staging_data / "最新行情快照.csv",
            snapshot_rows,
            list(snapshot_rows[0]),
        )
        _replace_directory(staging_data, run_id)
        return {
            "run_id": run_id,
            "as_of": manifest["as_of"],
            "instrument_count": len(instruments),
            "warnings": len(warnings),
            "output": str(AUX_DATA_DIR),
        }
    finally:
        core_updater.DATA_DIR = original_data_dir
        if lock_fd is not None:
            os.close(lock_fd)
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="更新辅助观察ETF行情，不改变固定8只决策池")
    parser.add_argument("--as-of", help="截止日期 YYYY-MM-DD；默认上海当前日期")
    parser.add_argument("--force-full", action="store_true", help="忽略已有辅助数据并全量重建")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(SHANGHAI).date()
        result = run(as_of=as_of, force_full=args.force_full)
    except (AuxiliaryUpdateError, core_updater.UpdateError, ValueError, OSError) as exc:
        print(f"辅助行情更新失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
