# -*- coding: utf-8 -*-
"""Replace one configured ETF data set without running investment analysis.

This utility only downloads public OHLCV data, aggregates daily rows into weekly
and monthly files, verifies preserved file hashes, validates the replacement
data, and atomically rebuilds the data manifest. It never generates a score,
ranking, price level, position, backtest, or report conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import update_data as updater


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_preserved_files(old_manifest: dict, replaced_code: str) -> None:
    for record in old_manifest["files"]:
        path = ROOT / record["path"]
        if path.name.startswith(f"{replaced_code}_") or path.name == "fund_snapshot.csv":
            continue
        if not path.exists():
            raise updater.UpdateError(f"待保留文件不存在：{path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise updater.UpdateError(f"待保留文件哈希不一致：{path}")


def _copy_preserved(staging: Path, run_id: str, replaced_code: str) -> tuple[list[dict], dict[str, list[dict]]]:
    files: list[dict] = []
    daily_by_code: dict[str, list[dict]] = {}
    for inst in updater.INSTRUMENTS:
        if inst["code"] == replaced_code:
            continue
        for chinese in ("日线", "周线", "月线"):
            source = DATA_DIR / f"{inst['code']}_{inst['name']}_{chinese}.csv"
            rows = _read_csv(source)
            if not rows:
                raise updater.UpdateError(f"待保留文件为空：{source}")
            for row in rows:
                row["run_id"] = run_id
            target = staging / "数据" / source.name
            updater._write_csv(target, rows)
            files.append(updater._file_record(target, rows, staging))
            if chinese == "日线":
                daily_by_code[inst["code"]] = rows
    return files, daily_by_code


def _metadata(staging: Path, run_id: str, as_of: str, new_inst: dict, old_code: str) -> tuple[Path, list[dict]]:
    old_rows = _read_csv(DATA_DIR / "fund_snapshot.csv")
    rows = []
    for row in old_rows:
        if row["code"] == old_code:
            continue
        row["run_id"] = run_id
        rows.append(row)
    rows.append({
        "code": new_inst["code"], "metadata_date": as_of,
        "nav": "", "iopv": "", "premium_discount_pct": "", "turnover_pct": "",
        "median_amount_20d": "", "aum_cny": "",
        "tracking_error_60d_pct": "", "tracking_error_252d_pct": "",
        "benchmark_code": new_inst["benchmark"]["code"], "metadata_status": "degraded",
        "missing_fields": "nav|iopv|premium_discount_pct|turnover_pct|median_amount_20d|aum_cny|tracking_error_60d_pct|tracking_error_252d_pct",
        "source": "静态官方映射；本次仅替换行情", "source_url": new_inst["official_source"], "run_id": run_id,
    })
    order = {inst["code"]: index for index, inst in enumerate(updater.INSTRUMENTS)}
    rows.sort(key=lambda row: order[row["code"]])
    target = staging / "数据" / "fund_snapshot.csv"
    updater._write_csv(target, rows, list(rows[0]))
    return target, rows


def run(new_code: str, old_code: str, as_of: date) -> dict:
    matching = [inst for inst in updater.INSTRUMENTS if inst["code"] == new_code]
    if len(matching) != 1:
        raise updater.UpdateError(f"配置中未唯一找到新标的：{new_code}")
    new_inst = matching[0]
    old_manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    _verify_preserved_files(old_manifest, old_code)

    now = datetime.now(updater.SHANGHAI)
    run_id = now.strftime("%Y%m%dT%H%M%S")
    timestamp = now.isoformat(timespec="seconds")
    staging = ROOT / f".replace_staging_{run_id}"
    lock = ROOT / ".update.lock"
    fd = None
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, run_id.encode("ascii"))
        requests, _ = updater._dependencies()
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})

        files, daily_by_code = _copy_preserved(staging, run_id, new_code)
        daily, warnings = updater._build_daily(session, new_inst, as_of, run_id, timestamp)
        weekly = updater._aggregate(daily, "weekly")
        monthly = updater._aggregate(daily, "monthly")
        for timeframe, chinese, rows in (("daily", "日线", daily), ("weekly", "周线", weekly), ("monthly", "月线", monthly)):
            warnings.extend(updater._validate(rows, new_inst, timeframe))
            target = staging / "数据" / f"{new_code}_{new_inst['name']}_{chinese}.csv"
            updater._write_csv(target, rows)
            files.append(updater._file_record(target, rows, staging))
        daily_by_code[new_code] = daily

        latest_dates = {code: rows[-1]["date"] for code, rows in daily_by_code.items()}
        if len(daily_by_code) != 8 or len(set(latest_dates.values())) != 1:
            raise updater.UpdateError(f"替换后8只ETF最新日期不一致：{latest_dates}")
        common_as_of = min(latest_dates.values())

        metadata_path, metadata_rows = _metadata(staging, run_id, common_as_of, new_inst, old_code)
        files.append(updater._file_record(metadata_path, metadata_rows, staging, date_key="metadata_date"))
        warnings = [warning for warning in old_manifest.get("warnings", []) if old_code not in warning] + warnings
        warnings.append(f"{new_code}: 基金元数据未联网补全，状态为degraded")
        manifest = {
            "schema_version": "2.0.0", "run_id": run_id, "generated_at": timestamp,
            "report_status": "degraded", "market_status": "closed", "as_of": common_as_of,
            "latest_dates": latest_dates, "quality_status": "passed_with_warnings",
            "warnings": warnings, "files": sorted(files, key=lambda row: row["path"]),
        }
        (staging / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        snapshot_rows = []
        for inst in updater.INSTRUMENTS:
            last = daily_by_code[inst["code"]][-1]
            snapshot_rows.append({
                "code": inst["code"], "name": inst["name"], "exchange": inst["exchange"],
                "date": last["date"], "raw_close": last["raw_close"], "adj_close": last["adj_close"],
                "amount": last.get("amount"), "data_status": "closed", "run_id": run_id,
            })
        updater._write_csv(staging / "最新行情快照.csv", snapshot_rows, list(snapshot_rows[0]))
        update_log = {
            "schema_version": "2.0.0", "run_id": run_id, "generated_at": timestamp,
            "as_of": common_as_of, "mode": "close", "status": "degraded", "warnings": warnings,
            "replacement": {"old_code": old_code, "new_code": new_code},
        }
        (staging / "update_log.json").write_text(json.dumps(update_log, ensure_ascii=False, indent=2), encoding="utf-8")
        updater._commit(staging, manifest, update_log)
        return {"run_id": run_id, "as_of": common_as_of, "old_code": old_code, "new_code": new_code}
    finally:
        if fd is not None:
            os.close(fd)
        if lock.exists():
            lock.unlink()
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description="只替换一个ETF的数据文件，不执行投资分析")
    parser.add_argument("--new-code", required=True)
    parser.add_argument("--old-code", required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    try:
        result = run(args.new_code, args.old_code, date.fromisoformat(args.as_of))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
