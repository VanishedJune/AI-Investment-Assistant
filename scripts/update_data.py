# -*- coding: utf-8 -*-
"""Transactional V2 market-data updater.

Usage:
    python scripts/update_data.py --mode close|draft [--as-of YYYY-MM-DD] [--force-full]

The instrument and benchmark mapping is read-only configuration. Formal data is
committed only after all eight instruments and the manifest have been generated
and validated in a same-volume staging directory. Draft data stays outside the
formal path. Report analysis and HTML generation are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time as time_module
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "数据"
CONFIG = json.loads((ROOT / "config" / "instruments.json").read_text(encoding="utf-8"))
INSTRUMENTS = CONFIG["instruments"]
SHANGHAI = ZoneInfo("Asia/Shanghai")
FIELDS = [
    "code", "name", "exchange", "date",
    "raw_open", "raw_high", "raw_low", "raw_close",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_factor",
    "volume", "amount", "source", "source_timestamp", "run_id",
]


class UpdateError(RuntimeError):
    pass


STALE_LOCK_SECONDS = 12 * 60 * 60
STALE_STAGING_SECONDS = 24 * 60 * 60


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_matches_data_dir(manifest: dict) -> bool:
    if not DATA_DIR.is_dir() or not isinstance(manifest.get("files"), list):
        return False
    for record in manifest["files"]:
        rel = Path(str(record.get("path") or ""))
        if not rel.parts or rel.parts[0] != DATA_DIR.name:
            return False
        path = (ROOT / rel).resolve()
        if DATA_DIR.resolve() not in path.parents:
            return False
        if not path.is_file():
            return False
        if _sha256_file(path) != record.get("sha256"):
            return False
    return True


def _recover_interrupted_data_swap(*, use_integrity_digests: bool = False) -> None:
    """Repair the small crash window inside ``_commit``.

    ``_commit`` swaps the whole ``数据`` directory before replacing the manifest.
    A process kill between those two ``os.replace`` calls leaves either the new
    data without a matching manifest (roll forward/back by hash) or a missing
    ``数据`` directory (restore the preserved old directory).
    """

    old_dirs = sorted(ROOT.glob(".previous_data_*"))
    if not old_dirs:
        return
    if len(old_dirs) != 1:
        raise UpdateError("存在多个未恢复的数据回滚目录，请人工检查: " + ", ".join(path.name for path in old_dirs))
    old_data = old_dirs[0]
    if not use_integrity_digests and DATA_DIR.exists():
        raise UpdateError("检测到未完成的数据目录交换；当前禁用文件摘要，无法自动判定新旧目录，请人工检查")
    manifest_path = ROOT / "data_manifest.json"
    if not DATA_DIR.exists():
        os.replace(old_data, DATA_DIR)
        return
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UpdateError("data_manifest.json无法解析，无法自动恢复上次数据更新") from exc
    if manifest is not None and _manifest_matches_data_dir(manifest):
        # Manifest already references the new data; the preserved old directory
        # is obsolete and can be removed.
        shutil.rmtree(old_data)
        return
    # Manifest still references the preserved directory: restore the old data
    # and quarantine the uncommitted new directory.
    failed_data = ROOT / f".failed_data_{manifest.get('run_id', 'unknown') if manifest else 'unknown'}"
    if failed_data.exists():
        raise UpdateError(f"存在未清理的失败数据目录: {failed_data.name}")
    os.replace(DATA_DIR, failed_data)
    try:
        os.replace(old_data, DATA_DIR)
    except Exception:
        if failed_data.exists() and not DATA_DIR.exists():
            os.replace(failed_data, DATA_DIR)
        raise
    if failed_data.exists():
        shutil.rmtree(failed_data)


def _cleanup_stale_update_artifacts() -> None:
    cutoff = time_module.time() - STALE_STAGING_SECONDS
    for pattern in (".update_staging_*", ".failed_data_*"):
        for path in ROOT.glob(pattern):
            try:
                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
            except OSError:
                continue


def _acquire_update_lock(lock: Path, run_id: str) -> int:
    if lock.exists():
        try:
            age = time_module.time() - lock.stat().st_mtime
        except OSError as exc:
            raise UpdateError(f"无法读取更新锁: {lock}") from exc
        if age < 0 or age < STALE_LOCK_SECONDS:
            raise UpdateError(f"已有行情更新正在运行或锁文件未过期: {lock}")
        # Age alone cannot establish that a process stopped owning this lock.
        raise UpdateError(f"更新锁已过期但归属未确认，保留锁并停止；确认原进程退出后人工处理: {lock}")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise UpdateError(f"已有行情更新正在运行: {lock}") from exc
    payload = json.dumps({"pid": os.getpid(), "run_id": run_id, "created_at": datetime.now(SHANGHAI).isoformat(timespec="seconds")}, ensure_ascii=False)
    try:
        os.write(fd, payload.encode("utf-8"))
    except BaseException:
        os.close(fd)
        lock.unlink(missing_ok=True)
        raise
    return fd


def _number(value, default=None):
    if value in (None, "", "-", "--"):
        return default
    if isinstance(value, str):
        value = value.replace(",", "").replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dependencies():
    try:
        import requests  # type: ignore
    except ImportError:
        class _StdlibResponse:
            def __init__(self, payload: bytes):
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return json.loads(self._payload.decode("utf-8"))

        class _StdlibSession:
            def __init__(self):
                self.headers: dict[str, str] = {}

            def get(self, url: str, *, params: dict, timeout: int):
                query = urllib.parse.urlencode(params)
                request = urllib.request.Request(
                    f"{url}?{query}",
                    headers=self.headers,
                    method="GET",
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return _StdlibResponse(response.read())

        class _StdlibRequests:
            Session = _StdlibSession

        requests = _StdlibRequests()
    try:
        import akshare  # type: ignore
    except ImportError:
        akshare = None
    return requests, akshare


def _retry_json(session, url: str, params: dict, tries: int = 3) -> dict:
    last = None
    for attempt in range(tries):
        try:
            response = session.get(url, params=params, timeout=25)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time_module.sleep(attempt + 1)
    raise UpdateError(f"行情请求失败：{url}，{last}")


def _fetch_tencent(session, symbol: str, start: date, end: date, *, adjusted: bool) -> dict[str, list]:
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get" if adjusted else "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
    merged: dict[str, list] = {}
    for year in range(start.year, end.year + 1):
        first = max(start, date(year, 1, 1))
        last = min(end, date(year, 12, 31))
        if first > last:
            continue
        suffix = ",qfq" if adjusted else ""
        param = f"{symbol},day,{first.isoformat()},{last.isoformat()},640{suffix}"
        payload = _retry_json(session, url, {"param": param})
        node = ((payload.get("data") or {}).get(symbol) or {})
        rows = node.get("qfqday") or node.get("day") or []
        for row in rows:
            if len(row) >= 6 and str(row[0]) <= end.isoformat():
                merged[str(row[0])] = list(row)
        time_module.sleep(0.08)

    if adjusted and merged:
        # 兼容腾讯按日期区间请求的前复权缓存滞后与精度损失：
        # 区间请求可能不含最新交易日（原始K线已含，qfqday 仍停留在上一日），
        # 或对最新交易日返回四舍五入后的低精度行。改为对尾部逐日单日请求，
        # 覆盖最近几个自然日并补取缺失日期（最多尝试 12 个自然日）。
        # 单日请求与区间请求同一量纲，仅覆盖尾部，不触碰更早历史行。
        have = max(date.fromisoformat(day) for day in merged)
        cursor = max(have + timedelta(days=1), end - timedelta(days=2))
        attempts = 0
        while cursor <= end and attempts < 12:
            attempts += 1
            if cursor.weekday() < 5:
                param = f"{symbol},day,{cursor.isoformat()},{cursor.isoformat()},640,qfq"
                payload = _retry_json(session, url, {"param": param})
                node = ((payload.get("data") or {}).get(symbol) or {})
                for row in node.get("qfqday") or node.get("day") or []:
                    if len(row) >= 6 and str(row[0]) == cursor.isoformat():
                        merged[str(row[0])] = list(row)
                time_module.sleep(0.08)
            cursor += timedelta(days=1)
    return merged


def _existing_amounts(inst: dict) -> dict[str, float | None]:
    path = DATA_DIR / f"{inst['code']}_{inst['name']}_日线.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows=list(csv.DictReader(handle))
    result={row['date']:_number(row.get('amount')) for row in rows}
    supplement=ROOT/'辅助数据/成交额补充.json'
    if supplement.exists():
        from app.turnover_evidence import amounts_for_rows
        evidence=json.loads(supplement.read_text(encoding='utf-8'))
        for day,amount in amounts_for_rows(evidence['records'],rows,inst['code']).items():
            if result.get(day) is None:result[day]=amount
    return result


def _existing_daily(inst: dict) -> list[dict]:
    path = DATA_DIR / f"{inst['code']}_{inst['name']}_日线.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _akshare_amounts(
    akshare,
    inst: dict,
    start: date,
    end: date,
    warnings: list[str],
) -> dict[str, float]:
    """Use AkShare daily history to fill missing turnover values.

    This is best-effort enrichment: missing values are only filled, never
    overwriting audited non-null amounts, and every failure degrades to a
    warning rather than failing the whole market-data update.
    """

    if akshare is None:
        return {}
    last_error = None
    for attempt in range(2):
        try:
            frame = akshare.fund_etf_hist_em(
                symbol=inst["code"],
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="",
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                time_module.sleep(1)
            continue
        amounts: dict[str, float] = {}
        for _, row in frame.iterrows():
            raw_date = row.get("日期")
            if hasattr(raw_date, "strftime"):
                date_text = raw_date.strftime("%Y-%m-%d")
            else:
                date_text = str(raw_date or "")[:10]
            amount = _number(row.get("成交额"))
            if date_text and amount is not None:
                amounts[date_text] = amount
        if amounts:
            return amounts
        last_error = RuntimeError(f"{inst['code']} AkShare成交额为空")
    warnings.append(f"{inst['code']} AkShare成交额补充失败：{last_error}")
    return {}


def _eastmoney_amounts(session, inst: dict, start: date, end: date, warnings: list[str]) -> dict[str, float]:
    """Use the public Eastmoney history endpoint as a dependency-free amount fallback."""

    market = "0" if str(inst["symbol"]).lower().startswith("sz") else "1"
    params = {
        "secid": f"{market}.{inst['code']}",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
        "klt": "101",
        "fqt": "0",
        "beg": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
    }
    try:
        payload = _retry_json(session, "https://push2his.eastmoney.com/api/qt/stock/kline/get", params)
        klines = ((payload.get("data") or {}).get("klines") or [])
        amounts: dict[str, float] = {}
        for line in klines:
            fields = str(line).split(",")
            if len(fields) < 7:
                continue
            amount = _number(fields[6])
            if amount is not None:
                amounts[fields[0]] = amount
        if amounts:
            return amounts
        warnings.append(f"{inst['code']} 东方财富成交额补充为空")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{inst['code']} 东方财富成交额补充失败：{exc}")
    return {}


def _build_daily(
    session,
    inst: dict,
    as_of: date,
    run_id: str,
    timestamp: str,
    *,
    force_full: bool = False,
    akshare=None,
) -> tuple[list[dict], list[str]]:
    existing = [] if force_full else _existing_daily(inst)
    inception = date.fromisoformat(inst["inception_date"])
    # Preserve the audited history and refetch a bounded overlap.  Providers
    # periodically rebase their entire qfq series; rebuilding all history can
    # therefore manufacture false jumps at an ETF's listing date.
    start = max(inception, date.fromisoformat(existing[-1]["date"]) - timedelta(days=45)) if existing else inception
    raw = _fetch_tencent(session, inst["symbol"], start, as_of, adjusted=False)
    qfq = _fetch_tencent(session, inst["symbol"], start, as_of, adjusted=True)
    if not raw:
        raise UpdateError(f"{inst['code']} 未取得腾讯原始日K")
    warnings: list[str] = []
    amount_by_date = _existing_amounts(inst)
    if existing:
        missing_existing = [
            date.fromisoformat(day)
            for day, amount in amount_by_date.items()
            if amount is None and day <= existing[-1]["date"]
        ]
        amount_fetch_start = min(missing_existing) if missing_existing else date.fromisoformat(existing[-1]["date"])
    else:
        amount_fetch_start = inception
    if amount_fetch_start <= as_of:
        supplemental_amounts = _akshare_amounts(akshare, inst, amount_fetch_start, as_of, warnings)
        if not supplemental_amounts:
            supplemental_amounts = _eastmoney_amounts(session, inst, amount_fetch_start, as_of, warnings)
        for date_text, amount in supplemental_amounts.items():
            if amount_by_date.get(date_text) is None:
                amount_by_date[date_text] = amount
    existing_by_date = {row["date"]: row for row in existing}
    last_existing_date = existing[-1]["date"] if existing else None
    qfq_scale = 1.0
    if existing:
        anchors = [day for day in qfq if day in existing_by_date and day <= last_existing_date]
        if not anchors:
            raise UpdateError(f"{inst['code']} 增量更新缺少复权重叠锚点")
        anchor = max(anchors)
        provider_adj_close = float(qfq[anchor][2])
        if provider_adj_close <= 0:
            raise UpdateError(f"{inst['code']} {anchor} 复权锚点非法")
        qfq_scale = float(existing_by_date[anchor]["adj_close"]) / provider_adj_close

        # Raw history is authoritative for execution.  A material revision in
        # the overlap is not silently accepted; the whole update fails closed.
        tick = float(inst.get("price_tick", 0.001))
        for day in sorted(set(raw).intersection(existing_by_date)):
            current = raw[day]
            prior = existing_by_date[day]
            observed = (float(current[1]), float(current[3]), float(current[4]), float(current[2]))
            preserved = tuple(float(prior[key]) for key in ("raw_open", "raw_high", "raw_low", "raw_close"))
            if any(abs(a - b) > tick + 1e-9 for a, b in zip(observed, preserved)):
                raise UpdateError(f"{inst['code']} {day} 原始OHLC与已验收历史冲突")

    last_factor = None
    rows: list[dict] = []
    for prior in existing:
        preserved = dict(prior)
        for key in (
            "raw_open", "raw_high", "raw_low", "raw_close",
            "adj_open", "adj_high", "adj_low", "adj_close", "adj_factor", "volume",
        ):
            preserved[key] = float(preserved[key])
        preserved["amount"] = amount_by_date.get(preserved["date"], _number(preserved.get("amount")))
        preserved["run_id"] = run_id
        rows.append(preserved)
    for date_text in sorted(raw):
        if last_existing_date is not None and date_text <= last_existing_date:
            continue
        values = raw[date_text]
        raw_open, raw_close, raw_high, raw_low = map(float, (values[1], values[2], values[3], values[4]))
        adjusted_values = qfq.get(date_text)
        if adjusted_values:
            adjusted_close = float(adjusted_values[2]) * qfq_scale
            factor = adjusted_close / raw_close
            last_factor = factor
        elif last_factor is not None:
            factor = last_factor
            adjusted_close = raw_close * factor
            warnings.append(f"{inst['code']} {date_text}: 前复权缺失，沿用最近已知复权因子")
        else:
            raise UpdateError(f"{inst['code']} {date_text}: 无法建立复权因子")
        rows.append({
            "code": inst["code"], "name": inst["name"], "exchange": inst["exchange"], "date": date_text,
            "raw_open": raw_open, "raw_high": raw_high, "raw_low": raw_low, "raw_close": raw_close,
            "adj_open": raw_open * factor, "adj_high": raw_high * factor,
            "adj_low": raw_low * factor, "adj_close": adjusted_close, "adj_factor": factor,
            "volume": float(values[5]) * 100.0,
            "amount": amount_by_date.get(date_text),
            "source": "腾讯日K/前复权日K", "source_timestamp": timestamp, "run_id": run_id,
        })
    return rows, warnings


def _aggregate(rows: list[dict], timeframe: str) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        trade_date = date.fromisoformat(row["date"])
        key = f"{trade_date.isocalendar().year}-W{trade_date.isocalendar().week:02d}" if timeframe == "weekly" else trade_date.strftime("%Y-%m")
        buckets.setdefault(key, []).append(row)
    output: list[dict] = []
    for values in buckets.values():
        first, last = values[0], values[-1]
        amounts = [float(row["amount"]) for row in values if row.get("amount") is not None]
        output.append({
            "code": last["code"], "name": last["name"], "exchange": last["exchange"], "date": last["date"],
            "raw_open": first["raw_open"], "raw_high": max(row["raw_high"] for row in values),
            "raw_low": min(row["raw_low"] for row in values), "raw_close": last["raw_close"],
            "adj_open": first["adj_open"], "adj_high": max(row["adj_high"] for row in values),
            "adj_low": min(row["adj_low"] for row in values), "adj_close": last["adj_close"],
            "adj_factor": last["adj_factor"], "volume": sum(row["volume"] for row in values),
            "amount": sum(amounts) if len(amounts)==len(values) else None, "source": last["source"],
            "source_timestamp": last["source_timestamp"], "run_id": last["run_id"],
        })
    return output


def _validate(rows: list[dict], inst: dict, timeframe: str) -> list[str]:
    if not rows:
        raise UpdateError(f"{inst['code']} {timeframe}为空")
    warnings: list[str] = []
    previous_date = ""
    previous_adjusted = None
    previous_raw = None
    threshold = (float(inst["price_limit_pct"]) + 1.0) / 100.0
    for row in rows:
        required = ("raw_open", "raw_high", "raw_low", "raw_close", "adj_open",
                    "adj_high", "adj_low", "adj_close", "adj_factor", "volume")
        for key in (*required, "amount"):
            if key == "amount" and row.get(key) is None:
                continue
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise UpdateError(f"{inst['code']} {row.get('date')} {key}缺失或不是数值") from exc
            if not math.isfinite(value):
                raise UpdateError(f"{inst['code']} {row.get('date')} {key}不是有限数值")
        if row["date"] <= previous_date:
            raise UpdateError(f"{inst['code']} {timeframe}日期重复或未递增：{row['date']}")
        previous_date = row["date"]
        ro, rh, rl, rc = (float(row[key]) for key in ("raw_open", "raw_high", "raw_low", "raw_close"))
        ao, ah, al, ac = (float(row[key]) for key in ("adj_open", "adj_high", "adj_low", "adj_close"))
        if min(ro, rh, rl, rc, ao, ah, al, ac) <= 0:
            raise UpdateError(f"{inst['code']} {row['date']}价格非正")
        tolerance = max(rh, ah) * 1e-9
        if rl > min(ro, rc) + tolerance or rh < max(ro, rc) - tolerance or al > min(ao, ac) + tolerance or ah < max(ao, ac) - tolerance:
            raise UpdateError(f"{inst['code']} {row['date']} OHLC非法")
        if row["volume"] < 0 or (row.get("amount") is not None and row["amount"] < 0):
            raise UpdateError(f"{inst['code']} {row['date']}量额为负")
        if abs(float(row["adj_factor"]) - ac / rc) > max(1e-8, abs(ac / rc) * 1e-7):
            raise UpdateError(f"{inst['code']} {row['date']}复权因子不一致")
        if timeframe == "daily" and previous_adjusted is not None:
            daily_return = ac / previous_adjusted - 1.0
            raw_return = rc / previous_raw - 1.0
            if abs(daily_return) > threshold + 1e-9:
                if abs(raw_return) > threshold + 1e-9:
                    raise UpdateError(
                        f"{inst['code']} {row['date']}复权收益{daily_return:.2%}、"
                        f"原始收益{raw_return:.2%}同时超过价格限制门禁"
                    )
                warnings.append(
                    f"{inst['code']} {row['date']}: 复权收益{daily_return:.2%}超过原始价格限制，"
                    f"原始收益{raw_return:.2%}仍合规，按分红复权跳变保留"
                )
            elif abs(raw_return) > threshold + 1e-9:
                warnings.append(
                    f"{inst['code']} {row['date']}: 原始收益{raw_return:.2%}超过价格限制，"
                    f"复权收益{daily_return:.2%}仍合规，按份额拆分保留"
                )
        previous_raw = rc
        previous_adjusted = ac
    if any(row.get("amount") is None for row in rows):
        warnings.append(f"{inst['code']} {timeframe}: 成交额存在缺失")
    return warnings


def _write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_log(run_id: str, payload: dict) -> None:
    """Append an immutable per-run log without disturbing the formal artifacts."""

    path = ROOT / "logs" / "updates" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _file_record(
    path: Path,
    rows: list[dict],
    staging: Path,
    *,
    date_key: str = "date",
    include_integrity_digest: bool = True,
) -> dict:
    record = {
        "path": path.relative_to(staging).as_posix(),
        "rows": len(rows),
        "start": rows[0][date_key], "end": rows[-1][date_key], "run_id": rows[-1]["run_id"],
    }
    if include_integrity_digest:
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def _spot_map(akshare, warnings: list[str]) -> dict[str, dict]:
    if akshare is None:
        warnings.append("AkShare不可用：基金元数据与跟踪误差降级")
        return {}
    last_error = None
    for attempt in range(3):
        try:
            frame = akshare.fund_etf_spot_em()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 2:
                time_module.sleep(attempt + 1)
            continue
        result = {}
        for _, row in frame.iterrows():
            code = str(row.get("代码") or row.get("基金代码") or "").zfill(6)
            if code:
                result[code] = row.to_dict()
        return result
    warnings.append(f"ETF元数据获取失败（已重试3次）：{last_error}")
    return {}


def _metadata_rows(akshare, run_id: str, as_of: str, warnings: list[str]) -> list[dict]:
    spot = _spot_map(akshare, warnings)
    result = []
    for inst in INSTRUMENTS:
        row = spot.get(inst["code"], {})
        nav = _number(row.get("单位净值") or row.get("基金净值"))
        iopv = _number(row.get("IOPV实时估值") or row.get("IOPV"))
        premium = _number(row.get("折溢价率") or row.get("基金折价率"))
        turnover = _number(row.get("换手率"))
        primary_values = {
            "nav": nav,
            "iopv": iopv,
            "premium_discount_pct": premium,
            "turnover_pct": turnover,
        }
        values = {
            **primary_values,
            # Derived liquidity and tracking statistics are intentionally blank.
            # They belong to the Agent's frozen analysis artifact, not this downloader.
            "median_amount_20d": None, "aum_cny": None,
            "tracking_error_60d_pct": None, "tracking_error_252d_pct": None,
        }
        missing = [key for key, value in primary_values.items() if value is None]
        result.append({
            "code": inst["code"], "metadata_date": as_of, **values,
            "benchmark_code": inst["benchmark"]["code"],
            "metadata_status": "complete" if not missing else "degraded",
            "missing_fields": "|".join(missing), "source": "AkShare公开接口+静态官方映射" if akshare else "unavailable",
            "source_url": inst["official_source"], "run_id": run_id,
        })
    return result


def _inject(name: str, requested: str | None) -> None:
    if requested == name:
        raise UpdateError(f"测试注入失败阶段：{name}")


def _commit(staging: Path, manifest: dict, update_log: dict) -> None:
    rollback = staging / "_rollback"
    rollback.mkdir()
    old_data = ROOT / f".previous_data_{manifest['run_id']}"
    if old_data.exists():
        raise UpdateError(f"存在未清理的回滚目录：{old_data}")
    file_targets = [
        ROOT / "data_manifest.json", ROOT / "update_log.json", ROOT / "最新行情快照.csv",
    ]
    backups: dict[Path, Path] = {}
    data_swapped = False
    original_moved = False
    try:
        for target in file_targets:
            if target.exists():
                backup = rollback / (str(len(backups)) + ".bak")
                shutil.copy2(target, backup)
                backups[target] = backup
        os.replace(DATA_DIR, old_data)
        original_moved = True
        os.replace(staging / "数据", DATA_DIR)
        data_swapped = True
        replacements = {
            ROOT / "最新行情快照.csv": staging / "最新行情快照.csv",
            ROOT / "update_log.json": staging / "update_log.json",
        }
        for target, source in replacements.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        # Manifest is the final commit marker.
        os.replace(staging / "data_manifest.json", ROOT / "data_manifest.json")
    except Exception:
        if original_moved:
            failed_data = ROOT / f".failed_data_{manifest['run_id']}"
            if data_swapped and DATA_DIR.exists():
                os.replace(DATA_DIR, failed_data)
            if old_data.exists():
                os.replace(old_data, DATA_DIR)
            if failed_data.exists():
                shutil.rmtree(failed_data)
        for target in file_targets:
            backup = backups.get(target)
            if backup and backup.exists():
                shutil.copy2(backup, target)
            elif target.exists() and target not in backups:
                target.unlink()
        raise
    if old_data.exists():
        shutil.rmtree(old_data)


def run(args: argparse.Namespace) -> dict:
    now = datetime.now(SHANGHAI)
    as_of = date.fromisoformat(args.as_of) if args.as_of else now.date()
    if args.mode == "close" and as_of >= now.date() and now.time() < time(15, 30):
        raise UpdateError("正式更新仅允许在上海时间15:30后执行；盘中请使用 --mode draft")
    run_id = now.strftime("%Y%m%dT%H%M%S")
    staging = ROOT / f".update_staging_{run_id}"
    lock = ROOT / ".update.lock"
    if staging.resolve().parent != ROOT.resolve():
        raise UpdateError("暂存目录不在项目同盘根目录")
    fd = None
    moved_to_draft = False
    try:
        fd = _acquire_update_lock(lock, run_id)
        _recover_interrupted_data_swap(use_integrity_digests=not args.no_integrity_digest)
        _cleanup_stale_update_artifacts()
        requests, akshare = _dependencies()
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
        warnings: list[str] = []
        files: list[dict] = []
        latest_dates: dict[str, str] = {}
        daily_by_code: dict[str, list[dict]] = {}
        timestamp = now.isoformat(timespec="seconds")
        _inject("download", args.fail_stage)
        for inst in INSTRUMENTS:
            daily, instrument_warnings = _build_daily(
                session, inst, as_of, run_id, timestamp, force_full=args.force_full, akshare=akshare
            )
            weekly = _aggregate(daily, "weekly")
            monthly = _aggregate(daily, "monthly")
            warnings.extend(instrument_warnings)
            daily_by_code[inst["code"]] = daily
            latest_dates[inst["code"]] = daily[-1]["date"]
            for timeframe, chinese, rows in (("daily", "日线", daily), ("weekly", "周线", weekly), ("monthly", "月线", monthly)):
                warnings.extend(_validate(rows, inst, timeframe))
                path = staging / "数据" / f"{inst['code']}_{inst['name']}_{chinese}.csv"
                _write_csv(path, rows)
                files.append(_file_record(path, rows, staging, include_integrity_digest=not args.no_integrity_digest))
        _inject("validate", args.fail_stage)
        if len(daily_by_code) != 8 or len(set(latest_dates.values())) != 1:
            raise UpdateError(f"8只ETF最新日期不一致：{latest_dates}")
        common_as_of = min(latest_dates.values())
        metadata = _metadata_rows(akshare, run_id, common_as_of, warnings)
        metadata_fields = list(metadata[0])
        metadata_path = staging / "数据" / "fund_snapshot.csv"
        _write_csv(metadata_path, metadata, metadata_fields)
        files.append(_file_record(
            metadata_path,
            metadata,
            staging,
            date_key="metadata_date",
            include_integrity_digest=not args.no_integrity_digest,
        ))
        degraded = sum(row["metadata_status"] != "complete" for row in metadata)
        manifest = {
            "schema_version": "2.0.0", "run_id": run_id, "generated_at": timestamp,
            "report_status": "draft" if args.mode == "draft" else "degraded" if degraded else "ready",
            "market_status": "draft" if args.mode == "draft" else "closed",
            "as_of": common_as_of, "latest_dates": latest_dates,
            "quality_status": "passed_with_warnings" if warnings else "passed",
            "integrity_mode": "not_evaluated" if args.no_integrity_digest else "digest_verified",
            "warnings": warnings, "files": files,
        }
        (staging / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        snapshot_rows = []
        for inst in INSTRUMENTS:
            last = daily_by_code[inst["code"]][-1]
            snapshot_rows.append({
                "code": inst["code"], "name": inst["name"], "exchange": inst["exchange"],
                "date": last["date"], "raw_close": last["raw_close"], "adj_close": last["adj_close"],
                "amount": last["amount"], "data_status": "draft" if args.mode == "draft" else "closed", "run_id": run_id,
            })
        _write_csv(staging / "最新行情快照.csv", snapshot_rows, list(snapshot_rows[0]))
        update_log = {
            "schema_version": "2.0.0", "run_id": run_id, "generated_at": timestamp,
            "as_of": common_as_of, "mode": args.mode, "status": manifest["report_status"], "warnings": warnings,
        }
        (staging / "update_log.json").write_text(json.dumps(update_log, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            _write_run_log(run_id, update_log)
        except OSError as exc:
            warnings.append(f"运行审计日志写入失败: {exc}")

        if args.mode == "draft":
            draft_root = ROOT / "drafts" / run_id
            draft_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, draft_root)
            moved_to_draft = True
            return {"mode": "draft", "run_id": run_id, "as_of": common_as_of, "output": str(draft_root), "next_step": "由Agent读取草稿数据并输出冻结报告"}
        _inject("commit", args.fail_stage)
        _commit(staging, manifest, update_log)
        return {"mode": "close", "run_id": run_id, "as_of": common_as_of, "warnings": len(warnings), "next_step": "由Agent读取正式数据并输出冻结报告"}
    finally:
        if fd is not None:
            os.close(fd)
            if lock.exists():
                lock.unlink()
            if staging.exists() and not moved_to_draft and staging.resolve().parent == ROOT.resolve():
                shutil.rmtree(staging)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Investment Assistant V2 数据更新")
    parser.add_argument("--mode", choices=("close", "draft"), required=True, help="close=正式收盘数据；draft=隔离的盘中数据草稿")
    parser.add_argument("--as-of", help="截止日期 YYYY-MM-DD；默认上海当前日期")
    parser.add_argument("--force-full", action="store_true", help="强制全量拉取（V2当前始终以全量校验为准）")
    parser.add_argument("--no-integrity-digest", action="store_true", default=True, help="默认已禁用文件指纹；仍执行结构、日期与行情逻辑校验")
    parser.add_argument("--fail-stage", choices=("download", "validate", "commit"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
