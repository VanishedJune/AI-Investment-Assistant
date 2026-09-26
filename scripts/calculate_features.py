# -*- coding: utf-8 -*-
"""Calculate deterministic, non-prescriptive ETF parameters.

The output is numerical evidence only. This program never creates a report,
orders instruments, or emits portfolio/trade instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from feature_contract import (
    PARAMETER_ENGINE,
    PARAMETER_VERSION,
    SCHEMA_VERSION,
    validate_snapshot,
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_iteration.rolling.market_sessions import completed_week

SHANGHAI = ZoneInfo("Asia/Shanghai")
EPSILON = 1e-12
TIMEFRAME_FILES = {"daily": "日线", "weekly": "周线"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, digits: int = 10) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if abs(number) < 5e-14:
        number = 0.0
    return round(number, digits)


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    numeric = [
        "raw_open", "raw_high", "raw_low", "raw_close",
        "adj_open", "adj_high", "adj_low", "adj_close",
        "volume", "amount",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _rolling_mad(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    median = np.median(finite)
    return float(np.median(np.abs(finite - median)))


def _robust_normalize(series: pd.Series, window: int) -> pd.Series:
    """Normalize one objective series by its own rolling robust scale."""

    minimum = max(12, window // 2)
    mad = series.rolling(window, min_periods=minimum).apply(_rolling_mad, raw=True)
    return series / (1.4826 * mad).clip(lower=EPSILON)


def _feature_frame(frame: pd.DataFrame, mad_window: int, annualization: int) -> pd.DataFrame:
    result = frame.copy()
    close = result["adj_close"].astype(float)
    result["sma5"] = close.rolling(5, min_periods=5).mean()
    result["sma10"] = close.rolling(10, min_periods=10).mean()
    result["sma20"] = close.rolling(20, min_periods=20).mean()
    result["sma60"] = close.rolling(60, min_periods=60).mean()
    result["ema12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    result["ema26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["dif"] = result["ema12"] - result["ema26"]
    result["dea"] = result["dif"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_hist"] = 2.0 * (result["dif"] - result["dea"])
    # True discrete derivatives.  Compute both raw derivatives first, then
    # normalize each against its own history.  Never difference a normalized
    # rolling slope: its changing denominator can reverse the apparent sign.
    result["dif_first_raw"] = result["dif"].diff()
    result["dif_second_raw"] = result["dif_first_raw"].diff()
    result["dif_first_normalized"] = _robust_normalize(result["dif_first_raw"], mad_window)
    result["dif_second_normalized"] = _robust_normalize(result["dif_second_raw"], mad_window)
    result["dif_first_display"] = result["dif_first_normalized"].clip(-3.0, 3.0)
    result["dif_second_display"] = result["dif_second_normalized"].clip(-3.0, 3.0)

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi14"] = 100 - 100 / (1 + rs)
    result.loc[(avg_loss == 0) & (avg_gain > 0), "rsi14"] = 100.0
    result.loc[(avg_loss == 0) & (avg_gain == 0), "rsi14"] = 50.0

    previous = result["raw_close"].shift(1)
    true_range = pd.concat(
        [
            result["raw_high"] - result["raw_low"],
            (result["raw_high"] - previous).abs(),
            (result["raw_low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    returns = close.pct_change(fill_method=None)
    for window in (20, 60, 252):
        result[f"volatility_{window}"] = returns.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(annualization)
    rolling_high = close.rolling(252, min_periods=20).max()
    result["max_drawdown_252"] = close / rolling_high - 1.0
    return result


def _timeframe_payload(frame: pd.DataFrame, *, partial: bool) -> dict[str, Any]:
    if frame.empty:
        return {
            "as_of": None, "is_partial": partial, "sample_count": 0,
            "adj_close": None, "sma5": None, "sma10": None,
            "sma20": None, "sma60": None,
            "ema12": None, "ema26": None, "dif": None, "dea": None,
            "macd_hist": None, "dif_first_raw": None, "dif_first_normalized": None,
            "dif_first_display": None, "dif_second_raw": None,
            "dif_second_normalized": None, "dif_second_display": None,
            "rsi14": None, "atr14": None,
            "volatility_20": None, "volatility_60": None,
            "volatility_252": None, "max_drawdown_252": None,
        }
    row = frame.iloc[-1]
    return {
        "as_of": row["date"].strftime("%Y-%m-%d"),
        "is_partial": bool(partial),
        "sample_count": int(len(frame)),
        "adj_close": _finite(row.get("adj_close")),
        "sma5": _finite(row.get("sma5")),
        "sma10": _finite(row.get("sma10")),
        "sma20": _finite(row.get("sma20")),
        "sma60": _finite(row.get("sma60")),
        "ema12": _finite(row.get("ema12")),
        "ema26": _finite(row.get("ema26")),
        "dif": _finite(row.get("dif")),
        "dea": _finite(row.get("dea")),
        "macd_hist": _finite(row.get("macd_hist")),
        "dif_first_raw": _finite(row.get("dif_first_raw")),
        "dif_first_normalized": _finite(row.get("dif_first_normalized")),
        "dif_first_display": _finite(row.get("dif_first_display")),
        "dif_second_raw": _finite(row.get("dif_second_raw")),
        "dif_second_normalized": _finite(row.get("dif_second_normalized")),
        "dif_second_display": _finite(row.get("dif_second_display")),
        "rsi14": _finite(row.get("rsi14")),
        "atr14": _finite(row.get("atr14")),
        "volatility_20": _finite(row.get("volatility_20")),
        "volatility_60": _finite(row.get("volatility_60")),
        "volatility_252": _finite(row.get("volatility_252")),
        "max_drawdown_252": _finite(row.get("max_drawdown_252")),
    }


def _percentile(series: pd.Series, value: float | None) -> float | None:
    clean = series.dropna()
    if value is None or not math.isfinite(value) or clean.empty:
        return None
    return float((clean <= value).mean())


def _stats(events: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if not events:
        return {"count": 0, "positive_rate": None, "mean": None, "median": None, "worst_adverse": None}
    returns = [item[0] for item in events]
    adverse = [item[1] for item in events]
    return {
        "count": len(events),
        "positive_rate": sum(value > 0 for value in returns) / len(returns),
        "mean": statistics.fmean(returns),
        "median": statistics.median(returns),
        "worst_adverse": min(adverse),
    }


def _series_value_at(frame: pd.DataFrame, when: pd.Timestamp, column: str) -> float | None:
    eligible = frame[frame["date"] <= when]
    if eligible.empty:
        return None
    return _finite(eligible.iloc[-1][column])


def _similarity_payload(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
) -> dict[str, Any]:
    current = (
        _finite(weekly.iloc[-1]["dif_first_normalized"]) if not weekly.empty else None,
        _finite(daily.iloc[-1]["dif_first_normalized"]) if not daily.empty else None,
    )
    base = {
        "method": "nearest_weekly_daily_dif_state",
        "maximum_events": 30,
        "sample_count_4w": 0, "positive_rate_4w": None, "mean_return_4w": None,
        "median_return_4w": None, "worst_adverse_4w": None,
        "sample_count_8w": 0, "positive_rate_8w": None, "mean_return_8w": None,
        "median_return_8w": None, "worst_adverse_8w": None,
    }
    if any(value is None for value in current):
        return base
    date_to_index = {date: index for index, date in enumerate(daily["date"])}
    candidates: list[tuple[float, int, float, float, float, float]] = []
    for _, week in weekly.iloc[:-1].iterrows():
        when = week["date"]
        index = date_to_index.get(when)
        if index is None or index + 40 >= len(daily):
            continue
        values = (
            _finite(week.get("dif_first_normalized")),
            _series_value_at(daily, when, "dif_first_normalized"),
        )
        if any(value is None for value in values):
            continue
        distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(values, current)))
        start = float(daily.iloc[index]["adj_close"])
        close20 = float(daily.iloc[index + 20]["adj_close"])
        close40 = float(daily.iloc[index + 40]["adj_close"])
        adverse20 = float(daily.iloc[index + 1:index + 21]["adj_close"].min() / start - 1.0)
        adverse40 = float(daily.iloc[index + 1:index + 41]["adj_close"].min() / start - 1.0)
        candidates.append((distance, index, close20 / start - 1.0, adverse20, close40 / start - 1.0, adverse40))
    candidates.sort(key=lambda item: item[0])

    def select(cooldown: int, ret_index: int, adverse_index: int) -> list[tuple[float, float]]:
        chosen: list[tuple[float, float]] = []
        indices: list[int] = []
        for item in candidates:
            index = item[1]
            if all(abs(index - existing) >= cooldown for existing in indices):
                indices.append(index)
                chosen.append((item[ret_index], item[adverse_index]))
                if len(chosen) == 30:
                    break
        return chosen

    four = _stats(select(20, 2, 3))
    eight = _stats(select(40, 4, 5))
    return {
        "method": base["method"],
        "maximum_events": 30,
        "sample_count_4w": four["count"],
        "positive_rate_4w": _finite(four["positive_rate"]),
        "mean_return_4w": _finite(four["mean"]),
        "median_return_4w": _finite(four["median"]),
        "worst_adverse_4w": _finite(four["worst_adverse"]),
        "sample_count_8w": eight["count"],
        "positive_rate_8w": _finite(eight["positive_rate"]),
        "mean_return_8w": _finite(eight["mean"]),
        "median_return_8w": _finite(eight["median"]),
        "worst_adverse_8w": _finite(eight["worst_adverse"]),
    }


def _performance(returns: list[float]) -> tuple[float | None, float | None, float | None]:
    if not returns:
        return None, None, None
    series = pd.Series(returns, dtype=float)
    wealth = (1.0 + series).cumprod()
    cumulative = float(wealth.iloc[-1] - 1.0)
    sharpe = None
    if len(series) > 1 and float(series.std(ddof=1)) > EPSILON:
        sharpe = float(series.mean() / series.std(ddof=1) * math.sqrt(52))
    drawdown = wealth / wealth.cummax() - 1.0
    return cumulative, sharpe, float(drawdown.min())


def _research_backtest(
    codes: list[str],
    frames: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    common_dates: set[pd.Timestamp] | None = None
    for code in codes:
        dates = set(frames[code]["daily"]["date"])
        common_dates = dates if common_dates is None else common_dates & dates
    dates = sorted(common_dates or [])
    if len(dates) < 300:
        dates = []
    calendar_frame = pd.DataFrame({"date": dates})
    if not calendar_frame.empty:
        iso = calendar_frame["date"].dt.isocalendar()
        calendar_frame["year"] = iso.year
        calendar_frame["week"] = iso.week
        evaluation_dates = calendar_frame.groupby(["year", "week"])["date"].max().tolist()
    else:
        evaluation_dates = []

    portfolio_returns: list[float] = []
    benchmark_returns: list[float] = []
    previous = {code: 0.0 for code in codes}
    previous_cash = 1.0
    turnover_total = 0.0
    for index in range(len(evaluation_dates) - 2):
        signal_date = evaluation_dates[index]
        execution_date = next((date for date in dates if date > signal_date), None)
        next_signal = evaluation_dates[index + 1]
        next_execution = next((date for date in dates if date > next_signal), None)
        if execution_date is None or next_execution is None:
            continue
        selected: list[str] = []
        period_returns: dict[str, float] = {}
        valid = True
        for code in codes:
            daily = frames[code]["daily"]
            weekly = frames[code]["weekly"]
            daily_value = _series_value_at(daily, signal_date, "dif_first_normalized")
            weekly_value = _series_value_at(weekly, signal_date, "dif_first_normalized")
            start_row = daily[daily["date"] == execution_date]
            end_row = daily[daily["date"] == next_execution]
            if start_row.empty or end_row.empty:
                valid = False
                break
            start = float(start_row.iloc[0]["adj_open"])
            end = float(end_row.iloc[0]["adj_open"])
            period_returns[code] = end / start - 1.0
            if daily_value is not None and weekly_value is not None:
                if weekly_value > 0.0 and daily_value > 0.0:
                    selected.append(code)
        if not valid:
            continue
        each = min(0.20, 0.80 / len(selected)) if selected else 0.0
        weights = {code: (each if code in selected else 0.0) for code in codes}
        cash = 1.0 - sum(weights.values())
        turnover = sum(abs(weights[code] - previous[code]) for code in codes) + abs(cash - previous_cash)
        cost = turnover * 0.0005
        portfolio_returns.append(sum(weights[code] * period_returns[code] for code in codes) - cost)
        benchmark_returns.append(statistics.fmean(period_returns.values()))
        turnover_total += turnover
        previous = weights
        previous_cash = cash

    cumulative, sharpe, max_drawdown = _performance(portfolio_returns)
    benchmark_cumulative, benchmark_sharpe, benchmark_drawdown = _performance(benchmark_returns)
    points = len(portfolio_returns)
    return {
        "rule_version": "weekly_primary_daily_confirmation_zero_axis-v2.0.0",
        "status": "available" if points >= 52 else "insufficient_history",
        "weekly_points": points,
        "net_cumulative_return": _finite(cumulative),
        "sharpe": _finite(sharpe),
        "max_drawdown": _finite(max_drawdown),
        "turnover": _finite(turnover_total),
        "benchmark_net_cumulative_return": _finite(benchmark_cumulative),
        "benchmark_sharpe": _finite(benchmark_sharpe),
        "benchmark_max_drawdown": _finite(benchmark_drawdown),
        "roundtrip_cost_bps": 10,
    }


def _agent_track() -> dict[str, Any]:
    report_dir = ROOT / "app" / "reports" / "weekly"
    points = 0
    for path in sorted(report_dir.glob("????-??-??.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        evaluation = payload.get("evaluation_portfolio")
        if isinstance(evaluation, dict) and isinstance(evaluation.get("weights_pct"), dict):
            points += 1
    return {
        "status": "available" if points >= 52 else "insufficient_history",
        "weekly_points": points,
        "net_cumulative_return": None,
        "sharpe": None,
        "max_drawdown": None,
        "roundtrip_cost_bps": 10,
    }


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def calculate(manifest_path: Path, output_dir: Path, *, verify_integrity_digest: bool = False, skip_research_metrics: bool = False) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config" / "instruments.json").read_text(encoding="utf-8"))
    configured = {item["code"]: item for item in config["instruments"]}
    priority_path = ROOT / "model_iteration" / "configs" / "investment_priority.json"
    if priority_path.is_file():
        priority = json.loads(priority_path.read_text(encoding="utf-8"))["priority"]
        if [item["code"] for item in config["instruments"]] != list(priority):
            raise ValueError("investment priority and instrument configuration differ (order and membership must match)")
        instruments = [configured[code] for code in priority]
    else:
        instruments = config["instruments"]
    codes = [item["code"] for item in instruments]
    run_id = str(manifest["run_id"])
    as_of = pd.Timestamp(manifest["as_of"])
    warnings: list[str] = [str(item) for item in manifest.get("warnings", [])]

    manifest_records = {Path(record["path"]).name: record for record in manifest["files"]}
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for item in instruments:
        code = item["code"]
        frames[code] = {}
        for key, chinese in TIMEFRAME_FILES.items():
            name = f"{code}_{item['name']}_{chinese}.csv"
            path = ROOT / "数据" / name
            record = manifest_records.get(name)
            if record is None:
                raise ValueError(f"manifest missing {name}")
            if verify_integrity_digest:
                if _sha256(path) != record["sha256"]:
                    raise ValueError(f"integrity mismatch {name}")
            raw = _read_csv(path)
            if set(raw["run_id"].astype(str)) != {run_id}:
                raise ValueError(f"run_id mismatch {name}")
            if key == "daily":
                frames[code][key] = _feature_frame(raw, 60, 252)
            else:
                frames[code][key] = _feature_frame(raw, 52, 52)

    closes = pd.concat(
        [frames[code]["daily"].set_index("date")["adj_close"].rename(code) for code in codes],
        axis=1,
        sort=True,
    )
    basket_returns: dict[int, float | None] = {}
    for window in (20, 60):
        values = closes.iloc[-1] / closes.shift(window).iloc[-1] - 1.0 if len(closes) > window else pd.Series(dtype=float)
        basket_returns[window] = float(values.dropna().mean()) if not values.dropna().empty else None

    output_instruments: dict[str, Any] = {}
    # A pre-Friday aggregate is final only when every remaining weekday is a
    # documented exchange closure. Never infer closure from absent source rows.
    weekly_is_partial = not completed_week(as_of.date(), datetime.now(SHANGHAI).date())
    for item in instruments:
        code = item["code"]
        daily = frames[code]["daily"]
        weekly = frames[code]["weekly"]
        raw_daily = daily
        last = raw_daily.iloc[-1]
        volume_median = raw_daily["volume"].tail(20).median()
        amount_median = raw_daily["amount"].tail(20).median()
        close = raw_daily["adj_close"]
        return20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) > 20 else None
        return60 = float(close.iloc[-1] / close.iloc[-61] - 1.0) if len(close) > 60 else None
        current_vol = _finite(daily.iloc[-1]["volatility_60"])
        output_instruments[code] = {
            "code": code,
            "name": item["name"],
            "daily": _timeframe_payload(daily, partial=False),
            "weekly": _timeframe_payload(weekly, partial=weekly_is_partial),
            "volume": {
                "latest_volume": _finite(last.get("volume")),
                "latest_amount": _finite(last.get("amount")),
                "volume_to_20d_median": _finite(last.get("volume") / volume_median if volume_median and not pd.isna(volume_median) else None),
                "amount_to_20d_median": _finite(last.get("amount") / amount_median if amount_median and not pd.isna(amount_median) else None),
                "median_amount_20d": _finite(amount_median),
            },
            "risk": {
                "volatility_percentile_60d": _finite(_percentile(daily["volatility_60"], current_vol)),
                "drawdown_from_252d_high": _finite(last.get("max_drawdown_252")),
                "history_days": int(len(daily)),
            },
            "relative_strength": {
                "return_20d": _finite(return20),
                "return_60d": _finite(return60),
                "vs_equal_weight_20d": _finite(return20 - basket_returns[20] if return20 is not None and basket_returns[20] is not None else None),
                "vs_equal_weight_60d": _finite(return60 - basket_returns[60] if return60 is not None and basket_returns[60] is not None else None),
            },
            "similarity": _similarity_payload(daily, weekly),
        }

    returns = closes.pct_change(fill_method=None).dropna(how="all")
    recent = returns.tail(252)
    matrix: dict[str, dict[str, float | None]] = {}
    observations: dict[str, dict[str, int]] = {}
    for left in codes:
        matrix[left] = {}
        observations[left] = {}
        for right in codes:
            if left == right:
                paired = recent[[left]].dropna()
                observations[left][right] = int(len(paired))
                matrix[left][right] = 1.0 if len(paired) >= 120 else None
                continue
            paired = recent[[left, right]].dropna()
            observations[left][right] = int(len(paired))
            matrix[left][right] = _finite(paired[left].corr(paired[right])) if len(paired) >= 120 else None

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "parameter_engine": PARAMETER_ENGINE,
        "parameter_version": PARAMETER_VERSION,
        "data_run_id": run_id,
        "as_of": as_of.strftime("%Y-%m-%d"),
        "generated_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "integrity_mode": "digest_verified" if verify_integrity_digest else "not_evaluated",
        "instrument_order": codes,
        "instruments": output_instruments,
        "correlation": {
            "window_sessions": 252,
            "minimum_overlap": 120,
            "observations": observations,
            "matrix": matrix,
        },
        "backtest_metrics": {
            "research_rule": ({"status": "not_recomputed_instrument_migration", "weekly_points": 0,
                "rule_version": "not_executed", "net_cumulative_return": None, "sharpe": None,
                "max_drawdown": None, "turnover": None, "roundtrip_cost_bps": None,
                "benchmark_max_drawdown": None, "benchmark_net_cumulative_return": None,
                "benchmark_sharpe": None} if skip_research_metrics else _research_backtest(codes, frames)),
            "agent_decisions": ({"status": "not_recomputed_instrument_migration", "weekly_points": 0,
                "net_cumulative_return": None, "sharpe": None, "max_drawdown": None,
                "roundtrip_cost_bps": None} if skip_research_metrics else _agent_track()),
        },
        "quality": {"instrument_count": len(codes), "warnings": warnings},
    }
    errors = validate_snapshot(snapshot, codes)
    if errors:
        raise ValueError("; ".join(errors))
    content = (json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    run_path = output_dir / f"{run_id}.json"
    latest_path = output_dir / "latest.json"
    _write_atomic(run_path, content)
    _write_atomic(latest_path, content)
    if run_path.read_bytes() != latest_path.read_bytes():
        raise ValueError("feature snapshots differ after write")
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="计算周、日确定性客观参数")
    parser.add_argument("--manifest", default="data_manifest.json")
    parser.add_argument("--output-dir", default="app/features")
    parser.add_argument("--skip-research-metrics", action="store_true", help="只计算客观指标，不运行研究回测或Agent收益统计")
    parser.add_argument("--no-integrity-digest", action="store_true", default=True, help="默认已禁用文件指纹；仍执行特征契约与数据一致性校验")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if "reports" in {part.lower() for part in output_dir.parts}:
        print("ERROR: parameter output cannot target a report directory")
        return 1
    try:
        snapshot = calculate(manifest_path, output_dir, verify_integrity_digest=not args.no_integrity_digest, skip_research_metrics=args.skip_research_metrics)
    except Exception as exc:  # contract boundary: fail closed
        print(f"ERROR: {exc}")
        return 1
    print(
        f"参数已冻结：run_id={snapshot['data_run_id']} · "
        f"instruments={len(snapshot['instrument_order'])} · "
        f"weekly_points={snapshot['backtest_metrics']['research_rule']['weekly_points']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
