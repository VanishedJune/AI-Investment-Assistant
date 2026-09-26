# -*- coding: utf-8 -*-
"""Calculate passive control-group metrics for the frozen 165-week study.

This script only produces historical numerical evidence.  It does not emit
current ETF selections, rankings, eligibility, trade instructions or position
recommendations.  Final interpretation remains the Agent's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EPSILON = 1e-12
ONE_WAY_COST = 0.0005  # 10bp round trip
INITIAL_CAPITAL = 100_000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
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


def _performance(returns: list[float]) -> dict[str, float]:
    series = pd.Series(returns, dtype=float)
    wealth = (1.0 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    standard_deviation = float(series.std(ddof=1))
    sharpe = float(series.mean() / standard_deviation * math.sqrt(52)) if standard_deviation > EPSILON else 0.0
    cumulative = float(wealth.iloc[-1] - 1.0)
    return {
        "ending_value": round(INITIAL_CAPITAL * (1.0 + cumulative), 2),
        "cumulative_return": round(cumulative, 10),
        "max_drawdown": round(float(drawdown.min()), 10),
        "sharpe": round(sharpe, 10),
    }


def _daily_frames(config: dict[str, Any], manifest: dict[str, Any]) -> tuple[list[str], dict[str, pd.DataFrame]]:
    codes = [item["code"] for item in config["instruments"]]
    records = {Path(item["path"]).name: item for item in manifest["files"]}
    frames: dict[str, pd.DataFrame] = {}
    for item in config["instruments"]:
        filename = f"{item['code']}_{item['name']}_日线.csv"
        path = ROOT / "数据" / filename
        manifest_record = records.get(filename)
        if manifest_record is None:
            raise ValueError(f"manifest missing {filename}")
        if _sha256(path) != manifest_record["sha256"]:
            raise ValueError(f"hash mismatch {filename}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if set(frame["run_id"].astype(str)) != {str(manifest["run_id"])}:
            raise ValueError(f"run_id mismatch {filename}")
        frame["date"] = pd.to_datetime(frame["date"])
        frames[item["code"]] = frame.sort_values("date").reset_index(drop=True)
    return codes, frames


def _weekly_period_returns(codes: list[str], frames: dict[str, pd.DataFrame]) -> tuple[list[pd.Timestamp], list[dict[str, float]]]:
    common_dates: set[pd.Timestamp] | None = None
    for code in codes:
        dates = set(frames[code]["date"])
        common_dates = dates if common_dates is None else common_dates & dates
    dates = sorted(common_dates or [])
    calendar = pd.DataFrame({"date": dates})
    iso = calendar["date"].dt.isocalendar()
    calendar["year"] = iso.year
    calendar["week"] = iso.week
    evaluation_dates = calendar.groupby(["year", "week"])["date"].max().tolist()

    signal_dates: list[pd.Timestamp] = []
    weekly_returns: list[dict[str, float]] = []
    for index in range(len(evaluation_dates) - 2):
        signal_date = evaluation_dates[index]
        execution_date = next((date for date in dates if date > signal_date), None)
        next_signal = evaluation_dates[index + 1]
        next_execution = next((date for date in dates if date > next_signal), None)
        if execution_date is None or next_execution is None:
            continue
        period: dict[str, float] = {}
        for code in codes:
            frame = frames[code]
            start = frame.loc[frame["date"] == execution_date, "adj_open"]
            end = frame.loc[frame["date"] == next_execution, "adj_open"]
            if start.empty or end.empty:
                period = {}
                break
            period[code] = float(end.iloc[0] / start.iloc[0] - 1.0)
        if period:
            signal_dates.append(signal_date)
            weekly_returns.append(period)
    return signal_dates, weekly_returns


def _fixed_exposure(
    codes: list[str],
    periods: list[dict[str, float]],
    exposure: float,
    *,
    apply_rebalance_cost: bool,
) -> dict[str, float]:
    target_assets = {code: exposure / len(codes) for code in codes}
    target_cash = 1.0 - exposure
    current_assets = {code: 0.0 for code in codes}
    current_cash = 1.0
    net_returns: list[float] = []
    turnover_total = 0.0

    for period in periods:
        turnover = sum(abs(target_assets[code] - current_assets[code]) for code in codes)
        turnover += abs(target_cash - current_cash)
        cost = turnover * ONE_WAY_COST if apply_rebalance_cost else 0.0
        gross_return = sum(target_assets[code] * period[code] for code in codes)
        growth = (1.0 - cost) * (1.0 + gross_return)
        net_returns.append(growth - 1.0)
        turnover_total += turnover

        ending_assets = {
            code: (1.0 - cost) * target_assets[code] * (1.0 + period[code])
            for code in codes
        }
        ending_cash = (1.0 - cost) * target_cash
        total = sum(ending_assets.values()) + ending_cash
        current_assets = {code: value / total for code, value in ending_assets.items()}
        current_cash = ending_cash / total

    result = _performance(net_returns)
    result["gross_exposure"] = round(exposure, 10)
    result["cash_fraction"] = round(1.0 - exposure, 10)
    result["turnover"] = round(turnover_total, 10)
    return result


def _solve_exposure(
    codes: list[str],
    periods: list[dict[str, float]],
    target: float,
    metric: str,
) -> dict[str, float]:
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2.0
        result = _fixed_exposure(codes, periods, middle, apply_rebalance_cost=True)
        value = abs(result[metric]) if metric == "max_drawdown" else result[metric]
        if value < target:
            low = middle
        else:
            high = middle
    return _fixed_exposure(codes, periods, (low + high) / 2.0, apply_rebalance_cost=True)


def main() -> int:
    config = json.loads((ROOT / "config" / "instruments.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    codes, frames = _daily_frames(config, manifest)
    signal_dates, periods = _weekly_period_returns(codes, frames)
    if len(periods) != 165:
        raise ValueError(f"expected 165 complete weeks, got {len(periods)}")

    uncosted = _fixed_exposure(codes, periods, 1.0, apply_rebalance_cost=False)
    if abs(uncosted["cumulative_return"] - 0.3853) > 0.0001:
        raise ValueError(f"benchmark reconciliation failed: {uncosted['cumulative_return']}")

    groups = {
        "equal_weight_uncosted_reconciliation": uncosted,
        "equal_weight_costed": _fixed_exposure(codes, periods, 1.0, apply_rebalance_cost=True),
        "equal_weight_80pct_costed": _fixed_exposure(codes, periods, 0.8, apply_rebalance_cost=True),
        "equal_weight_50pct_costed": _fixed_exposure(codes, periods, 0.5, apply_rebalance_cost=True),
        "drawdown_matched_to_agent_costed": _solve_exposure(codes, periods, 0.1085, "max_drawdown"),
        "return_matched_to_agent_costed": _solve_exposure(codes, periods, 0.1559866, "cumulative_return"),
    }
    payload = {
        "schema_version": "control-groups-v1.0.0",
        "parameter_engine": "deterministic_local_calculator",
        "data_run_id": str(manifest["run_id"]),
        "source_manifest_sha256": _sha256(manifest_path),
        "initial_capital": INITIAL_CAPITAL,
        "capital_policy": "weekly_compounded_current_equity",
        "weekly_points": len(periods),
        "first_signal_date": signal_dates[0].strftime("%Y-%m-%d"),
        "last_signal_date": signal_dates[-1].strftime("%Y-%m-%d"),
        "execution_policy": "signal_week_data_only_then_next_common_session_adj_open",
        "cost_policy": "10bp_roundtrip_with_weekly_target_rebalance",
        "cash_return": 0.0,
        "groups": groups,
    }
    output = ROOT / "app" / "features" / f"control_groups_{manifest['run_id']}.json"
    _write_atomic(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
