# -*- coding: utf-8 -*-
"""Backtest the Agent-fixed weekly/daily two-timeframe research protocol.

The Agent fixes the protocol before execution.  Python only replays historical
numbers and emits aggregate metrics; it does not emit current selections,
rankings, trade instructions or recommended weights.
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

from calculate_control_groups import (
    _daily_frames,
    _fixed_exposure,
    _performance,
    _solve_exposure,
    _weekly_period_returns,
)
from calculate_features import _feature_frame, _read_csv, _series_value_at, _sha256


ROOT = Path(__file__).resolve().parents[1]
INITIAL_CAPITAL = 100_000.0
WEEKLY_POSITIVE = 0.5
DAILY_POSITIVE = 0.5
JOINT_NEGATIVE = -0.5
BASE_MAX_INSTRUMENTS = 4
CASH20_MIN_INSTRUMENTS = 4
CASH20_MAX_INSTRUMENTS = 5
PER_INSTRUMENT = 0.20
MINIMUM_TOTAL = 0.80
MAXIMUM_TOTAL = 1.00
BASE_FRACTION_EACH = 0.10
TACTICAL_POOL_MAX = 0.20
TACTICAL_INCREMENT_CAP = 0.10
ONE_WAY_COST = 0.0005


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


def _weekly_frames(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    records = {Path(item["path"]).name: item for item in manifest["files"]}
    result: dict[str, pd.DataFrame] = {}
    for item in config["instruments"]:
        filename = f"{item['code']}_{item['name']}_周线.csv"
        path = ROOT / "数据" / filename
        record = records.get(filename)
        if record is None:
            raise ValueError(f"manifest missing {filename}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"hash mismatch {filename}")
        raw = _read_csv(path)
        if set(raw["run_id"].astype(str)) != {str(manifest["run_id"])}:
            raise ValueError(f"run_id mismatch {filename}")
        result[item["code"]] = _feature_frame(raw, mad_window=52, annualization=52)
    return result


def _simulate(
    codes: list[str],
    daily_features: dict[str, pd.DataFrame],
    weekly_features: dict[str, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    periods: list[dict[str, float]],
) -> dict[str, float | int]:
    active: list[str] = []
    previous = {code: 0.0 for code in codes}
    previous_cash = 1.0
    returns: list[float] = []
    exposures: list[float] = []
    turnover_total = 0.0
    changed_weeks = 0

    for signal_date, period in zip(signal_dates, periods, strict=True):
        values: dict[str, tuple[float | None, float | None]] = {}
        for code in codes:
            daily_value = _series_value_at(daily_features[code], signal_date, "dif_first_normalized")
            weekly_value = _series_value_at(weekly_features[code], signal_date, "dif_first_normalized")
            values[code] = (daily_value, weekly_value)

        retained: list[str] = []
        for code in active:
            daily_value, weekly_value = values[code]
            joint_weak = (
                daily_value is not None
                and weekly_value is not None
                and daily_value < JOINT_NEGATIVE
                and weekly_value < JOINT_NEGATIVE
            )
            if not joint_weak:
                retained.append(code)

        available_slots = BASE_MAX_INSTRUMENTS - len(retained)
        candidates: list[tuple[float, float, str]] = []
        if available_slots > 0:
            for code in codes:
                if code in retained:
                    continue
                daily_value, weekly_value = values[code]
                if (
                    daily_value is not None
                    and weekly_value is not None
                    and daily_value > DAILY_POSITIVE
                    and weekly_value > WEEKLY_POSITIVE
                ):
                    candidates.append((weekly_value, daily_value, code))
        candidates.sort(reverse=True)
        active = retained + [code for _, _, code in candidates[:available_slots]]

        weights = {code: (PER_INSTRUMENT if code in active else 0.0) for code in codes}
        cash = 1.0 - sum(weights.values())
        turnover = sum(abs(weights[code] - previous[code]) for code in codes) + abs(cash - previous_cash)
        cost = turnover * ONE_WAY_COST
        returns.append(sum(weights[code] * period[code] for code in codes) - cost)
        exposures.append(sum(weights.values()))
        turnover_total += turnover
        if turnover > 1e-12:
            changed_weeks += 1
        previous = weights
        previous_cash = cash

    metrics = _performance(returns)
    return {
        **metrics,
        "weekly_points": len(returns),
        "turnover": round(turnover_total, 10),
        "changed_weeks": changed_weeks,
        "average_gross_exposure": round(statistics.fmean(exposures), 10),
        "average_cash_fraction": round(1.0 - statistics.fmean(exposures), 10),
        "fully_invested_weeks": sum(abs(value - 0.80) < 1e-12 for value in exposures),
        "zero_exposure_weeks": sum(value < 1e-12 for value in exposures),
    }


def _simulate_cash20(
    codes: list[str],
    daily_features: dict[str, pd.DataFrame],
    weekly_features: dict[str, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    periods: list[dict[str, float]],
) -> dict[str, float | int]:
    """Replay the Agent-fixed two-timeframe protocol with an 80% exposure floor."""
    active: list[str] = []
    previous = {code: 0.0 for code in codes}
    previous_cash = 1.0
    returns: list[float] = []
    exposures: list[float] = []
    turnover_total = 0.0
    changed_weeks = 0

    for signal_date, period in zip(signal_dates, periods, strict=True):
        values: dict[str, tuple[float | None, float | None]] = {}
        for code in codes:
            daily_value = _series_value_at(daily_features[code], signal_date, "dif_first_normalized")
            weekly_value = _series_value_at(weekly_features[code], signal_date, "dif_first_normalized")
            values[code] = (daily_value, weekly_value)

        retained: list[str] = []
        for code in active:
            daily_value, weekly_value = values[code]
            joint_weak = (
                daily_value is not None
                and weekly_value is not None
                and daily_value < JOINT_NEGATIVE
                and weekly_value < JOINT_NEGATIVE
            )
            if not joint_weak:
                retained.append(code)

        available_slots = CASH20_MAX_INSTRUMENTS - len(retained)
        finite_candidates: list[tuple[float, float, str]] = []
        for code in codes:
            if code in retained:
                continue
            daily_value, weekly_value = values[code]
            if daily_value is not None and weekly_value is not None:
                finite_candidates.append((weekly_value, daily_value, code))
        finite_candidates.sort(reverse=True)

        confirmed = [
            item for item in finite_candidates
            if item[0] > WEEKLY_POSITIVE and item[1] > DAILY_POSITIVE
        ]
        fallback = [item for item in finite_candidates if item not in confirmed]
        additions = confirmed[:available_slots]
        remaining_minimum = max(0, CASH20_MIN_INSTRUMENTS - len(retained) - len(additions))
        additions += fallback[:remaining_minimum]
        active = retained + [code for _, _, code in additions]
        if not CASH20_MIN_INSTRUMENTS <= len(active) <= CASH20_MAX_INSTRUMENTS:
            raise ValueError(f"unable to maintain 4-5 instruments on {signal_date:%Y-%m-%d}")

        weights = {code: (PER_INSTRUMENT if code in active else 0.0) for code in codes}
        exposure = sum(weights.values())
        cash = 1.0 - exposure
        if exposure + 1e-12 < MINIMUM_TOTAL or exposure > MAXIMUM_TOTAL + 1e-12:
            raise ValueError(f"exposure range violated on {signal_date:%Y-%m-%d}")
        if cash < -1e-12 or cash > 0.20 + 1e-12:
            raise ValueError(f"cash cap violated on {signal_date:%Y-%m-%d}")
        turnover = sum(abs(weights[code] - previous[code]) for code in codes) + abs(cash - previous_cash)
        cost = turnover * ONE_WAY_COST
        returns.append(sum(weights[code] * period[code] for code in codes) - cost)
        exposures.append(exposure)
        turnover_total += turnover
        if turnover > 1e-12:
            changed_weeks += 1
        previous = weights
        previous_cash = cash

    metrics = _performance(returns)
    return {
        **metrics,
        "weekly_points": len(returns),
        "turnover": round(turnover_total, 10),
        "changed_weeks": changed_weeks,
        "average_gross_exposure": round(statistics.fmean(exposures), 10),
        "average_cash_fraction": round(1.0 - statistics.fmean(exposures), 10),
        "fully_invested_weeks": sum(abs(value - 1.00) < 1e-12 for value in exposures),
        "minimum_exposure_weeks": sum(abs(value - 0.80) < 1e-12 for value in exposures),
        "zero_exposure_weeks": sum(value < 1e-12 for value in exposures),
    }


def _simulate_all8(
    codes: list[str],
    daily_features: dict[str, pd.DataFrame],
    weekly_features: dict[str, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    periods: list[dict[str, float]],
) -> dict[str, float | int]:
    """Replay the all-eight core plus weekly/daily tactical-overlay protocol."""
    if len(codes) != 8:
        raise ValueError(f"expected 8 instruments, got {len(codes)}")
    tactical_active: list[str] = []
    previous = {code: 0.0 for code in codes}
    previous_cash = 1.0
    returns: list[float] = []
    exposures: list[float] = []
    tactical_counts: list[int] = []
    turnover_total = 0.0
    changed_weeks = 0

    for signal_date, period in zip(signal_dates, periods, strict=True):
        values: dict[str, tuple[float | None, float | None]] = {}
        for code in codes:
            daily_value = _series_value_at(daily_features[code], signal_date, "dif_first_normalized")
            weekly_value = _series_value_at(weekly_features[code], signal_date, "dif_first_normalized")
            values[code] = (daily_value, weekly_value)

        retained: list[str] = []
        for code in tactical_active:
            daily_value, weekly_value = values[code]
            joint_weak = (
                daily_value is not None
                and weekly_value is not None
                and daily_value < JOINT_NEGATIVE
                and weekly_value < JOINT_NEGATIVE
            )
            if not joint_weak:
                retained.append(code)

        additions: list[str] = []
        for code in codes:
            if code in retained:
                continue
            daily_value, weekly_value = values[code]
            if (
                daily_value is not None
                and weekly_value is not None
                and daily_value > DAILY_POSITIVE
                and weekly_value > WEEKLY_POSITIVE
            ):
                additions.append(code)
        tactical_active = retained + additions

        tactical_count = len(tactical_active)
        if tactical_count == 0:
            tactical_each = 0.0
        elif tactical_count == 1:
            tactical_each = TACTICAL_INCREMENT_CAP
        else:
            tactical_each = min(TACTICAL_INCREMENT_CAP, TACTICAL_POOL_MAX / tactical_count)
        weights = {
            code: BASE_FRACTION_EACH + (tactical_each if code in tactical_active else 0.0)
            for code in codes
        }
        exposure = sum(weights.values())
        cash = 1.0 - exposure
        if exposure + 1e-12 < MINIMUM_TOTAL or exposure > MAXIMUM_TOTAL + 1e-12:
            raise ValueError(f"exposure range violated on {signal_date:%Y-%m-%d}")
        if cash < -1e-12 or cash > 0.20 + 1e-12:
            raise ValueError(f"cash cap violated on {signal_date:%Y-%m-%d}")
        if any(weight > PER_INSTRUMENT + 1e-12 for weight in weights.values()):
            raise ValueError(f"instrument cap violated on {signal_date:%Y-%m-%d}")

        turnover = sum(abs(weights[code] - previous[code]) for code in codes) + abs(cash - previous_cash)
        cost = turnover * ONE_WAY_COST
        returns.append(sum(weights[code] * period[code] for code in codes) - cost)
        exposures.append(exposure)
        tactical_counts.append(tactical_count)
        turnover_total += turnover
        if turnover > 1e-12:
            changed_weeks += 1
        previous = weights
        previous_cash = cash

    metrics = _performance(returns)
    return {
        **metrics,
        "weekly_points": len(returns),
        "turnover": round(turnover_total, 10),
        "changed_weeks": changed_weeks,
        "average_gross_exposure": round(statistics.fmean(exposures), 10),
        "average_cash_fraction": round(1.0 - statistics.fmean(exposures), 10),
        "average_tactical_instruments": round(statistics.fmean(tactical_counts), 10),
        "fully_invested_weeks": sum(abs(value - 1.00) < 1e-12 for value in exposures),
        "ninety_percent_weeks": sum(abs(value - 0.90) < 1e-12 for value in exposures),
        "minimum_exposure_weeks": sum(abs(value - 0.80) < 1e-12 for value in exposures),
        "zero_exposure_weeks": 0,
    }


def _simulate_unbounded(
    codes: list[str],
    daily_features: dict[str, pd.DataFrame],
    weekly_features: dict[str, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    periods: list[dict[str, float]],
) -> dict[str, float | int]:
    """Replay an unrestricted-count, unrestricted-single-weight protocol."""
    active: list[str] = []
    previous = {code: 0.0 for code in codes}
    previous_cash = 1.0
    returns: list[float] = []
    exposures: list[float] = []
    active_counts: list[int] = []
    maximum_weights: list[float] = []
    turnover_total = 0.0
    changed_weeks = 0

    for signal_date, period in zip(signal_dates, periods, strict=True):
        values: dict[str, tuple[float | None, float | None]] = {}
        for code in codes:
            daily_value = _series_value_at(daily_features[code], signal_date, "dif_first_normalized")
            weekly_value = _series_value_at(weekly_features[code], signal_date, "dif_first_normalized")
            values[code] = (daily_value, weekly_value)

        retained: list[str] = []
        for code in active:
            daily_value, weekly_value = values[code]
            joint_weak = (
                daily_value is not None
                and weekly_value is not None
                and daily_value < JOINT_NEGATIVE
                and weekly_value < JOINT_NEGATIVE
            )
            if not joint_weak:
                retained.append(code)
        additions: list[str] = []
        for code in codes:
            if code in retained:
                continue
            daily_value, weekly_value = values[code]
            if (
                daily_value is not None
                and weekly_value is not None
                and daily_value > DAILY_POSITIVE
                and weekly_value > WEEKLY_POSITIVE
            ):
                additions.append(code)
        active = retained + additions

        def strength(code: str) -> float:
            daily_value, weekly_value = values[code]
            daily_clipped = max(-3.0, min(3.0, daily_value if daily_value is not None else -3.0))
            weekly_clipped = max(-3.0, min(3.0, weekly_value if weekly_value is not None else -3.0))
            return weekly_clipped + 0.5 * daily_clipped

        weights = {code: 0.0 for code in codes}
        convictions = {code: max(0.0, strength(code)) for code in active}
        total_conviction = sum(convictions.values())
        if total_conviction > 0:
            # A single instrument can reach 100% only at the maximum displayed
            # weekly/day strength (3 + 0.5*3 = 4.5). Multiple confirmed
            # instruments can jointly lift total exposure to 100%.
            exposure = min(1.0, total_conviction / 4.5)
            weights.update({
                code: exposure * conviction / total_conviction
                for code, conviction in convictions.items()
            })
            cash = 1.0 - exposure
        else:
            exposure = 0.0
            cash = 1.0

        exposure = sum(weights.values())
        if exposure < -1e-12 or exposure > MAXIMUM_TOTAL + 1e-12:
            raise ValueError(f"exposure range violated on {signal_date:%Y-%m-%d}")
        if cash < -1e-12 or cash > 1.0 + 1e-12:
            raise ValueError(f"cash range violated on {signal_date:%Y-%m-%d}")

        turnover = sum(abs(weights[code] - previous[code]) for code in codes) + abs(cash - previous_cash)
        cost = turnover * ONE_WAY_COST
        returns.append(sum(weights[code] * period[code] for code in codes) - cost)
        exposures.append(exposure)
        active_counts.append(len(active) if active else 1)
        maximum_weights.append(max(weights.values()))
        turnover_total += turnover
        if turnover > 1e-12:
            changed_weeks += 1
        previous = weights
        previous_cash = cash

    metrics = _performance(returns)
    return {
        **metrics,
        "weekly_points": len(returns),
        "turnover": round(turnover_total, 10),
        "changed_weeks": changed_weeks,
        "average_gross_exposure": round(statistics.fmean(exposures), 10),
        "average_cash_fraction": round(1.0 - statistics.fmean(exposures), 10),
        "average_active_instruments": round(statistics.fmean(active_counts), 10),
        "average_maximum_instrument_fraction": round(statistics.fmean(maximum_weights), 10),
        "maximum_observed_instrument_fraction": round(max(maximum_weights), 10),
        "fully_invested_weeks": sum(abs(value - 1.00) < 1e-12 for value in exposures),
        "minimum_exposure_weeks": sum(abs(value - 0.80) < 1e-12 for value in exposures),
        "zero_exposure_weeks": 0,
    }


def main() -> int:
    config = json.loads((ROOT / "config" / "instruments.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    codes, daily_raw = _daily_frames(config, manifest)
    signal_dates, periods = _weekly_period_returns(codes, daily_raw)
    if len(periods) != 165:
        raise ValueError(f"expected 165 complete weeks, got {len(periods)}")

    daily_features = {
        code: _feature_frame(frame, mad_window=60, annualization=252)
        for code, frame in daily_raw.items()
    }
    weekly_features = _weekly_frames(config, manifest)
    prior_unconstrained_metrics = _simulate(codes, daily_features, weekly_features, signal_dates, periods)
    prior_limited_metrics = _simulate_cash20(codes, daily_features, weekly_features, signal_dates, periods)
    prior_all8_core_metrics = _simulate_all8(codes, daily_features, weekly_features, signal_dates, periods)
    metrics = _simulate_unbounded(codes, daily_features, weekly_features, signal_dates, periods)
    average_exposure = float(metrics["average_gross_exposure"])
    comparison_controls = {
        "same_average_exposure_equal_weight": _fixed_exposure(
            codes, periods, average_exposure, apply_rebalance_cost=True
        ),
        "full_equal_weight_costed": _fixed_exposure(
            codes, periods, 1.0, apply_rebalance_cost=True
        ),
        "same_return_equal_weight": _solve_exposure(
            codes, periods, float(metrics["cumulative_return"]), "cumulative_return"
        ),
    }
    payload = {
        "schema_version": "weekly-daily-pk-v1.4.0",
        "parameter_engine": "deterministic_local_calculator",
        "data_run_id": str(manifest["run_id"]),
        "source_manifest_sha256": _sha256(manifest_path),
        "initial_capital": INITIAL_CAPITAL,
        "capital_policy": "weekly_compounded_current_equity",
        "first_signal_date": signal_dates[0].strftime("%Y-%m-%d"),
        "last_signal_date": signal_dates[-1].strftime("%Y-%m-%d"),
        "execution_policy": "signal_week_data_only_then_next_common_session_adj_open",
        "cost_policy": "10bp_roundtrip",
        "rule_version": "agent_weekly_daily_unbounded-v1.4.0",
        "numeric_protocol": {
            "weekly_positive_threshold": WEEKLY_POSITIVE,
            "daily_positive_threshold": DAILY_POSITIVE,
            "joint_negative_threshold": JOINT_NEGATIVE,
            "universe_size": len(codes),
            "allocation_method": "positive_clipped_weekly_plus_half_daily_proportional",
            "exposure_method": "min_one_sum_positive_strength_divided_by_4_5",
            "holding_count_limit_applied": False,
            "single_instrument_max_fraction": 1.00,
            "minimum_total_fraction": 0.00,
            "maximum_total_fraction": MAXIMUM_TOTAL,
            "minimum_cash_fraction": 0.00,
            "maximum_cash_fraction": 1.00,
            "correlation_exclusion_applied": False,
        },
        "metrics": metrics,
        "prior_all8_core_metrics": prior_all8_core_metrics,
        "prior_limited_metrics": prior_limited_metrics,
        "prior_unconstrained_metrics": prior_unconstrained_metrics,
        "comparison_controls": comparison_controls,
    }
    output = ROOT / "app" / "features" / f"weekly_daily_pk_{manifest['run_id']}.json"
    _write_atomic(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
