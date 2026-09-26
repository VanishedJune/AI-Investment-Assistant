# -*- coding: utf-8 -*-
"""Objective ablation study for the eight-instrument core allocation.

The variants are fixed in source before execution. Python emits aggregate
historical metrics only and never emits current selections or instructions.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from calculate_control_groups import _daily_frames, _performance, _weekly_period_returns
from calculate_features import _feature_frame, _series_value_at, _sha256
from calculate_weekly_daily_pk import (
    DAILY_POSITIVE,
    JOINT_NEGATIVE,
    ONE_WAY_COST,
    WEEKLY_POSITIVE,
    _weekly_frames,
)


ROOT = Path(__file__).resolve().parents[1]
INITIAL_CAPITAL = 100_000.0
DEVELOPMENT_WEEKS = 110
ZERO_AXIS_THRESHOLD = 0.0
MINIMUM_TRADE_GAP_WEEKS = 1


def _transition_allowed(
    week_index: int,
    last_transition_week: int | None,
    minimum_gap_weeks: int,
) -> bool:
    """Return whether an instrument may change tactical state this week."""
    if minimum_gap_weeks < 1:
        raise ValueError("minimum trade gap must be at least one week")
    return last_transition_week is None or week_index - last_transition_week >= minimum_gap_weeks


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


def _simulate(
    codes: list[str],
    daily_features: dict[str, pd.DataFrame],
    weekly_features: dict[str, pd.DataFrame],
    signal_dates: list[pd.Timestamp],
    periods: list[dict[str, float]],
    *,
    core_fraction: float,
    overlay_fraction: float,
    overlay_method: str,
    sweep_unused: bool,
    activation_threshold: float = WEEKLY_POSITIVE,
    deactivation_threshold: float = JOINT_NEGATIVE,
    minimum_trade_gap_weeks: int = MINIMUM_TRADE_GAP_WEEKS,
) -> dict[str, Any]:
    if abs(core_fraction + overlay_fraction - 1.0) > 1e-12:
        raise ValueError("core and overlay fractions must sum to one")
    active: list[str] = []
    current_assets = {code: 0.0 for code in codes}
    current_cash = 1.0
    weekly_returns: list[float] = []
    exposures: list[float] = []
    turnover_total = 0.0
    changed_weeks = 0
    state_transitions = 0
    last_transition_week: dict[str, int | None] = {code: None for code in codes}
    observed_transition_gaps: list[int] = []

    for week_index, (signal_date, period) in enumerate(zip(signal_dates, periods, strict=True)):
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
                and weekly_value < deactivation_threshold
                and daily_value < deactivation_threshold
            )
            may_transition = _transition_allowed(
                week_index,
                last_transition_week[code],
                minimum_trade_gap_weeks,
            )
            if joint_weak and may_transition:
                prior_week = last_transition_week[code]
                if prior_week is not None:
                    observed_transition_gaps.append(week_index - prior_week)
                last_transition_week[code] = week_index
                state_transitions += 1
            else:
                retained.append(code)
        additions = []
        for code in codes:
            if code in retained:
                continue
            daily_value, weekly_value = values[code]
            if (
                daily_value is not None
                and weekly_value is not None
                and weekly_value > activation_threshold
                and daily_value > activation_threshold
                and _transition_allowed(
                    week_index,
                    last_transition_week[code],
                    minimum_trade_gap_weeks,
                )
            ):
                additions.append(code)
        for code in additions:
            prior_week = last_transition_week[code]
            if prior_week is not None:
                observed_transition_gaps.append(week_index - prior_week)
            last_transition_week[code] = week_index
            state_transitions += 1
        active = retained + additions

        target = {code: core_fraction / len(codes) for code in codes}
        overlay_used = 0.0
        overlay_targets = active if active else (codes if sweep_unused else [])
        if overlay_targets:
            if not sweep_unused and len(overlay_targets) == 1:
                overlay_used = min(overlay_fraction, 0.10)
            else:
                overlay_used = overlay_fraction
            if overlay_method == "equal":
                for code in overlay_targets:
                    target[code] += overlay_used / len(overlay_targets)
            elif overlay_method == "strength":
                strengths: dict[str, float] = {}
                for code in overlay_targets:
                    daily_value, weekly_value = values[code]
                    daily_clipped = max(-3.0, min(3.0, daily_value if daily_value is not None else 0.0))
                    weekly_clipped = max(-3.0, min(3.0, weekly_value if weekly_value is not None else 0.0))
                    strengths[code] = max(0.0, weekly_clipped + 0.5 * daily_clipped)
                denominator = sum(strengths.values())
                if denominator <= 1e-12:
                    for code in overlay_targets:
                        target[code] += overlay_used / len(overlay_targets)
                else:
                    for code, strength in strengths.items():
                        target[code] += overlay_used * strength / denominator
            else:
                raise ValueError(f"unknown overlay method {overlay_method}")

        cash_target = 1.0 - sum(target.values())
        turnover = sum(abs(target[code] - current_assets[code]) for code in codes)
        turnover += abs(cash_target - current_cash)
        cost = turnover * ONE_WAY_COST
        gross_return = sum(target[code] * period[code] for code in codes)
        growth = (1.0 - cost) * (1.0 + gross_return)
        weekly_returns.append(growth - 1.0)
        exposures.append(sum(target.values()))
        turnover_total += turnover
        if turnover > 1e-12:
            changed_weeks += 1

        ending_assets = {
            code: (1.0 - cost) * target[code] * (1.0 + period[code])
            for code in codes
        }
        ending_cash = (1.0 - cost) * cash_target
        total = sum(ending_assets.values()) + ending_cash
        current_assets = {code: value / total for code, value in ending_assets.items()}
        current_cash = ending_cash / total

    full = _performance(weekly_returns)
    development = _performance(weekly_returns[:DEVELOPMENT_WEEKS])
    validation = _performance(weekly_returns[DEVELOPMENT_WEEKS:])
    return {
        "full_165": full,
        "development_110": development,
        "validation_55": validation,
        "turnover": round(turnover_total, 10),
        "changed_weeks": changed_weeks,
        "average_gross_exposure": round(statistics.fmean(exposures), 10),
        "average_cash_fraction": round(1.0 - statistics.fmean(exposures), 10),
        "tactical_state_transitions": state_transitions,
        "minimum_observed_transition_gap_weeks": (
            min(observed_transition_gaps) if observed_transition_gaps else None
        ),
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

    specifications = {
        "legacy_core80_overlay20": (0.80, 0.20, "equal", False, WEEKLY_POSITIVE, JOINT_NEGATIVE),
        "full_core80_overlay20_equal": (0.80, 0.20, "equal", True, WEEKLY_POSITIVE, JOINT_NEGATIVE),
        "full_core80_overlay20_zero_axis": (
            0.80,
            0.20,
            "equal",
            True,
            ZERO_AXIS_THRESHOLD,
            ZERO_AXIS_THRESHOLD,
        ),
        "full_core90_overlay10_equal": (0.90, 0.10, "equal", True, WEEKLY_POSITIVE, JOINT_NEGATIVE),
        "full_core95_overlay05_equal": (0.95, 0.05, "equal", True, WEEKLY_POSITIVE, JOINT_NEGATIVE),
        "full_core80_overlay20_strength": (0.80, 0.20, "strength", True, WEEKLY_POSITIVE, JOINT_NEGATIVE),
        "full_equal_weight": (1.00, 0.00, "equal", True, WEEKLY_POSITIVE, JOINT_NEGATIVE),
    }
    variants: dict[str, Any] = {}
    for name, (core, overlay, method, sweep, activation, deactivation) in specifications.items():
        variants[name] = {
            "numeric_specification": {
                "core_fraction": core,
                "overlay_fraction": overlay,
                "overlay_method": method,
                "sweep_unused_to_equal_weight": sweep,
                "primary_timeframe": "weekly",
                "confirmation_timeframe": "daily",
                "weekly_activation_slope_threshold": activation,
                "daily_confirmation_slope_threshold": activation,
                "weekly_deactivation_slope_threshold": deactivation,
                "daily_deactivation_confirmation_slope_threshold": deactivation,
                "minimum_trade_gap_weeks": MINIMUM_TRADE_GAP_WEEKS,
            },
            "metrics": _simulate(
                codes,
                daily_features,
                weekly_features,
                signal_dates,
                periods,
                core_fraction=core,
                overlay_fraction=overlay,
                overlay_method=method,
                sweep_unused=sweep,
                activation_threshold=activation,
                deactivation_threshold=deactivation,
                minimum_trade_gap_weeks=MINIMUM_TRADE_GAP_WEEKS,
            ),
        }

    payload = {
        "schema_version": "core8-optimization-v1.1.0",
        "parameter_engine": "deterministic_local_calculator",
        "data_run_id": str(manifest["run_id"]),
        "source_manifest_sha256": _sha256(manifest_path),
        "initial_capital": INITIAL_CAPITAL,
        "capital_policy": "weekly_compounded_current_equity",
        "weekly_points": len(periods),
        "development_weeks": DEVELOPMENT_WEEKS,
        "validation_weeks": len(periods) - DEVELOPMENT_WEEKS,
        "validation_start_signal_date": signal_dates[DEVELOPMENT_WEEKS].strftime("%Y-%m-%d"),
        "execution_policy": "signal_week_data_only_then_next_common_session_adj_open",
        "cost_policy": "10bp_roundtrip_with_actual_pretrade_weight_drift",
        "variants": variants,
    }
    output = ROOT / "app" / "features" / f"core8_optimization_{manifest['run_id']}.json"
    _write_atomic(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
