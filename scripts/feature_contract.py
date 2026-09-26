# -*- coding: utf-8 -*-
"""Contract validation for objective feature snapshots.

This module validates numerical evidence only. It intentionally has no report,
portfolio, ranking or trade-decision concepts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "features-v1.3.0"
PARAMETER_ENGINE = "deterministic_local_calculator"
PARAMETER_VERSION = "features-v1.3.0"
LEGACY_SCHEMA_VERSION = "features-v1.2.0"

FORBIDDEN_KEYS = {
    "decision",
    "buy",
    "sell",
    "hold",
    "ranking",
    "eligibility",
    "recommended_weight",
    "suggested_weight",
    "target_position",
    "entry_action",
    "model_gate_passed",
}
FORBIDDEN_TEXT = ("买入", "卖出", "持有", "推荐", "建议仓位", "目标仓位")

ROOT_KEYS = {
    "schema_version",
    "parameter_engine",
    "parameter_version",
    "data_run_id",
    "as_of",
    "generated_at",
    "source_manifest_sha256",
    "instrument_order",
    "instruments",
    "correlation",
    "backtest_metrics",
    "quality",
}
ROOT_KEYS_WITHOUT_DIGEST = (ROOT_KEYS - {"source_manifest_sha256"}) | {"integrity_mode"}
INSTRUMENT_KEYS = {
    "code",
    "name",
    "daily",
    "weekly",
    "volume",
    "risk",
    "relative_strength",
    "similarity",
}
LEGACY_TIMEFRAME_KEYS = {
    "as_of",
    "is_partial",
    "sample_count",
    "adj_close",
    "sma20",
    "sma60",
    "ema12",
    "ema26",
    "dif",
    "dea",
    "macd_hist",
    "dif_first_raw",
    "dif_first_normalized",
    "dif_first_display",
    "dif_second_raw",
    "dif_second_normalized",
    "dif_second_display",
    "rsi14",
    "atr14",
    "volatility_20",
    "volatility_60",
    "volatility_252",
    "max_drawdown_252",
}
TIMEFRAME_KEYS = LEGACY_TIMEFRAME_KEYS | {"sma5", "sma10"}
VOLUME_KEYS = {
    "latest_volume",
    "latest_amount",
    "volume_to_20d_median",
    "amount_to_20d_median",
    "median_amount_20d",
}
RISK_KEYS = {"volatility_percentile_60d", "drawdown_from_252d_high", "history_days"}
RELATIVE_KEYS = {"return_20d", "return_60d", "vs_equal_weight_20d", "vs_equal_weight_60d"}
SIMILARITY_KEYS = {
    "method",
    "maximum_events",
    "sample_count_4w",
    "positive_rate_4w",
    "mean_return_4w",
    "median_return_4w",
    "worst_adverse_4w",
    "sample_count_8w",
    "positive_rate_8w",
    "mean_return_8w",
    "median_return_8w",
    "worst_adverse_8w",
}
CORRELATION_KEYS = {"window_sessions", "minimum_overlap", "observations", "matrix"}
BACKTEST_KEYS = {"research_rule", "agent_decisions"}
RESEARCH_KEYS = {
    "rule_version",
    "status",
    "weekly_points",
    "net_cumulative_return",
    "sharpe",
    "max_drawdown",
    "turnover",
    "benchmark_net_cumulative_return",
    "benchmark_sharpe",
    "benchmark_max_drawdown",
    "roundtrip_cost_bps",
}
AGENT_TRACK_KEYS = {
    "status",
    "weekly_points",
    "net_cumulative_return",
    "sharpe",
    "max_drawdown",
    "roundtrip_cost_bps",
}
QUALITY_KEYS = {"instrument_count", "warnings"}


def _check_keys(value: dict[str, Any], allowed: set[str], location: str) -> list[str]:
    errors: list[str] = []
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        errors.append(f"{location}: unknown fields {unknown}")
    if missing:
        errors.append(f"{location}: missing fields {missing}")
    return errors


def _walk(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                errors.append(f"{location}: forbidden field {key}")
            errors.extend(_walk(item, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_walk(item, f"{location}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location}: non-finite number")
    elif isinstance(value, str) and any(token in value for token in FORBIDDEN_TEXT):
        errors.append(f"{location}: forbidden instruction text")
    return errors


def validate_snapshot(snapshot: dict[str, Any], expected_codes: list[str] | None = None) -> list[str]:
    root_keys = ROOT_KEYS_WITHOUT_DIGEST if snapshot.get("integrity_mode") == "not_evaluated" else ROOT_KEYS
    errors = _check_keys(snapshot, root_keys, "root")
    errors.extend(_walk(snapshot))
    schema_version = snapshot.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        expected_parameter_version = PARAMETER_VERSION
        timeframe_keys = TIMEFRAME_KEYS
    elif schema_version == LEGACY_SCHEMA_VERSION:
        expected_parameter_version = LEGACY_SCHEMA_VERSION
        timeframe_keys = LEGACY_TIMEFRAME_KEYS
    else:
        errors.append("root.schema_version mismatch")
        expected_parameter_version = PARAMETER_VERSION
        timeframe_keys = TIMEFRAME_KEYS
    if snapshot.get("parameter_engine") != PARAMETER_ENGINE:
        errors.append("root.parameter_engine mismatch")
    if snapshot.get("parameter_version") != expected_parameter_version:
        errors.append("root.parameter_version mismatch")

    order = snapshot.get("instrument_order")
    instruments = snapshot.get("instruments")
    if not isinstance(order, list) or not all(isinstance(code, str) for code in order):
        errors.append("root.instrument_order must be a string list")
        order = []
    if expected_codes is not None and order != expected_codes:
        errors.append("root.instrument_order does not match configured order")
    if not isinstance(instruments, dict):
        errors.append("root.instruments must be an object")
        instruments = {}
    if list(instruments) != order:
        errors.append("root.instruments order does not match instrument_order")

    for code in order:
        item = instruments.get(code)
        if not isinstance(item, dict):
            errors.append(f"instruments.{code}: missing object")
            continue
        errors.extend(_check_keys(item, INSTRUMENT_KEYS, f"instruments.{code}"))
        if item.get("code") != code:
            errors.append(f"instruments.{code}.code mismatch")
        for frame in ("daily", "weekly"):
            value = item.get(frame)
            if not isinstance(value, dict):
                errors.append(f"instruments.{code}.{frame}: missing object")
            else:
                errors.extend(_check_keys(value, timeframe_keys, f"instruments.{code}.{frame}"))
        for section, allowed in (
            ("volume", VOLUME_KEYS),
            ("risk", RISK_KEYS),
            ("relative_strength", RELATIVE_KEYS),
            ("similarity", SIMILARITY_KEYS),
        ):
            value = item.get(section)
            if not isinstance(value, dict):
                errors.append(f"instruments.{code}.{section}: missing object")
            else:
                errors.extend(_check_keys(value, allowed, f"instruments.{code}.{section}"))

    correlation = snapshot.get("correlation")
    if isinstance(correlation, dict):
        errors.extend(_check_keys(correlation, CORRELATION_KEYS, "correlation"))
    else:
        errors.append("root.correlation must be an object")
    backtests = snapshot.get("backtest_metrics")
    if isinstance(backtests, dict):
        errors.extend(_check_keys(backtests, BACKTEST_KEYS, "backtest_metrics"))
        if isinstance(backtests.get("research_rule"), dict):
            errors.extend(_check_keys(backtests["research_rule"], RESEARCH_KEYS, "backtest_metrics.research_rule"))
        if isinstance(backtests.get("agent_decisions"), dict):
            errors.extend(_check_keys(backtests["agent_decisions"], AGENT_TRACK_KEYS, "backtest_metrics.agent_decisions"))
    else:
        errors.append("root.backtest_metrics must be an object")
    quality = snapshot.get("quality")
    if isinstance(quality, dict):
        errors.extend(_check_keys(quality, QUALITY_KEYS, "quality"))
    else:
        errors.append("root.quality must be an object")
    return errors


def load_and_validate(path: Path, expected_codes: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    return snapshot, validate_snapshot(snapshot, expected_codes)
