# -*- coding: utf-8 -*-
"""021528基金净值的独立回放、冠军评估和最新直接分析。

该适配器复用现有159915冠军参数、8周预测期限、PIT状态桶、预测晋级门禁
和固定候选语法。输入只来自021528真实公布净值；基金没有OHLCV，因此成交量、
盘中开高低价及ETF流动性字段保持不可用，不参与候选评估。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from rolling.forecast import build_forecast, compute_bucket_stats  # noqa: E402
from rolling.spec import compute_position  # noqa: E402
from rolling.market_sessions import completed_week  # noqa: E402
from scripts.features import rsi, slope_frame  # noqa: E402


CODE = "021528"
NAME = "财通成长优选混合C"
NAV_PATH = PROJECT / "基金数据" / f"{CODE}_{NAME}_净值历史.csv"
CONFIG_PATH = ROOT / "configs" / f"fund_{CODE}.json"
WORKSPACE = ROOT / f"fund_{CODE}"
STATE_PATH = WORKSPACE / "weekly_rolling" / "state.json"
ITERATION_PATH = WORKSPACE / "iterations" / f"ITERATION_001_{CODE}.json"
ANALYSIS_PATH = PROJECT / "app" / "features" / f"fund_{CODE}_latest.json"
SYNTAX_PATH = ROOT / "etf_159915" / "sanitized_workspace" / "spec_syntax.json"
ETF_CONFIG_PATH = ROOT / "configs" / "etf_159915.json"

UNAVAILABLE_FEATURES = {"atr_pct", "rel_strength", "d_vol_ma5", "d_vol_chg", "w_vol_ma5"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_nav() -> pd.DataFrame:
    frame = pd.read_csv(NAV_PATH, encoding="utf-8-sig", dtype={"code": str})
    required = {"code", "name", "date", "unit_nav", "cum_nav", "source", "source_timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"021528净值缺少字段: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["unit_nav"] = pd.to_numeric(frame["unit_nav"], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if frame.empty or (frame["unit_nav"] <= 0).any():
        raise RuntimeError("021528净值为空或存在非正值")
    if not frame["date"].is_monotonic_increasing:
        raise RuntimeError("021528净值日期没有严格升序")
    return frame


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _indicator(close: pd.Series, annualization: int) -> pd.DataFrame:
    sl = slope_frame(close)
    dif = _ema(close, 12) - _ema(close, 26)
    dea = _ema(dif, 9)
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["dif"] = dif
    out["dea"] = dea
    out["macd_hist"] = 2.0 * (dif - dea)
    out["slope"] = sl["slope_norm"]
    out["dif1"] = sl["dif1"]
    out["dif2"] = sl["dif2"]
    out["rsi14"] = rsi(close, 14)
    out["vol20"] = close.pct_change().rolling(20).std(ddof=1) * math.sqrt(annualization) * 100.0
    out["mom5"] = close / close.shift(5) - 1.0
    out["mom20"] = close / close.shift(20) - 1.0
    out["mom60"] = close / close.shift(60) - 1.0
    out["dd20"] = close / close.rolling(20).max() - 1.0
    for window in (5, 10, 20, 60):
        out[f"ma{window}"] = close.rolling(window).mean()
    out["ma20_slope"] = out["ma20"].pct_change(5 if annualization == 252 else 1)
    out["low20"] = close.rolling(20).min()
    out["high20"] = close.rolling(20).max()
    return out


def _expanding_percentile(series: pd.Series, min_periods: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for index, value in enumerate(values):
        history = values[:index]
        history = history[np.isfinite(history)]
        if index >= min_periods and np.isfinite(value) and len(history):
            result[index] = float(np.mean(history <= value))
    return pd.Series(result, index=series.index)


def build_frames(nav: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = nav[["date", "unit_nav"]].rename(columns={"unit_nav": "nav"}).copy()
    daily_ind = _indicator(daily["nav"], 252)
    daily = pd.concat([daily, daily_ind], axis=1)

    periods = daily["date"].dt.to_period("W-FRI")
    weekly = daily.groupby(periods, sort=True).tail(1)[["date", "nav"]].reset_index(drop=True)
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    weekly = weekly[weekly["date"].map(
        lambda value: completed_week(value.date(), today.date())
    )].reset_index(drop=True)
    weekly_ind = _indicator(weekly["nav"], 52)
    weekly = pd.concat([weekly, weekly_ind], axis=1)

    anchors = weekly[["date", "nav"]].rename(columns={"date": "signal", "nav": "anchor_close"}).copy()
    nav_by_date = daily.set_index("date")["nav"]
    dates = daily["date"].to_numpy()
    exec_dates: list[pd.Timestamp | None] = []
    exec_navs: list[float] = []
    for signal in anchors["signal"]:
        later = dates[dates > np.datetime64(signal)]
        if len(later):
            execution = pd.Timestamp(later[0])
            exec_dates.append(execution)
            exec_navs.append(float(nav_by_date.loc[execution]))
        else:
            exec_dates.append(None)
            exec_navs.append(np.nan)
    anchors["exec"] = exec_dates
    anchors["exec_nav"] = exec_navs
    closes = anchors["anchor_close"].to_numpy(dtype=float)
    for horizon in (1, 2, 4, 8):
        forward = np.full(len(anchors), np.nan)
        if len(anchors) > horizon:
            forward[:-horizon] = closes[horizon:] / closes[:-horizon] - 1.0
        anchors[f"fwd{horizon}"] = forward
    return daily, weekly, anchors, build_features(daily, weekly, anchors)


def build_features(daily: pd.DataFrame, weekly: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    daily_index = daily.set_index("date")
    weekly_index = weekly.set_index("date")
    rows: list[dict] = []
    for _, anchor in anchors.iterrows():
        signal = anchor["signal"]
        d = daily_index.loc[:signal].iloc[-1]
        w = weekly_index.loc[:signal].iloc[-1]
        close = float(d["nav"])
        rows.append(
            {
                "d_slope": float(d["slope"]), "d_dif1": float(d["dif1"]), "d_dif2": float(d["dif2"]),
                "w_slope": float(w["slope"]), "w_dif1": float(w["dif1"]), "w_dif2": float(w["dif2"]),
                "rsi14": float(d["rsi14"]), "atr_pct": np.nan, "vol20": float(d["vol20"]),
                "mom20": float(d["mom20"]), "mom60": float(d["mom60"]), "dd20": float(d["dd20"]),
                "rel_strength": np.nan, "sim8_win": 0.5, "close": close,
                "d_high20": float(d["high20"]), "d_low20": float(d["low20"]),
                "w_sma20": float(w["ma20"]), "w_low4": float(weekly_index.loc[:signal, "nav"].tail(4).min()),
                "mom5": float(d["mom5"]),
                "d_ma5": float(d["ma5"]), "d_ma10": float(d["ma10"]), "d_ma20": float(d["ma20"]),
                "d_ma5_bias": close / float(d["ma5"]) - 1.0,
                "d_ma10_bias": close / float(d["ma10"]) - 1.0,
                "d_ma20_bias": close / float(d["ma20"]) - 1.0,
                "d_ma_align": float(close > d["ma5"] > d["ma10"] > d["ma20"]),
                "d_ma20_slope": float(d["ma20_slope"]),
                "d_vol_ma5": np.nan, "d_vol_chg": np.nan,
                "d_low20_dist": close / float(d["low20"]) - 1.0,
                "d_high20_dist": close / float(d["high20"]) - 1.0,
                "w_ma5": float(w["ma5"]), "w_ma10": float(w["ma10"]), "w_ma20": float(w["ma20"]),
                "w_ma_align": float(close > w["ma5"] > w["ma10"] > w["ma20"]),
                "w_ma20_slope": float(w["ma20_slope"]),
                "w_low4_dist": close / float(weekly_index.loc[:signal, "nav"].tail(4).min()) - 1.0,
                "w_vol_ma5": np.nan,
            }
        )
    features = pd.DataFrame(rows)
    daily_vol5 = daily["nav"].pct_change().rolling(5).std(ddof=1) * math.sqrt(252) * 100.0
    daily_vol20 = daily["nav"].pct_change().rolling(20).std(ddof=1) * math.sqrt(252) * 100.0
    vol_map = pd.Series(daily_vol5.to_numpy() / daily_vol20.to_numpy(), index=daily["date"])
    features["vol_ratio"] = [float(vol_map.loc[:signal].iloc[-1]) for signal in anchors["signal"]]
    features["w_slope_pct"] = _expanding_percentile(features["w_slope"], 30)
    features["d_slope_pct"] = _expanding_percentile(features["d_slope"], 5)
    features["vol_pct"] = _expanding_percentile(features["vol20"], 5)

    buckets: dict[tuple[int, int], list[float]] = {}
    for index, row in features.iterrows():
        if not np.isfinite(row["w_slope"]) or not np.isfinite(row["d_slope"]):
            continue
        key = (int(row["w_slope"] > 0), int(row["d_slope"] > 0))
        history = buckets.get(key, [])
        features.loc[index, "sim8_win"] = sum(history) / len(history) if len(history) >= 10 else 0.5
        if index + 8 < len(anchors) and np.isfinite(anchors.loc[index, "fwd8"]):
            buckets.setdefault(key, []).append(float(anchors.loc[index, "fwd8"] > 0))
    return features


def used_features(spec: dict) -> set[str]:
    result = {term["feature"] for term in (spec.get("score") or {}).get("terms") or []}
    result.update(gate["feature"] for gate in spec.get("gates") or [])
    ramp = spec.get("ramp") or {}
    result.update(gate["feature"] for gate in ramp.get("confirm_gate") or [])
    result.update(spec.get("features") or [])
    return result


def candidate_specs(seed: dict) -> tuple[list[dict], list[dict]]:
    syntax = json.loads(SYNTAX_PATH.read_text(encoding="utf-8"))
    accepted = [{"id": "SEED_159915_CHAMPION", "source": "159915_current_champion", "spec": deepcopy(seed)}]
    rejected: list[dict] = []
    for family in ("prediction_roles", "strategy_roles"):
        for role, item in (syntax.get(family) or {}).items():
            spec = deepcopy(item["spec"])
            features = used_features(spec)
            candidate_id = f"{family}:{role}"
            if spec.get("kind") == "learning_ridge":
                rejected.append({"id": candidate_id, "reason": "walkforward_learning_not_supported_by_fund_adapter"})
            elif features & UNAVAILABLE_FEATURES:
                rejected.append({"id": candidate_id, "reason": "requires_unavailable_fund_fields", "fields": sorted(features & UNAVAILABLE_FEATURES)})
            else:
                accepted.append({"id": candidate_id, "source": "existing_159915_spec_syntax", "spec": spec})
    return accepted, rejected


def metric_slice(values: list[dict], start: int = 0) -> dict:
    rows = values[start:]
    returns = np.array([r["return"] for r in rows], dtype=float)
    nav = np.cumprod(1.0 + returns) if len(returns) else np.array([1.0])
    peak = np.maximum.accumulate(nav)
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return {
        "cumulative_return": float(nav[-1] - 1.0) if len(nav) else 0.0,
        "annualized_return": float(nav[-1] ** (52.0 / len(returns)) - 1.0) if len(returns) and nav[-1] > 0 else 0.0,
        "sharpe": float(np.mean(returns) / std * math.sqrt(52)) if std > 0 else 0.0,
        "max_drawdown": float(np.min(nav / peak - 1.0)) if len(nav) else 0.0,
        "win_rate": float(np.mean(returns > 0)) if len(returns) else 0.0,
        "weeks": int(len(returns)),
    }


def prediction_metrics(forecasts: list[dict], anchors: pd.DataFrame, start: int, end: int) -> dict:
    result: dict[str, float] = {}
    for horizon in (1, 2):
        maes: list[float] = []
        briers: list[float] = []
        correct: list[float] = []
        for index in range(start, min(end, len(forecasts))):
            actual = float(anchors.loc[index, f"fwd{horizon}"])
            if not np.isfinite(actual):
                continue
            forecast = forecasts[index]
            maes.append(abs(float(forecast[f"expected_return_{horizon}w"]) - actual))
            p_up = float(forecast[f"p_up_{horizon}w"])
            briers.append((p_up - float(actual > 0)) ** 2)
            direction = int(forecast[f"direction_{horizon}w"])
            if direction:
                correct.append(float(np.sign(actual) == direction))
        result[f"mae_{horizon}w"] = float(np.mean(maes)) if maes else 0.0
        result[f"brier_{horizon}w"] = float(np.mean(briers)) if briers else 0.0
        result[f"accuracy_{horizon}w"] = float(np.mean(correct)) if correct else 0.0
    return result


def evaluate_model(item: dict, anchors: pd.DataFrame, features: pd.DataFrame, cost_bps: float) -> dict:
    fwd_arrays = {h: anchors[f"fwd{h}"].to_numpy(dtype=float) for h in (1, 2, 4, 8)}
    forecasts: list[dict] = []
    positions: list[float] = []
    equity_history: list[float] = [1.0]
    rows: list[dict] = []
    previous = 0.0
    for index, row in features.iterrows():
        stats = compute_bucket_stats(features, fwd_arrays, index)
        forecast = build_forecast(item["spec"], row, index, fwd_arrays, stats, features=features)
        position = compute_position(item["spec"], row, {"position_history": positions, "equity_history": equity_history})
        forecasts.append(forecast)
        positions.append(position)
        if index + 1 >= len(anchors) or not np.isfinite(anchors.loc[index, "exec_nav"]) or not np.isfinite(anchors.loc[index + 1, "exec_nav"]):
            continue
        underlying = float(anchors.loc[index + 1, "exec_nav"] / anchors.loc[index, "exec_nav"] - 1.0)
        strategy_return = position * underlying - abs(position - previous) * cost_bps / 10000.0
        previous = position
        equity_history.append(equity_history[-1] * (1.0 + strategy_return))
        rows.append({"anchor_index": int(index), "signal": anchors.loc[index, "signal"].date().isoformat(), "return": strategy_return})
    valid_start = next((i for i, row in features.iterrows() if np.isfinite(row["mom60"]) and np.isfinite(row["w_slope"])), 0)
    rows = [row for row in rows if row["anchor_index"] >= valid_start]
    recent_start_row = max(0, len(rows) - 26)
    first_anchor = rows[0]["anchor_index"] if rows else valid_start
    recent_anchor = rows[recent_start_row]["anchor_index"] if rows else valid_start
    return {
        "id": item["id"], "source": item["source"], "spec": item["spec"],
        "full": metric_slice(rows), "validation_26w": metric_slice(rows, recent_start_row),
        "prediction_full": prediction_metrics(forecasts, anchors, first_anchor, len(anchors)),
        "prediction_validation_26w": prediction_metrics(forecasts, anchors, recent_anchor, len(anchors)),
        "turnover": float(sum(abs(positions[i] - positions[i - 1]) for i in range(1, len(positions)))),
        "latest_position": float(positions[-1]), "position_path": [float(x) for x in positions],
    }


def noninferior(candidate: float, baseline: float) -> bool:
    return candidate <= baseline * 1.05


def prediction_gate(candidate: dict, baseline: dict, require_improvement: bool) -> bool:
    checks = [
        noninferior(candidate["mae_1w"], baseline["mae_1w"]),
        noninferior(candidate["mae_2w"], baseline["mae_2w"]),
        noninferior(candidate["brier_1w"], baseline["brier_1w"]),
        candidate["accuracy_1w"] >= baseline["accuracy_1w"] - 0.02,
        candidate["accuracy_2w"] >= baseline["accuracy_2w"] - 0.02,
    ]
    if require_improvement:
        improvements = []
        for key in ("mae_1w", "mae_2w", "brier_1w"):
            base = baseline[key]
            improvements.append((base - candidate[key]) / base >= 0.05 if base > 0 else candidate[key] <= base)
        checks.append(any(improvements))
    return all(checks)


def round_numbers(value):
    if isinstance(value, dict):
        return {key: round_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_numbers(item) for item in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def latest_analysis(nav: pd.DataFrame, daily: pd.DataFrame, weekly: pd.DataFrame, champion: dict) -> dict:
    d = daily.iloc[-1]
    w = weekly.iloc[-1]
    direction = (
        "看多" if d["dif1"] > 0 and w["dif1"] > 0
        else "看空" if d["dif1"] < 0 and w["dif1"] < 0
        else "震荡修复" if d["dif2"] > 0 or w["dif2"] > 0
        else "震荡减弱"
    )
    def side(value: float, ma: float) -> str:
        return "上方" if value > ma else "下方"
    return {
        "schema_version": "fund-direct-analysis-v1",
        "code": CODE, "name": NAME,
        "nav_as_of": nav.iloc[-1]["date"].date().isoformat(),
        "completed_week_as_of": weekly.iloc[-1]["date"].date().isoformat(),
        "unit_nav": float(nav.iloc[-1]["unit_nav"]),
        "data_source": str(nav.iloc[-1]["source"]),
        "analysis_mode": "own_nav_direct_analysis",
        "direction": direction,
        "champion_id": champion["id"],
        "champion_position": champion["latest_position"],
        "daily": {
            "dif": float(d["dif"]), "dea": float(d["dea"]), "macd_hist": float(d["macd_hist"]),
            "dif_first": float(d["dif1"]), "dif_second": float(d["dif2"]),
            "ma5": float(d["ma5"]), "ma10": float(d["ma10"]), "ma20": float(d["ma20"]), "ma60": float(d["ma60"]),
            "ma_position": {f"ma{x}": side(float(d["nav"]), float(d[f"ma{x}"])) for x in (5, 10, 20, 60)},
            "rsi14": float(d["rsi14"]), "mom5": float(d["mom5"]), "mom20": float(d["mom20"]),
            "mom60": float(d["mom60"]), "vol20_annualized_pct": float(d["vol20"]),
            "nav_low20": float(d["low20"]), "nav_high20": float(d["high20"]),
            "volume": None,
        },
        "weekly": {
            "dif": float(w["dif"]), "dea": float(w["dea"]), "macd_hist": float(w["macd_hist"]),
            "dif_first": float(w["dif1"]), "dif_second": float(w["dif2"]),
            "ma5": float(w["ma5"]), "ma10": float(w["ma10"]), "ma20": float(w["ma20"]),
            "mom5": float(w["mom5"]), "mom20": float(w["mom20"]),
            "volume": None,
        },
        "unavailable_fields": ["OHLC", "volume", "intraday_execution_price"],
        "execution_note": "场外基金按申购赎回确认规则和基金净值成交；当前只生成研究信号，不生成ETF式开盘成交指令。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只计算并打印摘要，不写工作区")
    args = parser.parse_args()
    config = load_config()
    nav = load_nav()
    daily, weekly, anchors, features = build_frames(nav)
    etf_config = json.loads(ETF_CONFIG_PATH.read_text(encoding="utf-8"))
    seed_spec = deepcopy(etf_config["champion"]["spec"])
    specs, rejected = candidate_specs(seed_spec)
    results = [evaluate_model(item, anchors, features, float(config["cost_bps"])) for item in specs]
    baseline = results[0]
    eligible = []
    for result in results[1:]:
        passed = (
            prediction_gate(result["prediction_full"], baseline["prediction_full"], True)
            and prediction_gate(result["prediction_validation_26w"], baseline["prediction_validation_26w"], False)
            and result["full"]["cumulative_return"] > baseline["full"]["cumulative_return"]
        )
        result["promotion_gate_passed"] = bool(passed)
        if passed:
            eligible.append(result)
    eligible.sort(
        key=lambda item: (
            -(item["full"]["sharpe"] - baseline["full"]["sharpe"]),
            -(item["full"]["cumulative_return"] - baseline["full"]["cumulative_return"]),
            item["id"],
        )
    )
    winner = eligible[0] if eligible else baseline
    promoted = winner["id"] != baseline["id"]
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    state = {
        "schema_version": "fund-nav-rolling-v1",
        "instrument_type": "open_end_fund",
        "code": CODE, "name": NAME,
        "coverage": {"start": nav.iloc[0]["date"].date().isoformat(), "end": nav.iloc[-1]["date"].date().isoformat()},
        "completed_week_as_of": weekly.iloc[-1]["date"].date().isoformat(),
        "horizon_weeks": int(config["horizon"]), "cost_bps_research": float(config["cost_bps"]),
        "initial_champion_provenance": {
            "source_code": "159915", "source_champion_id": etf_config["champion"]["id"],
            "mode": "initial_parameters_only",
        },
        "champion": {"id": winner["id"], "spec": winner["spec"], "latest_position": winner["latest_position"]},
        "promotion_history": ([{"from": baseline["id"], "to": winner["id"], "reason": "existing_prediction_and_full_period_gates"}] if promoted else []),
        "status": "FULL_CYCLE_COMPLETED_PROMOTED" if promoted else "FULL_CYCLE_COMPLETED_NO_PROMOTION",
        "completed_at": completed,
        "field_availability": {"unit_nav": True, "cum_nav": True, "ohlc": False, "volume": False},
        "release_status": "RESEARCH_ONLY_FUND_EXECUTION_FEES_INCOMPLETE",
    }
    iteration = {
        "schema_version": "fund-iteration-record-v1", "iteration_id": f"ITERATION_001_{CODE}",
        "completed_at": completed, "data_rows": int(len(nav)), "weekly_anchors": int(len(anchors)),
        "usable_models": int(len(results)), "rejected_models": rejected,
        "seed": baseline, "winner": winner,
        "leaderboard": sorted(results, key=lambda x: -x["full"]["cumulative_return"]),
        "promotion_gate": "existing_159915_prediction_gates_plus_full_period_return",
        "notes": [
            "基金使用自身公布净值独立回放，未继承创业板历史业绩或训练进度。",
            "OHLCV和盘中成交价不可用，依赖这些字段的既有候选已拒绝参与。",
            "10bp仅为与现有模型一致的研究成本；真实申购赎回费率未完整登记，结果不得视为实盘收益。",
        ],
    }
    analysis = latest_analysis(nav, daily, weekly, winner)
    summary = {
        "status": state["status"], "data_end": state["coverage"]["end"],
        "completed_week": state["completed_week_as_of"], "weekly_anchors": len(anchors),
        "champion": winner["id"], "latest_position": winner["latest_position"],
        "direction": analysis["direction"], "eligible_challengers": len(eligible),
    }
    if not args.check:
        atomic_json(STATE_PATH, round_numbers(state))
        atomic_json(ITERATION_PATH, round_numbers(iteration))
        atomic_json(ANALYSIS_PATH, round_numbers(analysis))
    print(json.dumps(round_numbers(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
