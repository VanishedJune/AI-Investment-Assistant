# -*- coding: utf-8 -*-
"""2026 年七方案交易 PK（独立研究回放，不改写生产状态）。

七组账户使用相同的行情、成交价、整手、费用、现金利息和估值日期：

1. B0207 每 28 天完整调仓；非决策周仅执行五参数看空保护。
2. B0207 每周完整调仓。
3. Agent-v70 每 28 天完整调仓；非决策周仅执行五参数看空保护。
4. Agent-v70 每周完整调仓。
5. B0207 每 28 天完整调仓；非决策周使用 Agent-v70 看空保护。
6. 首个名义决策日按 B0207 建仓；此后仅在持仓触发 Agent-v70 看空
   保护时，按固定结构优先级把保护卖出份额接力到看多 ETF，不主动
   留存为现金。
7. 在第五组基础上增加单标的持有期高点止损：信号日收盘相对持有期间
   日收盘最高价回撤超过 10% 时，无需满足其他V70条件即可保护减持。

重要边界：
- B0207 使用当前冻结冠军参数回放历史行情，不训练、不推进迭代台账。
- Agent 使用当前 v70 规则的确定性、逐期重建版本；它不是历史真实 Agent 成绩。
- 两类信号都只读取当期信号日及以前的特征；信号在下一交易日 raw_open 执行。
- 最后一个没有下一交易日执行价的信号只记录，不提前成交。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

# 直接以文件路径启动时，scripts/calendar.py会遮蔽Python标准库calendar，进而
# 令pandas导入失败。模块运行本来没有此问题，这里同时兼容两种启动方式。
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
while _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT / "model_iteration"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from rolling.ca import load_ca  # noqa: E402
from rolling.calendar import NAMES, build_anchors, load_daily  # noqa: E402
from rolling.challenge import position_path  # noqa: E402
from rolling.features import (  # noqa: E402
    ALL_FEATURES,
    analysis_frames,
    build_extended_features,
    smooth_signal_features,
)
from rolling.inference import current_champion  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts.investment_priority import allocate, load_config  # noqa: E402


INITIAL_CAPITAL = 200_000.0
LOT = 100
COMMISSION_RATE = 0.00025
MIN_COMMISSION = 5.0
CASH_APR = 0.0175
CASH_WEEKLY_FACTOR = (1.0 + CASH_APR) ** (1.0 / 52.0)
GUARD_SELL_FRACTION = 0.70
STOP_LOSS_THRESHOLD = 0.10
STOP_LOSS_SELL_FRACTION = 1.00
MOM20_GUARD = -0.05
FIRST_NOMINAL_DECISION = date(2026, 1, 10)

GROUPS = {
    "G1_B0207_4W_GUARD": "第一组：B0207四周决策＋周度五参数保护",
    "G2_B0207_WEEKLY": "第二组：B0207每周决策",
    "G3_AGENT_4W_GUARD": "第三组：Agent-v70四周决策＋周度五参数保护",
    "G4_AGENT_WEEKLY": "第四组：Agent-v70每周决策",
    "G5_B0207_4W_AGENT_GUARD": "第五组：B0207四周决策＋Agent-v70看空保护",
    "G6_PRIORITY_ROTATION_AGENT_GUARD": "第六组：B0207首期建仓＋Agent-v70优先级接力保护",
    "G7_B0207_4W_AGENT_GUARD_STOP": "第七组：B0207四周决策＋Agent-v70保护＋单标的10%止损清仓＋原有持仓回投",
}


class PKError(RuntimeError):
    """回放输入或账户约束不满足。"""


def _iso(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def trade_cost(notional: float) -> float:
    return max(notional * COMMISSION_RATE, MIN_COMMISSION) if notional > 0 else 0.0


def _rolling_normalized(series: pd.Series, window: int = 20) -> pd.Series:
    """仅用截至当期的滚动中位绝对偏差归一化，不读取未来值。"""
    values = pd.to_numeric(series, errors="coerce")
    center = values.rolling(window, min_periods=5).median()
    scale = (values - center).abs().rolling(window, min_periods=5).median()
    return values / scale.replace(0.0, np.nan)


def _load_market_bundle(codes: list[str]) -> dict[str, dict]:
    """绕过训练台账固定锚点门禁，只读装配当前行情与冻结冠军。"""
    raw_daily: dict[str, pd.DataFrame] = {}
    analysis: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]] = {}
    for code in codes:
        set_workspace(code)
        ca = load_ca(code)
        daily_a, weekly_a = analysis_frames(code, ca)
        daily = load_daily(code)
        raw_daily[code] = daily.copy()
        analysis[code] = (ca, daily_a, weekly_a)

    # 与当前冻结推理一致：相对强弱只比较生产配置内的8只ETF。
    all_daily = {code: frame.copy() for code, frame in raw_daily.items()}
    result: dict[str, dict] = {}
    for code in codes:
        ca, daily_a, weekly_a = analysis[code]
        daily = raw_daily[code]
        anchors = build_anchors(code, daily_a, weekly_a, daily=daily)
        raw_features = build_extended_features(
            anchors,
            code,
            ca,
            frames=(daily_a, weekly_a),
            all_daily=all_daily,
        )
        cfg, champion, _workspace = current_champion(PROJECT, code)
        smooth_days = int((cfg.get("execution_policy") or {}).get("signal_smooth_days") or 0)
        if smooth_days > 0:
            raw_features = smooth_signal_features(
                anchors, daily_a, weekly_a, raw_features, window=smooth_days
            )
        mask = ~raw_features[ALL_FEATURES].isna().any(axis=1)
        usable = anchors.loc[mask].reset_index(drop=True)
        features = raw_features.loc[mask].reset_index(drop=True)
        if usable.empty or len(usable) != len(features):
            raise PKError(f"{code}可用锚点为空或错位")
        if not np.allclose(
            features["close"].to_numpy(dtype=float),
            usable["anchor_close"].to_numpy(dtype=float),
            rtol=0,
            atol=1e-9,
        ):
            raise PKError(f"{code}特征与锚点收盘价错位")
        positions = position_path(champion["spec"], features, len(features) - 1)
        if len(positions) != len(features) or not np.isfinite(positions).all():
            raise PKError(f"{code}冻结冠军仓位路径无效")
        features = features.copy()
        features["d_dif2_norm"] = _rolling_normalized(features["d_dif2"])
        features["w_dif2_norm"] = _rolling_normalized(features["w_dif2"])
        by_signal: dict[str, dict] = {}
        for idx, row in usable.iterrows():
            signal = _iso(row["signal"])
            by_signal[signal] = {
                "index": idx,
                "exec": None if pd.isna(row["exec"]) else _iso(row["exec"]),
                "exec_raw_open": None if pd.isna(row["exec_raw_open"]) else float(row["exec_raw_open"]),
                "feature": features.iloc[idx],
                "champion_position": float(positions[idx]),
            }
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily_by_date = {
            _iso(row["date"]): {
                "raw_open": float(row["raw_open"]),
                "raw_close": float(row["raw_close"]),
            }
            for _, row in daily.iterrows()
        }
        result[code] = {
            "name": NAMES[code],
            "ca": ca,
            "usable": usable,
            "features": features,
            "by_signal": by_signal,
            "daily": daily,
            "daily_by_date": daily_by_date,
            "champion_id": champion["id"],
        }
    return result


def _agent_v70_view(row: pd.Series) -> dict:
    """把v70既有R001-R008按固定优先顺序转成可回放方向。

    这不是训练新模型，也没有用累计收益挑选阈值。数值幅度只用于让同一规则在
    多只同时看多时向B0207分配器表达证据强弱；不声称是校准概率。
    """
    d1 = _finite(row.get("d_dif1"))
    d2 = _finite(row.get("d_dif2"))
    w1 = _finite(row.get("w_dif1"))
    w2 = _finite(row.get("w_dif2"))
    d1n = _finite(row.get("d_slope"))
    w1n = _finite(row.get("w_slope"))
    d2n = _finite(row.get("d_dif2_norm"))
    w2n = _finite(row.get("w_dif2_norm"))
    close = _finite(row.get("close"))
    d_ma5 = _finite(row.get("d_ma5"))
    d_ma20 = _finite(row.get("d_ma20"))
    w_ma20 = _finite(row.get("w_ma20"))
    d_ma20_slope = _finite(row.get("d_ma20_slope"))
    w_ma20_slope = _finite(row.get("w_ma20_slope"))
    mom5 = _finite(row.get("mom5"))
    mom20 = _finite(row.get("mom20"))
    rsi = _finite(row.get("rsi14"), 50.0)
    d_volume = _finite(row.get("d_vol_ma5"))
    w_volume = _finite(row.get("w_vol_ma5"))
    rel = _finite(row.get("rel_strength"), 0.5)
    daily_above = close > d_ma20 > 0
    weekly_above = close > w_ma20 > 0
    volume_confirmed = d_volume >= 0 or w_volume >= 0
    thin_volume = d_volume < -0.40 and w_volume < -0.40

    hard_break = (
        d1 < 0 and d2 < 0 and w1 < 0 and w2 < 0
        and not daily_above and not weekly_above and mom20 < MOM20_GUARD
    )
    severe = w2n <= -1.0 and d1n < 0 and d2n < 0 and thin_volume
    r006_repair = (
        w1 < 0 and w2 > 0 and d1 > 0
        and (d2 >= 0 or mom5 > 0) and close >= d_ma5 > 0
    )
    r008_conditional_bear = d1 < 0 and w1 < 0 and d2 > 0 and w2 > 0
    exhausted = (
        w1 > 0 and w2 < 0 and d1 < 0 and d2 < 0
        and ((rsi >= 70 and mom20 >= 0.10) or (mom20 >= 0.08 and not volume_confirmed))
    )
    full_resonance = (
        d1 > 0 and d2 >= 0 and w1 > 0 and w2 >= 0
        and daily_above and weekly_above
    )
    continuation = (
        w1 > 0 and d1 > 0 and (w2 >= 0 or d2 >= 0)
        and (daily_above or weekly_above or d_ma20_slope > 0 or w_ma20_slope > 0)
    )
    weekly_repair = (
        w1 > 0 and w2 > 0 and d2 > 0
        and (d1 >= 0 or mom5 >= 0) and not hard_break
    )

    if hard_break or severe:
        direction, strength, rule = "bearish", 0.0, "R002"
    elif r008_conditional_bear and not (daily_above and mom5 > 0 and volume_confirmed):
        direction, strength, rule = "bearish", 0.0, "R008"
    elif exhausted:
        direction, strength, rule = "bearish", 0.0, "R003"
    elif full_resonance:
        direction, strength, rule = "bullish", 1.0, "R005"
    elif r006_repair:
        direction, strength, rule = "bullish", 0.60, "R006"
    elif continuation:
        direction, strength, rule = "bullish", 0.80, "R001"
    elif weekly_repair:
        direction, strength, rule = "bullish", 0.55, "R001/R004"
    else:
        direction, strength, rule = "neutral", 0.0, "R004/R007"

    # 高位缩量只降低证据强度，不把“已上涨”直接等同于未来看空。
    overheated_thin = rsi >= 70 and mom20 >= 0.10 and not volume_confirmed
    if direction == "bullish" and overheated_thin:
        strength = min(strength, 0.40)
        rule = f"{rule}+R003"
    # 相对强弱只作幅度修正，不能单独改变方向。
    if direction == "bullish":
        strength *= 0.90 + 0.20 * min(1.0, max(0.0, rel))
        strength = min(1.0, max(0.05, strength))

    return {
        "direction": direction,
        "position": strength if direction == "bullish" else 0.0,
        "rule": rule,
        "d_dif1": d1,
        "d_dif2": d2,
        "w_dif1": w1,
        "w_dif2": w2,
        "mom20": mom20,
        "rsi14": rsi,
        "daily_above_ma20": daily_above,
        "weekly_above_ma20": weekly_above,
        "volume_confirmed": volume_confirmed,
    }


def _five_parameter_guard(row: pd.Series) -> bool:
    return (
        _finite(row.get("d_dif1"), 1.0) < 0
        and _finite(row.get("d_dif2"), 1.0) < 0
        and _finite(row.get("w_dif1"), 1.0) < 0
        and _finite(row.get("w_dif2"), 1.0) < 0
        and _finite(row.get("mom20"), 0.0) < MOM20_GUARD
    )


def _priority_rotation_recipient(
    codes: list[str], seller: str, views: dict[str, dict]
) -> str | None:
    """为看空持仓选择固定优先级下的接力标的。

    先寻找比卖出标的优先级更高的看多 ETF；若没有，则只有在卖出标的
    之前的所有更高优先级 ETF 都同样看空时，才允许向较低优先级中排名
    最靠前的看多 ETF 继续接力。这样黄金看空而创业板看多时只能转入
    创业板；创业板和黄金同时看空时，黄金才可继续向后寻找看多标的。
    中性高优先级不被当作“已触发看空”，因此会阻断向后越级。
    """
    if seller not in codes:
        raise PKError(f"优先级列表中缺少卖出标的: {seller}")
    seller_index = codes.index(seller)
    higher = codes[:seller_index]
    higher_bullish = [
        code for code in higher if views[code]["direction"] == "bullish"
    ]
    if higher_bullish:
        return higher_bullish[0]
    if not all(views[code]["direction"] == "bearish" for code in higher):
        return None
    lower_bullish = [
        code for code in codes[seller_index + 1 :]
        if views[code]["direction"] == "bullish"
    ]
    return lower_bullish[0] if lower_bullish else None


def nominal_signal_dates(
    signal_dates: Iterable[str], end_as_of: str, first_nominal: date = FIRST_NOMINAL_DECISION
) -> dict[str, str]:
    """名义周六对应其前一日及以前最新完整周信号。"""
    signals = sorted(date.fromisoformat(value) for value in signal_dates)
    end_date = date.fromisoformat(end_as_of)
    mapping: dict[str, str] = {}
    nominal = first_nominal
    while nominal <= end_date:
        cutoff = nominal - timedelta(days=1)
        eligible = [value for value in signals if value <= cutoff]
        if eligible:
            mapping[nominal.isoformat()] = eligible[-1].isoformat()
        nominal += timedelta(days=28)
    return mapping


def _signal_targets(
    market: dict[str, dict], codes: list[str], signals: list[str]
) -> tuple[dict[str, dict], list[dict]]:
    targets: dict[str, dict] = {}
    audit_rows: list[dict] = []
    for signal in signals:
        missing = [code for code in codes if signal not in market[code]["by_signal"]]
        if missing:
            raise PKError(f"{signal}缺少共同信号: {missing}")

        b_positions = {
            code: market[code]["by_signal"][signal]["champion_position"] for code in codes
        }
        b_alloc = allocate(
            {code: b_positions[code] > 0 for code in codes}, b_positions
        )
        agent_views = {
            code: _agent_v70_view(market[code]["by_signal"][signal]["feature"])
            for code in codes
        }
        a_positions = {code: agent_views[code]["position"] for code in codes}
        a_alloc = allocate(
            {code: agent_views[code]["direction"] == "bullish" for code in codes},
            a_positions,
        )
        exec_dates = {market[code]["by_signal"][signal]["exec"] for code in codes}
        if len(exec_dates) != 1:
            raise PKError(f"{signal}的8只ETF执行日不一致: {sorted(exec_dates, key=str)}")
        exec_date = exec_dates.pop()
        targets[signal] = {
            "exec_date": exec_date,
            "b0207": {**b_alloc["weights"], "CASH": b_alloc["cash_pct"]},
            "agent": {**a_alloc["weights"], "CASH": a_alloc["cash_pct"]},
            "agent_views": agent_views,
        }
        for code in codes:
            audit_rows.append(
                {
                    "signal_date": signal,
                    "execution_date": exec_date or "",
                    "code": code,
                    "name": market[code]["name"],
                    "b0207_champion_position": b_positions[code],
                    "b0207_target_pct": b_alloc["weights"][code],
                    "agent_direction": agent_views[code]["direction"],
                    "agent_rule": agent_views[code]["rule"],
                    "agent_evidence_strength": agent_views[code]["position"],
                    "agent_target_pct": a_alloc["weights"][code],
                    "d_dif1": agent_views[code]["d_dif1"],
                    "d_dif2": agent_views[code]["d_dif2"],
                    "w_dif1": agent_views[code]["w_dif1"],
                    "w_dif2": agent_views[code]["w_dif2"],
                    "mom20": agent_views[code]["mom20"],
                    "rsi14": agent_views[code]["rsi14"],
                }
            )
        audit_rows.append(
            {
                "signal_date": signal,
                "execution_date": exec_date or "",
                "code": "CASH",
                "name": "现金",
                "b0207_champion_position": "",
                "b0207_target_pct": b_alloc["cash_pct"],
                "agent_direction": "neutral",
                "agent_rule": "cash_residual",
                "agent_evidence_strength": "",
                "agent_target_pct": a_alloc["cash_pct"],
                "d_dif1": "",
                "d_dif2": "",
                "w_dif1": "",
                "w_dif2": "",
                "mom20": "",
                "rsi14": "",
            }
        )
    return targets, audit_rows


@dataclass
class Account:
    group_id: str
    codes: list[str]
    cash: float = INITIAL_CAPITAL
    shares: dict[str, int] = field(default_factory=dict)
    average_cost: dict[str, float] = field(default_factory=dict)
    peak_close: dict[str, float] = field(default_factory=dict)
    guard_episode: dict[str, bool] = field(default_factory=dict)
    stop_loss_episode: dict[str, bool] = field(default_factory=dict)
    trade_rows: list[dict] = field(default_factory=list)
    protection_rows: list[dict] = field(default_factory=list)
    stop_loss_review_rows: list[dict] = field(default_factory=list)
    nav_rows: list[dict] = field(default_factory=list)
    costs: float = 0.0
    traded_notional: float = 0.0
    guard_count: int = 0
    agent_guard_count: int = 0
    stop_loss_count: int = 0

    def __post_init__(self) -> None:
        self.shares = {code: 0 for code in self.codes}
        self.average_cost = {code: 0.0 for code in self.codes}
        self.peak_close = {code: 0.0 for code in self.codes}
        self.guard_episode = {code: False for code in self.codes}
        self.stop_loss_episode = {code: False for code in self.codes}

    def value(self, prices: dict[str, float]) -> float:
        return self.cash + sum(self.shares[code] * prices[code] for code in self.codes)

    def _update_cost_after_buy(
        self, code: str, quantity: int, notional: float, fee: float
    ) -> None:
        prior_shares = self.shares[code]
        total_shares = prior_shares + quantity
        if quantity <= 0 or total_shares <= 0:
            raise PKError(f"{self.group_id}买入成本更新参数无效: {code}")
        prior_cost = self.average_cost[code] * prior_shares
        self.average_cost[code] = (prior_cost + notional + fee) / total_shares
        buy_price = notional / quantity
        self.peak_close[code] = (
            buy_price if prior_shares == 0 else max(self.peak_close[code], buy_price)
        )

    def _update_cost_after_sell(self, code: str, quantity: int) -> None:
        if quantity <= 0 or quantity > self.shares[code]:
            raise PKError(f"{self.group_id}卖出成本更新参数无效: {code}")
        if quantity == self.shares[code]:
            self.average_cost[code] = 0.0
            self.peak_close[code] = 0.0
            self.stop_loss_episode[code] = False

    def mark_close_peaks(self, close_prices: dict[str, float]) -> None:
        """用已发生的日收盘更新各持仓的持有期高水位。"""
        for code in self.codes:
            if self.shares[code] > 0:
                self.peak_close[code] = max(
                    self.peak_close[code], float(close_prices[code])
                )
            else:
                self.peak_close[code] = 0.0

    def stop_loss_snapshot(self, signal_close_prices: dict[str, float]) -> dict[str, dict]:
        """冻结交易前可见的单标的持有期高点回撤。"""
        snapshot: dict[str, dict] = {}
        for code in self.codes:
            peak = self.peak_close[code]
            current = float(signal_close_prices[code])
            drawdown = current / peak - 1.0 if self.shares[code] > 0 and peak > 0 else 0.0
            snapshot[code] = {
                "held_before_rebalance": self.shares[code] > 0,
                "shares_before_rebalance": self.shares[code],
                "peak_close": peak,
                "signal_raw_close": current,
                "drawdown_from_peak": drawdown,
                "triggered": self.shares[code] > 0 and drawdown < -STOP_LOSS_THRESHOLD,
            }
        return snapshot

    def _record_trade(
        self,
        *,
        signal: str,
        execution: str,
        code: str,
        side: str,
        shares: int,
        price: float,
        cost: float,
        reason: str,
    ) -> None:
        notional = shares * price
        self.trade_rows.append(
            {
                "group_id": self.group_id,
                "group_name": GROUPS[self.group_id],
                "signal_date": signal,
                "execution_date": execution,
                "code": code,
                "name": NAMES[code],
                "side": side,
                "shares": shares,
                "price": price,
                "notional": notional,
                "commission": cost,
                "reason": reason,
            }
        )
        self.costs += cost
        self.traded_notional += notional

    def rebalance(
        self,
        *,
        signal: str,
        execution: str,
        target: dict[str, float],
        open_prices: dict[str, float],
        reason: str,
    ) -> None:
        nav = self.value(open_prices)
        target_shares = {
            code: int((nav * float(target.get(code, 0.0)) / 100.0) / open_prices[code] // LOT) * LOT
            for code in self.codes
        }
        # 先卖后买，避免顺序造成虚构融资。
        for code in self.codes:
            delta = target_shares[code] - self.shares[code]
            if delta >= 0:
                continue
            quantity = -delta
            price = open_prices[code]
            notional = quantity * price
            fee = trade_cost(notional)
            self.cash += notional - fee
            self._update_cost_after_sell(code, quantity)
            self.shares[code] -= quantity
            self._record_trade(
                signal=signal, execution=execution, code=code, side="SELL",
                shares=quantity, price=price, cost=fee, reason=reason,
            )
        for code in self.codes:
            desired = target_shares[code] - self.shares[code]
            if desired <= 0:
                continue
            price = open_prices[code]
            quantity = desired
            while quantity >= LOT:
                notional = quantity * price
                fee = trade_cost(notional)
                if notional + fee <= self.cash + 1e-9:
                    break
                quantity -= LOT
            if quantity < LOT:
                continue
            notional = quantity * price
            fee = trade_cost(notional)
            self.cash -= notional + fee
            self._update_cost_after_buy(code, quantity, notional, fee)
            self.shares[code] += quantity
            self._record_trade(
                signal=signal, execution=execution, code=code, side="BUY",
                shares=quantity, price=price, cost=fee, reason=reason,
            )
        if self.cash < -1e-6:
            raise PKError(f"{self.group_id}调仓后现金为负: {self.cash}")
        self.guard_episode = {code: False for code in self.codes}

    def apply_guard(
        self,
        *,
        signal: str,
        execution: str,
        market: dict[str, dict],
        open_prices: dict[str, float],
    ) -> None:
        for code in self.codes:
            row = market[code]["by_signal"][signal]["feature"]
            triggered = _five_parameter_guard(row)
            if not triggered:
                self.guard_episode[code] = False
                continue
            if self.guard_episode[code] or self.shares[code] <= 0:
                self.guard_episode[code] = True
                continue
            quantity = int(self.shares[code] * GUARD_SELL_FRACTION // LOT) * LOT
            self.guard_episode[code] = True
            if quantity <= 0:
                continue
            price = open_prices[code]
            notional = quantity * price
            fee = trade_cost(notional)
            self.cash += notional - fee
            self._update_cost_after_sell(code, quantity)
            self.shares[code] -= quantity
            self.guard_count += 1
            self._record_trade(
                signal=signal, execution=execution, code=code, side="SELL",
                shares=quantity, price=price, cost=fee,
                reason="five_parameter_guard_70pct_once_per_episode",
            )

    def apply_agent_guard(
        self,
        *,
        signal: str,
        execution: str,
        targets: dict[str, dict],
        open_prices: dict[str, float],
    ) -> None:
        """Agent-v70看空即保护；幅度与五参数组一致以隔离触发逻辑差异。"""
        views = targets[signal]["agent_views"]
        for code in self.codes:
            view = views[code]
            triggered = view["direction"] == "bearish"
            if not triggered:
                self.guard_episode[code] = False
                continue
            if self.guard_episode[code] or self.shares[code] <= 0:
                self.guard_episode[code] = True
                continue
            quantity = int(self.shares[code] * GUARD_SELL_FRACTION // LOT) * LOT
            self.guard_episode[code] = True
            if quantity <= 0:
                continue
            price = open_prices[code]
            notional = quantity * price
            fee = trade_cost(notional)
            self.cash += notional - fee
            self._update_cost_after_sell(code, quantity)
            self.shares[code] -= quantity
            self.guard_count += 1
            self.agent_guard_count += 1
            self._record_trade(
                signal=signal,
                execution=execution,
                code=code,
                side="SELL",
                shares=quantity,
                price=price,
                cost=fee,
                reason=f"agent_v70_bearish_guard_{view['rule']}_70pct_once_per_episode",
            )

    def apply_agent_guard_with_stop_loss(
        self,
        *,
        signal: str,
        execution: str,
        targets: dict[str, dict],
        stop_loss_state: dict[str, dict],
        open_prices: dict[str, float],
        enable_agent_guard: bool = True,
        stop_loss_sell_fraction: float = STOP_LOSS_SELL_FRACTION,
        agent_guard_sell_fraction: float = GUARD_SELL_FRACTION,
    ) -> None:
        """第五组V70保护叠加单标的持有期高点止损。

        信号日收盘价相对连续持有期间的日收盘最高价回撤严格超过10%时，
        不依赖V70方向即可在下一交易日清仓100%。V70看空正式口径减持70%，
        对照回放可显式传入其他比例；两者分别维护连续区间，任一条件新触发
        即可保护，同日同时触发只执行一次。
        """
        views = targets[signal]["agent_views"]
        for code in self.codes:
            view = views[code]
            state = stop_loss_state[code]
            agent_triggered = enable_agent_guard and view["direction"] == "bearish"
            stop_triggered = bool(state["triggered"])
            drawdown = float(state["drawdown_from_peak"])
            if state["held_before_rebalance"]:
                self.stop_loss_review_rows.append(
                    {
                        "signal_date": signal,
                        "execution_date": execution,
                        "code": code,
                        "name": NAMES[code],
                        "shares_before_rebalance": state.get("shares_before_rebalance", ""),
                        "shares_before_protection": self.shares[code],
                        "holding_peak_raw_close": state["peak_close"],
                        "signal_raw_close": state["signal_raw_close"],
                        "drawdown_from_holding_peak": drawdown,
                        "stop_loss_triggered": stop_triggered,
                        "nominal_week": not enable_agent_guard,
                        "agent_direction": view["direction"],
                        "agent_rule": view["rule"],
                    }
                )
            new_agent = agent_triggered and not self.guard_episode[code]
            new_stop = stop_triggered and not self.stop_loss_episode[code]

            if not agent_triggered:
                self.guard_episode[code] = False
            if not stop_triggered:
                self.stop_loss_episode[code] = False
            if self.shares[code] <= 0 or not (new_agent or new_stop):
                if self.shares[code] <= 0:
                    self.guard_episode[code] = agent_triggered
                    self.stop_loss_episode[code] = stop_triggered
                continue

            sell_fraction = stop_loss_sell_fraction if new_stop else agent_guard_sell_fraction
            if not 0 < sell_fraction <= 1:
                raise PKError("保护卖出比例必须在0至1之间")
            quantity = (
                self.shares[code]
                if sell_fraction >= 1.0
                else int(self.shares[code] * sell_fraction // LOT) * LOT
            )
            self.guard_episode[code] = agent_triggered
            self.stop_loss_episode[code] = stop_triggered
            if quantity <= 0:
                continue
            price = open_prices[code]
            notional = quantity * price
            fee = trade_cost(notional)
            self.cash += notional - fee
            self._update_cost_after_sell(code, quantity)
            self.shares[code] -= quantity
            self.guard_count += 1
            if new_agent:
                self.agent_guard_count += 1
            if new_stop:
                self.stop_loss_count += 1
            trigger_labels = []
            if new_agent:
                trigger_labels.append(f"agent_{view['rule']}")
            if new_stop:
                trigger_labels.append(f"holding_peak_stop_loss_{drawdown:.6f}")
            self.protection_rows.append(
                {
                    "group_id": self.group_id,
                    "group_name": GROUPS[self.group_id],
                    "signal_date": signal,
                    "execution_date": execution,
                    "code": code,
                    "name": NAMES[code],
                    "agent_triggered": new_agent,
                    "agent_rule": view["rule"] if new_agent else "",
                    "stop_loss_triggered": new_stop,
                    "holding_peak_raw_close": state["peak_close"],
                    "signal_raw_close": state["signal_raw_close"],
                    "drawdown_from_holding_peak": drawdown,
                    "nominal_week": not enable_agent_guard,
                    "sell_fraction": sell_fraction,
                    "shares_sold": quantity,
                }
            )
            self._record_trade(
                signal=signal,
                execution=execution,
                code=code,
                side="SELL",
                shares=quantity,
                price=price,
                cost=fee,
                reason=(
                    f"agent_v70_plus_10pct_holding_peak_stop_{sell_fraction:.0%}_"
                    + "_and_".join(trigger_labels)
                    + "_once_per_condition_episode"
                ),
            )

    def reinvest_protection_cash_into_prior_holdings(
        self,
        *,
        signal: str,
        execution: str,
        prior_shares: dict[str, int],
        protected_codes: set[str],
        available_cash: float,
        open_prices: dict[str, float],
    ) -> None:
        """把保护卖出新增现金按原有其他持仓市值比例回投。

        仅供回放对照使用：不买入保护前未持有标的，也不回买本次被保护卖出的标的。
        整手与佣金造成的零头继续留在现金。
        """
        eligible = [
            code for code in self.codes
            if prior_shares.get(code, 0) > 0
            and code not in protected_codes
            and self.shares[code] > 0
        ]
        if available_cash <= 0 or not eligible:
            return
        weights = {
            code: prior_shares[code] * open_prices[code]
            for code in eligible
        }
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return
        remaining_budget = min(available_cash, self.cash)
        for index, code in enumerate(eligible):
            budget = (
                remaining_budget
                if index == len(eligible) - 1
                else available_cash * weights[code] / total_weight
            )
            price = open_prices[code]
            quantity = int(budget / price // LOT) * LOT
            while quantity >= LOT:
                notional = quantity * price
                fee = trade_cost(notional)
                if notional + fee <= self.cash + 1e-9:
                    break
                quantity -= LOT
            if quantity < LOT:
                continue
            notional = quantity * price
            fee = trade_cost(notional)
            self.cash -= notional + fee
            remaining_budget = max(0.0, remaining_budget - notional - fee)
            self._update_cost_after_buy(code, quantity, notional, fee)
            self.shares[code] += quantity
            self._record_trade(
                signal=signal,
                execution=execution,
                code=code,
                side="BUY",
                shares=quantity,
                price=price,
                cost=fee,
                reason="research_reinvest_protection_cash_into_prior_holdings_pro_rata",
            )

    def rebalance_with_nominal_stop_overlay(
        self,
        *,
        signal: str,
        execution: str,
        target: dict[str, float],
        targets: dict[str, dict],
        stop_loss_state: dict[str, dict],
        open_prices: dict[str, float],
        reinvest_protection_cash: bool,
        stop_loss_sell_fraction: float,
    ) -> None:
        """先把名义周止损覆盖合并进目标，再执行一次净额调仓。"""
        prior_shares = dict(self.shares)
        prior_stop_episode = dict(self.stop_loss_episode)
        adjusted = {code: float(target.get(code, 0.0)) for code in self.codes}
        protected = {
            code for code in self.codes
            if stop_loss_state[code]["triggered"]
            and not prior_stop_episode[code]
            and prior_shares[code] > 0
        }
        if not 0 < stop_loss_sell_fraction <= 1:
            raise PKError("持有期止损卖出比例必须在0至1之间")
        removed_weight = sum(
            adjusted[code] * stop_loss_sell_fraction for code in protected
        )
        for code in protected:
            adjusted[code] *= 1.0 - stop_loss_sell_fraction
        if reinvest_protection_cash and removed_weight > 0:
            eligible = [
                code for code in self.codes
                if prior_shares[code] > 0
                and code not in protected
                and adjusted[code] > 0
            ]
            denominator = sum(prior_shares[code] * open_prices[code] for code in eligible)
            if denominator > 0:
                for code in eligible:
                    adjusted[code] += removed_weight * (
                        prior_shares[code] * open_prices[code] / denominator
                    )
        self.rebalance(
            signal=signal,
            execution=execution,
            target=adjusted,
            open_prices=open_prices,
            reason="nominal_28day_net_rebalance_with_holding_peak_stop_overlay",
        )
        views = targets[signal]["agent_views"]
        for code in self.codes:
            state = stop_loss_state[code]
            if state["held_before_rebalance"]:
                self.stop_loss_review_rows.append(
                    {
                        "signal_date": signal,
                        "execution_date": execution,
                        "code": code,
                        "name": NAMES[code],
                        "shares_before_rebalance": state.get("shares_before_rebalance", ""),
                        "shares_before_protection": prior_shares[code],
                        "holding_peak_raw_close": state["peak_close"],
                        "signal_raw_close": state["signal_raw_close"],
                        "drawdown_from_holding_peak": state["drawdown_from_peak"],
                        "stop_loss_triggered": bool(state["triggered"]),
                        "nominal_week": True,
                        "agent_direction": views[code]["direction"],
                        "agent_rule": views[code]["rule"],
                    }
                )
            if code in protected:
                self.stop_loss_episode[code] = True
                self.guard_count += 1
                self.stop_loss_count += 1
                self.protection_rows.append(
                    {
                        "group_id": self.group_id,
                        "group_name": GROUPS[self.group_id],
                        "signal_date": signal,
                        "execution_date": execution,
                        "code": code,
                        "name": NAMES[code],
                        "agent_triggered": False,
                        "agent_rule": "",
                        "stop_loss_triggered": True,
                        "holding_peak_raw_close": state["peak_close"],
                        "signal_raw_close": state["signal_raw_close"],
                        "drawdown_from_holding_peak": state["drawdown_from_peak"],
                        "nominal_week": True,
                        "sell_fraction": stop_loss_sell_fraction,
                        "shares_sold": max(0, prior_shares[code] - self.shares[code]),
                    }
                )
            elif not state["triggered"]:
                self.stop_loss_episode[code] = False

    def apply_priority_rotation_guard(
        self,
        *,
        signal: str,
        execution: str,
        targets: dict[str, dict],
        open_prices: dict[str, float],
    ) -> None:
        """把V70保护卖出款接力到固定优先级下可用的看多ETF。

        每个连续看空区间仍只保护一次、卖出当前份额的70%，与第五组保持
        相同触发及幅度，仅把卖出款的去向由现金改为优先级接力。实际交易
        受100份整手和佣金约束，无法购买一手时不执行该笔卖出。
        """
        views = targets[signal]["agent_views"]
        for code in self.codes:
            view = views[code]
            triggered = view["direction"] == "bearish"
            if not triggered:
                self.guard_episode[code] = False
                continue
            if self.guard_episode[code] or self.shares[code] <= 0:
                self.guard_episode[code] = True
                continue

            recipient = _priority_rotation_recipient(self.codes, code, views)
            if recipient is None:
                # 没有符合接力约束的看多标的时不卖出，也不转为主动现金。
                self.guard_episode[code] = False
                continue

            sell_quantity = int(self.shares[code] * GUARD_SELL_FRACTION // LOT) * LOT
            if sell_quantity <= 0:
                self.guard_episode[code] = False
                continue
            sell_price = open_prices[code]
            sell_notional = sell_quantity * sell_price
            sell_fee = trade_cost(sell_notional)
            transfer_budget = sell_notional - sell_fee

            buy_price = open_prices[recipient]
            buy_quantity = int(transfer_budget / buy_price // LOT) * LOT
            while buy_quantity >= LOT:
                buy_notional = buy_quantity * buy_price
                buy_fee = trade_cost(buy_notional)
                if buy_notional + buy_fee <= transfer_budget + 1e-9:
                    break
                buy_quantity -= LOT
            if buy_quantity < LOT:
                self.guard_episode[code] = False
                continue

            self.cash += transfer_budget
            self._update_cost_after_sell(code, sell_quantity)
            self.shares[code] -= sell_quantity
            self._record_trade(
                signal=signal,
                execution=execution,
                code=code,
                side="SELL",
                shares=sell_quantity,
                price=sell_price,
                cost=sell_fee,
                reason=(
                    f"agent_v70_priority_rotation_{view['rule']}_70pct_to_{recipient}"
                    "_once_per_episode"
                ),
            )

            buy_notional = buy_quantity * buy_price
            buy_fee = trade_cost(buy_notional)
            self.cash -= buy_notional + buy_fee
            self._update_cost_after_buy(recipient, buy_quantity, buy_notional, buy_fee)
            self.shares[recipient] += buy_quantity
            self._record_trade(
                signal=signal,
                execution=execution,
                code=recipient,
                side="BUY",
                shares=buy_quantity,
                price=buy_price,
                cost=buy_fee,
                reason=f"priority_rotation_from_{code}_to_highest_eligible_bullish",
            )
            self.guard_episode[code] = True
            self.guard_count += 1
            self.agent_guard_count += 1
            if self.cash < -1e-6:
                raise PKError(f"{self.group_id}优先级接力后现金为负: {self.cash}")


def _simulate_accounts(
    market: dict[str, dict],
    codes: list[str],
    signals: list[str],
    targets: dict[str, dict],
    nominal_signals: set[str],
    start_signal: str,
    end_as_of: str,
    stop_loss_sell_fraction: float = STOP_LOSS_SELL_FRACTION,
    reinvest_g7_protection_cash: bool = False,
    g7_agent_guard_sell_fraction: float = GUARD_SELL_FRACTION,
) -> dict[str, Account]:
    accounts = {group: Account(group, codes) for group in GROUPS}
    executable = {
        targets[signal]["exec_date"]: signal
        for signal in signals
        if targets[signal]["exec_date"] is not None
        and targets[signal]["exec_date"] <= end_as_of
    }
    weekly_execution_dates = set(executable)
    trading_dates = [
        value for value in market[codes[0]]["daily_by_date"]
        if start_signal <= value <= end_as_of
    ]
    if not trading_dates or trading_dates[0] != start_signal or trading_dates[-1] != end_as_of:
        raise PKError("主行情未覆盖完整起止日期")

    for current_date in trading_dates:
        close_prices = {
            code: market[code]["daily_by_date"][current_date]["raw_close"] for code in codes
        }
        if current_date == start_signal:
            for account in accounts.values():
                nav = account.value(close_prices)
                account.nav_rows.append(
                    {
                        "group_id": account.group_id,
                        "group_name": GROUPS[account.group_id],
                        "date": current_date,
                        "nav": nav,
                        "cash": account.cash,
                        "exposure_pct": 0.0,
                        "drawdown_pct": 0.0,
                    }
                )
            continue

        if current_date in weekly_execution_dates:
            signal = executable[current_date]
            open_prices = {
                code: market[code]["daily_by_date"][current_date]["raw_open"] for code in codes
            }
            signal_close_prices = {
                code: market[code]["daily_by_date"][signal]["raw_close"] for code in codes
            }
            g7_stop_loss_state = accounts[
                "G7_B0207_4W_AGENT_GUARD_STOP"
            ].stop_loss_snapshot(signal_close_prices)
            for account in accounts.values():
                account.cash *= CASH_WEEKLY_FACTOR
            # 周频组每周完整换仓。
            accounts["G2_B0207_WEEKLY"].rebalance(
                signal=signal, execution=current_date, target=targets[signal]["b0207"],
                open_prices=open_prices, reason="weekly_full_rebalance",
            )
            accounts["G4_AGENT_WEEKLY"].rebalance(
                signal=signal, execution=current_date, target=targets[signal]["agent"],
                open_prices=open_prices, reason="weekly_full_rebalance",
            )
            # 四周组只在名义决策周完整换仓，其余周仅看空保护。
            for group_id, source in (
                ("G1_B0207_4W_GUARD", "b0207"),
                ("G3_AGENT_4W_GUARD", "agent"),
                ("G5_B0207_4W_AGENT_GUARD", "b0207"),
                ("G7_B0207_4W_AGENT_GUARD_STOP", "b0207"),
            ):
                account = accounts[group_id]
                if signal in nominal_signals:
                    if group_id == "G7_B0207_4W_AGENT_GUARD_STOP":
                        account.rebalance_with_nominal_stop_overlay(
                            signal=signal,
                            execution=current_date,
                            target=targets[signal][source],
                            targets=targets,
                            stop_loss_state=g7_stop_loss_state,
                            open_prices=open_prices,
                            reinvest_protection_cash=reinvest_g7_protection_cash,
                            stop_loss_sell_fraction=stop_loss_sell_fraction,
                        )
                    else:
                        account.rebalance(
                            signal=signal, execution=current_date,
                            target=targets[signal][source], open_prices=open_prices,
                            reason="nominal_28day_full_rebalance",
                        )
                elif group_id == "G5_B0207_4W_AGENT_GUARD":
                    account.apply_agent_guard(
                        signal=signal,
                        execution=current_date,
                        targets=targets,
                        open_prices=open_prices,
                    )
                elif group_id == "G7_B0207_4W_AGENT_GUARD_STOP":
                    cash_before = account.cash
                    shares_before = dict(account.shares)
                    event_count_before = len(account.protection_rows)
                    account.apply_agent_guard_with_stop_loss(
                        signal=signal,
                        execution=current_date,
                        targets=targets,
                        stop_loss_state=g7_stop_loss_state,
                        open_prices=open_prices,
                        stop_loss_sell_fraction=stop_loss_sell_fraction,
                        agent_guard_sell_fraction=g7_agent_guard_sell_fraction,
                    )
                    if reinvest_g7_protection_cash:
                        new_events = account.protection_rows[event_count_before:]
                        account.reinvest_protection_cash_into_prior_holdings(
                            signal=signal,
                            execution=current_date,
                            prior_shares=shares_before,
                            protected_codes={row["code"] for row in new_events},
                            available_cash=max(0.0, account.cash - cash_before),
                            open_prices=open_prices,
                        )
                else:
                    account.apply_guard(
                        signal=signal, execution=current_date, market=market,
                        open_prices=open_prices,
                    )

            # 第六组只在首次名义决策建立B0207初始仓位；此后名义周与普通周
            # 完全同口径，只有持仓触发V70看空时才进行优先级接力换仓。
            rotation = accounts["G6_PRIORITY_ROTATION_AGENT_GUARD"]
            if signal == start_signal:
                rotation.rebalance(
                    signal=signal,
                    execution=current_date,
                    target=targets[signal]["b0207"],
                    open_prices=open_prices,
                    reason="initial_nominal_b0207_seed",
                )
            rotation.apply_priority_rotation_guard(
                signal=signal,
                execution=current_date,
                targets=targets,
                open_prices=open_prices,
            )

        for account in accounts.values():
            nav = account.value(close_prices)
            holdings = nav - account.cash
            prior_peak = max((row["nav"] for row in account.nav_rows), default=INITIAL_CAPITAL)
            peak = max(prior_peak, nav)
            account.nav_rows.append(
                {
                    "group_id": account.group_id,
                        "group_name": GROUPS[account.group_id],
                    "date": current_date,
                    "nav": nav,
                    "cash": account.cash,
                    "exposure_pct": holdings / nav * 100.0 if nav > 0 else 0.0,
                    "drawdown_pct": (nav / peak - 1.0) * 100.0 if peak > 0 else 0.0,
                }
            )
            account.mark_close_peaks(close_prices)
    return accounts


def _account_metrics(account: Account, signal_dates: set[str]) -> dict:
    frame = pd.DataFrame(account.nav_rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["daily_return"] = frame["nav"].pct_change()
    peak = frame["nav"].cummax()
    drawdown = frame["nav"] / peak - 1.0
    trough_idx = int(drawdown.idxmin())
    peak_idx = int(frame.loc[:trough_idx, "nav"].idxmax())
    final_nav = float(frame.iloc[-1]["nav"])
    cumulative = final_nav / INITIAL_CAPITAL - 1.0
    days = max(1, int((frame.iloc[-1]["date"] - frame.iloc[0]["date"]).days))
    annual = (final_nav / INITIAL_CAPITAL) ** (365.0 / days) - 1.0
    daily = frame["daily_return"].dropna()
    rf_daily = (1.0 + CASH_APR) ** (1.0 / 252.0) - 1.0
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float((daily.mean() - rf_daily) / std * math.sqrt(252.0)) if std > 0 else 0.0
    volatility = std * math.sqrt(252.0)
    weekly = frame.loc[frame["date"].dt.date.astype(str).isin(signal_dates), ["date", "nav"]].copy()
    weekly["return"] = weekly["nav"].pct_change()
    wr = weekly["return"].dropna()
    positive = wr[wr > 0]
    negative = wr[wr < 0]
    win_rate = float((wr > 0).mean()) if len(wr) else 0.0
    pnl_ratio = (
        float(positive.mean() / abs(negative.mean()))
        if len(positive) and len(negative) and negative.mean() != 0 else float("nan")
    )
    return {
        "group_id": account.group_id,
        "group_name": GROUPS[account.group_id],
        "initial_capital": INITIAL_CAPITAL,
        "final_nav": final_nav,
        "profit_cny": final_nav - INITIAL_CAPITAL,
        "cumulative_return": cumulative,
        "annualized_return": annual,
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_peak_date": frame.loc[peak_idx, "date"].date().isoformat(),
        "max_drawdown_trough_date": frame.loc[trough_idx, "date"].date().isoformat(),
        "annualized_sharpe": sharpe,
        "annualized_volatility": volatility,
        "weekly_win_rate": win_rate,
        "weekly_profit_loss_ratio": pnl_ratio,
        "average_exposure": float(frame["exposure_pct"].mean()) / 100.0,
        "turnover_over_initial": account.traded_notional / INITIAL_CAPITAL,
        "trade_orders": len(account.trade_rows),
        "guard_triggers": account.guard_count,
        "agent_guard_triggers": account.agent_guard_count,
        "stop_loss_triggers": account.stop_loss_count,
        "commission_total": account.costs,
        "final_cash": account.cash,
        "final_positions": {code: account.shares[code] for code in account.codes},
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise PKError(f"拒绝写入空表: {path.name}")
    columns = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def run(output_dir: Path | None = None) -> dict:
    config = load_config()
    codes = list(config["priority"])
    market = _load_market_bundle(codes)
    master_signals = sorted(market[codes[0]]["by_signal"])
    end_as_of = min(_iso(market[code]["daily"]["date"].iloc[-1]) for code in codes)
    available = [value for value in master_signals if value <= end_as_of]
    if "2026-01-09" not in available:
        raise PKError("缺少2026年首个名义决策所需信号2026-01-09")
    signals = [value for value in available if value >= "2026-01-09"]
    targets, signal_rows = _signal_targets(market, codes, signals)
    nominal_map = nominal_signal_dates(signals, end_as_of)
    nominal_signals = set(nominal_map.values())
    accounts = _simulate_accounts(
        market, codes, signals, targets, nominal_signals, signals[0], end_as_of,
        stop_loss_sell_fraction=STOP_LOSS_SELL_FRACTION,
        reinvest_g7_protection_cash=True,
    )
    comparison_accounts_70 = _simulate_accounts(
        market, codes, signals, targets, nominal_signals, signals[0], end_as_of,
        stop_loss_sell_fraction=GUARD_SELL_FRACTION,
        reinvest_g7_protection_cash=True,
    )
    cash_wait_accounts = _simulate_accounts(
        market, codes, signals, targets, nominal_signals, signals[0], end_as_of,
        stop_loss_sell_fraction=STOP_LOSS_SELL_FRACTION,
        reinvest_g7_protection_cash=False,
    )
    agent_guard_100_accounts = _simulate_accounts(
        market, codes, signals, targets, nominal_signals, signals[0], end_as_of,
        stop_loss_sell_fraction=STOP_LOSS_SELL_FRACTION,
        reinvest_g7_protection_cash=True,
        g7_agent_guard_sell_fraction=1.0,
    )
    signal_set = set(signals)
    metrics = [_account_metrics(accounts[group], signal_set) for group in GROUPS]
    metrics.sort(key=lambda row: row["cumulative_return"], reverse=True)
    metric_by_id = {row["group_id"]: row for row in metrics}
    stop_loss_70_metric = _account_metrics(
        comparison_accounts_70["G7_B0207_4W_AGENT_GUARD_STOP"], signal_set
    )
    stop_loss_100_metric = metric_by_id["G7_B0207_4W_AGENT_GUARD_STOP"]
    reinvest_metric = stop_loss_100_metric
    cash_wait_metric = _account_metrics(
        cash_wait_accounts["G7_B0207_4W_AGENT_GUARD_STOP"], signal_set
    )
    agent_guard_70_account = accounts["G7_B0207_4W_AGENT_GUARD_STOP"]
    agent_guard_100_account = agent_guard_100_accounts["G7_B0207_4W_AGENT_GUARD_STOP"]
    agent_guard_70_metric = stop_loss_100_metric
    agent_guard_100_metric = _account_metrics(agent_guard_100_account, signal_set)

    agent_guard_fraction_comparison = {
        "controlled_variable": "only_v70_bearish_sell_fraction",
        "unchanged": [
            "b0207_28day_rebalance",
            "holding_peak_stop_strictly_over_10pct_sell_100pct",
            "reinvest_into_other_prior_holdings_pro_rata",
            "execution_and_cost_assumptions",
        ],
        "v70_sell_70_pct": agent_guard_70_metric,
        "v70_sell_100_pct": agent_guard_100_metric,
        "v70_sell_100_minus_70_return_pp": 100.0 * (
            agent_guard_100_metric["cumulative_return"]
            - agent_guard_70_metric["cumulative_return"]
        ),
        "v70_sell_100_minus_70_mdd_pp": 100.0 * (
            agent_guard_100_metric["max_drawdown"]
            - agent_guard_70_metric["max_drawdown"]
        ),
    }
    reinvest_comparison = {
        "cash_wait": cash_wait_metric,
        "reinvest_prior_holdings": reinvest_metric,
        "reinvest_minus_cash_return_pp": 100.0 * (
            reinvest_metric["cumulative_return"]
            - cash_wait_metric["cumulative_return"]
        ),
        "reinvest_minus_cash_mdd_pp": 100.0 * (
            reinvest_metric["max_drawdown"] - cash_wait_metric["max_drawdown"]
        ),
    }
    stop_loss_fraction_comparison = {
        "sell_70_pct": stop_loss_70_metric,
        "sell_100_pct": stop_loss_100_metric,
        "sell_100_minus_70_return_pp": 100.0 * (
            stop_loss_100_metric["cumulative_return"] - stop_loss_70_metric["cumulative_return"]
        ),
        "sell_100_minus_70_mdd_pp": 100.0 * (
            stop_loss_100_metric["max_drawdown"] - stop_loss_70_metric["max_drawdown"]
        ),
    }
    comparisons = {
        "b0207_four_week_minus_weekly_return_pp": 100.0 * (
            metric_by_id["G1_B0207_4W_GUARD"]["cumulative_return"]
            - metric_by_id["G2_B0207_WEEKLY"]["cumulative_return"]
        ),
        "agent_weekly_minus_four_week_return_pp": 100.0 * (
            metric_by_id["G4_AGENT_WEEKLY"]["cumulative_return"]
            - metric_by_id["G3_AGENT_4W_GUARD"]["cumulative_return"]
        ),
        "four_week_b0207_minus_agent_return_pp": 100.0 * (
            metric_by_id["G1_B0207_4W_GUARD"]["cumulative_return"]
            - metric_by_id["G3_AGENT_4W_GUARD"]["cumulative_return"]
        ),
        "weekly_b0207_minus_agent_return_pp": 100.0 * (
            metric_by_id["G2_B0207_WEEKLY"]["cumulative_return"]
            - metric_by_id["G4_AGENT_WEEKLY"]["cumulative_return"]
        ),
        "agent_guard_minus_five_parameter_guard_return_pp": 100.0 * (
            metric_by_id["G5_B0207_4W_AGENT_GUARD"]["cumulative_return"]
            - metric_by_id["G1_B0207_4W_GUARD"]["cumulative_return"]
        ),
        "agent_guard_minus_five_parameter_guard_mdd_pp": 100.0 * (
            metric_by_id["G5_B0207_4W_AGENT_GUARD"]["max_drawdown"]
            - metric_by_id["G1_B0207_4W_GUARD"]["max_drawdown"]
        ),
        "priority_rotation_minus_agent_cash_guard_return_pp": 100.0 * (
            metric_by_id["G6_PRIORITY_ROTATION_AGENT_GUARD"]["cumulative_return"]
            - metric_by_id["G5_B0207_4W_AGENT_GUARD"]["cumulative_return"]
        ),
        "priority_rotation_minus_agent_cash_guard_mdd_pp": 100.0 * (
            metric_by_id["G6_PRIORITY_ROTATION_AGENT_GUARD"]["max_drawdown"]
            - metric_by_id["G5_B0207_4W_AGENT_GUARD"]["max_drawdown"]
        ),
        "stop_loss_plus_agent_guard_minus_agent_guard_return_pp": 100.0 * (
            metric_by_id["G7_B0207_4W_AGENT_GUARD_STOP"]["cumulative_return"]
            - metric_by_id["G5_B0207_4W_AGENT_GUARD"]["cumulative_return"]
        ),
        "stop_loss_plus_agent_guard_minus_agent_guard_mdd_pp": 100.0 * (
            metric_by_id["G7_B0207_4W_AGENT_GUARD_STOP"]["max_drawdown"]
            - metric_by_id["G5_B0207_4W_AGENT_GUARD"]["max_drawdown"]
        ),
    }

    stop_loss_reviews = accounts["G7_B0207_4W_AGENT_GUARD_STOP"].stop_loss_review_rows
    worst_stop_loss_review = (
        min(stop_loss_reviews, key=lambda row: row["drawdown_from_holding_peak"])
        if stop_loss_reviews else None
    )
    payload = {
        "schema_version": "seven-way-2026-pk-v1",
        "status": "RETROSPECTIVE_RECONSTRUCTION",
        "window": {
            "first_nominal_decision_date": FIRST_NOMINAL_DECISION.isoformat(),
            "first_signal_date": signals[0],
            "first_execution_date": targets[signals[0]]["exec_date"],
            "market_data_as_of": end_as_of,
            "last_executed_signal_date": max(
                signal for signal in signals if targets[signal]["exec_date"] is not None
                and targets[signal]["exec_date"] <= end_as_of
            ),
            "unexecuted_signal_dates": [
                signal for signal in signals
                if targets[signal]["exec_date"] is None or targets[signal]["exec_date"] > end_as_of
            ],
            "nominal_decision_to_signal": nominal_map,
        },
        "account_assumptions": {
            "initial_capital_cny": INITIAL_CAPITAL,
            "external_cashflows": 0,
            "execution_price": "next_trading_day_raw_open",
            "valuation_price": "raw_close",
            "lot_size": LOT,
            "commission_rate_one_way": COMMISSION_RATE,
            "minimum_commission_cny": MIN_COMMISSION,
            "stamp_duty": 0,
            "cash_apr": CASH_APR,
            "five_parameter_guard": {
                "conditions": [
                    "d_dif1<0", "d_dif2<0", "w_dif1<0", "w_dif2<0", "mom20<-5%"
                ],
                "sell_fraction": GUARD_SELL_FRACTION,
                "once_per_continuous_episode": True,
            },
            "agent_bearish_guard": {
                "trigger": "agent_v70_direction_is_bearish",
                "bearish_rules": ["R002", "R003", "R008"],
                "sell_fraction": GUARD_SELL_FRACTION,
                "once_per_continuous_episode": True,
                "comparison_design": "与第一组保持相同卖出比例，只替换保护触发条件",
            },
            "priority_rotation_guard": {
                "initial_position": "first_nominal_decision_b0207_target",
                "subsequent_rebalance": "agent_v70_bearish_trigger_only",
                "sell_fraction": GUARD_SELL_FRACTION,
                "once_per_continuous_episode": True,
                "priority_order": codes,
                "recipient": (
                    "最高优先级看多ETF；向较低优先级接力前，卖出标的之前的"
                    "所有高优先级ETF必须同时看空"
                ),
                "cash_treatment": "保护卖出款尽量全部买入接力标的，仅保留整手和佣金零头",
            },
            "single_instrument_stop_loss": {
                "base_group": "G5_B0207_4W_AGENT_GUARD",
                "reference": "signal_day_raw_close_vs_highest_daily_raw_close_during_continuous_holding",
                "trigger": "single_instrument_drawdown_strictly_below_minus_10pct",
                "threshold": STOP_LOSS_THRESHOLD,
                "sell_fraction": STOP_LOSS_SELL_FRACTION,
                "execution": "next_trading_day_raw_open",
                "cash_treatment": "reinvest_into_other_prior_holdings_pro_rata; cash_only_if_no_eligible_recipient",
                "scope": "all_weekly_signals; on_nominal_weeks_stop_loss_overlays_b0207_target",
                "independent_of_agent_v70": True,
                "once_per_continuous_below_threshold_episode": True,
            },
        },
        "method_limits": [
            "B0207历史信号由当前冻结冠军参数在历史行情上回放，未重新训练，但含当前模型选择的事后性。",
            "Agent结果由当前v70规则逐期确定性重建，不是当时真实Agent建议，也不是独立样本外成绩。",
            "所有当期动作只使用信号日及以前特征；未发生的下一交易日不成交。",
        ],
        "champions": {code: market[code]["champion_id"] for code in codes},
        "comparisons": comparisons,
        "stop_loss_sell_fraction_comparison": stop_loss_fraction_comparison,
        "agent_guard_sell_fraction_comparison": agent_guard_fraction_comparison,
        "g7_reinvestment_comparison": reinvest_comparison,
        "stop_loss_events": [
            row
            for row in accounts["G7_B0207_4W_AGENT_GUARD_STOP"].protection_rows
            if row["stop_loss_triggered"]
        ],
        "stop_loss_review_summary": {
            "observations": len(stop_loss_reviews),
            "worst_observation": worst_stop_loss_review,
        },
        "metrics": metrics,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for rank, row in enumerate(metrics, 1):
            summary_rows.append(
                {
                    "收益排名": rank,
                    "组别": row["group_name"],
                    "期初资金": round(row["initial_capital"], 2),
                    "期末资产": round(row["final_nav"], 2),
                    "盈亏金额": round(row["profit_cny"], 2),
                    "累计收益率": _fmt_pct(row["cumulative_return"]),
                    "年化收益率": _fmt_pct(row["annualized_return"]),
                    "最大回撤": _fmt_pct(row["max_drawdown"]),
                    "最大回撤区间": f"{row['max_drawdown_peak_date']}→{row['max_drawdown_trough_date']}",
                    "年化Sharpe": round(row["annualized_sharpe"], 3),
                    "年化波动率": _fmt_pct(row["annualized_volatility"]),
                    "周胜率": _fmt_pct(row["weekly_win_rate"]),
                    "周盈亏比": "" if math.isnan(row["weekly_profit_loss_ratio"]) else round(row["weekly_profit_loss_ratio"], 3),
                    "平均持仓": _fmt_pct(row["average_exposure"]),
                    "换手率_成交额除以期初": _fmt_pct(row["turnover_over_initial"]),
                    "成交指令数": row["trade_orders"],
                    "保护触发次数": row["guard_triggers"],
                    "Agent保护触发次数": row["agent_guard_triggers"],
                    "10%止损触发次数": row["stop_loss_triggers"],
                    "累计佣金": round(row["commission_total"], 2),
                    "期末现金": round(row["final_cash"], 2),
                }
            )
        _write_csv(output_dir / "七方案PK汇总.csv", summary_rows)
        _write_csv(
            output_dir / "第七组V70卖出70与100对比.csv",
            [
                {
                    "V70看空卖出比例": label,
                    "期末资产": round(row["final_nav"], 2),
                    "累计收益率": _fmt_pct(row["cumulative_return"]),
                    "最大回撤": _fmt_pct(row["max_drawdown"]),
                    "最大回撤区间": f"{row['max_drawdown_peak_date']}→{row['max_drawdown_trough_date']}",
                    "年化Sharpe": round(row["annualized_sharpe"], 3),
                    "平均持仓": _fmt_pct(row["average_exposure"]),
                    "成交指令数": row["trade_orders"],
                    "V70保护触发次数": row["agent_guard_triggers"],
                    "10%止损触发次数": row["stop_loss_triggers"],
                    "累计佣金": round(row["commission_total"], 2),
                    "期末现金": round(row["final_cash"], 2),
                }
                for label, row in (("70%（当前第七组）", agent_guard_70_metric), ("100%（新增对照）", agent_guard_100_metric))
            ],
        )
        _write_csv(
            output_dir / "第七组V70卖出100逐笔交易.csv",
            agent_guard_100_account.trade_rows,
        )
        _write_csv(
            output_dir / "第七组V70卖出100逐日净值.csv",
            agent_guard_100_account.nav_rows,
        )
        _write_csv(
            output_dir / "第七组止损70与100对比.csv",
            [
                {
                    "止损卖出比例": label,
                    "期末资产": round(row["final_nav"], 2),
                    "累计收益率": _fmt_pct(row["cumulative_return"]),
                    "最大回撤": _fmt_pct(row["max_drawdown"]),
                    "最大回撤区间": f"{row['max_drawdown_peak_date']}→{row['max_drawdown_trough_date']}",
                    "年化Sharpe": round(row["annualized_sharpe"], 3),
                    "平均持仓": _fmt_pct(row["average_exposure"]),
                    "成交指令数": row["trade_orders"],
                    "止损触发次数": row["stop_loss_triggers"],
                    "累计佣金": round(row["commission_total"], 2),
                    "期末现金": round(row["final_cash"], 2),
                }
                for label, row in (("70%", stop_loss_70_metric), ("100%", stop_loss_100_metric))
            ],
        )
        _write_csv(
            output_dir / "第七组保护卖出现金与回投对比.csv",
            [
                {
                    "方案": label,
                    "期末资产": round(row["final_nav"], 2),
                    "累计收益": _fmt_pct(row["cumulative_return"]),
                    "最大回撤": _fmt_pct(row["max_drawdown"]),
                    "年化波动": _fmt_pct(row["annualized_volatility"]),
                    "Sharpe": round(row["annualized_sharpe"], 3),
                    "平均持仓": _fmt_pct(row["average_exposure"]),
                    "成交指令数": row["trade_orders"],
                    "累计佣金": round(row["commission_total"], 2),
                    "期末现金": round(row["final_cash"], 2),
                }
                for label, row in (
                    ("保护卖出转现金", cash_wait_metric),
                    ("保护卖出款回投其他原有持仓", reinvest_metric),
                )
            ],
        )
        _write_csv(
            output_dir / "第七组回投对照逐笔交易.csv",
            accounts["G7_B0207_4W_AGENT_GUARD_STOP"].trade_rows,
        )
        _write_csv(
            output_dir / "第七组回投对照逐日净值.csv",
            accounts["G7_B0207_4W_AGENT_GUARD_STOP"].nav_rows,
        )
        nav_rows = [row for group in GROUPS for row in accounts[group].nav_rows]
        _write_csv(output_dir / "逐日净值与回撤.csv", nav_rows)
        trade_rows = [row for group in GROUPS for row in accounts[group].trade_rows]
        _write_csv(output_dir / "逐笔交易.csv", trade_rows)
        _write_csv(output_dir / "逐周信号与目标仓位.csv", signal_rows)
        _write_csv(
            output_dir / "第七组止损触发明细.csv",
            payload["stop_loss_events"],
            fieldnames=[
                "group_id", "group_name", "signal_date", "execution_date", "code", "name",
                "agent_triggered", "agent_rule", "stop_loss_triggered",
                "holding_peak_raw_close", "signal_raw_close", "drawdown_from_holding_peak",
                "nominal_week", "sell_fraction", "shares_sold",
            ],
        )
        _write_csv(
            output_dir / "第七组止损逐周检查.csv",
            stop_loss_reviews,
        )
        (output_dir / "七方案PK结果.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

        lines = [
            "# 2026年七方案交易PK",
            "",
            f"回放范围：名义决策日 {FIRST_NOMINAL_DECISION.isoformat()} 起，"
            f"首笔于 {targets[signals[0]]['exec_date']} 开盘执行，估值截至 {end_as_of} 收盘。",
            "统一期初资金：200,000元；无追加资金。",
            "",
            "| 排名 | 方案 | 期末资产 | 累计收益 | 年化收益 | 最大回撤 | Sharpe | 周胜率 | 周盈亏比 | 平均持仓 | 换手率 | 交易数 | 保护次数 | 止损次数 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(metrics, 1):
            ratio = "-" if math.isnan(row["weekly_profit_loss_ratio"]) else f"{row['weekly_profit_loss_ratio']:.2f}"
            lines.append(
                f"| {rank} | {row['group_name']} | {row['final_nav']:,.2f} | "
                f"{row['cumulative_return']:.2%} | {row['annualized_return']:.2%} | "
                f"{row['max_drawdown']:.2%} | {row['annualized_sharpe']:.2f} | "
                f"{row['weekly_win_rate']:.2%} | {ratio} | {row['average_exposure']:.2%} | "
                f"{row['turnover_over_initial']:.2%} | {row['trade_orders']} | {row['guard_triggers']} | "
                f"{row['stop_loss_triggers']} |"
            )
        lines += [
            "",
            "## 关键比较",
            "",
            f"- B0207四周组比B0207每周组高 {comparisons['b0207_four_week_minus_weekly_return_pp']:.2f} 个收益百分点；"
            f"每周组换手率为 {metric_by_id['G2_B0207_WEEKLY']['turnover_over_initial']:.2%}，"
            f"四周组为 {metric_by_id['G1_B0207_4W_GUARD']['turnover_over_initial']:.2%}。",
            f"- Agent每周组比Agent四周组高 {comparisons['agent_weekly_minus_four_week_return_pp']:.2f} 个收益百分点，"
            f"但最大回撤由 {metric_by_id['G3_AGENT_4W_GUARD']['max_drawdown']:.2%} 扩大到 "
            f"{metric_by_id['G4_AGENT_WEEKLY']['max_drawdown']:.2%}，换手也明显增加。",
            f"- 同为四周频率时，B0207比Agent高 {comparisons['four_week_b0207_minus_agent_return_pp']:.2f} 个收益百分点；"
            f"同为每周频率时高 {comparisons['weekly_b0207_minus_agent_return_pp']:.2f} 个收益百分点。",
            f"- 四周B0207仅在 {metric_by_id['G1_B0207_4W_GUARD']['guard_triggers']} 个连续看空区间触发保护；"
            "本窗口的主要差异来自调仓频率与目标组合变化，不是大量保护卖出。",
            f"- 第五组Agent看空保护相对第一组的累计收益差为 "
            f"{comparisons['agent_guard_minus_five_parameter_guard_return_pp']:+.2f} 个百分点，"
            f"最大回撤差为 {comparisons['agent_guard_minus_five_parameter_guard_mdd_pp']:+.2f} 个百分点；"
            f"Agent保护共触发 {metric_by_id['G5_B0207_4W_AGENT_GUARD']['guard_triggers']} 次。",
            f"- 第六组优先级接力相对第五组现金保护的累计收益差为 "
            f"{comparisons['priority_rotation_minus_agent_cash_guard_return_pp']:+.2f} 个百分点，"
            f"最大回撤差为 {comparisons['priority_rotation_minus_agent_cash_guard_mdd_pp']:+.2f} 个百分点；"
            f"共完成 {metric_by_id['G6_PRIORITY_ROTATION_AGENT_GUARD']['guard_triggers']} 次保护接力。",
            f"- 第七组相对第五组的累计收益差为 "
            f"{comparisons['stop_loss_plus_agent_guard_minus_agent_guard_return_pp']:+.2f} 个百分点，"
            f"最大回撤差为 {comparisons['stop_loss_plus_agent_guard_minus_agent_guard_mdd_pp']:+.2f} 个百分点；"
            f"10%持有期高点回撤止损触发 {metric_by_id['G7_B0207_4W_AGENT_GUARD_STOP']['stop_loss_triggers']} 次。",
            f"- 将10%持有期高点止损从卖出70%改为清仓100%后，累计收益变化 "
            f"{stop_loss_fraction_comparison['sell_100_minus_70_return_pp']:+.2f} 个百分点，最大回撤变化 "
            f"{stop_loss_fraction_comparison['sell_100_minus_70_mdd_pp']:+.2f} 个百分点；"
            "正值表示100%清仓口径更高或回撤较浅。",
            f"- 第七组保护资金回投其他原有持仓，相对卖出转现金的累计收益差为 "
            f"{reinvest_comparison['reinvest_minus_cash_return_pp']:+.2f} 个百分点，最大回撤差为 "
            f"{reinvest_comparison['reinvest_minus_cash_mdd_pp']:+.2f} 个百分点。",
            f"- 仅将第七组V70看空卖出比例由70%提高至100%后，全周期累计收益变化 "
            f"{agent_guard_fraction_comparison['v70_sell_100_minus_70_return_pp']:+.2f} 个百分点，"
            f"最大回撤变化 {agent_guard_fraction_comparison['v70_sell_100_minus_70_mdd_pp']:+.2f} 个百分点；"
            "10%持有期高点止损仍为清仓100%，其他规则不变。",
            (
                f"- 第七组共检查 {len(stop_loss_reviews)} 个周度持仓观察（含名义决策周）；"
                f"最深单标的持有期高点回撤为 {worst_stop_loss_review['drawdown_from_holding_peak']:.2%}"
                f"（{worst_stop_loss_review['signal_date']}，{worst_stop_loss_review['code']} "
                f"{worst_stop_loss_review['name']}）。"
                if worst_stop_loss_review else "- 第七组没有可检查的持仓观察。"
            ),
            "",
            "## 统一口径",
            "",
            "- 信号只使用当期周收盘及以前数据，下一交易日raw_open成交，最终按raw_close估值。",
            "- 100份整手；单边佣金为成交额0.025%，每笔最低5元；ETF不计印花税。",
            "- 现金年化1.75%，按周复利；七组使用相同公司行动记录。",
            "- 五参数保护要求日/周DIF一阶和二阶全部为负且20日动量低于-5%，每个连续触发区间只减一次、卖出70%现有份额。",
            "- Agent保护在非决策周出现v70看空判断时触发；为公平比较，同样每个连续看空区间只减一次、卖出70%现有份额。",
            "- 第六组首次按B0207建仓，之后名义决策周和普通周都不做常规整体调仓；只有持仓V70看空时才把70%保护份额按固定优先级接力到看多ETF。",
            "- 第六组若存在更高优先级看多ETF则向上接力；只有全部更高优先级ETF同时看空时才允许向后接力。保护卖出款不主动留作现金，整手与佣金零头除外。",
            "- 第七组保留第五组B0207四周完整调仓与V70保护；V70新看空区间仍卖出70%。任一周若信号日raw_close相对本次连续持有期间的日收盘最高价回撤严格超过10%，无需其他条件即可在下一交易日清仓100%。保护卖出款按其他原有持仓市值比例回投，不回买被保护ETF、不新建此前未持有ETF；没有合格接收标的时才留现金。名义决策周先合并目标、止损和回投后一次性净额成交。",
            "",
            "## 解释边界",
            "",
            "- 项目正式策略ID为B0207；M0207是旧称，本报告不把它们当作两套模型。",
            "- B0207使用当前冻结冠军回放历史行情，因此比较的是当前模型规则的事后回放，不是每个历史时点当时真实发布的机械记录。",
            "- Agent使用当前v70规则逐期重建，不是历史真实Agent建议，不能作为独立样本外收益证明。",
            f"- {', '.join(payload['window']['unexecuted_signal_dates']) or '无'}没有可用的下一交易日执行价，因此未成交。",
            "",
            "逐日净值、逐笔交易、逐周信号、第七组止损逐周检查、触发明细、止损比例对比及V70卖出比例对比见同目录CSV。",
        ]
        (output_dir / "七方案PK报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="2026年七方案交易PK")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "champion_vs_dca" / "2026七方案交易PK",
    )
    parser.add_argument("--json", action="store_true", help="同时在终端输出JSON")
    args = parser.parse_args()
    payload = run(args.output_dir.resolve())
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        for row in payload["metrics"]:
            print(
                f"{row['group_name']}: 期末 {row['final_nav']:.2f}, "
                f"收益 {row['cumulative_return']:.2%}, 回撤 {row['max_drawdown']:.2%}, "
                f"Sharpe {row['annualized_sharpe']:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
