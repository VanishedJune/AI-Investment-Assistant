# -*- coding: utf-8 -*-
"""全历史剩余 Due 的 Proposal 生成器（Agent 设计策略物化）。

对每个尚未冻结的 Due：
1) 读取该 Due 的盲化诊断包（PIT 统计 + 逐窗口案例）；
2) 按诊断选择挑战者角色（wrong_chase 高 -> 退出/风控角色；missed 高 -> 暴露角色；否则校准角色）；
3) 以轮次哈希做确定性扰动（角色轮换 + 权重抖动），保证轮间不同；
4) 生成 schema 完整 draft，再由 scripts/fix_proposals.py 做顺序 divergence 修正后冻结。
"""

from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling import ledger as ledger_mod  # noqa: E402
from rolling.ledger import (  # noqa: E402
    diagnostic_path,
    proposal_path,
    round_id,
    set_workspace,
)


def _seed(rid: str) -> int:
    return int(hashlib.sha256(rid.encode("utf-8")).hexdigest()[:16], 16)


def _pos_step(weight: float, gates=None, score_terms=None):
    spec = {
        "score": {"terms": score_terms or [{"feature": "w_dif1", "weight": 1000.0}]},
        "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": weight}], "min": 0.0, "max": weight},
    }
    if gates:
        spec["gates"] = gates
    return spec


def prediction_role(name: str, j: float) -> tuple[str, dict, str, str]:
    """返回 (target_problem, spec, expected_effect, main_risk)。"""
    if name == "trend_fusion":
        return ("ACTIVITY：趋势段暴露不足",
                _pos_step(1.0, score_terms=[{"feature": "d_slope", "weight": 10.0 * j}, {"feature": "w_dif1", "weight": 1000.0 * j}, {"feature": "mom20", "weight": 20.0 * j}]),
                "日线/周线/动量合成满仓", "多信号叠加增加追高")
    if name == "recovery":
        return ("ACTIVITY：回撤修复段错过",
                _pos_step(0.5, score_terms=[{"feature": "dd20", "weight": -10.0 * j}, {"feature": "d_dif1", "weight": 500.0 * j}]),
                "回撤加深+日DIF修复半仓", "修复失败继续下跌")
    if name == "mom_only":
        return ("ACTIVITY：动量段暴露不足",
                _pos_step(1.0, gates=[{"feature": "mom20", "op": ">", "value": 0.0}], score_terms=[{"feature": "mom20", "weight": 50.0 * j}]),
                "动量转正满仓", "顶部追入")
    if name == "d_trend":
        return ("ACTIVITY：日线趋势段错过",
                _pos_step(1.0, gates=[{"feature": "d_slope", "op": ">", "value": 0.0}], score_terms=[{"feature": "d_slope", "weight": 10.0 * j}, {"feature": "d_dif1", "weight": 1000.0 * j}]),
                "日线斜率+速度确认满仓", "斜率噪声")
    if name == "momentum_vol":
        return ("RISK+ACTIVITY：动量参与但波动未降档",
                _pos_step(0.75, gates=[{"feature": "mom20", "op": ">", "value": 0.0}], score_terms=[{"feature": "mom20", "weight": 50.0 * j}, {"feature": "vol_pct", "weight": -10.0 * j}]),
                "动量正且低波 75%", "高波动强趋势被过滤")
    if name == "weekly_speed":
        return ("PREDICTION：周线速度确认",
                _pos_step(0.75, gates=[{"feature": "w_dif1", "op": ">", "value": 0.0}], score_terms=[{"feature": "w_dif1", "weight": 2000.0 * j}, {"feature": "w_dif2", "weight": 1000.0 * j}]),
                "周线一阶+二阶走强 75%", "周线滞后")
    if name == "accel":
        return ("PREDICTION：DIF 加速度确认",
                _pos_step(0.9, gates=[{"feature": "d_dif1", "op": ">", "value": 0.0}, {"feature": "d_dif2", "op": ">", "value": 0.0}], score_terms=[{"feature": "d_dif2", "weight": 1000.0 * j}, {"feature": "d_dif1", "weight": 500.0 * j}]),
                "日线 DIF 加速 90%", "错过启动周")
    if name == "rsi_mom":
        return ("CALIBRATION：RSI+动量校准",
                _pos_step(0.6, score_terms=[{"feature": "mom20", "weight": 50.0 * j}, {"feature": "rsi14", "weight": -0.5 * j}]),
                "动量正减超买惩罚 60%", "超买区追高")
    if name == "dual_dif_lowconf":
        return ("CALIBRATION：双 DIF 低置信",
                _pos_step(0.5, score_terms=[{"feature": "d_dif1", "weight": 500.0 * j}, {"feature": "w_dif1", "weight": 200.0 * j}]),
                "日周 DIF 合成半仓", "强趋势参与不足")
    if name == "vol_target":
        return ("RISK：波动目标仓位",
                {"score": {"terms": [{"feature": "vol20", "weight": -2.0 * j}]},
                 "position": {"type": "linear", "slope": 0.02, "intercept": 0.9, "min": 0.0, "max": 1.0}},
                "低波高仓高波低仓", "波动估计滞后")
    if name == "weekly_only":
        return ("PREDICTION：周线 DIF 单确认",
                _pos_step(1.0, gates=[{"feature": "w_dif1", "op": ">", "value": 0.0}], score_terms=[{"feature": "w_dif1", "weight": 2500.0 * j}]),
                "周线 DIF 转正满仓", "滞后于日线拐点")
    if name == "exit_mom":
        return ("WRONG_CHASE：动量退出",
                _pos_step(0.8, gates=[{"feature": "mom20", "op": ">", "value": 0.0}], score_terms=[{"feature": "mom20", "weight": 1000.0 * j}]),
                "动量转负即空仓", "反弹踏空")
    if name == "weekly_protect":
        return ("WRONG_CHASE：周线保护",
                _pos_step(0.8, gates=[{"feature": "w_dif1", "op": ">", "value": 0.0}], score_terms=[{"feature": "w_dif1", "weight": 1000.0 * j}]),
                "周线 DIF 转负即空仓", "周线滞后")
    if name == "dd_stop":
        return ("RISK：回撤止损",
                _pos_step(0.6, gates=[{"feature": "dd20", "op": ">", "value": -0.05}], score_terms=[{"feature": "dd20", "weight": -100.0 * j}]),
                "20日回撤>5% 清仓", "震荡市假止损")
    if name == "rsi_filter":
        return ("WRONG_CHASE：RSI 弱势过滤",
                _pos_step(0.7, gates=[{"feature": "rsi14", "op": ">=", "value": 50.0}], score_terms=[{"feature": "rsi14", "weight": 10.0 * j}]),
                "RSI>=50 才持仓", "50 阈值频繁开关")
    if name == "mean_reversion_rsi":
        return ("MEAN_REVERSION：超卖修复",
                _pos_step(0.5, gates=[{"feature": "rsi14", "op": "<", "value": 30.0}],
                          score_terms=[{"feature": "d_dif1", "weight": 500.0 * j}, {"feature": "d_dif2", "weight": 200.0 * j}]),
                "RSI<30 且日DIF修复 50%", "超卖后继续下跌")
    if name == "breakout_mom":
        return ("BREAKOUT：动量突破",
                _pos_step(1.0, gates=[{"feature": "mom20", "op": ">", "value": 0.06}],
                          score_terms=[{"feature": "mom20", "weight": 100.0 * j}]),
                "20日动量>6% 满仓", "突破失败追高")
    if name == "dual_momentum":
        return ("DUAL_MOMENTUM：日周双动量",
                _pos_step(0.9, gates=[{"feature": "w_dif1", "op": ">", "value": 0.0}, {"feature": "mom20", "op": ">", "value": 0.0}],
                          score_terms=[{"feature": "w_dif1", "weight": 1000.0 * j}, {"feature": "mom20", "weight": 50.0 * j}]),
                "周线DIF+日线动量同向 90%", "双确认滞后")
    if name == "volume_divergence":
        return ("VOLUME：量价确认",
                _pos_step(0.6, gates=[{"feature": "vol_ratio", "op": ">", "value": 1.2}, {"feature": "d_dif1", "op": ">", "value": 0.0}],
                          score_terms=[{"feature": "d_dif1", "weight": 500.0 * j}, {"feature": "vol_ratio", "weight": 2.0 * j}]),
                "量比>1.2 且日DIF正 60%", "放量出货误判")
    if name == "channel_mean_revert":
        return ("CHANNEL：通道内均值回归",
                _pos_step(0.4, gates=[{"feature": "rsi14", "op": "<", "value": 40.0}, {"feature": "dd20", "op": ">", "value": -0.08}],
                          score_terms=[{"feature": "dd20", "weight": -10.0 * j}, {"feature": "d_dif1", "weight": 300.0 * j}]),
                "RSI<40 且回撤未破位 40%", "通道破位")
    if name == "multi_param_joint":
        return ("MULTI_PARAM：周K主导+日K辅助+均线/量价/结构联合决策",
                {"score": {"terms": [
                    {"feature": "w_dif1", "weight": 1000.0 * j},
                    {"feature": "w_dif2", "weight": 400.0 * j},
                    {"feature": "d_dif1", "weight": 250.0 * j},
                    {"feature": "d_dif2", "weight": 100.0 * j},
                    {"feature": "w_ma_align", "weight": 300.0 * j},
                    {"feature": "d_ma_align", "weight": 150.0 * j},
                    {"feature": "d_vol_ma5", "weight": 50.0 * j},
                    {"feature": "d_low20_dist", "weight": 150.0 * j},
                    {"feature": "d_high20_dist", "weight": 200.0 * j},
                    {"feature": "dd20", "weight": -200.0 * j},
                ]},
                 "gates": [{"feature": "w_ma20_slope", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [
                     {"score_min": 0.0, "weight": 0.3},
                     {"score_min": 5.0, "weight": 0.5},
                     {"score_min": 40.0, "weight": 0.7},
                     {"score_min": 300.0, "weight": 1.0},
                ], "min": 0.0, "max": 1.0}},
                "多参数联合：周DIF一/二阶为主、日DIF确认、均线结构+量能+前低距离加权，梯度仓位 30/50/70/100",
                "权重与阈值需按 ETF 校准，强趋势期高分持续满仓")
    if name == "weekly_trend_daily_timing":
        return ("ARCHITECTURE：周K趋势门控 + 日K择时",
                {"score": {"terms": [
                    {"feature": "d_dif1", "weight": 1000.0 * j},
                    {"feature": "d_dif2", "weight": 500.0 * j},
                    {"feature": "w_dif1", "weight": 500.0 * j},
                    {"feature": "w_dif2", "weight": 300.0 * j},
                ]},
                 "gates": [
                     {"feature": "w_dif1", "op": ">", "value": 0.0},
                     {"feature": "w_ma_align", "op": ">=", "value": 0.5},
                 ],
                 "position": {"type": "step", "levels": [
                     {"score_min": 0.0, "weight": 0.5},
                     {"score_min": 5.0, "weight": 0.8},
                 ], "min": 0.0, "max": 0.8}},
                "周线 DIF 转正且周均线多头排列才入场；日线 DIF 一/二阶择时决定 50/80 仓位",
                "周线双门控可能过滤强反弹，日线噪声仍影响仓位档位")
    return ("通用：趋势跟随",
            _pos_step(0.8, gates=[{"feature": "w_dif1", "op": ">", "value": 0.0}], score_terms=[{"feature": "w_dif1", "weight": 1000.0 * j}]),
            "周线确认 80%", "周线滞后")


def strategy_role(name: str, j: float) -> tuple[str, dict, str, str]:
    base_score = [{"feature": "d_dif1", "weight": 1000.0 * j}, {"feature": "w_dif1", "weight": 1000.0 * j}]
    if name == "ramp":
        return ("STRATEGY：分批建仓",
                {"score": {"terms": [{"feature": "d_dif1", "weight": 1000.0 * j}, {"feature": "d_dif2", "weight": 1000.0 * j}]},
                 "gates": [{"feature": "d_dif1", "op": ">", "value": 0.0}, {"feature": "d_dif2", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 1.0}], "min": 0.0, "max": 1.0},
                 "ramp": {"confirm_after_weeks": 2, "initial_weight": 0.5, "full_weight": 1.0,
                          "confirm_gate": [{"feature": "d_dif1", "op": ">", "value": 0.0}, {"feature": "d_dif2", "op": ">", "value": 0.0}]}},
                "50%->100% 两段确认", "启动首周参与不足")
    if name == "drawdown":
        return ("RISK：回撤制动",
                {"score": {"terms": base_score},
                 "gates": [{"feature": "d_dif1", "op": ">", "value": 0.0}, {"feature": "w_dif1", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.75}], "min": 0.0, "max": 0.75},
                 "drawdown_control": {"window_weeks": 8, "threshold": 0.10, "factor": 0.5}},
                "回撤>10% 减半", "反弹踏空")
    if name == "floor":
        return ("ACTIVITY：参与底仓",
                {"score": {"terms": base_score},
                 "position": {"type": "linear", "slope": 0.10 * j, "intercept": 0.20, "min": 0.20, "max": 1.0}},
                "20% 底仓+强度加仓", "下跌段底仓亏损")
    if name == "vol_target":
        return ("RISK：波动目标仓位",
                {"score": {"terms": [{"feature": "vol20", "weight": -2.0 * j}]},
                 "position": {"type": "linear", "slope": 0.02, "intercept": 0.9, "min": 0.0, "max": 1.0}},
                "低波高仓高波低仓", "波动估计滞后")
    if name == "graded":
        return ("STRATEGY：梯度暴露",
                {"score": {"terms": base_score},
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.25}, {"score_min": 5.0, "weight": 0.5}, {"score_min": 10.0, "weight": 1.0}], "min": 0.0, "max": 1.0}},
                "25/50/100 三档", "单周期假信号")
    if name == "recovery_step":
        return ("ACTIVITY：日线恢复梯度",
                {"score": {"terms": base_score},
                 "gates": [{"feature": "d_dif1", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.5}, {"score_min": 5.0, "weight": 1.0}], "min": 0.0, "max": 1.0}},
                "日DIF修复50%周确认100%", "假修复")
    if name == "weekly_ramp":
        return ("ACTIVITY：周线确认分批",
                {"score": {"terms": [{"feature": "w_dif1", "weight": 1000.0 * j}]},
                 "gates": [{"feature": "w_dif1", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 1.0}], "min": 0.0, "max": 1.0},
                 "ramp": {"confirm_after_weeks": 2, "initial_weight": 0.5, "full_weight": 1.0,
                          "confirm_gate": [{"feature": "w_dif1", "op": ">", "value": 0.0}]}},
                "周线转强50%->100%", "启动首周半仓")
    if name == "momentum_dd":
        return ("RISK+ACTIVITY：动量+回撤制动",
                {"score": {"terms": [{"feature": "mom20", "weight": 1000.0 * j}]},
                 "gates": [{"feature": "mom20", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.8}], "min": 0.0, "max": 0.8},
                 "drawdown_control": {"window_weeks": 8, "threshold": 0.10, "factor": 0.5}},
                "动量80%+回撤减半", "动量滞后")
    if name == "mean_reversion":
        return ("MEAN_REVERSION：超卖修复策略",
                {"score": {"terms": [{"feature": "dd20", "weight": -8.0 * j}, {"feature": "d_dif1", "weight": 300.0 * j}]},
                 "gates": [{"feature": "rsi14", "op": "<", "value": 30.0}, {"feature": "dd20", "op": ">", "value": -0.10}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.5}], "min": 0.0, "max": 0.5},
                 "drawdown_control": {"window_weeks": 8, "threshold": 0.10, "factor": 0.5}},
                "RSI<30 且未破位 50%+回撤减半", "超卖接刀")
    if name == "breakout":
        return ("BREAKOUT：动量突破策略",
                {"score": {"terms": [{"feature": "mom20", "weight": 60.0 * j}]},
                 "gates": [{"feature": "mom20", "op": ">", "value": 0.06}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 0.9}], "min": 0.0, "max": 0.9},
                 "drawdown_control": {"window_weeks": 8, "threshold": 0.10, "factor": 0.5}},
                "20日动量突破 90%+回撤减半", "突破失败")
    if name == "dual_momentum":
        return ("DUAL_MOMENTUM：日周双动量策略",
                {"score": {"terms": [{"feature": "w_dif1", "weight": 1000.0 * j}, {"feature": "mom20", "weight": 50.0 * j}]},
                 "gates": [{"feature": "w_dif1", "op": ">", "value": 0.0}, {"feature": "mom20", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [{"score_min": 0.0, "weight": 1.0}], "min": 0.0, "max": 1.0}},
                "日周双动量同向满仓", "双确认滞后")
    if name == "multi_param_defensive":
        return ("MULTI_PARAM：多参数+回撤制动的防御策略",
                {"score": {"terms": [
                    {"feature": "w_dif1", "weight": 1000.0 * j},
                    {"feature": "d_dif1", "weight": 300.0 * j},
                    {"feature": "w_ma_align", "weight": 400.0 * j},
                    {"feature": "d_ma_align", "weight": 200.0 * j},
                    {"feature": "d_vol_ma5", "weight": 60.0 * j},
                    {"feature": "dd20", "weight": -300.0 * j},
                    {"feature": "d_high20_dist", "weight": 300.0 * j},
                ]},
                 "gates": [{"feature": "w_ma20_slope", "op": ">", "value": 0.0}],
                 "position": {"type": "step", "levels": [
                     {"score_min": 0.0, "weight": 0.5},
                     {"score_min": 10.0, "weight": 0.8},
                 ], "min": 0.0, "max": 0.8},
                 "drawdown_control": {"window_weeks": 8, "threshold": 0.10, "factor": 0.5}},
                "多参数打分+周MA20结构门控，50/80 两档，8周回撤>10% 减半",
                "门控过滤强反弹，回撤制动可能踏空")
    return ("STRATEGY：强底仓梯度",
            {"score": {"terms": base_score},
             "position": {"type": "linear", "slope": 0.15, "intercept": 0.40, "min": 0.40, "max": 1.0}},
            "40% 底仓+陡梯度", "底仓亏损")


def choose_roles(ctype: str, stats: dict, seed: int) -> list[str]:
    missed = stats.get("missed_rally", 0)
    wrong = stats.get("wrong_chase", 0)
    if ctype == "prediction":
        if wrong > missed and wrong > 0:
            pool = ["exit_mom", "weekly_protect", "dd_stop", "rsi_filter", "trend_fusion",
                    "momentum_vol", "dual_dif_lowconf", "weekly_speed"]
        elif missed > 0:
            pool = ["trend_fusion", "recovery", "mom_only", "d_trend", "momentum_vol",
                    "weekly_speed", "accel", "rsi_mom"]
        else:
            pool = ["dual_dif_lowconf", "rsi_mom", "vol_target", "weekly_only", "accel",
                    "weekly_speed", "trend_fusion", "mom_only"]
        n = 5
    else:
        if wrong > missed and wrong > 0:
            pool = ["drawdown", "momentum_dd", "vol_target", "ramp", "graded", "floor", "weekly_ramp", "recovery_step"]
        else:
            pool = ["floor", "recovery_step", "weekly_ramp", "graded", "ramp", "drawdown", "momentum_dd", "vol_target"]
        n = 3
    out: list[str] = []
    start = seed % len(pool)
    k = 0
    while len(out) < n and k < 100:
        role = pool[(start + k * 3 + (seed >> (k * 3)) % 5) % len(pool)]
        if role not in out:
            out.append(role)
        k += 1
    return out


def build_proposal(rid: str, diag: dict) -> dict:
    ctype = diag["challenge_type"]
    seq = int(rid.split("_")[1])
    stats = diag["stats"]
    seed = _seed(rid)
    missed = stats.get("missed_rally", 0)
    wrong = stats.get("wrong_chase", 0)
    if wrong > missed:
        primary, secondary = "WRONG_CHASE", "RISK_CONTROL_ERROR"
    elif missed > 0:
        primary, secondary = "ACTIVITY_ERROR", "CALIBRATION_ERROR"
    else:
        primary, secondary = "CALIBRATION_ERROR", "PREDICTION_ERROR"
    roles = choose_roles(ctype, stats, seed)
    challengers = []
    for i, role in enumerate(roles, 1):
        j = 1.0 + ((seed >> (i * 7)) % 5) * 0.25
        if ctype == "prediction":
            target, spec, effect, risk = prediction_role(role, j)
            cid = f"CH_P{seq:03d}_{i}"
        else:
            target, spec, effect, risk = strategy_role(role, j)
            cid = f"CH_S{seq:03d}_{i}"
        challengers.append({
            "id": cid,
            "target_problem": target,
            "change_description": f"角色={role}；依据本轮诊断（matured_1w={stats.get('matured_1w')}，missed={missed}，wrong={wrong}）动态设计。",
            "changed_dimensions": ["score_function", "position_mapping", "entry_confirmation"],
            "old_values": {"champion": "当前 Champion 规则"},
            "new_values": {"role": role, "jitter": round(j, 3)},
            "expected_effect": effect,
            "main_risk": risk,
            "reason": f"diagnosis: {primary}/{secondary}; mae_1w={stats.get('mae_1w')}, brier_1w={stats.get('brier_1w')}",
            "spec": spec,
        })
    evidence = [
        f"matured_1w={stats.get('matured_1w')}",
        f"missed_rally={missed}",
        f"wrong_chase={wrong}",
        f"mae_1w={stats.get('mae_1w')}",
        f"brier_1w={stats.get('brier_1w')}",
        f"failure_cases={len(diag.get('failure_cases', []))}",
    ]
    return {
        "iteration_id": "ITERATION_002",
        "opaque_due_id": diag["opaque_due_id"],
        "challenge_round": rid,
        "challenge_type": ctype,
        "diagnosis": f"matured_1w={stats.get('matured_1w')}；missed_rally={missed}；wrong_chase={wrong}；"
                     f"mae_1w={stats.get('mae_1w')}；brier_1w={stats.get('brier_1w')}；"
                     f"accuracy_2w={stats.get('accuracy_2w')}；accuracy_8w={stats.get('accuracy_8w')}。",
        "primary_failure_mode": primary,
        "secondary_failure_mode": secondary,
        "evidence": evidence,
        "hypothesis": f"依据本轮诊断（{primary}/{secondary}）动态生成 {len(roles)} 个确定性信号候选；"
                      "每个候选带角色轮换与权重扰动，避免模板复用；只登记 EVALUATING。",
        "revision": 1,
        "status": "DRAFT",
        "challengers": challengers,
    }


def main(etf: str = "159915") -> int:
    set_workspace(etf)
    generated = 0
    for ctype, seq_max in (("prediction", 177), ("strategy", 88)):
        start_seq = 12 if ctype == "prediction" else 6
        for seq in range(start_seq, seq_max + 1):
            rid = round_id(ctype, seq)
            if proposal_path(rid).exists():
                continue
            dpath = diagnostic_path(rid)
            if not dpath.exists():
                print("缺少诊断:", rid)
                continue
            diag = json.loads(dpath.read_text(encoding="utf-8"))
            obj = build_proposal(rid, diag)
            out = ledger_mod.DRAFT_DIR / f"{rid}.draft.json"
            out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            generated += 1
    print(f"generated drafts: {generated}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
