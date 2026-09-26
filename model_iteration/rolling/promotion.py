# -*- coding: utf-8 -*-
"""晋级检查点（promotion-v2.0.0 核心瘦身）。

门禁只保留核心指标：
- Prediction：1W/2W 方向准确率、MAE、p_up Brier（RECENT 仅 non-inferiority；
  EXPANDING 还需 1W/2W 至少一项相对改进 >=5%）。
- Strategy：累计收益、Sharpe、MDD（RECENT non-inferiority；EXPANDING 需 cum>=base+1pp）。

REGIME、强涨/强跌 Brier、4W/8W 预测指标、胜率、中位、参与率、平均仓位、
空仓率、换手、FULL_AVAILABLE_PIT 全部只进诊断输出，不进入 PASS/FAIL。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .account import anchor_exec_prices, run_account, weekly_metrics
from .challenge import champion_path, position_path, spec_cache_key
from .config import load_etf_config
from .statindex import EvalIndex, index_from_events

MIN_1W = 12
MIN_8W = 4
MAX_AGE = 104
MIN_REGIME = 6
DIVERGENCE = 0.05


def matured_week_indices(start: int, end_idx: int, fwd: np.ndarray, horizon: int = 1) -> list[int]:
    """截至 anchor end_idx 已成熟的窗口：j+horizon <= end_idx（严格 PIT，禁止未来窗口）。"""
    return [
        j for j in range(start, min(end_idx, len(fwd)))
        if j + horizon <= end_idx and not np.isnan(fwd[j])
    ]


def regime_label(close: np.ndarray, j: int) -> str | None:
    if j < 20 or np.isnan(close[j]) or np.isnan(close[j - 20]):
        return None
    r20 = close[j] / close[j - 20] - 1.0
    return "UP" if r20 >= 0.02 else "DOWN" if r20 <= -0.02 else "SIDEWAYS"


def bucket_metrics(returns: np.ndarray, navs: np.ndarray, week_indices: list[int], labels: dict[int, str]) -> dict:
    """REGIME 分桶指标（仅诊断，不参与门禁）。"""
    out = {}
    for label in ("UP", "DOWN", "SIDEWAYS"):
        idx = [j for j in week_indices if labels.get(j) == label]
        if len(idx) < MIN_REGIME:
            out[label] = {"n": len(idx), "status": "INSUFFICIENT_SAMPLE"}
            continue
        ret = np.asarray([returns[j] for j in idx])
        bucket_nav = np.cumprod(1.0 + ret)
        bucket_return = float(bucket_nav[-1] - 1.0)
        peak = np.maximum.accumulate(bucket_nav)
        mdd = float(np.nanmin(bucket_nav / peak - 1.0))
        out[label] = {"n": len(idx), "status": "EVALUATED", "bucket_return": round(bucket_return, 6), "bucket_mdd": round(mdd, 6)}
    return out


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _mae_brier_acc_from_index(
    idx: EvalIndex, model_id: str, start: int, end: int
) -> tuple[dict[int, float | None], dict[int, float | None], dict[int, float | None], dict[int, float | None]]:
    """返回 (MAE 均值, Brier(p_up) 均值, 方向准确率, Brier_strong 诊断均值) 按 horizon 分桶。"""
    mae: dict[int, float | None] = {}
    brier: dict[int, float | None] = {}
    brier_strong: dict[int, float | None] = {}
    acc: dict[int, float | None] = {}
    for k in (1, 2, 4, 8):
        m, b, a, s = idx.range_stats(model_id, k, start, end)
        mae[k], brier[k], acc[k], brier_strong[k] = m, b, a, s
    return mae, brier, acc, brier_strong


def _mae_brier_acc_from_events(
    events: list[dict], model_id: str, start: int, end: int
) -> tuple[dict[int, float | None], dict[int, float | None], dict[int, float | None], dict[int, float | None]]:
    """兼容测试/旧接口：由事件列表构建索引后查询。"""
    return _mae_brier_acc_from_index(index_from_events(events), model_id, start, end)


def _relative_improvement(ch: float | None, base: float | None) -> bool:
    if ch is None or base is None:
        return False
    if base > 0:
        return (base - ch) / base >= 0.05
    return ch <= base + 1e-9


def _noninferior(ch: float | None, base: float | None, factor: float = 1.05) -> bool:
    if ch is None or base is None:
        return False
    return ch <= base * factor


def _prediction_gates(ch: dict, base: dict, recent: bool) -> tuple[bool, dict]:
    """Prediction 核心门禁：仅 1W/2W 的 MAE、p_up Brier、方向准确率。"""
    def _num(x):
        return 0.0 if x is None else float(x)
    checks = {
        "mae_1w": _noninferior(ch.get("mae_1w"), base.get("mae_1w")),
        "mae_2w": _noninferior(ch.get("mae_2w"), base.get("mae_2w")),
        "brier_1w": _noninferior(ch.get("brier_1w"), base.get("brier_1w")),
        "acc_1w": _num(ch.get("acc_1w")) >= _num(base.get("acc_1w")) - 0.02,
        "acc_2w": _num(ch.get("acc_2w")) >= _num(base.get("acc_2w")) - 0.02,
    }
    if not recent:
        checks["improvement"] = (
            _relative_improvement(ch.get("mae_1w"), base.get("mae_1w"))
            or _relative_improvement(ch.get("mae_2w"), base.get("mae_2w"))
            or _relative_improvement(ch.get("brier_1w"), base.get("brier_1w"))
        )
    return all(checks.values()), checks


def _strategy_gates(ch: dict, base: dict, recent: bool, penalize_dd: bool = True) -> tuple[bool, dict]:
    """Strategy 核心门禁：累计收益、Sharpe、MDD（回撤是否扣分可配置）。"""
    def _num(x):
        return 0.0 if x is None else float(x)
    cum_floor = _num(base.get("cum")) + (-0.01 if recent else 0.01)
    if penalize_dd:
        checks = {
            "cum": _num(ch.get("cum")) >= cum_floor,
            "sharpe": _num(ch.get("sharpe")) >= _num(base.get("sharpe")) - 0.05,
        }
        checks["mdd"] = _num(ch.get("mdd")) >= _num(base.get("mdd")) - 0.03
        if not recent:
            checks["hard_risk"] = _num(ch.get("mdd")) >= -0.45
    else:
        # 收益最大化模式：只看累计收益，不看 Sharpe/MDD
        checks = {"cum": _num(ch.get("cum")) >= cum_floor}
    return all(checks.values()), checks


def sort_candidates(candidates: list[dict], return_only: bool = False) -> None:
    """唯一 Winner 冻结排序：EXPANDING Sharpe 差→cum 差→−MAE_1W 差→−Brier_1W 差→MDD 差→turnover 差→id。"""
    if return_only:
        # 只看总收益率：按累计收益差排序，其余指标不参与
        candidates.sort(key=lambda c: (-float(c.get("_cum", 0.0)), c["id"]))
        return
    candidates.sort(
        key=lambda c: (
            -float(c.get("_sharpe", 0.0)),
            -float(c.get("_cum", 0.0)),
            -float(c.get("_mae1_diff", 0.0)),
            -float(c.get("_brier1_diff", 0.0)),
            -float(c.get("_mdd_diff", 0.0)),
            float(c.get("_turnover", 0.0)),
            c["id"],
        )
    )


def checkpoint(
    state: dict,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    daily: pd.DataFrame,
    ca_payload: dict,
    due_idx: int,
    eval_index: EvalIndex,
    fwd1: np.ndarray,
    fwd8: np.ndarray,
    close: np.ndarray,
    cost_bps: int = 10,
    benchmark_refs: dict | None = None,
    fwd_arrays: dict[int, np.ndarray] | None = None,
    benchmark_frames=None,
    path_cache: dict | None = None,
) -> dict:
    """对 EVALUATING/SHADOW Challenger 做晋级判定；返回本轮 promotion 事件与诊断。"""
    out: dict = {
        "checked": [],
        "promoted": None,
        "candidates": [],
        "regime_diagnostics": [],
        "forecast_diagnostics": [],
    }
    policy = load_etf_config(state.get("etf", "159915")).get("promotion_policy", {})
    execution_policy = load_etf_config(state.get("etf", "159915")).get("execution_policy") or None
    min_weeks = int(policy.get("min_oos_weeks", MIN_1W))
    dd_penalized = bool(policy.get("drawdown_penalized", True))
    if fwd_arrays is None:
        fwd_arrays = {k: anchors[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    if path_cache is None:
        path_cache = {}
    full_end = len(features) - 1

    def cached_path(spec: dict, start_idx: int, end_idx: int) -> np.ndarray:
        """按 (spec, start_idx) 缓存全长仓位路径，跨检查点复用；返回长度 end_idx+1 的前缀。
        position_path 的顺序 ctx（ramp/drawdown）只依赖更早的行，前缀与 end_idx 无关。"""
        key = (spec_cache_key(spec), start_idx)
        if key not in path_cache:
            path_cache[key] = position_path(
                spec, features, full_end, start_idx=start_idx, fwd_arrays=fwd_arrays
            )
        full = path_cache[key]
        if len(full) >= end_idx + 1:
            return full[: end_idx + 1]
        path_cache[key] = position_path(
            spec, features, end_idx, start_idx=start_idx, fwd_arrays=fwd_arrays
        )
        return path_cache[key]

    champion_spec = state["champion"]["spec"]
    champion_positions = cached_path(state["champion"]["spec"], 0, due_idx)
    exec_prices = anchor_exec_prices(anchors, daily)
    champion_account = run_account(
        champion_positions, anchors, daily, ca_payload, exec_prices=exec_prices,
        execution_policy=execution_policy, features=features,
    )
    champion_full_metrics = weekly_metrics(
        champion_account["nav"],
        np.asarray(champion_account["positions"]),
        champion_account["returns"],
    )

    for ch in state.get("active_challengers", []):
        if ch["status"] not in ("EVALUATING", "SHADOW_EVALUATION"):
            continue
        start = ch["forward_oos_start"]
        weeks1 = matured_week_indices(start, due_idx, fwd1, 1)
        weeks8 = matured_week_indices(start, due_idx, fwd8, 8)
        age1 = len(weeks1)
        age8 = len(weeks8)
        ch["matured_counts"] = {"w1": age1, "w8": age8}
        if age1 < min_weeks or age8 < MIN_8W:
            ch["status"] = "EVALUATING"
            out["checked"].append({"id": ch["id"], "age1": age1, "status": "EVALUATING"})
            continue
        if ch.get("non_decisional"):
            # 全历史上下文生成的候选：可竞争、可记录，但永不晋级（NON_DECISIONAL）
            ch["status"] = "SHADOW_EVALUATION"
            out["checked"].append({
                "id": ch["id"], "age1": age1, "status": "SHADOW_EVALUATION",
                "all_pass": False, "non_decisional": True,
            })
            continue
        # 周K主导架构硬门禁：晋级候选必须同时满足
        # 1) 门控包含至少一个周线特征（周K判大趋势）；
        # 2) 打分包含至少一个日线特征（日K定短期择时/幅度）。
        if policy.get("weekly_primary_hard_gate", False):
            spec = ch.get("spec") or {}
            gate_feats = [
                str(g.get("feature", ""))
                for g in spec.get("gates") or []
            ]
            score_feats = [
                str(t.get("feature", ""))
                for t in (spec.get("score") or {}).get("terms") or []
            ]
            has_weekly_gate = any(f.startswith("w_") for f in gate_feats)
            has_daily_score = any(not f.startswith("w_") for f in score_feats)
            if not (has_weekly_gate and has_daily_score):
                ch["status"] = "SHADOW_EVALUATION"
                ch["reject_reason"] = "WEEKLY_PRIMARY_GATE"
                out["checked"].append({
                    "id": ch["id"], "age1": age1, "status": "SHADOW_EVALUATION",
                    "all_pass": False, "reason": "WEEKLY_PRIMARY_GATE",
                    "arch": {"weekly_gate": has_weekly_gate, "daily_score": has_daily_score},
                })
                continue

        oos_weeks = [j for j in weeks1 if j < due_idx]
        k_recent = 4 if ch.get("challenge_type") == "prediction" else 8
        recent_weeks = oos_weeks[-k_recent:]

        # 账户路径（challenger vs parent baseline，同一快照起步、同一成本/执行规则）
        ch_pos = cached_path(ch["spec"], start, due_idx)
        ch_acct = run_account(
            ch_pos, anchors, daily, ca_payload, start_snapshot=ch.get("account_snapshot"),
            start_anchor=start, exec_prices=exec_prices,
            execution_policy=execution_policy, features=features,
        )
        base_spec = (
            {"kind": "champion_zero_axis"}
            if ch.get("parent_champion_id", "") == "champion_zero_axis"
            else ch.get("parent_spec", {})
        )
        base_pos = cached_path(base_spec, start, due_idx)
        base_acct = run_account(
            base_pos, anchors, daily, ca_payload, start_snapshot=ch.get("account_snapshot"),
            start_anchor=start, exec_prices=exec_prices,
            execution_policy=execution_policy, features=features,
        )

        ret_len = min(len(ch_acct["returns"]), len(base_acct["returns"]))
        nav_c = ch_acct["nav"][: ret_len + 1]
        nav_b = base_acct["nav"][: ret_len + 1]
        ret_c = ch_acct["returns"][:ret_len]
        ret_b = base_acct["returns"][:ret_len]
        # 指标与 LOW_EFFECT 使用实际执行仓位（B0207 式策略下不等于每周目标仓位）
        pos_c = np.asarray(ch_acct["positions"][:ret_len])
        pos_b = np.asarray(base_acct["positions"][:ret_len])

        # LOW_EFFECT：创建期零建仓或 OOS 内零建仓 → 到点直接 REJECTED
        low_effect = bool(ch.get("low_effect")) or (
            len(pos_c) > 0 and float(np.max(pos_c)) == 0.0
        )
        if age1 >= min_weeks and low_effect:
            ch["status"] = "REJECTED"
            ch["reject_reason"] = "LOW_EFFECT"
            out["checked"].append({
                "id": ch["id"], "age1": age1, "status": "REJECTED",
                "all_pass": False, "reason": "LOW_EFFECT",
            })
            continue

        # EXPANDING（全部 Forward OOS）与 RECENT（最近 k 个成熟 1W）分窗口口径
        ch_full = weekly_metrics(nav_c, pos_c, ret_c)
        base_full = weekly_metrics(nav_b, pos_b, ret_b)
        anchor_map = {offset + start: offset for offset in range(ret_len)}
        offs = sorted(anchor_map[j] for j in recent_weeks if j in anchor_map)
        if offs:
            ch_recent = weekly_metrics(nav_c, pos_c, ret_c, offs[0], offs[-1] + 2)
            base_recent = weekly_metrics(nav_b, pos_b, ret_b, offs[0], offs[-1] + 2)
        else:
            ch_recent, base_recent = dict(ch_full), dict(base_full)

        is_pred = ch.get("challenge_type") == "prediction"
        ch_mae, ch_brier, ch_acc, ch_brier_strong = _mae_brier_acc_from_index(eval_index, ch["id"], start, due_idx)
        base_mae, base_brier, base_acc, base_brier_strong = _mae_brier_acc_from_index(
            eval_index, ch.get("parent_champion_id", "champion_zero_axis"), start, due_idx
        )

        # REGIME 与 4W/8W/强涨跌只作诊断
        week_offs = sorted(anchor_map[j] for j in range(start, due_idx) if j in anchor_map)
        labels = {
            anchor_map[j]: regime_label(close, j)
            for j in range(start, due_idx) if j in anchor_map
        }
        out["regime_diagnostics"].append({
            "id": ch["id"],
            "challenger": bucket_metrics(ret_c, nav_c, week_offs, labels),
            "baseline": bucket_metrics(ret_b, nav_b, week_offs, labels),
        })
        out["forecast_diagnostics"].append({
            "id": ch["id"],
            "mae_4w": ch_mae[4], "mae_8w": ch_mae[8],
            "brier_4w": ch_brier[4], "brier_8w": ch_brier[8],
            "brier_strong_1w": ch_brier_strong[1], "brier_strong_2w": ch_brier_strong[2],
        })

        # RECENT 预测层按最近成熟窗口独立统计
        if is_pred and recent_weeks and dd_penalized:
            rmin, rmax = min(recent_weeks), max(recent_weeks) + 1
            ch_mae_r, ch_brier_r, ch_acc_r, _ = _mae_brier_acc_from_index(eval_index, ch["id"], rmin, rmax)
            base_mae_r, base_brier_r, base_acc_r, _ = _mae_brier_acc_from_index(
                eval_index, ch.get("parent_champion_id", "champion_zero_axis"), rmin, rmax
            )
            rec_ok, rec_checks = _prediction_gates(
                {"mae_1w": ch_mae_r[1], "mae_2w": ch_mae_r[2], "brier_1w": ch_brier_r[1], "acc_1w": ch_acc_r[1], "acc_2w": ch_acc_r[2]},
                {"mae_1w": base_mae_r[1], "mae_2w": base_mae_r[2], "brier_1w": base_brier_r[1], "acc_1w": base_acc_r[1], "acc_2w": base_acc_r[2]},
                recent=True,
            )
        else:
            rec_ok, rec_checks = True, {}

        if is_pred:
            if dd_penalized:
                exp_ok, exp_checks = _prediction_gates(
                    {"mae_1w": ch_mae[1], "mae_2w": ch_mae[2], "brier_1w": ch_brier[1], "acc_1w": ch_acc[1], "acc_2w": ch_acc[2]},
                    {"mae_1w": base_mae[1], "mae_2w": base_mae[2], "brier_1w": base_brier[1], "acc_1w": base_acc[1], "acc_2w": base_acc[2]},
                    recent=False,
                )
            else:
                exp_ok, exp_checks = True, {}
        else:
            exp_ok, exp_checks = _strategy_gates(
                {"cum": ch_full["cumulative_return"], "sharpe": ch_full["sharpe"], "mdd": ch_full["max_drawdown"]},
                {"cum": base_full["cumulative_return"], "sharpe": base_full["sharpe"], "mdd": base_full["max_drawdown"]},
                recent=False,
                penalize_dd=policy.get("drawdown_penalized", True),
            )
            if offs:
                rec_ok, rec_checks = _strategy_gates(
                    {"cum": ch_recent["cumulative_return"], "sharpe": ch_recent["sharpe"], "mdd": ch_recent["max_drawdown"]},
                    {"cum": base_recent["cumulative_return"], "sharpe": base_recent["sharpe"], "mdd": base_recent["max_drawdown"]},
                    recent=True,
                    penalize_dd=policy.get("drawdown_penalized", True),
                )

        all_pass = exp_ok and rec_ok
        # 全周期晋级硬门禁：挑战者规则从 anchor 0 到当前检查点必须跑赢在任冠军
        # （回撤计分时：累计收益不低于、Sharpe/MDD 不劣于容忍线；
        #   回撤不扣分时：只要求全周期累计收益严格跑赢，收益最大化）。
        if policy.get("full_period_hard_gate", False) and all_pass:
            fp_ch_pos = cached_path(ch["spec"], 0, due_idx)
            fp_ch_acct = run_account(
                fp_ch_pos, anchors, daily, ca_payload, start_anchor=0,
                exec_prices=exec_prices,
                execution_policy=execution_policy, features=features,
            )
            ret_len_fp = min(len(fp_ch_acct["returns"]), len(champion_account["returns"]))
            m_ch_fp = weekly_metrics(
                fp_ch_acct["nav"][: ret_len_fp + 1],
                np.asarray(fp_ch_acct["positions"][:ret_len_fp]),
                fp_ch_acct["returns"][:ret_len_fp],
            )
            m_base_fp = weekly_metrics(
                champion_account["nav"][: ret_len_fp + 1],
                np.asarray(champion_account["positions"][:ret_len_fp]),
                champion_account["returns"][:ret_len_fp],
            )
            if dd_penalized:
                fp_ok = bool(
                    m_ch_fp["cumulative_return"] >= m_base_fp["cumulative_return"]
                    and m_ch_fp["sharpe"] >= m_base_fp["sharpe"] - float(
                        policy.get("full_period_sharpe_tol", 0.05)
                    )
                    and m_ch_fp["max_drawdown"] >= m_base_fp["max_drawdown"] - float(
                        policy.get("full_period_mdd_tol", 0.03)
                    )
                )
            else:
                fp_ok = bool(m_ch_fp["cumulative_return"] > m_base_fp["cumulative_return"])
            ch["_full_period"] = {
                "pass": fp_ok,
                "challenger": {
                    "cum": round(float(m_ch_fp["cumulative_return"]), 4),
                    "sharpe": round(float(m_ch_fp["sharpe"]), 4),
                    "mdd": round(float(m_ch_fp["max_drawdown"]), 4),
                },
                "champion": {
                    "cum": round(float(m_base_fp["cumulative_return"]), 4),
                    "sharpe": round(float(m_base_fp["sharpe"]), 4),
                    "mdd": round(float(m_base_fp["max_drawdown"]), 4),
                },
                "window": f"0..{due_idx}",
            }
            if not fp_ok:
                all_pass = False
                ch["reject_reason"] = "FULL_PERIOD_GATE"
        # 全历史 MDD 硬约束：创建时样本内全周期回撤太深（未经历 2015/2018 等大顶的窗口易漏判）
        if (
            policy.get("long_horizon_mdd_hard")
            and dd_penalized
            and (ch.get("long_horizon") or {}).get("mdd") is not None
        ):
            if float(ch["long_horizon"]["mdd"]) < -0.45:
                all_pass = False
                ch["reject_reason"] = "LONG_HORIZON_MDD_GATE"
        # 同现金流晋级门禁：每周2000，挑战者必须同时跑赢现任 Champion 与智能定投（TWR 主判）
        if all_pass and benchmark_frames is not None:
            from .benchmark import cashflow_gate

            gate = cashflow_gate(
                state.get("etf", "159915"),
                ch["spec"],
                state["champion"]["spec"],
                start_anchor=start,
                end_anchor=due_idx,
                _frames=benchmark_frames,
                penalize_dd=dd_penalized,
            )
            ch["_cashflow"] = {
                "pass": bool(gate["pass"]),
                "challenger_twr": round(float(gate["challenger_twr"]), 4),
                "champion_twr": round(float(gate["champion_twr"]), 4),
                "dca_twr": round(float(gate["dca_twr"]), 4),
                "challenger_irr": round(float(gate["challenger_irr"]), 4),
                "champion_irr": round(float(gate["champion_irr"]), 4),
                "dca_irr": round(float(gate["dca_irr"]), 4),
                "challenger_mdd": round(float(gate["challenger_mdd"]), 4),
                "champion_mdd": round(float(gate["champion_mdd"]), 4),
                "dca_mdd": round(float(gate["dca_mdd"]), 4),
            }
            if not gate["pass"] and policy.get("benchmark_hard_gate", True):
                all_pass = False
                ch["reject_reason"] = "CASHFLOW_GATE"
        # 多重检验控制：52 周内晋级次数上限
        if all_pass:
            limit = int(policy.get("max_promotions_per_52w", 2))
            recent = sum(
                1 for p in state.get("promotion_history", [])
                if due_idx - 52 <= p.get("anchor", -1) <= due_idx
            )
            if recent >= limit:
                all_pass = False
                ch["reject_reason"] = "PROMOTION_RATE_LIMIT"
        # 长周期稳健性对照（NON_DECISIONAL reference）：跑不赢基准至少标注，可配置硬门禁
        under = None
        if benchmark_refs and (ch.get("long_horizon") or {}).get("sharpe") is not None:
            lh = ch["long_horizon"]
            bh = benchmark_refs["buy_hold"]
            dca = benchmark_refs["smart_dca"]
            mm = benchmark_refs["momentum_dd"]
            ref_sharpe_max = max(float(bh["sharpe"]), float(dca["sharpe"]), float(mm["sharpe"]))
            ref_mdd_min = min(float(bh["mdd"]), float(dca["mdd"]), float(mm["mdd"]))
            if dd_penalized:
                under = bool(
                    float(lh["sharpe"]) < ref_sharpe_max - 0.05
                    or float(lh.get("cum", 0.0)) < float(bh["cum"]) * 0.8
                    or float(lh.get("mdd", 0.0)) < ref_mdd_min - 0.10
                )
            else:
                # 收益最大化模式：基准门禁只看累计收益是否守住买入持有的 0.8 倍
                under = bool(float(lh.get("cum", 0.0)) < float(bh["cum"]) * 0.8)
            ch["_benchmark"] = {
                "underperform": under,
                "ref_sharpe_max": round(ref_sharpe_max, 4),
                "ref_mdd_min": round(ref_mdd_min, 4),
                "buy_hold_cum": round(float(bh["cum"]), 4),
                "momentum_dd_sharpe": round(float(mm["sharpe"]), 4),
            }
            if under and policy.get("benchmark_hard_gate"):
                all_pass = False
                ch["reject_reason"] = "BENCHMARK_GATE"
        forced = age1 >= MAX_AGE
        if all_pass:
            ch["status"] = "PROMOTION_CANDIDATE"
            # 真实 EXPANDING 指标，用于唯一 Winner 排序（禁止占位常量）
            ch["_sharpe"] = float(ch_full["sharpe"] - base_full["sharpe"])
            ch["_cum"] = float(ch_full["cumulative_return"] - base_full["cumulative_return"])
            ch["_mae1_diff"] = float((base_mae[1] - ch_mae[1])) if base_mae[1] is not None and ch_mae[1] is not None else 0.0
            ch["_brier1_diff"] = float((base_brier[1] - ch_brier[1])) if base_brier[1] is not None and ch_brier[1] is not None else 0.0
            ch["_mdd_diff"] = float(ch_full["max_drawdown"] - base_full["max_drawdown"])
            ch["_turnover"] = float(ch_full.get("turnover", 0.0) - base_full.get("turnover", 0.0))
            out["candidates"].append(ch)
        elif forced:
            ch["status"] = "REJECTED"
        else:
            ch["status"] = "SHADOW_EVALUATION"
        out["checked"].append({
            "id": ch["id"], "age1": age1, "status": ch["status"],
            "all_pass": all_pass, "expanding_checks": exp_checks, "recent_checks": rec_checks,
            "benchmark": ch.get("_benchmark"),
            "long_horizon": ch.get("long_horizon"),
            "cashflow": ch.get("_cashflow"),
            "full_period": ch.get("_full_period"),
        })

    candidates = out["candidates"]
    if candidates:
        sort_candidates(candidates, return_only=not dd_penalized)
        for ch in candidates:
            ch["status"] = "PROMOTION_CANDIDATE"
        # 机器硬门禁只产出候选；PROMOTED 必须再经 Agent APPROVE（auto_approve=false）
        out["recommended"] = candidates[0]["id"]
    return out
