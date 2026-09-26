# -*- coding: utf-8 -*-
"""Challenger 创建期筛查：spec 合法、PIT 可行、divergence、样本内回放（non-decisional）。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .account import anchor_exec_prices, run_account, weekly_metrics
from .config import integrity_digests_enabled, load_etf_config
from .forecast import champion_position
from .spec import compute_position, validate_spec


def position_path(
    spec: dict,
    features: pd.DataFrame,
    end_idx: int,
    start_idx: int = 0,
    prices: np.ndarray | None = None,
    fwd_arrays: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """逐周目标仓位路径；回填 equity_history 使 drawdown_control 在回测中真实生效。

    prices 默认用 features['close']（analysis 收盘价）作为净值代理；
    传入真实执行价（如 exec raw_open）可获得更精确的回撤制动。
    """
    positions = np.zeros(end_idx + 1)
    if prices is None:
        prices = features["close"].to_numpy(dtype=float)
    ctx: dict = {"position_history": [], "equity_history": [1.0]}
    equity = 1.0
    prev_price = float(prices[start_idx]) if start_idx < len(prices) else 1.0
    for i in range(start_idx, end_idx + 1):
        kind = spec.get("kind")
        if kind == "learning_ridge":
            if fwd_arrays is None:
                raise ValueError("learning_ridge 需要 fwd_arrays")
            from .learning import walkforward_position

            pos, _ = walkforward_position(features, fwd_arrays, i, spec)
        elif kind == "champion_zero_axis":
            pos = champion_position(features.iloc[i])
        else:
            pos = compute_position(spec, features.iloc[i], ctx)
        positions[i] = pos
        ctx["position_history"].append(pos)
        price = float(prices[i]) if i < len(prices) else prev_price
        if i > start_idx and prev_price > 0:
            ret = price / prev_price - 1.0
            equity *= 1.0 + pos * ret
        ctx["equity_history"].append(equity)
        prev_price = price
    return positions


def champion_path(features: pd.DataFrame, end_idx: int, start_idx: int = 0) -> np.ndarray:
    return position_path({"kind": "champion_zero_axis"}, features, end_idx, start_idx)


def forecast_vector(feature_row: pd.Series, spec: dict) -> np.ndarray:
    """预测向量：状态信号 + 特征归一值（用于 Prediction divergence）。"""
    if spec.get("kind") == "champion_zero_axis":
        score = 1.0 if (feature_row["w_slope"] > 0 and feature_row["d_slope"] > 0) else 0.0
    else:
        score = 0.0
        for term in (spec.get("score") or {}).get("terms", []):
            score += float(feature_row[term["feature"]]) * float(term.get("weight", 1.0))
    vec = np.asarray([score, feature_row["w_slope_pct"], feature_row["d_slope_pct"], feature_row["vol_pct"]], dtype=float)
    return np.nan_to_num(vec, nan=0.0)


def divergence_prediction(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(np.linalg.norm(a), np.linalg.norm(b), 1e-9)
    return float(np.linalg.norm(a - b) / denom)


def divergence_position(a: np.ndarray, b: np.ndarray) -> float:
    valid = ~np.isnan(a) & ~np.isnan(b)
    if not valid.any():
        return 0.0
    return float(np.mean(a[valid] != b[valid]))


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def spec_cache_key(spec: dict) -> str:
    """Exact canonical spec text used only as an in-memory cache key."""
    return json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def create_screening(
    proposal: dict,
    due_idx: int,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    daily: pd.DataFrame,
    ca_payload: dict,
    champion_spec: dict,
    challenge_type: str,
    parent_champion_id: str | None = None,
    existing_challengers: list[dict] | None = None,
    fwd_arrays: dict[int, np.ndarray] | None = None,
    path_cache: dict[str, np.ndarray] | None = None,
    execution_policy: dict | None = None,
) -> list[dict]:
    if fwd_arrays is None:
        fwd_arrays = {k: anchors[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    if path_cache is None:
        path_cache = {}
    full_end = len(features) - 1

    def cached_path(spec: dict, end_idx: int) -> np.ndarray:
        """按 spec 哈希缓存全长仓位路径，按 end_idx 切片；前缀与 end_idx 无关（顺序 ctx）。"""
        h = spec_cache_key(spec)
        if h not in path_cache:
            path_cache[h] = position_path(spec, features, full_end, fwd_arrays=fwd_arrays)
        full = path_cache[h]
        if len(full) >= end_idx + 1:
            return full[: end_idx + 1]
        path_cache[h] = position_path(spec, features, end_idx, fwd_arrays=fwd_arrays)
        return path_cache[h]

    champion_positions = cached_path(champion_spec, due_idx)
    exec_prices = anchor_exec_prices(anchors, daily)
    champion_account = run_account(
        champion_positions, anchors, daily, ca_payload, exec_prices=exec_prices,
        execution_policy=execution_policy, features=features,
    )
    champion_snapshot = {
        "cash": champion_account["final_cash"],
        "shares": champion_account["final_shares"],
        "nav": round(float(champion_account["nav"][-1]), 6) if len(champion_account["nav"]) else None,
    }
    champ_vec = forecast_vector(features.iloc[due_idx], champion_spec)
    parent_id = parent_champion_id or champion_spec.get("id", "champion_zero_axis")
    context_scope = proposal.get("context_scope", "full_context")
    non_decisional = context_scope != "sanitized_only"
    active_statuses = {"EVALUATING", "SHADOW_EVALUATION", "PROMOTION_CANDIDATE", "NON_WINNING_CANDIDATE", "ACTIVE"}
    registered = []
    paths = {}
    vectors = {}
    existing_vectors = {}
    existing_paths = {}
    for ex in existing_challengers or []:
        if ex.get("status") not in active_statuses:
            continue
        if challenge_type == "prediction":
            existing_vectors[ex["id"]] = forecast_vector(features.iloc[due_idx], ex.get("spec", {}))
        else:
            existing_paths[ex["id"]] = cached_path(ex.get("spec", {}), due_idx)
    for ch in proposal["challengers"]:
        spec = ch["spec"]
        errors = validate_spec(spec)
        if errors:
            ch["screening"] = {"valid": False, "errors": errors}
            continue
        pos = cached_path(spec, due_idx)
        paths[ch["id"]] = pos
        vectors[ch["id"]] = forecast_vector(features.iloc[due_idx], spec)
        ch["screening"] = {
            "valid": True,
            "low_effect": bool(len(pos) and float(np.max(pos)) == 0.0),
        }

    ids = [ch["id"] for ch in proposal["challengers"] if ch.get("screening", {}).get("valid")]
    # divergence：Prediction 比预测向量，Strategy 比仓位路径；与 parent Champion 及两两比较
    for ch_id in ids:
        if challenge_type == "prediction":
            vs_champion = divergence_prediction(vectors[ch_id], champ_vec)
            vs_others = [
                divergence_prediction(vectors[ch_id], vectors[other])
                for other in ids if other != ch_id
            ]
        else:
            vs_champion = divergence_position(paths[ch_id], champion_positions)
            vs_others = [
                divergence_position(paths[ch_id], paths[other])
                for other in ids if other != ch_id
            ]
        identical = vs_champion < 0.05 or any(d < 0.05 for d in vs_others)
        if challenge_type == "prediction":
            vs_existing = [
                divergence_prediction(vectors[ch_id], v)
                for v in existing_vectors.values()
            ]
        else:
            vs_existing = [
                divergence_position(paths[ch_id], p)
                for p in existing_paths.values()
            ]
        identical_existing = any(d < 0.05 for d in vs_existing)
        for ch in proposal["challengers"]:
            if ch["id"] == ch_id:
                ch["screening"]["divergence_ok"] = not identical and not identical_existing
                ch["screening"]["vs_champion"] = round(vs_champion, 4)
                ch["screening"]["vs_others"] = [round(d, 4) for d in vs_others]
                ch["screening"]["vs_existing"] = [round(d, 4) for d in vs_existing]
                ch["screening"]["identical"] = identical or identical_existing

    for ch in proposal["challengers"]:
        s = ch.get("screening", {})
        if not s.get("valid") or s.get("identical"):
            continue
        pos = paths[ch["id"]]
        account = run_account(
            pos, anchors, daily, ca_payload, exec_prices=exec_prices,
            execution_policy=execution_policy, features=features,
        )
        lm = weekly_metrics(account["nav"], pos, account["returns"])
        record = {
                "id": ch["id"],
                "spec": ch["spec"],
                "status": "EVALUATING",
                "challenge_type": challenge_type,
                "created_anchor": due_idx,
                "forward_oos_start": due_idx + 1,
                "account_snapshot": champion_snapshot,
                "parent_champion_id": parent_id,
                "parent_spec": champion_spec,
                "baseline_branch_id": f"baseline_{parent_id}@{due_idx}",
                "baseline_start_anchor": due_idx + 1,
                "matured_counts": {"w1": 0, "w8": 0},
                "low_effect": bool(ch.get("screening", {}).get("low_effect")),
                "context_scope": context_scope,
                "non_decisional": non_decisional,
                "long_horizon": {
                    "cum": lm["cumulative_return"],
                    "sharpe": lm["sharpe"],
                    "mdd": lm["max_drawdown"],
                    "win_rate": lm["win_rate"],
                    "avg_pos": lm["average_position"],
                },
                "evaluation_shadow_nav": round(float(account["nav"][-1]), 6) if len(account["nav"]) else None,
                "in_sample_replay": {
                    "non_decisional_in_sample": True,
                    "cumulative_return": round(float(account["nav"][-1] / account["nav"][0] - 1.0), 6) if len(account["nav"]) > 1 else None,
                },
            }
        asset = str((proposal.get("orchestrated") or {}).get("asset") or "159915")
        if integrity_digests_enabled(asset):
            record["spec_hash"] = spec_hash(ch["spec"])
            record["parent_champion_spec_hash"] = spec_hash(champion_spec)
            record["baseline_account_snapshot_hash"] = spec_hash(champion_snapshot)
        registered.append(record)
    return registered


def rebase_stale(
    state: dict,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    daily: pd.DataFrame,
    ca_payload: dict,
    anchor: int,
    policy: dict,
    fwd_arrays: dict[int, np.ndarray] | None = None,
) -> list[str]:
    """打破 STALE 锁死：长周期 Sharpe 达标的 STALE 候选以新 Champion 为 parent 重新登记。"""
    from .ledger import append_event

    execution_policy = load_etf_config(state.get("etf", "159915")).get("execution_policy") or None
    if fwd_arrays is None:
        fwd_arrays = {k: anchors[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    threshold = float(policy.get("rebase_min_sharpe", 0.7))
    max_gen = int(policy.get("max_rebase_generations", 2))
    champion_spec = state["champion"]["spec"]
    parent = state["champion"]["id"]
    rebased: list[str] = []
    for ch in list(state.get("active_challengers", [])):
        if ch.get("status") != "STALE_SHADOW" or ch.get("rebased_at"):
            continue
        lh = ch.get("long_horizon") or {}
        if float(lh.get("sharpe", -1.0)) < threshold:
            continue
        gen = int(ch.get("rebase_generation", 0)) + 1
        if gen > max_gen:
            continue
        clone_id = f"{ch['id']}@R{anchor}"
        proposal = {
            "iteration_id": "ITERATION_002",
            "opaque_due_id": "DUE-REBASE",
            "challenge_round": f"ROUND_REBASE_{anchor}",
            "challenge_type": ch.get("challenge_type", "strategy"),
            "diagnosis": "rebase 强原型：长周期 Sharpe 达标，避免 STALE 锁死",
            "primary_failure_mode": "ACTIVITY_ERROR",
            "secondary_failure_mode": "RISK_CONTROL_ERROR",
            "evidence": [f"long_horizon_sharpe={lh.get('sharpe')}"],
            "hypothesis": "同 spec 在新 Champion 下重新积累 Forward OOS",
            "revision": 1,
            "status": "FINAL",
            "context_scope": "sanitized_only",
            "challengers": [{
                "id": clone_id,
                "target_problem": "强原型延续",
                "change_description": f"同 spec，parent={parent}，anchor={anchor} rebase",
                "changed_dimensions": [],
                "old_values": {"parent": ch.get("parent_champion_id")},
                "new_values": {"parent": parent},
                "expected_effect": "保留长周期强势规则参与竞争",
                "main_risk": "短期窗口表现不确定",
                "reason": "长周期稳健性达标（rebase_min_sharpe）",
                "spec": ch["spec"],
            }],
        }
        reg = create_screening(
            proposal, anchor, features, anchors, daily, ca_payload,
            champion_spec, ch.get("challenge_type", "strategy"),
            parent_champion_id=parent,
            existing_challengers=state["active_challengers"],
            fwd_arrays=fwd_arrays,
            execution_policy=execution_policy,
        )
        if reg:
            reg[0]["rebase_generation"] = gen
            reg[0]["lineage_id"] = ch.get("lineage_id", ch["id"])
            state["active_challengers"].extend(reg)
            ch["rebased_at"] = anchor
            append_event({
                "type": "challenger_rebased",
                "old_id": ch["id"],
                "new_id": reg[0]["id"],
                "anchor": anchor,
                "parent_champion_id": parent,
            })
            rebased.append(reg[0]["id"])
    return rebased
