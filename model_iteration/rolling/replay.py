# -*- coding: utf-8 -*-
"""周滚动主循环：PIT 回放、成熟、Due 暂停、Proposal 校验后继续。

修复基线（方案第 0 节）：
- 数据装配唯一合法方式：features = feat_raw[mask].reset_index(drop=True)，启动断言 fail-closed；
- 诊断包必须包含逐窗口 failure_cases，成熟>=4 且为空 → DIAGNOSTIC_INCOMPLETE 暂停；
- Promotion 机器门禁只产出候选（PROMOTION_CANDIDATE + pending_ai_reviews），
  PROMOTED 必须经 Agent APPROVE（auto_approve=false）。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import challenge as challenge_mod
from . import promotion as promotion_mod
from .benchmark import compute_reference_benchmarks
from .ca import ca_events_hash, load_ca
from .config import integrity_digests_enabled, load_etf_config
from .data import DataAssemblyError, load_aligned_data
from .diagnose import sanitize, write_diagnostic
from .forecast import build_forecast, compute_bucket_stats
from .ledger import (
    append_event,
    ledger_hash,
    load_state,
    proposal_path,
    read_events,
    round_id,
    save_state,
)
from .proposal import validate_proposal
from .review import apply_ai_review
from .statindex import EvalIndex, index_from_events


def _print_ledger_checkpoint(etf: str, state: dict) -> None:
    if integrity_digests_enabled(etf):
        print(f"LEDGER_HASH={ledger_hash()}")
        return
    print(
        "LEDGER_CHECKPOINT="
        f"anchor:{state.get('latest_anchor_index', -1)};"
        f"resume:{state.get('resume_from', 0)};"
        f"events:{len(read_events())}"
    )


def _models_at(state: dict, i: int, include_promoted: bool = False) -> list[dict]:
    models = [dict(state["champion"])]
    active_statuses = {"EVALUATING", "SHADOW_EVALUATION", "PROMOTION_CANDIDATE", "NON_WINNING_CANDIDATE"}
    if include_promoted:
        active_statuses = active_statuses | {"PROMOTED"}
    for ch in state.get("active_challengers", []):
        if ch["status"] in active_statuses and ch["forward_oos_start"] <= i:
            models.append(ch)
    return models


def _diagnostic_stats_from_index(
    eval_index: EvalIndex, model_id: str, matured_w1: int, miss_counts: tuple[int, int] | None = None
) -> dict:
    out: dict = {}
    for k in (1, 2, 4, 8):
        mae, brier, acc, _ = eval_index.range_stats(model_id, k, 0, 10**9)
        if mae is not None:
            out[f"accuracy_{k}w"] = round(acc, 4) if acc is not None else None
            out[f"mae_{k}w"] = round(mae, 6)
            out[f"brier_{k}w"] = round(brier, 6)
    out["matured_1w"] = matured_w1
    missed, wrong = miss_counts or (0, 0)
    out["missed_rally"] = missed
    out["wrong_chase"] = wrong
    return out


def build_failure_cases(
    eval_index: EvalIndex,
    features: pd.DataFrame,
    model_id: str,
    due_idx: int,
    max_windows: int = 12,
) -> list[dict]:
    """最近 max_windows 个成熟 1W 窗口的盲化相对案例（禁止日期/代码/绝对价格/anchor_index）。"""
    start = max(0, due_idx - max_windows)
    cases = []
    for r in eval_index.range_cases(model_id, 1, start, due_idx):
        j = r["anchor"]
        row = features.iloc[j]
        if row["w_slope"] > 0 and row["d_slope"] > 0:
            state = "日强周强"
        elif row["w_slope"] < 0 and row["d_slope"] < 0:
            state = "日弱周弱"
        elif row["d_slope"] > 0:
            state = "日强周弱"
        elif row["w_slope"] > 0:
            state = "周强日弱"
        else:
            state = "震荡"
        cases.append(
            {
                "week_offset": j - due_idx,
                "state": state,
                "direction_correct": r["direction_correct"],
                "position": round(float(r["position"]), 4) if r["position"] is not None else None,
                "actual_return": round(float(r["actual_return"]), 4) if r["actual_return"] is not None else None,
                "missed_rally": r["missed_rally"],
                "wrong_chase": r["wrong_chase"],
                "over_cash": r["over_cash"],
                "late_entry": r["late_entry"],
                "conflict_correct": r["conflict_correct"],
            }
        )
    return cases


def _due_specs(matured_w1: int, last_triggered: dict) -> list[tuple[str, int]]:
    due_types: list[tuple[str, int]] = []
    if matured_w1 > 0 and matured_w1 % 4 == 0 and matured_w1 > last_triggered.get("prediction", 0):
        due_types.append(("prediction", matured_w1 // 4))
    if matured_w1 > 0 and matured_w1 % 8 == 0 and matured_w1 > last_triggered.get("strategy", 0):
        due_types.append(("strategy", matured_w1 // 8))
    return due_types


def _process_due(
    i: int,
    due_types: list[tuple[str, int]],
    state: dict,
    eval_index: EvalIndex,
    miss_counter: dict[str, list[int]],
    diag_model_id: str,
    features: pd.DataFrame,
    usable: pd.DataFrame,
    daily: pd.DataFrame,
    ca_payload: dict,
    etf: str = "159915",
    orchestrated_done: set | None = None,
    path_cache: dict | None = None,
) -> bool:
    orchestrated_done = set() if orchestrated_done is None else orchestrated_done
    all_ready = True
    for ctype, seq in due_types:
        rid = round_id(ctype, seq)
        stats = _diagnostic_stats_from_index(
            eval_index, diag_model_id, state["matured_counts"]["w1"],
            tuple(miss_counter.get(diag_model_id, [0, 0])),
        )
        cases = build_failure_cases(eval_index, features, diag_model_id, i)
        payload = sanitize(rid, ctype, stats, cases)
        write_diagnostic(rid, payload)
        if not payload.get("complete"):
            all_ready = False
            print(f"DIAGNOSTIC_INCOMPLETE: {rid} failure_cases 为空但 matured_1w>=4")
            continue
        path = proposal_path(rid)
        if not path.exists():
            all_ready = False
            print(f"AGENT_REVIEW_REQUIRED: 缺少 Proposal {rid} (anchor index {i})")
            continue
        proposal = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_proposal(proposal, etf=etf)
        if errors or proposal.get("status") != "FINAL":
            all_ready = False
            print(f"AGENT_REVIEW_REQUIRED: Proposal {rid} 未冻结或校验失败: {errors}")
            continue
        if proposal.get("opaque_due_id") != payload["opaque_due_id"]:
            all_ready = False
            print(f"AGENT_REVIEW_REQUIRED: Proposal {rid} 诊断不匹配（opaque_due_id 不一致）")
            continue
        registered = challenge_mod.create_screening(
            proposal, i, features, usable, daily, ca_payload,
            state["champion"]["spec"], ctype,
            parent_champion_id=state["champion"]["id"],
            existing_challengers=state["active_challengers"],
            path_cache=path_cache,
            execution_policy=load_etf_config(state.get("etf", "159915")).get("execution_policy") or None,
        )
        existing_ids = {ch["id"] for ch in state["active_challengers"]}
        registered = [c for c in registered if c["id"] not in existing_ids]
        if registered:
            state["active_challengers"].extend(registered)
            if (rid, i) not in orchestrated_done:
                append_event({
                    "type": "proposal_orchestrated",
                    "round": rid,
                    "anchor_index": i,
                    "anchor": usable.iloc[i]["signal"].date().isoformat(),
                    "asset": state.get("etf", "159915"),
                })
                orchestrated_done.add((rid, i))
            print(f"anchor[{i}] {rid}: 登记 {len(registered)} 个 EVALUATING Challenger")
        state["last_triggered"][ctype] = state["matured_counts"]["w1"]
    return all_ready


def _freeze_t_forecasts(
    i: int,
    models_for_t: list[dict],
    features: pd.DataFrame,
    usable: pd.DataFrame,
    fwd_arrays: dict,
    forecast_index: dict,
    forecast_models_at: dict,
) -> None:
    stats = compute_bucket_stats(features.iloc[:i], fwd_arrays, up_to=i)
    row = features.iloc[i]
    forecast_models_at[i] = [m["id"] for m in models_for_t]
    for model in models_for_t:
        if (i, model["id"]) in forecast_index:
            continue
        fc = build_forecast(model["spec"], row, i, fwd_arrays, stats, features=features)
        event = {"type": "forecast", "anchor_index": i, "model_id": model["id"], "forecast": fc}
        event["anchor"] = usable.iloc[i]["signal"].date().isoformat()
        append_event(event)
        forecast_index[(i, model["id"])] = event


def run_replay(
    etf: str = "159915",
    resume: bool = False,
    mode: str = "generate",
    stop_after: int | None = None,
    auto_review: bool = False,
) -> int:
    try:
        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    except DataAssemblyError as exc:
        print(f"DATA_ASSEMBLY_FAILED: {exc}")
        return 2
    print(f"anchors usable: {len(usable)} ({usable['signal'].iloc[0].date()} ~ {usable['signal'].iloc[-1].date()})")

    state = load_state()
    state["etf"] = etf
    if integrity_digests_enabled(etf):
        state["ca_events_hash"] = ca_events_hash(ca)
    else:
        coverage = ca.get("coverage") or {}
        state.pop("ca_events_hash", None)
        state["corporate_actions_reference"] = {
            "etf": etf,
            "coverage_start": coverage.get("start"),
            "coverage_end": coverage.get("end"),
            "verified": coverage.get("verified"),
            "event_count": len(ca.get("events") or []),
        }
    events = read_events()
    eval_index = index_from_events(events)
    forecast_models_at: dict[int, list[str]] = {}
    for e in events:
        if e.get("type") == "forecast":
            forecast_models_at.setdefault(e.get("anchor_index"), []).append(e.get("model_id"))
    miss_counter: dict[str, list[int]] = {}
    for e in events:
        if e.get("type") == "evaluation" and e.get("horizon") == 1 and not e.get("pending"):
            row = miss_counter.setdefault(e.get("model_id", ""), [0, 0])
            row[0] += int(e.get("missed_rally", False))
            row[1] += int(e.get("wrong_chase", False))
    forecast_index: dict[tuple[int, str], dict] = {
        (e.get("anchor_index"), e.get("model_id")): e
        for e in events
        if e.get("type") == "forecast"
    }
    eval_keys = {
        (e.get("anchor_index"), e.get("horizon"), e.get("model_id"))
        for e in events
        if e.get("type") == "evaluation" and not e.get("pending")
    }
    orchestrated_done = {
        (e.get("round"), e.get("anchor_index"))
        for e in events
        if e.get("type") == "proposal_orchestrated"
    }
    fwd_arrays = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    screen_path_cache: dict = {}
    checkpoint_path_cache: dict = {}

    if resume:
        if state.get("paused_at") is not None:
            paused = state["paused_at"]
            if state.get("paused_stage") == "pre_t":
                # AI 复核暂停：已批准/拒绝后，先由旧 Champion 冻结 T 预测，再处理 Due（parent=当前 Champion）
                old_champion = state.get("paused_champion", state["champion"])
                models_for_t = _models_at(
                    {"champion": old_champion, "active_challengers": state.get("active_challengers", [])},
                    paused,
                    include_promoted=True,
                )
                _freeze_t_forecasts(paused, models_for_t, features, usable, fwd_arrays, forecast_index, forecast_models_at)
                due_types = _due_specs(state.get("matured_counts", {}).get("w1", 0), state["last_triggered"])
                diag_id = old_champion.get("id", state["champion"]["id"])
                ok = _process_due(
                    paused, due_types, state, eval_index, miss_counter, diag_id,
                    features, usable, daily, ca, etf, orchestrated_done,
                    path_cache=screen_path_cache,
                )
                if not ok:
                    save_state(state)
                    _print_ledger_checkpoint(etf, state)
                    return 3
                state["paused_at"] = None
                state["paused_stage"] = None
                state.pop("paused_champion", None)
                state["latest_anchor_index"] = paused
                state["resume_from"] = paused + 1
                save_state(state)
            else:
                # 普通 Due 暂停：补登记（不重成熟/不重预测/不重复 append）
                due_types = _due_specs(state.get("matured_counts", {}).get("w1", 0), state["last_triggered"])
                diag_id = state["champion"]["id"]
                if not _process_due(
                    paused, due_types, state, eval_index, miss_counter, diag_id,
                    features, usable, daily, ca, etf, orchestrated_done,
                    path_cache=screen_path_cache,
                ):
                    save_state(state)
                    _print_ledger_checkpoint(etf, state)
                    return 3
                state["paused_at"] = None
                state["latest_anchor_index"] = paused
                state["resume_from"] = paused + 1
                save_state(state)
        start = state.get("resume_from", 0)
    else:
        if state.get("paused_at") is not None:
            print(f"回放已暂停于 anchor {state['paused_at']}，请使用 --resume 继续（避免重复追加）")
            return 3
        start = state.get("latest_anchor_index", -1) + 1

    end = len(usable) if stop_after is None else min(stop_after, len(usable))
    for i in range(start, end):
        if i % 50 == 0:
            print(f"anchor[{i}] ...", flush=True)
        # 1) 成熟
        for k in (1, 2, 4, 8):
            j = i - k
            if j >= 0 and not np.isnan(fwd_arrays[k][j]):
                for model_id in forecast_models_at.get(j, []):
                    fc_event = forecast_index.get((j, model_id))
                    if fc_event is None:
                        continue
                    if (j, k, model_id) in eval_keys:
                        continue
                    actual = float(fwd_arrays[k][j])
                    sub = float(fwd_arrays[k // 2][j]) if k // 2 in fwd_arrays and not np.isnan(fwd_arrays[k // 2][j]) else None
                    from .evaluate import build_evaluation

                    ev = build_evaluation(
                        j, k, model_id, fc_event["forecast"], actual, sub,
                        features.iloc[j], daily_a, usable,
                    )
                    ev["type"] = "evaluation"
                    ev["anchor"] = usable.iloc[j]["signal"].date().isoformat()
                    append_event(ev)
                    eval_keys.add((j, k, model_id))
                    eval_index.add(ev)
                    if k == 1:
                        miss_counter.setdefault(model_id, [0, 0])
                        miss_counter[model_id][0] += int(ev.get("missed_rally", False))
                        miss_counter[model_id][1] += int(ev.get("wrong_chase", False))

        # 成熟计数与模型无关
        matured_counts = {}
        for k in (1, 2, 4, 8):
            matured_counts[f"w{k}"] = sum(
                1 for j in range(max(0, i - k + 1)) if not np.isnan(fwd_arrays[k][j])
            )
        matured_w1 = matured_counts["w1"]
        state["matured_counts"] = matured_counts

        pred_due = matured_w1 > 0 and matured_w1 % 4 == 0 and matured_w1 > state["last_triggered"]["prediction"]
        strat_due = matured_w1 > 0 and matured_w1 % 8 == 0 and matured_w1 > state["last_triggered"]["strategy"]
        models_for_t = _models_at(state, i, include_promoted=True)
        pre_promo_champion = dict(state["champion"])

        # 2) Promotion checkpoint（机器门禁，产出候选；AI 复核后生效）
        if pred_due or strat_due:
            refs = compute_reference_benchmarks(
                etf, daily_a, weekly_a, daily, anchors, usable, features, end_idx=i
            )
            promo = promotion_mod.checkpoint(
                state, features, usable, daily, ca, i, eval_index,
                fwd_arrays[1], fwd_arrays[8],
                usable["anchor_close"].to_numpy(dtype=float),
                benchmark_refs=refs,
                fwd_arrays=fwd_arrays,
                benchmark_frames=(ca, daily_a, weekly_a, daily, anchors, usable, features),
                path_cache=checkpoint_path_cache,
            )
            if promo.get("checked"):
                checkpoint_ref = append_event({
                    "type": "promotion_checkpoint",
                    "anchor": i,
                    "policy_version": state.get("promotion_policy_version"),
                    "checked": promo["checked"],
                    "candidates": [
                        {
                            "id": c["id"],
                            "metrics": {
                                "sharpe_diff": round(float(c.get("_sharpe", 0.0)), 4),
                                "cum_diff": round(float(c.get("_cum", 0.0)), 6),
                                "mae1_diff": round(float(c.get("_mae1_diff", 0.0)), 6),
                                "brier1_diff": round(float(c.get("_brier1_diff", 0.0)), 6),
                                "mdd_diff": round(float(c.get("_mdd_diff", 0.0)), 4),
                            "turnover_diff": round(float(c.get("_turnover", 0.0)), 4),
                            },
                            "benchmark": c.get("_benchmark"),
                            "long_horizon": c.get("long_horizon"),
                            "cashflow": c.get("_cashflow"),
                        }
                        for c in promo["candidates"]
                    ],
                    "recommended": promo.get("recommended"),
                })
                for c in promo["candidates"]:
                    review_item = {
                        "id": c["id"],
                        "anchor": i,
                        "recommended": c["id"] == promo.get("recommended"),
                        "metrics": {
                            "sharpe_diff": round(float(c.get("_sharpe", 0.0)), 4),
                            "cum_diff": round(float(c.get("_cum", 0.0)), 6),
                            "mae1_diff": round(float(c.get("_mae1_diff", 0.0)), 6),
                            "brier1_diff": round(float(c.get("_brier1_diff", 0.0)), 6),
                            "mdd_diff": round(float(c.get("_mdd_diff", 0.0)), 4),
                            "turnover_diff": round(float(c.get("_turnover", 0.0)), 4),
                        },
                        "benchmark": c.get("_benchmark"),
                        "long_horizon": c.get("long_horizon"),
                        "cashflow": c.get("_cashflow"),
                    }
                    if integrity_digests_enabled(etf):
                        review_item["checkpoint_hash"] = checkpoint_ref
                    else:
                        review_item["checkpoint_ref"] = checkpoint_ref
                    state.setdefault("pending_ai_reviews", []).append(review_item)
            if state.get("pending_ai_reviews"):
                if auto_review:
                    for r in list(state.get("pending_ai_reviews", [])):
                        decision = "APPROVE" if r.get("recommended", True) else "REJECT"
                        rc2, msg = apply_ai_review(
                            state, r["id"], r["anchor"], decision,
                            "auto_review: 机器硬门禁通过且为推荐候选，批准晋级（事件留痕）",
                        )
                        print(f"auto_review anchor[{r['anchor']}] {r['id']}: {msg}", flush=True)
                    cfg = load_etf_config(etf)
                    reb = challenge_mod.rebase_stale(
                        state, features, usable, daily, ca, i, cfg.get("promotion_policy", {})
                    )
                    for rid_ in reb:
                        print(f"rebase anchor[{i}] {rid_}", flush=True)
                    save_state(state)
                else:
                    # AI 复核暂停：T 预测尚未冻结，批准后 resume 先冻结 T 再处理 Due
                    state["paused_at"] = i
                    state["resume_from"] = i + 1
                    state["paused_stage"] = "pre_t"
                    state["paused_champion"] = pre_promo_champion
                    save_state(state)
                    print(f"AI_REVIEW_REQUIRED: anchor[{i}] 有 {len(state['pending_ai_reviews'])} 个候选待 Agent 复核")
                    _print_ledger_checkpoint(etf, state)
                    return 4

        # 3) 冻结 T 预测（归属晋级前的 Champion 快照）
        _freeze_t_forecasts(i, models_for_t, features, usable, fwd_arrays, forecast_index, forecast_models_at)

        # 4) Due 诊断/Proposal
        if pred_due or strat_due:
            due_types = _due_specs(matured_w1, state["last_triggered"])
            if not _process_due(
                i, due_types, state, eval_index, miss_counter, pre_promo_champion["id"],
                features, usable, daily, ca, etf, orchestrated_done,
                path_cache=screen_path_cache,
            ):
                state["paused_at"] = i
                state["resume_from"] = i + 1
                save_state(state)
                _print_ledger_checkpoint(etf, state)
                return 3

        state["latest_anchor_index"] = i
        state["resume_from"] = i + 1
        state["paused_at"] = None
        state["paused_stage"] = None
        state.pop("paused_champion", None)
        save_state(state)

    if state.get("pending_ai_reviews"):
        print(f"AI_REVIEW_REQUIRED: 仍有 {len(state['pending_ai_reviews'])} 个候选待 Agent 复核")
        _print_ledger_checkpoint(etf, state)
        return 4
    print("replay completed")
    _print_ledger_checkpoint(etf, state)
    return 0


def run_weekly(etf: str = "159915", anchor: str | None = None) -> int:
    """实时单锚点推进：只处理下一个锚点，遇 Due/AI 复核则按协议暂停。"""
    state = load_state()
    if state.get("paused_at") is not None:
        print(f"当前暂停于 anchor {state['paused_at']}，请先完成 Proposal/AI 复核再 weekly")
        return 3
    stop = state.get("latest_anchor_index", -1) + 1
    if anchor:
        print(f"weekly 单步推进（--anchor {anchor} 仅作记录，实际推进到 anchor {stop}）")
    return run_replay(etf=etf, resume=True, mode="generate", stop_after=stop)


def run_diagnose_all(etf: str = "159915") -> int:
    """Champion 专用 PIT 回放：为全部 Due 生成带逐窗口案例的盲化诊断包（不写 events/state）。"""
    try:
        ca = load_ca(etf)
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    except DataAssemblyError as exc:
        print(f"DATA_ASSEMBLY_FAILED: {exc}")
        return 2
    fwd_arrays = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}

    eval_index = EvalIndex()
    miss_counter: dict[str, list[int]] = {}
    forecast_index: dict[tuple[int, str], dict] = {}
    cfg = load_etf_config(etf)
    champion_cfg = cfg.get("champion") or {}
    champion_spec = dict(champion_cfg.get("spec") or {"kind": "champion_zero_axis"})
    champion_spec["id"] = champion_cfg.get("id", "champion_zero_axis")
    champion_id = champion_spec["id"]
    written = 0
    for i in range(len(usable)):
        for k in (1, 2, 4, 8):
            j = i - k
            if j >= 0 and not np.isnan(fwd_arrays[k][j]):
                fc_event = forecast_index.get((j, champion_id))
                if fc_event is None:
                    continue
                actual = float(fwd_arrays[k][j])
                sub = float(fwd_arrays[k // 2][j]) if k // 2 in fwd_arrays and not np.isnan(fwd_arrays[k // 2][j]) else None
                from .evaluate import build_evaluation

                ev = build_evaluation(
                    j, k, champion_id, fc_event["forecast"], actual, sub,
                    features.iloc[j], daily_a, usable,
                )
                ev["type"] = "evaluation"
                ev["anchor"] = usable.iloc[j]["signal"].date().isoformat()
                eval_index.add(ev)
                if k == 1:
                    row = miss_counter.setdefault(champion_id, [0, 0])
                    row[0] += int(ev.get("missed_rally", False))
                    row[1] += int(ev.get("wrong_chase", False))

        matured_counts = {}
        for k in (1, 2, 4, 8):
            matured_counts[f"w{k}"] = sum(
                1 for j in range(max(0, i - k + 1)) if not np.isnan(fwd_arrays[k][j])
            )
        mw1 = matured_counts["w1"]
        stats = compute_bucket_stats(features.iloc[:i], fwd_arrays, up_to=i)
        fc = build_forecast(champion_spec, features.iloc[i], i, fwd_arrays, stats, features=features)
        forecast_index[(i, champion_id)] = {"forecast": fc}

        stats_payload = _diagnostic_stats_from_index(
            eval_index, champion_id, mw1,
            tuple(miss_counter.get(champion_id, [0, 0])),
        )
        cases = build_failure_cases(eval_index, features, champion_id, i)
        if mw1 > 0 and mw1 % 4 == 0:
            rid = round_id("prediction", mw1 // 4)
            write_diagnostic(rid, sanitize(rid, "prediction", stats_payload, cases))
            written += 1
        if mw1 > 0 and mw1 % 8 == 0:
            rid = round_id("strategy", mw1 // 8)
            write_diagnostic(rid, sanitize(rid, "strategy", stats_payload, cases))
            written += 1
    print(f"diagnose-all completed: {written} diagnostics")
    return 0
