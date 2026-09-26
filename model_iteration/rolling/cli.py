# -*- coding: utf-8 -*-
"""CLI 入口：rolling_replay / rolling_proposal / rolling_checkpoint / rolling_weekly。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import challenge as challenge_mod
from . import promotion as promotion_mod
from .benchmark import compute_reference_benchmarks
from .ca import load_ca, validate_coverage
from .config import integrity_digests_enabled, load_etf_config
from .data import DataAssemblyError, load_aligned_data
from .ledger import (
    append_event,
    diagnostic_path,
    load_state,
    proposal_path,
    read_events,
    save_state,
    set_workspace,
)
from .proposal import freeze_final_proposal, validate_proposal
from .replay import run_diagnose_all, run_replay, run_weekly
from .review import apply_ai_review
from .statindex import index_from_events


def _replay(args) -> int:
    set_workspace(args.etf)
    return run_replay(etf=args.etf, resume=args.resume, mode=args.mode)


def _diagnose_all(args) -> int:
    set_workspace(args.etf)
    return run_diagnose_all(etf=args.etf)


def _proposal(args) -> int:
    set_workspace(args.etf)
    path = Path(args.file)
    obj = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_proposal(obj, etf=args.etf)
    if errors:
        print("Proposal 校验失败:")
        for e in errors:
            print(" -", e)
        return 1
    out = freeze_final_proposal(obj, etf=args.etf)
    print("已冻结 FINAL Proposal:", out)
    return 0


def _run_screening(path: Path, etf: str) -> int:
    """冻结前筛查：schema + PIT + divergence，返回可登记数量（不写任何台账）。"""
    set_workspace(etf)
    obj = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_proposal(obj, etf=etf)
    if errors:
        print("Proposal 校验失败:")
        for e in errors:
            print(" -", e)
        return 1
    try:
        rid = obj["challenge_round"]
        ctype = obj["challenge_type"]
        seq = int(rid.split("_")[1])
        due_idx = seq * 4 if ctype == "prediction" else seq * 8
        cfg = load_etf_config(etf)
        coverage = cfg.get("coverage") or {}
        ca = load_ca(etf)
        if coverage.get("start") and coverage.get("end"):
            validate_coverage(ca, coverage["start"], coverage["end"])
        daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
        fwd_arrays = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
        state = load_state()
        registered = challenge_mod.create_screening(
            obj, due_idx, features, usable, daily, ca,
            state["champion"]["spec"], ctype,
            parent_champion_id=state["champion"]["id"],
            existing_challengers=state["active_challengers"],
            fwd_arrays=fwd_arrays,
            execution_policy=cfg.get("execution_policy") or None,
        )
        print(f"precheck {rid}: 可登记 {len(registered)}/{len(obj['challengers'])}")
        for ch in obj["challengers"]:
            s = ch.get("screening", {})
            print(
                f"  {ch['id']}: valid={s.get('valid')} divergence_ok={s.get('divergence_ok')} "
                f"vs_champion={s.get('vs_champion')} vs_others={s.get('vs_others')}"
            )
        return 0 if len(registered) == len(obj["challengers"]) else 1
    except Exception as exc:  # noqa: BLE001
        print("precheck 失败:", exc)
        return 2


def _precheck(args) -> int:
    return _run_screening(Path(args.file), args.etf)


def _challenge(args) -> int:
    """rolling_challenge --proposal：创建期筛查（与 precheck 同一口径）。"""
    return _run_screening(Path(args.proposal), args.etf)


def _checkpoint(args) -> int:
    """独立晋级检查：在当前暂停锚点运行 promotion.checkpoint 并持久化状态。"""
    etf = args.etf
    set_workspace(etf)
    try:
        cfg = load_etf_config(etf)
        coverage = cfg.get("coverage") or {}
        ca = load_ca(etf)
        if coverage.get("start") and coverage.get("end"):
            validate_coverage(ca, coverage["start"], coverage["end"])
    except Exception as exc:  # noqa: BLE001
        print("DATA_CA_INCOMPLETE:", exc)
        return 2
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    state = load_state()
    due_idx = state.get("paused_at")
    if due_idx is None:
        due_idx = state.get("latest_anchor_index")
    if due_idx is None:
        print("无可检查的锚点")
        return 1
    eval_index = index_from_events(read_events())
    refs = compute_reference_benchmarks(etf, daily_a, weekly_a, daily, anchors, usable, features, due_idx)
    promo = promotion_mod.checkpoint(
        state, features, usable, daily, ca, due_idx, eval_index,
        usable["fwd1"].to_numpy(dtype=float),
        usable["fwd8"].to_numpy(dtype=float),
        usable["anchor_close"].to_numpy(dtype=float),
        benchmark_refs=refs,
        benchmark_frames=(ca, daily_a, weekly_a, daily, anchors, usable, features),
    )
    if promo.get("checked"):
        checkpoint_ref = append_event({
            "type": "promotion_checkpoint",
            "anchor": due_idx,
            "policy_version": state.get("promotion_policy_version"),
            "checked": promo["checked"],
            "candidates": [
                {"id": c["id"], "metrics": {
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
                "anchor": due_idx,
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
    save_state(state)
    print(f"checkpoint@{due_idx}: checked={len(promo['checked'])} candidates={len(promo['candidates'])} recommended={promo.get('recommended')}")
    for row in promo["checked"]:
        print(f"  {row['id']}: {row['status']} age1={row['age1']} all_pass={row.get('all_pass')}")
    return 4 if state.get("pending_ai_reviews") else 0


def _approve(args) -> int:
    """AI 晋级复核：APPROVE / REJECT / HOLD；无 APPROVE 不得 PROMOTED（auto_approve=false）。"""
    set_workspace(args.etf)
    state = load_state()
    rc, msg = apply_ai_review(state, args.candidate, args.anchor, args.decision, args.reason or "")
    print(msg)
    save_state(state)
    return rc


def _diagnose(args) -> int:
    set_workspace(args.etf)
    path = diagnostic_path(args.round)
    if not path.exists():
        print("诊断包不存在:", path)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def _weekly(args) -> int:
    set_workspace(args.etf)
    return run_weekly(etf=args.etf, anchor=args.anchor)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(prog="rolling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser("replay")
    p_replay.add_argument("--etf", required=True)
    p_replay.add_argument("--resume", action="store_true")
    p_replay.add_argument("--mode", choices=["generate", "verify"], default="generate")
    p_replay.set_defaults(func=_replay)

    p_diag_all = sub.add_parser("diagnose-all")
    p_diag_all.add_argument("--etf", required=True)
    p_diag_all.set_defaults(func=_diagnose_all)

    p_prop = sub.add_parser("proposal")
    p_prop.add_argument("--file", required=True)
    p_prop.add_argument("--etf", required=True)
    p_prop.set_defaults(func=_proposal)

    p_pre = sub.add_parser("precheck")
    p_pre.add_argument("--file", required=True)
    p_pre.add_argument("--etf", required=True)
    p_pre.set_defaults(func=_precheck)

    p_ck = sub.add_parser("checkpoint")
    p_ck.add_argument("--etf", required=True)
    p_ck.set_defaults(func=_checkpoint)

    p_ch = sub.add_parser("challenge")
    p_ch.add_argument("--proposal", required=True)
    p_ch.add_argument("--etf", required=True)
    p_ch.set_defaults(func=_challenge)

    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("--round", required=True)
    p_diag.add_argument("--etf", required=True)
    p_diag.set_defaults(func=_diagnose)

    p_wk = sub.add_parser("weekly")
    p_wk.add_argument("--etf", required=True)
    p_wk.add_argument("--anchor", default=None)
    p_wk.set_defaults(func=_weekly)

    p_ap = sub.add_parser("approve")
    p_ap.add_argument("--candidate", required=True)
    p_ap.add_argument("--anchor", type=int, required=True)
    p_ap.add_argument("--decision", choices=["APPROVE", "REJECT", "HOLD"], default="APPROVE")
    p_ap.add_argument("--reason", default=None)
    p_ap.add_argument("--etf", required=True)
    p_ap.set_defaults(func=_approve)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
