# -*- coding: utf-8 -*-
"""AI 晋级复核：apply_ai_review 由 CLI 与全历史驱动循环共用。"""

from __future__ import annotations

from .ledger import append_event


def apply_ai_review(state: dict, candidate_id: str, anchor: int, decision: str, reason: str = "") -> tuple[int, str]:
    decision = decision.upper()
    if state.get("auto_approve"):
        return 0, "auto_approve=true，机器门禁通过即晋级，无需人工批准"
    idx = None
    for n, r in enumerate(state.get("pending_ai_reviews", [])):
        if r.get("id") == candidate_id and r.get("anchor") == anchor:
            idx = n
            break
    if idx is None:
        return 1, f"未找到待复核候选: {candidate_id}@{anchor}"
    review = state["pending_ai_reviews"].pop(idx)
    if decision not in ("APPROVE", "REJECT", "HOLD"):
        state.setdefault("pending_ai_reviews", []).append(review)
        return 1, "decision 必须为 APPROVE / REJECT / HOLD"
    ch = next((c for c in state["active_challengers"] if c["id"] == candidate_id), None)
    if ch is None:
        state.setdefault("pending_ai_reviews", []).append(review)
        return 1, "候选不在 active_challengers"
    if decision == "APPROVE":
        ch["status"] = "PROMOTED"
        old_id = state["champion"]["id"]
        state["promotion_history"].append({
            "old_champion_id": old_id,
            "new_champion_id": ch["id"],
            "anchor": anchor,
            "effective_anchor": anchor + 1,
            "ai_decision": "APPROVE",
            "reason": reason,
        })
        promoted_champion = {
            "id": ch["id"],
            "spec": ch["spec"],
            "effective_anchor": anchor + 1,
        }
        if "spec_hash" in ch:
            promoted_champion["spec_hash"] = ch["spec_hash"]
        state["champion"] = promoted_champion
        for r in state.get("pending_ai_reviews", []):
            if r.get("anchor") == anchor:
                other = next((c for c in state["active_challengers"] if c["id"] == r["id"]), None)
                if other and other["status"] == "PROMOTION_CANDIDATE":
                    other["status"] = "NON_WINNING_CANDIDATE"
        state["pending_ai_reviews"] = [
            r for r in state.get("pending_ai_reviews", []) if r.get("anchor") != anchor
        ]
        for c in state["active_challengers"]:
            if c["id"] == candidate_id:
                continue
            if c.get("parent_champion_id") != state["champion"]["id"]:
                c["status"] = "STALE_SHADOW"
        msg = f"APPROVED: {candidate_id} 自 anchor {anchor + 1} 生效"
    elif decision == "REJECT":
        ch["status"] = "REJECTED"
        ch["reject_reason"] = reason or "AI_REJECT"
        msg = f"REJECTED: {candidate_id}"
    else:  # HOLD
        state.setdefault("pending_ai_reviews", []).append(review)
        return 0, f"HOLD: {candidate_id} 保持候选"
    review_event = {
        "type": "ai_review",
        "anchor": anchor,
        "candidate": candidate_id,
        "decision": decision,
        "reason": reason,
        "effective_anchor": anchor + 1 if decision == "APPROVE" else None,
    }
    if review.get("checkpoint_ref") is not None:
        review_event["checkpoint_ref"] = review["checkpoint_ref"]
    elif review.get("checkpoint_hash") is not None:
        review_event["checkpoint_hash"] = review["checkpoint_hash"]
    append_event(review_event)
    return 0, msg
