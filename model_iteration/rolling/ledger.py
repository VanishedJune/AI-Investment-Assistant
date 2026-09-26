# -*- coding: utf-8 -*-
"""台账：events.jsonl（只追加）+ state.json（派生快照）+ state_history/ + diagnostics/ + proposals/。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "weekly_rolling"
EVENTS = LEDGER_DIR / "events.jsonl"
STATE = LEDGER_DIR / "state.json"
STATE_HISTORY = LEDGER_DIR / "state_history"
DIAGNOSTICS = LEDGER_DIR / "diagnostics"
PROPOSALS = LEDGER_DIR / "proposals"
CA_DIR = LEDGER_DIR / "corporate_actions"
DRAFT_DIR = ROOT / "scripts" / "proposal_drafts"
_CURRENT_ETF = "159915"


def set_workspace(etf: str = "159915") -> Path:
    """按 ETF 切换工作区（历史/台账/草稿/CA 全部收进 etf_<code>/ 目录）。"""
    from .config import load_etf_config

    cfg = load_etf_config(etf)
    if cfg.get("training_disabled"):
        raise ValueError(f"{etf}为冻结参数迁移标的；禁止创建/推进训练台账，请使用只读 monitoring_baseline_report 推理")
    ws = ROOT / cfg.get("workspace", f"etf_{etf}")
    ws = ws.resolve()
    global LEDGER_DIR, EVENTS, STATE, STATE_HISTORY, DIAGNOSTICS, PROPOSALS, CA_DIR, DRAFT_DIR, _CURRENT_ETF
    _CURRENT_ETF = etf
    LEDGER_DIR = ws / "weekly_rolling"
    EVENTS = LEDGER_DIR / "events.jsonl"
    STATE = LEDGER_DIR / "state.json"
    STATE_HISTORY = LEDGER_DIR / "state_history"
    DIAGNOSTICS = LEDGER_DIR / "diagnostics"
    PROPOSALS = LEDGER_DIR / "proposals"
    CA_DIR = LEDGER_DIR / "corporate_actions"
    DRAFT_DIR = ws / "proposal_drafts"
    ensure_dirs()
    return ws


def set_ledger_root(path: Path) -> Path:
    """将台账读写整体重定向到指定目录（用于隔离 verify，不触碰正式台账）。"""
    global LEDGER_DIR, EVENTS, STATE, STATE_HISTORY, DIAGNOSTICS, PROPOSALS, CA_DIR
    path = path.resolve()
    LEDGER_DIR = path
    EVENTS = path / "events.jsonl"
    STATE = path / "state.json"
    STATE_HISTORY = path / "state_history"
    DIAGNOSTICS = path / "diagnostics"
    PROPOSALS = path / "proposals"
    CA_DIR = path / "corporate_actions"
    ensure_dirs()
    return path


def ensure_dirs() -> None:
    for d in (LEDGER_DIR, STATE_HISTORY, DIAGNOSTICS, PROPOSALS, CA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def append_event(obj: dict) -> str:
    ensure_dirs()
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    event_number = 1
    if EVENTS.exists():
        with open(EVENTS, encoding="utf-8") as existing:
            event_number += sum(1 for value in existing if value.strip())
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    from .config import integrity_digests_enabled

    if not integrity_digests_enabled(_CURRENT_ETF):
        return f"EVENT-{event_number:08d}"
    return line_hash(line)


def read_events() -> list[dict]:
    if not EVENTS.exists():
        return []
    out = []
    with open(EVENTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ledger_hash() -> str:
    """events.jsonl + state.json 的 SHA-256，用于 REPLAY/VERIFY 确定性校验。"""
    return ledger_hash_dir(LEDGER_DIR)


def ledger_hash_dir(directory: Path) -> str:
    """计算指定台账目录的 events.jsonl + state.json SHA-256。"""
    h = hashlib.sha256()
    for name in ("events.jsonl", "state.json"):
        p = Path(directory) / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def default_state(etf: str | None = None) -> dict:
    from .config import integrity_digests_enabled, load_etf_config

    if etf is None:
        etf = _CURRENT_ETF
    champion_cfg = load_etf_config(etf).get("champion") or {}
    champion_spec = champion_cfg.get("spec") or {"kind": "champion_zero_axis"}
    champion_id = champion_cfg.get("id") or "champion_zero_axis"
    champion = {"id": champion_id, "spec": champion_spec, "effective_anchor": 0}
    if integrity_digests_enabled(etf):
        champion["spec_hash"] = hashlib.sha256(
            json.dumps(champion_spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    state = {
        "schema_version": "rolling-v1.0.0",
        "etf": etf,
        "latest_anchor_index": -1,
        "paused_at": None,
        "resume_from": 0,
        "matured_counts": {"w1": 0, "w2": 0, "w4": 0, "w8": 0},
        "last_triggered": {"prediction": 0, "strategy": 0},
        "champion": champion,
        "active_challengers": [],
        "promotion_history": [],
        "next_challenge_due": {"prediction": None, "strategy": None},
        "promotion_policy_version": "promotion-v2.0.0",
        "pending_ai_reviews": [],
        "auto_approve": False,
        "diagnostic_ok": True,
    }
    state["integrity_mode"] = "digest_verified" if integrity_digests_enabled(etf) else "not_evaluated"
    if integrity_digests_enabled(etf):
        state["ca_events_hash"] = ""
    return state


def load_state() -> dict:
    ensure_dirs()
    if not STATE.exists():
        return default_state()
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    ensure_dirs()
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    seq = len(list(STATE_HISTORY.glob("*.json"))) + 1
    (STATE_HISTORY / f"{seq:06d}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def diagnostic_path(round_id: str) -> Path:
    return DIAGNOSTICS / f"{round_id}.json"


def proposal_path(round_id: str) -> Path:
    return PROPOSALS / f"{round_id}.json"


def round_id(challenge_type: str, sequence: int) -> str:
    return f"ROUND_{sequence:03d}_{challenge_type}"


# 默认工作区：159915（多 ETF 时由 CLI set_workspace 按 --etf 切换）
set_workspace("159915")
