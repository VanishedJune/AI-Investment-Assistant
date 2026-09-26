# -*- coding: utf-8 -*-
"""REPLAY/VERIFY 驱动：回放 -> AI 复核（固定理由）-> 续跑，直到完成并打印台账哈希。"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.ledger import ledger_hash, load_state, save_state, set_workspace  # noqa: E402
from rolling.replay import run_replay  # noqa: E402
from rolling.review import apply_ai_review  # noqa: E402


def main(
    etf: str = "159915",
    mode: str = "verify",
    resume: bool = False,
    stop_after: int | None = None,
    isolated: bool = False,
) -> int:
    from rolling import ledger as ledger_mod

    set_workspace(etf)
    main_ledger = Path(ledger_mod.LEDGER_DIR)
    if isolated:
        iso = main_ledger.parent / "weekly_rolling_verify"
        if iso.exists():
            shutil.rmtree(iso)
        iso.mkdir(parents=True, exist_ok=True)
        for sub in ("corporate_actions", "diagnostics", "proposals"):
            src = main_ledger / sub
            if src.exists():
                shutil.copytree(src, iso / sub)
        ledger_mod.set_ledger_root(iso)
        resume = False  # 隔离台账必须从零 verify，禁止 resume 正式台账
        print("MAIN_LEDGER_HASH", ledger_mod.ledger_hash_dir(main_ledger), flush=True)
        print("ISOLATED_LEDGER_DIR", iso, flush=True)
    reason = "AI 复核：机器门禁+同现金流门禁全过且为推荐候选，批准晋级"
    rc = run_replay(etf=etf, resume=resume, mode=mode, stop_after=stop_after)
    approvals = 0
    while rc == 4:
        state = load_state()
        pend = state.get("pending_ai_reviews", [])
        if not pend:
            print("no pending; abort", flush=True)
            break
        r = pend[0]
        rc2, msg = apply_ai_review(state, r["id"], r["anchor"], "APPROVE", reason)
        print(f"{mode} approve {msg}", flush=True)
        save_state(state)
        approvals += 1
        rc = run_replay(etf=etf, resume=True, mode=mode, stop_after=stop_after)
    print(f"{mode.upper()}_RC {rc} approvals {approvals}", flush=True)
    print("LEDGER_HASH", ledger_hash(), flush=True)
    if isolated:
        main_hash = ledger_mod.ledger_hash_dir(main_ledger)
        verify_hash = ledger_mod.ledger_hash()
        print("MAIN_LEDGER_HASH", main_hash, flush=True)
        print("VERIFY_LEDGER_HASH", verify_hash, flush=True)
        print("HASH_MATCH", main_hash == verify_hash, flush=True)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    parser.add_argument("--mode", default="verify", choices=["generate", "verify"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(args.etf, args.mode, args.resume, args.stop_after, args.isolated))
