"""批量冻结隔离草稿为 FINAL Proposal（单进程装配一次数据）。

用法（model_iteration 目录）：
    python -m scripts.freeze_all_proposals --etf <code>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from rolling.proposal import freeze_final_proposal  # noqa: E402


def main(etf: str) -> int:
    ws = set_workspace(etf)
    drafts_dir = ws / "sanitized_workspace" / "outputs"
    drafts = sorted(drafts_dir.glob("ROUND_*.draft.json"))
    if not drafts:
        print("no drafts found:", drafts_dir)
        return 1
    _, _, _, _, usable, _ = load_aligned_data(etf)
    ok = 0
    failed: list[tuple[str, str]] = []
    for draft in drafts:
        obj = json.loads(draft.read_text(encoding="utf-8"))
        rid = obj.get("challenge_round", draft.stem)
        try:
            freeze_final_proposal(obj, etf=etf, usable=usable)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed.append((rid, str(exc)[:240]))
    print(f"freeze completed: ok={ok} failed={len(failed)}")
    for rid, err in failed:
        print(f"FAIL {rid}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
