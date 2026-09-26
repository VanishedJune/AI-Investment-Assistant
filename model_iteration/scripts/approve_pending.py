"""批量 AI 晋级复核：每个暂停锚点批准机器推荐的候选（留痕），其余同锚点候选
由晋级逻辑自动转为 NON_WINNING_CANDIDATE。

用法（model_iteration 目录）：
    python -m scripts.approve_pending --etf <code>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling.ledger import load_state, save_state, set_workspace  # noqa: E402
from rolling.review import apply_ai_review  # noqa: E402


def main(etf: str) -> int:
    set_workspace(etf)
    state = load_state()
    pending = list(state.get("pending_ai_reviews", []))
    if not pending:
        print("no pending ai reviews")
        return 0
    by_anchor: dict[int, list[dict]] = {}
    for review in pending:
        by_anchor.setdefault(int(review["anchor"]), []).append(review)
    for anchor in sorted(by_anchor):
        reviews = by_anchor[anchor]
        recommended = next((r for r in reviews if r.get("recommended", False)), reviews[0])
        rc, msg = apply_ai_review(
            state,
            recommended["id"],
            anchor,
            "APPROVE",
            "批量复核：机器硬门禁通过且为推荐候选，批准晋级（事件留痕）",
        )
        print(f"anchor[{anchor}] {recommended['id']}: {msg}")
        if rc != 0:
            print(f"  WARN approve failed: {msg}")
    save_state(state)
    remaining = len(state.get("pending_ai_reviews", []))
    print(f"remaining pending: {remaining}")
    return 1 if remaining else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
