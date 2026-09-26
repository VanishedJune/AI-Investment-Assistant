# -*- coding: utf-8 -*-
"""顺序登记干跑：按 Due 顺序模拟 create_screening（不写台账），输出每轮登记数与拒绝原因。"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling import challenge as challenge_mod  # noqa: E402
from rolling import ledger as ledger_mod  # noqa: E402
from rolling.ca import load_ca  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402


def main(etf: str = "159915") -> int:
    set_workspace(etf)
    ca = load_ca(etf)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    champion_cfg = load_etf_config(etf).get("champion") or {}
    champion_spec = champion_cfg.get("spec") or {"kind": "champion_zero_axis"}
    champion_id = champion_cfg.get("id") or "champion_zero_axis"
    registered: list[dict] = []
    drafts = sorted(ledger_mod.DRAFT_DIR.glob("*.draft.json"))
    for path in drafts:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rid = obj["challenge_round"]
        ctype = obj["challenge_type"]
        seq = int(rid.split("_")[1])
        due_idx = seq * 4 if ctype == "prediction" else seq * 8
        before = len(registered)
        reg = challenge_mod.create_screening(
            obj, due_idx, features, usable, daily, ca,
            champion_spec, ctype,
            parent_champion_id=champion_id,
            existing_challengers=registered,
        )
        registered.extend(reg)
        ok_ids = {r["id"] for r in reg}
        for ch in obj["challengers"]:
            s = ch.get("screening", {})
            mark = "OK " if ch["id"] in ok_ids else "REJ"
            print(
                f"{mark} {rid} {ch['id']}: valid={s.get('valid')} div_ok={s.get('divergence_ok')} "
                f"vs_champion={s.get('vs_champion')} vs_others={s.get('vs_others')} "
                f"vs_existing={s.get('vs_existing')}"
            )
        print(f"  -> {rid}: {len(reg)}/{len(obj['challengers'])} (已有 {len(registered)} )")
        _ = before
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
