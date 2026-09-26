# -*- coding: utf-8 -*-
"""8 ETF champion model vs smart DCA, last-5-year window (read-only PK).

Window: 2021-08-10 ~ last usable anchor (2026-08-07).  Both sides use the
same cash flow: weekly 2000, next raw_open execution, 10bp cost.  The
champion side applies the ETF's current champion spec over the window (a
backtest with today's rule; late-effective champions are hindsight).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.benchmark import rule_with_deposits, smart_dca  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402

CODES = ["159941", "159915", "518600", "512800", "512690", "512010", "159622", "516150"]
START = "2021-08-10"
BASE_AMOUNT = 2000.0
COST_BPS = 10


def champion_spec(code: str) -> tuple[str, dict]:
    root = ROOT / f"etf_{code}"
    best = None
    best_anchor = -1
    for ws in sorted(p for p in root.iterdir() if p.is_dir()):
        st = ws / "state.json"
        if not st.is_file():
            continue
        try:
            payload = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        anchor = int(payload.get("latest_anchor_index", -1) or -1)
        if anchor > best_anchor:
            best_anchor = anchor
            best = payload
    champion = (best or {}).get("champion") or {}
    return str(champion.get("id", "")), champion.get("spec") or {}


def main() -> int:
    print(
        f"{'ETF':<8}{'champion':<14}{'window':<24}{'side':<10}"
        f"{'TWR':>9}{'IRR':>9}{'Sharpe':>8}{'MDD':>9}{'invested':>10}{'final':>12}"
    )
    rows = []
    for code in CODES:
        set_workspace(code)
        try:
            daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(code)
        except Exception as exc:  # noqa: BLE001
            print(f"{code}: load failed {exc}")
            continue
        signals = usable["signal"].dt.date.astype(str)
        candidates = np.flatnonzero(signals >= START)
        if len(candidates) == 0:
            print(f"{code}: no anchors after {START}")
            continue
        start_anchor = int(candidates[0])
        end_anchor = len(usable) - 1
        frames = (load_ca(code), daily_a, weekly_a, daily, anchors, usable, features)
        champ_id, spec = champion_spec(code)
        champ = rule_with_deposits(
            code, spec, base_amount=BASE_AMOUNT, cost_bps=COST_BPS,
            end_idx=end_anchor, start_anchor=start_anchor, _frames=frames,
        )
        dca = smart_dca(
            code, end_idx=end_anchor, base_amount=BASE_AMOUNT, cost_bps=COST_BPS,
            _frames=frames, start_anchor=start_anchor,
        )
        window = f"{usable.iloc[start_anchor]['signal'].date()}~{usable.iloc[end_anchor]['signal'].date()}"
        rows.append((code, champ_id, window, champ, dca))
        for side, r in (("champion", champ), ("dca", dca)):
            print(
                f"{code:<8}{champ_id[:12]:<14}{window:<24}{side:<10}"
                f"{r['twr']:>9.2%}{r['irr_annual']:>9.2%}{r['sharpe']:>8.2f}"
                f"{r['mdd']:>9.2%}{r['invested']:>10.0f}{r['final_nav']:>12.0f}"
            )
    print("\n=== champion - dca (TWR / IRR / MDD pp) ===")
    for code, champ_id, window, champ, dca in rows:
        print(
            f"{code:<8}{champ_id[:12]:<14}"
            f"TWR {champ['twr'] - dca['twr']:>+8.2%}  "
            f"IRR {champ['irr_annual'] - dca['irr_annual']:>+8.2%}  "
            f"MDD {champ['mdd'] - dca['mdd']:>+8.2%}"
        )
    return 0


if __name__ == "__main__":
    from rolling.ca import load_ca

    raise SystemExit(main())
