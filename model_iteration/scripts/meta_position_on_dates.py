# -*- coding: utf-8 -*-
"""查询 Meta 模型在指定日期的持仓策略（各相位目标 + Meta 加权目标）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import allocate_cfg, range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import OUT_DIR, WINDOW_5Y  # noqa: E402


def pct(x) -> str:
    return f"{float(x):.1%}"


def main() -> int:
    etfs = {c: load_etf(c) for c in CODES}
    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    start5y = range_idx(master_dates, *WINDOW_5Y)[0]

    meta = json.loads((OUT_DIR / "meta_weighted_model.json").read_text(encoding="utf-8"))
    weights = meta["weights"]
    data = json.loads((OUT_DIR / "b0207_phase_champions_v2_standard.json").read_text(encoding="utf-8"))
    phase_configs = [
        {"id": ch["model"]["id"], "order": ch["model"]["order"], "params": ch["model"]["params"]}
        for ch in data["champions"]
    ]

    def phase_target_at(cfg: dict, i: int) -> dict:
        date = master_dates[i]
        ex = master_execs[i]
        raw = {c: (et["by_date"].get(date, 0.0) if ex is not None and et["price_map"].get(ex, 0) > 0 else 0.0)
               for c, et in etfs.items()}
        floor = float(cfg["params"]["floor"])
        eff = {c: min(1.0, max(floor, raw[c])) for c in etfs}
        sig = {c: eff[c] > 0.0 for c in etfs}
        return dict(allocate_cfg(sig, eff, cfg["order"], cfg["params"])["weights"])

    for target in ("2021-12-12", "2026-06-23"):
        i = next((k for k in range(len(master_dates) - 1, -1, -1) if master_dates[k] <= target), None)
        if i is None:
            print(target, "no data")
            continue
        print("=" * 70)
        print(f"目标日期 {target} → 最近信号日 {master_dates[i]}，执行日 {master_execs[i]}")
        meta_target = {c: 0.0 for c in etfs}
        for p, cfg in enumerate(phase_configs):
            freq = int(cfg["params"]["rebalance"])
            last = i - ((i - (start5y + p)) % freq)
            if last < 0:
                tgt = {c: 0.0 for c in etfs}
                print(f"  相位{p} ({cfg['id']}，{freq}周)：尚无决策")
            else:
                tgt = phase_target_at(cfg, last)
                print(f"  相位{p} ({cfg['id']}，{freq}周) 最近决策 {master_dates[last]}："
                      + ", ".join(f"{NAMES[c]}{v:.0f}%" for c, v in tgt.items() if v > 0)
                      + f"（现金 {100 - sum(tgt.values()):.0f}%）")
            for c in etfs:
                meta_target[c] += weights[p] * tgt.get(c, 0.0)
        print(f"  Meta 目标："
              + ", ".join(f"{NAMES[c]}{v:.1f}%" for c, v in meta_target.items() if v > 0.05)
              + f"（现金 {100 - sum(meta_target.values()):.1f}%）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
