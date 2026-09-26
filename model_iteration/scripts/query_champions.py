# -*- coding: utf-8 -*-
"""查询 8 只 ETF 的当前冠军模型（取各 ETF 最新进度工作区）。

用法:
    python -m scripts.query_champions [--etf 512010] [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
MODEL = BASE / "model_iteration"
CODES = json.loads((MODEL / 'configs/investment_priority.json').read_text(encoding='utf-8'))['priority']
NAMES = {
    "159915": "创业板ETF易方达",
    "512010": "医药ETF易方达",
    "159941": "纳指ETF广发",
    "159611": "电力ETF广发",
    "518600": "广发黄金ETF",
    "517520": "黄金股ETF永赢",
    "515220": "煤炭ETF国泰",
    "159622": "创新药ETF东财",
    "512690": "鹏华酒ETF",
    "512800": "华宝银行ETF",
    "516150": "稀土ETF嘉实",
}


def champion_display_name(code: str, champion_id: str | None) -> str | None:
    """Build the user-facing name without changing the immutable ledger ID."""
    return f"{code}_{champion_id}" if champion_id else None


def spec_brief(spec: dict | None) -> dict:
    if not spec:
        return {}
    brief: dict = {}
    score = spec.get("score", {})
    if score.get("terms"):
        brief["terms"] = [
            {"feature": t.get("feature"), "weight": t.get("weight")}
            for t in score["terms"]
        ]
    gates = spec.get("gates")
    if gates:
        brief["gates"] = gates
    pos = spec.get("position")
    if pos:
        brief["position"] = pos
    dd = spec.get("drawdown_control")
    if dd:
        brief["drawdown_control"] = dd
    return brief


def newest_workspace_state(code: str) -> tuple[Path | None, dict | None]:
    config_path = MODEL / 'configs' / f'etf_{code}.json'
    if config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding='utf-8'))
        configured = (MODEL / cfg.get('workspace', f'etf_{code}') / 'weekly_rolling').resolve()
        if not configured.is_relative_to(MODEL.resolve()):
            raise ValueError('冠军工作区越界')
        if (configured / 'state.json').is_file():
            return configured, json.loads((configured / 'state.json').read_text(encoding='utf-8'))
    root = MODEL / f"etf_{code}"
    best_path: Path | None = None
    best: dict | None = None
    best_anchor = -1
    if not root.is_dir():
        return None, None
    for ws in sorted(p for p in root.iterdir() if p.is_dir()):
        st = ws / "state.json"
        if not st.is_file():
            continue
        try:
            s = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        anchor = int(s.get("latest_anchor_index", -1) or -1)
        if anchor > best_anchor:
            best_anchor = anchor
            best_path = ws
            best = s
    return best_path, best


def main() -> int:
    ap = argparse.ArgumentParser(description="查询 ETF 冠军模型")
    ap.add_argument("--etf", default=None, help="ETF 代码，缺省查询全部")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    codes = [args.etf] if args.etf else CODES
    rows: list[dict] = []
    for code in codes:
        ws, s = newest_workspace_state(code)
        champ = (s or {}).get("champion") or {}
        champion_id = champ.get("id")
        rows.append(
            {
                "code": code,
                "name": NAMES.get(code, code),
                "workspace": ws.name if ws else None,
                "latest_anchor_index": (s or {}).get("latest_anchor_index"),
                "champion_id": champion_id,
                "champion_name": champion_display_name(code, champion_id),
                "effective_anchor": champ.get("effective_anchor"),
                "spec": spec_brief(champ.get("spec")),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    print(f"{'ETF':<7}{'名称':<14}{'工作区':<22}{'进度':>6}  {'冠军名称':<24}{'生效':>6}")
    for r in rows:
        print(
            f"{r['code']:<7}{r['name']:<14}{str(r['workspace'] or '-'):<22}"
            f"{str(r['latest_anchor_index']):>6}  {str(r['champion_name'] or '-'):<24}"
            f"{str(r['effective_anchor']):>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
