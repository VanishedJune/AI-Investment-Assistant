# -*- coding: utf-8 -*-
"""查询用户确认的真实持仓历史；本脚本只读，不修改持仓或月报。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "app" / "portfolio_history" / "actual_holdings.json"


def _load() -> dict:
    payload = json.loads(HISTORY.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "actual-portfolio-history-v1":
        raise RuntimeError("真实持仓历史文件版本不受支持")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("真实持仓历史缺少entries数组")
    ids = [entry.get('event_id') for entry in entries]
    if any(not event_id for event_id in ids) or len(set(ids)) != len(ids):
        raise RuntimeError('真实持仓历史存在缺失或重复event_id，保留记录并停止汇总')
    return payload


def _summary(entry: dict) -> str:
    current = entry["current_snapshot"]
    values = current.get("market_values_cny") or {}
    weights = current.get("weights_pct") or {}
    # weights_pct stores the shared growth slot; member weights identify the
    # actual security. Never report the same 50% as both ETF and fund.
    weights = dict(weights)
    weights.update(current.get('growth_slot_member_weights_pct') or {})
    actual = current.get('actual_instrument_weights_pct')
    if actual is not None:
        if any(code in weights and value != weights[code] for code, value in actual.items()):
            raise ValueError('实际证券比例与共享槽位成员比例冲突：' + entry['event_id'])
        weights.update(actual)
    nonzero = []
    names = dict((("159915", "创业板"), ("021528", "财通成长优选混合C"),
                       ("517520", "黄金股"), ("518600", "黄金"), ("516150", "稀土"),
                       ("159622", "创新药"), ("159611", "电力"), ("159941", "纳指"), ("515220", "煤炭"),
                       ("512800", "银行"), ("512690", "酒")))
    codes = list(dict.fromkeys([*names, *weights, *values, *current.get('held_codes', [])]))
    for code in codes:
        name = names.get(code, code)
        if values.get(code) is None and (code in current.get("held_codes", []) or (weights.get(code) or 0) > 0):
            note = current.get("positions", {}).get(code, {}).get("note", "")
            weight = weights.get(code)
            label = f"{weight:g}%" if weight is not None else "比例待补"
            nonzero.append(f"{name}{label}（金额/份数待补；{note}）")
        elif float(values.get(code, 0) or 0) > 0:
            weight = weights.get(code)
            label = f"{weight:.2f}%" if weight is not None else "比例待补"
            nonzero.append(f"{name}{values[code]:,.2f}元（{label}）")
    cash_amount = current.get('cash_amount_cny')
    cash_weight = current.get('cash_pct')
    if cash_amount is None:
        nonzero.append(f"现金{cash_weight:g}%（金额待补）" if cash_weight is not None else "现金金额/比例待补")
    else:
        cash_label = f"{cash_weight:.2f}%" if cash_weight is not None else "比例待补"
        nonzero.append(f"现金{cash_amount:,.2f}元（{cash_label}）")
    return f"{entry['event_id']}  {entry['confirmed_at']}  " + "，".join(nonzero)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="查询用户确认的真实持仓历史（只读）")
    parser.add_argument("--event-id", help="显示指定事件的完整JSON")
    parser.add_argument("--date", help="按确认日期YYYY-MM-DD筛选")
    parser.add_argument("--json", action="store_true", help="输出匹配记录的JSON")
    args = parser.parse_args()

    entries = _load()["entries"]
    selected = entries
    if args.event_id:
        selected = [item for item in selected if item.get("event_id") == args.event_id]
    if args.date:
        selected = [item for item in selected if str(item.get("confirmed_at", "")).startswith(args.date)]
    if not selected:
        print("未找到匹配的真实持仓记录")
        return 1
    if args.json or args.event_id:
        print(json.dumps(selected[0] if len(selected) == 1 else selected, ensure_ascii=False, indent=2))
    else:
        corrections = {item['correction']['supersedes_event_id']: item['event_id'] for item in entries if item.get('correction', {}).get('supersedes_event_id')}
        for item in selected:
            suffix = f"  [金额登记已由{corrections[item['event_id']]}纠正，不作有效估值]" if item['event_id'] in corrections else ""
            print(_summary(item) + suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
