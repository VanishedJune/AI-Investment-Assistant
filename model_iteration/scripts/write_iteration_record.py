"""为已完成全历史迭代的 ETF 生成 ITERATION_001_<code>.json/.md 记录。

用法（model_iteration 目录）：
    python -m scripts.write_iteration_record --etf <code>

约定：记录不含任何哈希/摘要字段（全局规则）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rolling import ledger  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402


def main(etf: str) -> int:
    ws = ledger.set_workspace(etf)
    cfg = load_etf_config(etf)
    state = ledger.load_state()
    events = ledger.read_events()
    _, _, _, _, usable, _ = load_aligned_data(etf)
    proposals = sorted(ledger.PROPOSALS.glob("ROUND_*.json"))
    diagnostics = sorted(ledger.DIAGNOSTICS.glob("ROUND_*.json"))
    challengers = state.get("active_challengers", [])
    promo_history = state.get("promotion_history", [])
    chain = [r["old_champion_id"] for r in promo_history] + [state["champion"]["id"]]
    event_counts = dict(Counter(e.get("type", "?") for e in events))
    status_counts = dict(Counter(c.get("status", "?") for c in challengers))

    record = {
        "schema_version": "iteration-record-v1.0.0",
        "etf": etf,
        "name": cfg.get("name"),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule_version": "existing-weekly-model-target" if cfg.get("execution_policy") is None else "B0207-v3.0.0",
        "execution_policy": cfg.get("execution_policy"),
        "coverage": cfg.get("coverage"),
        "usable_anchors": len(usable),
        "latest_anchor_index": state.get("latest_anchor_index"),
        "resume_from": state.get("resume_from"),
        "pending_ai_reviews": len(state.get("pending_ai_reviews", [])),
        "initial_champion": (cfg.get("champion") or {}).get("id"),
        "final_champion": state["champion"]["id"],
        "promotion_chain": chain,
        "promotions": len(promo_history),
        "proposals_frozen": len(proposals),
        "rounds_registered": event_counts.get("proposal_orchestrated", 0),
        "diagnostics": len(diagnostics),
        "challengers_total": len(challengers),
        "challenger_status": status_counts,
        "events_total": len(events),
        "events_by_type": event_counts,
        "workspace": str(ws),
    }
    iteration_dir = ws / "iterations"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in iteration_dir.glob(f"ITERATION_*_{etf}.json"):
        try:
            existing.append(int(path.name.split("_")[1]))
        except (ValueError, IndexError):
            continue
    iteration_no = max(existing, default=0) + 1
    iteration_id = f"ITERATION_{iteration_no:03d}_{etf}"
    out_json = iteration_dir / f"{iteration_id}.json"
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {iteration_id}（{cfg.get('name')} ｜ 全历史回放迭代）",
        "",
        f"- ETF：{etf} {cfg.get('name')} ｜ 覆盖 {cfg.get('coverage', {}).get('start')} ~ {cfg.get('coverage', {}).get('end')} ｜ "
        f"{len(usable)} 个可用锚点 ｜ 8 周 horizon ｜ {cfg.get('cost_bps')}bp 成本",
        ("- 规则：项目原有冠军模型周频目标仓位直接执行；引擎 promotion-v2.0.0（sanitized_only 提案 + AI 复核晋级）"
         if cfg.get("execution_policy") is None else
         "- 规则：B0207 式执行策略 v3.0.0（4 周决策锚点 / 非决策周仅加仓 / 看空保护卖出）；引擎 promotion-v2.0.0（sanitized_only 提案 + AI 复核晋级）"),
        "",
        "## 结果",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 初始 Champion | **{record['initial_champion']}** |",
        f"| 最终 Champion | **{record['final_champion']}** |",
        f"| 晋级链 | {' → '.join(chain)} |",
        f"| 晋级次数 | {record['promotions']} |",
        f"| 冻结提案 | {record['proposals_frozen']}（成功登记 {record['rounds_registered']} 轮） |",
        f"| 挑战者 | {record['challengers_total']}（{json.dumps(status_counts, ensure_ascii=False)}） |",
        f"| 事件 | {record['events_total']}（{json.dumps(event_counts, ensure_ascii=False)}） |",
        f"| 待复核 | {record['pending_ai_reviews']} |",
        "",
        "## 说明",
        "",
        "- 全历史从头（anchor 0）以初始 Champion 规则重放；每 Due 轮由隔离工厂生成 sanitized_only 提案并冻结，"
        "机器门禁通过且推荐的候选经 AI 复核（APPROVE）晋级。",
        "- 本记录不含任何哈希/摘要字段（按全局规则省略）。",
        "",
    ]
    (iteration_dir / f"{iteration_id}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"record written: {out_json}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
