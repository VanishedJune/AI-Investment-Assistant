# -*- coding: utf-8 -*-
"""准备四周决策月报的客观参数与B0207机械基线。"""

from __future__ import annotations

import argparse
import json

from scripts.agent_multi_parameter_gate import GATE_VERSION
from scripts.monthly_policy import MonthlyPolicyError, parse_iso_date, prepare, status_for


def main() -> int:
    parser = argparse.ArgumentParser(description="准备B0207机械基线与Agent分析候选")
    parser.add_argument("--decision-date", required=True, help="28天周期名义决策日 YYYY-MM-DD")
    parser.add_argument("--check", action="store_true", help="只检查决策周期与本地数据，不写文件")
    args = parser.parse_args()
    try:
        nominal = parse_iso_date(args.decision_date)
        if args.check:
            status = status_for(nominal)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0 if status["status"] == "READY" else 2
        snapshot_path, candidate_path, snapshot_meta = prepare(nominal)
        print(
            json.dumps(
                {
                    "status": "PREPARED",
                    "snapshot": str(snapshot_path),
                    "snapshot_size_bytes": snapshot_meta["size_bytes"],
                    "snapshot_modified_utc": snapshot_meta["modified_utc"],
                    "agent_candidate": str(candidate_path),
                    "next_step": (
                        f"Agent先读取迭代模型三份权威文件并填写iteration_model_reference，再按{GATE_VERSION}复核。"
                        "逐只读取forward_evidence最近5日/4完整周，在forward_analysis分开填写当前状态、"
                        "第一优先级必须分别读取159915自身行情和021528自身净值分析，填写"
                        "first_priority_slot_reviewed=true后再决定槽位承接成员；禁止用创业板走势替代基金判断。"
                        "连续变化、指定持有期的价格收益假设和当前入场质量；解释涨幅是否已兑现、"
                        "剩余空间、下行风险、成本与不操作/现金/轮换的机会成本，给出三路径和复核失效条件。"
                        "不能用MA多头排列、指标全正、MACD变色或涨跌持续时间代替未来价格判断。"
                        "七项evidence_review只作必查清单，不计票，不是五参数保护触发器。"
                        "按V70填写v70_protection_review：R002/R003/R008依据、R007防追空复核、反证、"
                        "失效条件及连续区间是否已执行。名义决策候选以B0207周期基线为保护起点；"
                        "同时逐只填写holding_peak_stop_review：决策前是否持有、连续持有起始日、"
                        "持有期间最高日收盘、信号日收盘、从高点回撤及区间状态；回撤严格超过10%"
                        "时可独立触发保护，不要求同时满足V70看空规则。保护卖出款按保护前其他仍持有ETF"
                        "的市值比例回投，不回买被保护ETF、不新建此前未持有ETF；没有合格接收标的时才留现金。"
                        "V70的70%仅为PK参考；持有期高点回撤严格超过10%时必须建议清仓100%，实盘仍须用户确认。"
                        "填写entry_setup及position_size_reason，按固定ETF顺序提交对比表。"
                        "完成future_return_and_entry_reviewed后仍须等待用户明确确认，禁止擅自发布或修改HTML。"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except MonthlyPolicyError as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
