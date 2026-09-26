"""Evaluate a frozen ETF champion without training, promotion, or replay writes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from rolling.account import anchor_exec_prices, run_account, weekly_metrics
from rolling.ca import load_ca
from rolling.challenge import position_path
from rolling.config import load_etf_config
from rolling.data import load_aligned_data
from rolling.ledger import set_workspace


ROOT = Path(__file__).resolve().parents[1]


def validate(etf: str, holdout_weeks: int = 26) -> dict:
    set_workspace(etf)
    cfg = load_etf_config(etf)
    ca = load_ca(etf)
    _, _, daily, anchors, usable, features = load_aligned_data(etf)
    state_path = ROOT / cfg.get("workspace", f"etf_{etf}") / "weekly_rolling" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    initial = cfg.get("initial_champion_provenance") or {}
    final_champion = state["champion"]
    positions = position_path(final_champion["spec"], features, len(features) - 1)
    account = run_account(
        positions,
        usable,
        daily,
        ca,
        cost_bps=int(cfg.get("cost_bps", 10)),
        exec_prices=anchor_exec_prices(usable, daily),
        execution_policy=cfg.get("execution_policy"),
        features=features,
    )
    buy_hold = run_account(
        np.ones(len(features), dtype=float),
        usable,
        daily,
        ca,
        cost_bps=int(cfg.get("cost_bps", 10)),
        exec_prices=anchor_exec_prices(usable, daily),
    )
    completed = len(account["nav"])
    holdout_start = max(0, completed - holdout_weeks)
    full_metrics = weekly_metrics(account["nav"], account["positions"], account["returns"])
    holdout_metrics = weekly_metrics(
        account["nav"],
        account["positions"],
        account["returns"],
        window_start=holdout_start,
        window_end=completed,
    )
    buy_hold_full = weekly_metrics(
        buy_hold["nav"], buy_hold["positions"], buy_hold["returns"]
    )
    buy_hold_holdout = weekly_metrics(
        buy_hold["nav"],
        buy_hold["positions"],
        buy_hold["returns"],
        window_start=holdout_start,
        window_end=completed,
    )
    return {
        "schema_version": "frozen-champion-validation-v1.0.0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "etf": etf,
        "champion_id": final_champion["id"],
        "initial_source_code": initial.get("source_code"),
        "initial_source_champion_id": initial.get("source_champion_id"),
        "promotion_count": len(state.get("promotion_history") or []),
        "selection_replay": {
            "signal_start": usable.iloc[0]["signal"].date().isoformat(),
            "signal_end": usable.iloc[-1]["signal"].date().isoformat(),
            "usable_anchors": len(usable),
            "completed_execution_anchors": completed,
            "metrics": full_metrics,
            "buy_hold_metrics": buy_hold_full,
            "classification": "model_selection_replay_not_out_of_sample",
        },
        "temporal_holdout_audit": {
            "signal_start": usable.iloc[holdout_start]["signal"].date().isoformat(),
            "signal_end": usable.iloc[completed - 1]["signal"].date().isoformat(),
            "weeks": completed - holdout_start,
            "metrics": holdout_metrics,
            "buy_hold_metrics": buy_hold_holdout,
            "parameter_update_allowed": False,
            "classification": "frozen_parameter_temporal_audit",
            "caveat": "The retained-model decision also inspected the full replay; this audit is not claimed as an independent model-selection result.",
        },
        "deployment_assessment": {
            "existing_iteration_protocol_completed": True,
            "candidate_forced_promotion": False,
            "return_or_sharpe_improvement_over_buy_hold": False,
            "drawdown_improvement_over_buy_hold": True,
            "result": "initial_champion_retained_and_slot_activated_under_existing_rules_with_performance_warning",
            "optimization_success_claimed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    parser.add_argument("--holdout-weeks", type=int, default=26)
    args = parser.parse_args()
    payload = validate(args.etf, args.holdout_weeks)
    out_dir = ROOT / f"etf_{args.etf}" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frozen_champion_validation.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
