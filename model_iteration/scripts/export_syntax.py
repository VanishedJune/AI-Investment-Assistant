# -*- coding: utf-8 -*-
"""把固定策略语法导出为隔离工作区数据文件（sanitized_workspace/spec_syntax.json）。

该文件是 sanitized_only 生成器的唯一设计输入：角色模板、特征白名单、
选择池、divergence 阈值、Proposal schema。生成器不得读取项目代码/台账/全量数据。
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling.features import ALL_SPEC_FEATURES  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from scripts.proposal_factory import prediction_role, strategy_role  # noqa: E402


def _serialize_role(fn, name: str) -> dict:
    target, spec, effect, risk = fn(name, 1.0)
    return {"target_problem": target, "spec": spec, "expected_effect": effect, "main_risk": risk}


PREDICTION_ROLES = [
    "trend_fusion", "recovery", "mom_only", "d_trend", "momentum_vol", "weekly_speed",
    "accel", "rsi_mom", "dual_dif_lowconf", "vol_target", "weekly_only",
    "exit_mom", "weekly_protect", "dd_stop", "rsi_filter",
    "mean_reversion_rsi", "breakout_mom", "dual_momentum", "volume_divergence", "channel_mean_revert",
    "multi_param_joint", "weekly_trend_daily_timing",
]
STRATEGY_ROLES = [
    "ramp", "drawdown", "floor", "vol_target", "graded", "recovery_step",
    "weekly_ramp", "momentum_dd", "learning_ridge",
    "mean_reversion", "breakout", "dual_momentum",
    "multi_param_defensive",
]

LEARNING_RIDGE_TEMPLATE = {
    "kind": "learning_ridge",
    "alpha": 10.0,
    "horizon": 8,
    "embargo": 8,
    "min_train": 60,
    "standardize": True,
    "features": ["d_slope", "w_slope", "d_dif1", "w_dif1", "mom20", "vol20"],
    "position": {"type": "step", "weight": 1.0},
}

SELECTION_POOLS = {
    "prediction": {
        "wrong_chase": ["exit_mom", "weekly_protect", "dd_stop", "rsi_filter", "trend_fusion",
                        "momentum_vol", "dual_dif_lowconf", "weekly_speed", "breakout_mom",
                        "volume_divergence"],
        "activity": ["trend_fusion", "recovery", "mom_only", "d_trend", "momentum_vol",
                     "weekly_speed", "accel", "rsi_mom", "breakout_mom", "dual_momentum",
                     "multi_param_joint", "weekly_trend_daily_timing"],
        "calibration": ["dual_dif_lowconf", "rsi_mom", "vol_target", "weekly_only", "accel",
                        "weekly_speed", "trend_fusion", "mom_only", "mean_reversion_rsi",
                        "channel_mean_revert", "multi_param_joint", "weekly_trend_daily_timing"],
        "count": 5,
    },
    "strategy": {
        "wrong_chase": ["drawdown", "momentum_dd", "vol_target", "ramp", "graded", "floor",
                        "weekly_ramp", "recovery_step", "learning_ridge", "breakout", "dual_momentum",
                        "multi_param_defensive"],
        "activity": ["floor", "recovery_step", "weekly_ramp", "graded", "ramp", "drawdown",
                     "momentum_dd", "vol_target", "learning_ridge", "breakout", "mean_reversion",
                     "multi_param_defensive"],
        "count": 3,
    },
}

PROPOSAL_FIELDS = [
    "iteration_id", "opaque_due_id", "challenge_round", "challenge_type", "diagnosis",
    "primary_failure_mode", "secondary_failure_mode", "evidence", "hypothesis", "revision",
    "status", "context_scope", "challengers",
]
CHALLENGER_FIELDS = [
    "id", "target_problem", "change_description", "changed_dimensions", "old_values",
    "new_values", "expected_effect", "main_risk", "reason", "spec",
]


def main(etf: str = "159915") -> int:
    ws = set_workspace(etf)
    strategy_roles = {n: _serialize_role(strategy_role, n) for n in STRATEGY_ROLES}
    strategy_roles["learning_ridge"] = {
        "target_problem": "学轨：Ridge 滚动学习（超参冻结，embargo>=horizon，训练集内标准化）",
        "spec": LEARNING_RIDGE_TEMPLATE,
        "expected_effect": "每周用 <=t-8 成熟样本重训，动态跟随市场结构",
        "main_risk": "样本不足或市场结构变化导致退化",
    }
    syntax = {
        "schema_version": "sanitized-syntax-v1",
        "allowed_features": ALL_SPEC_FEATURES,
        "divergence_threshold": 0.05,
        "proposal_fields": PROPOSAL_FIELDS,
        "challenger_fields": CHALLENGER_FIELDS,
        "selection_pools": SELECTION_POOLS,
        "prediction_roles": {n: _serialize_role(prediction_role, n) for n in PREDICTION_ROLES},
        "strategy_roles": strategy_roles,
    }
    out = ws / "sanitized_workspace" / "spec_syntax.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(syntax, ensure_ascii=False, indent=2), encoding="utf-8")
    print("syntax exported:", out)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
