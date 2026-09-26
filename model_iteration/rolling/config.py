# -*- coding: utf-8 -*-
"""ETF 通用配置：所有硬编码（覆盖区间、期望锚点数、成本、champion）迁入 configs/etf_<code>.json。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

DEFAULTS = {
    "coverage": {"start": None, "end": None},
    "expected_anchors": None,
    "horizon": 8,
    "cost_bps": 10,
    "champion": {"kind": "champion_zero_axis"},
    "promotion_policy_version": "promotion-v2.0.0",
    "promotion_policy": {
        "benchmark_hard_gate": True,
        "rebase_min_sharpe": 0.7,
        "min_oos_weeks": 26,
        "max_promotions_per_52w": 2,
        "long_horizon_mdd_hard": True,
        "max_rebase_generations": 2,
        "full_period_hard_gate": False,
        "full_period_sharpe_tol": 0.05,
        "full_period_mdd_tol": 0.03,
        "drawdown_penalized": True,
        "weekly_primary_hard_gate": False,
    },
    "execution_policy": {
        "mode": "b0207_cycle",
        "decision_cycle_weeks": 4,
        "signal_smooth_days": None,
        "non_decision_graded": {
            "enabled": False,
            "feature": "d_dif1",
            "add_cash_frac": 0.5,
            "sell_holdings_frac": 0.5,
        },
        "bearish_guard": {
            "enabled": True,
            "sell_frac": 0.7,
            "per_episode": True,
            "reset_on_recovery": True,
            "conditions": [
                {"feature": "d_dif1", "op": "<", "value": 0.0},
                {"feature": "d_dif2", "op": "<", "value": 0.0},
                {"feature": "w_dif1", "op": "<", "value": 0.0},
                {"feature": "w_dif2", "op": "<", "value": 0.0},
                {"feature": "mom20", "op": "<", "value": -0.05},
            ],
        },
    },
}


def load_etf_config(etf: str) -> dict:
    path = CONFIG_DIR / f"etf_{etf}.json"
    if not path.exists():
        return {**DEFAULTS, "etf": etf, "config_source": "defaults"}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    merged = {**DEFAULTS, **cfg}
    merged["promotion_policy"] = {
        **DEFAULTS["promotion_policy"],
        **(cfg.get("promotion_policy") or {}),
    }
    # An explicit null preserves the original champion-model iteration path:
    # every weekly model target is executed directly.  Missing configuration
    # continues to receive the newer B0207 execution policy default.
    if "execution_policy" in cfg and cfg["execution_policy"] is None:
        merged["execution_policy"] = None
    else:
        merged["execution_policy"] = {
            **DEFAULTS["execution_policy"],
            **(cfg.get("execution_policy") or {}),
        }
        merged["execution_policy"]["bearish_guard"] = {
            **DEFAULTS["execution_policy"]["bearish_guard"],
            **((cfg.get("execution_policy") or {}).get("bearish_guard") or {}),
        }
    merged["config_source"] = str(path)
    return merged


def integrity_digests_enabled(etf: str) -> bool:
    """Whether this ETF's training ledger may calculate digest fingerprints.

    Existing ledgers retain their historical behavior.  New workspaces can opt
    out with ``integrity_mode=not_evaluated`` and use event/model identifiers,
    paths, dates and structural validation for traceability.
    """
    return load_etf_config(etf).get("integrity_mode", "digest_verified") == "digest_verified"
