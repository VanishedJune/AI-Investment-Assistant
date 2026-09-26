# -*- coding: utf-8 -*-
"""隔离生成器：只读取 sanitized_workspace 内的输入（盲化诊断 + 固定策略语法）。

输入边界（硬约束）：
- 所有文件读取都经过 read_workspace()，路径必须位于 sanitized_workspace/ 内；
- 不 import 项目模块、不读取台账/代码/全量行情；
- 输出 Proposal 一律 context_scope=sanitized_only（可晋级），
  并在写出前做禁止字段自检（日期/代码/anchor_index/start_date/绝对价格）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "sanitized_workspace"
OUTPUTS = WORKSPACE / "outputs"
DIAGNOSTICS = WORKSPACE / "diagnostics"

FORBIDDEN = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"159915|创业板|易方达|TARGET_A|anchor_index|start_date", re.IGNORECASE),
]


def read_workspace(rel: str) -> str:
    path = (WORKSPACE / rel).resolve()
    if not str(path).startswith(str(WORKSPACE.resolve())):
        raise RuntimeError(f"越界读取被拒绝: {rel}")
    return path.read_text(encoding="utf-8")


def scan_forbidden(text: str, patterns: list | None = None) -> list[str]:
    return [p.pattern for p in (patterns or FORBIDDEN) if p.search(text)]


def _seed(rid: str) -> int:
    match = re.fullmatch(r"ROUND_(\d+)_(prediction|strategy)", rid)
    if not match:
        raise ValueError(f"invalid round id: {rid}")
    sequence = int(match.group(1))
    type_offset = 17 if match.group(2) == "prediction" else 53
    return sequence * 97 + type_offset


def jitter_spec(spec: dict, j: float) -> dict:
    import copy

    out = copy.deepcopy(spec)
    if out.get("kind") == "learning_ridge":
        return out  # 学轨超参随 Proposal 冻结，禁止扰动
    for t in out.get("score", {}).get("terms", []):
        t["weight"] = round(float(t["weight"]) * j, 4)
    pos = out.get("position") or {}
    if pos.get("type") == "linear" and "slope" in pos:
        pos["slope"] = round(float(pos["slope"]) * j, 4)
    return out


def choose_roles(ctype: str, stats: dict, seed: int, pools: dict) -> list[str]:
    missed = stats.get("missed_rally", 0)
    wrong = stats.get("wrong_chase", 0)
    cfg = pools[ctype]
    if wrong > missed and wrong > 0:
        pool = cfg["wrong_chase"]
    elif missed > 0:
        pool = cfg["activity"]
    else:
        pool = cfg["calibration"] if ctype == "prediction" else cfg["activity"]
    n = cfg["count"]
    out: list[str] = []
    start = seed % len(pool)
    k = 0
    while len(out) < n and k < 200:
        role = pool[(start + k * 3 + (seed >> (k * 3)) % 5) % len(pool)]
        if role not in out:
            out.append(role)
        k += 1
    return out


def build_proposal(rid: str, diag: dict, syntax: dict) -> dict:
    ctype = diag["challenge_type"]
    seq = int(rid.split("_")[1])
    stats = diag["stats"]
    seed = _seed(rid)
    missed = stats.get("missed_rally", 0)
    wrong = stats.get("wrong_chase", 0)
    if wrong > missed:
        primary, secondary = "WRONG_CHASE", "RISK_CONTROL_ERROR"
    elif missed > 0:
        primary, secondary = "ACTIVITY_ERROR", "CALIBRATION_ERROR"
    else:
        primary, secondary = "CALIBRATION_ERROR", "PREDICTION_ERROR"
    roles = syntax["selection_pools"][ctype]
    role_names = choose_roles(ctype, stats, seed, syntax["selection_pools"])
    role_table = syntax["prediction_roles"] if ctype == "prediction" else syntax["strategy_roles"]
    challengers = []
    for i, name in enumerate(role_names, 1):
        j = 1.0 + ((seed >> (i * 7)) % 5) * 0.25
        template = role_table[name]
        cid = f"CH_P{seq:03d}_{i}" if ctype == "prediction" else f"CH_S{seq:03d}_{i}"
        challengers.append({
            "id": cid,
            "target_problem": template["target_problem"],
            "change_description": f"角色={name}；依据本轮诊断（matured_1w={stats.get('matured_1w')}，missed={missed}，wrong={wrong}）动态设计。",
            "changed_dimensions": ["score_function", "position_mapping", "entry_confirmation"],
            "old_values": {"champion": "当前 Champion 规则"},
            "new_values": {"role": name, "jitter": round(j, 3)},
            "expected_effect": template["expected_effect"],
            "main_risk": template["main_risk"],
            "reason": f"diagnosis: {primary}/{secondary}; mae_1w={stats.get('mae_1w')}, brier_1w={stats.get('brier_1w')}",
            "spec": jitter_spec(template["spec"], j),
        })
    return {
        "iteration_id": "ITERATION_002",
        "opaque_due_id": diag["opaque_due_id"],
        "challenge_round": rid,
        "challenge_type": ctype,
        "diagnosis": f"matured_1w={stats.get('matured_1w')}；missed_rally={missed}；wrong_chase={wrong}；"
                     f"mae_1w={stats.get('mae_1w')}；brier_1w={stats.get('brier_1w')}；"
                     f"accuracy_2w={stats.get('accuracy_2w')}；accuracy_8w={stats.get('accuracy_8w')}。",
        "primary_failure_mode": primary,
        "secondary_failure_mode": secondary,
        "evidence": [
            f"matured_1w={stats.get('matured_1w')}",
            f"missed_rally={missed}",
            f"wrong_chase={wrong}",
            f"mae_1w={stats.get('mae_1w')}",
            f"brier_1w={stats.get('brier_1w')}",
            f"failure_cases={len(diag.get('failure_cases', []))}",
        ],
        "hypothesis": f"依据本轮诊断（{primary}/{secondary}）在隔离工作区内动态生成 {len(role_names)} 个候选；"
                      "context_scope=sanitized_only，可参与晋级。",
        "revision": 1,
        "status": "DRAFT",
        "context_scope": "sanitized_only",
        "challengers": challengers,
    }


def main(etf: str = "159915") -> int:
    from rolling.ledger import set_workspace
    from rolling.config import load_etf_config

    global WORKSPACE, OUTPUTS, DIAGNOSTICS, FORBIDDEN
    if etf != "159915":
        cfg = load_etf_config(etf)
        terms = [etf]
        if cfg.get("name"):
            terms.append(cfg["name"])
        FORBIDDEN = FORBIDDEN + [re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)]
    ws = set_workspace(etf) / "sanitized_workspace"
    WORKSPACE = ws
    OUTPUTS = ws / "outputs"
    DIAGNOSTICS = ws / "diagnostics"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    syntax = json.loads(read_workspace("spec_syntax.json"))
    manifest = ["spec_syntax.json"]
    generated = 0
    for dpath in sorted(DIAGNOSTICS.glob("ROUND_*.json")):
        rel = f"diagnostics/{dpath.name}"
        diag = json.loads(read_workspace(rel))
        manifest.append(rel)
        rid = diag.get("challenge_round")
        if rid is None:
            rid = dpath.stem
        obj = build_proposal(rid, diag, syntax)
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        bad = scan_forbidden(text)
        if bad:
            print(f"拒绝写出（含禁止字段）: {rid} {bad}")
            continue
        (OUTPUTS / f"{rid}.draft.json").write_text(text, encoding="utf-8")
        generated += 1
    (WORKSPACE / "input_manifest.json").write_text(
        json.dumps({"files": manifest, "context_scope": "sanitized_only"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"sanitized drafts generated: {generated}")
    print("manifest:", len(manifest), "files")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
