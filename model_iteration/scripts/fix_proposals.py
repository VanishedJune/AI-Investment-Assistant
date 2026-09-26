# -*- coding: utf-8 -*-
"""precheck->修正循环（顺序登记版）：
按 Due 顺序模拟 create_screening，对 divergence 不达标（vs Champion / 轮内两两 / 已注册候选）的
challenger 做确定性小幅变异（权重缩放、加 d_dif2/rsi14/w_dif2 项、交换 gate 特征），直到全部通过。
"""

from __future__ import annotations

import copy
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rolling import challenge as challenge_mod  # noqa: E402
from rolling import ledger as ledger_mod  # noqa: E402
from rolling.ca import load_ca  # noqa: E402
from rolling.config import load_etf_config  # noqa: E402
from rolling.data import load_aligned_data  # noqa: E402
from rolling.ledger import set_workspace  # noqa: E402
from rolling.spec import validate_spec  # noqa: E402

THRESHOLD = 0.05


def mutation_candidates(spec: dict) -> list[dict]:
    out = []
    terms = (spec.get("score") or {}).get("terms", [])
    if not terms:
        return out
    for f in (1.5, 2.2, 3.2, 4.6, 6.5, 9.0, 13.0):
        s = copy.deepcopy(spec)
        for t in s["score"]["terms"]:
            t["weight"] = round(float(t["weight"]) * f, 4)
        out.append(s)
    for feature, w in (("d_dif2", 200.0), ("d_dif2", 600.0), ("d_dif2", 1500.0),
                       ("rsi14", 0.3), ("rsi14", 0.8), ("w_dif2", 800.0),
                       ("mom5", -50.0), ("vol_ratio", 3.0)):
        s = copy.deepcopy(spec)
        if all(t["feature"] != feature for t in s["score"]["terms"]):
            s["score"]["terms"].append({"feature": feature, "weight": w})
        else:
            for t in s["score"]["terms"]:
                if t["feature"] == feature:
                    t["weight"] = round(float(t["weight"]) + w, 4)
        out.append(s)
    swaps = {"d_dif1": "w_dif1", "w_dif1": "d_dif1", "d_dif2": "d_dif1", "mom20": "d_dif1",
             "rsi14": "d_dif1", "w_dif2": "w_dif1"}
    gates = spec.get("gates") or []
    for i, g in enumerate(gates):
        if g.get("feature") in swaps:
            s = copy.deepcopy(spec)
            s["gates"][i]["feature"] = swaps[g["feature"]]
            out.append(s)
    return out


def main(etf: str = "159915") -> int:
    set_workspace(etf)
    ca = load_ca(etf)
    daily_a, weekly_a, daily, anchors, usable, features = load_aligned_data(etf)
    fwd_arrays = {k: usable[f"fwd{k}"].to_numpy(dtype=float) for k in (1, 2, 4, 8)}
    champion_cfg = load_etf_config(etf).get("champion") or {}
    champion_spec = champion_cfg.get("spec") or {"kind": "champion_zero_axis"}
    champion_id = champion_cfg.get("id") or "champion_zero_axis"
    path_cache: dict = {}
    max_idx = len(features) - 1

    def cached_path(spec: dict, due_idx: int) -> np.ndarray:
        """全长路径只算一次，按 due_idx 切片（前缀与 end_idx 无关）。"""
        h = challenge_mod.spec_hash(spec)
        if h not in path_cache:
            path_cache[h] = challenge_mod.position_path(spec, features, max_idx, fwd_arrays=fwd_arrays)
        return path_cache[h][: due_idx + 1]

    drafts = sorted(ledger_mod.DRAFT_DIR.glob("*.draft.json"))
    # 断点续跑：每轮落盘 checkpoint，进程被杀后从断点恢复，不重算已登记候选
    cp_path = ledger_mod.DRAFT_DIR / ".fix_checkpoint.json"
    processed: set[str] = set()
    failed: set[str] = set()
    registered: list[dict] = []
    if cp_path.exists():
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        processed = set(cp.get("processed", []))
        failed = set(cp.get("failed", []))
        registered = cp.get("registered", [])
        print(f"resume: processed={len(processed)} failed={len(failed)} registered={len(registered)}")

    def save_checkpoint() -> None:
        cp_path.write_text(
            json.dumps(
                {"processed": sorted(processed), "failed": sorted(failed), "registered": registered},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    total_fixed = 0
    for path in drafts:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rid = obj["challenge_round"]
        if rid in processed:
            continue
        ctype = obj["challenge_type"]
        seq = int(rid.split("_")[1])
        due_idx = seq * 4 if ctype == "prediction" else seq * 8
        champ_vec = challenge_mod.forecast_vector(features.iloc[due_idx], champion_spec)
        champ_path = cached_path(champion_spec, due_idx)
        existing_vectors = [
            challenge_mod.forecast_vector(features.iloc[due_idx], ex.get("spec") or {})
            for ex in registered
        ]
        existing_paths = [
            cached_path(ex.get("spec") or {}, due_idx)
            for ex in registered
        ]
        round_fixed = 0
        for ch in obj["challengers"]:
            spec = ch.get("spec") or {}
            if validate_spec(spec):
                continue
            if ctype == "prediction":
                vec = challenge_mod.forecast_vector(features.iloc[due_idx], spec)
                d_champ = challenge_mod.divergence_prediction(vec, champ_vec)
                d_others = [
                    challenge_mod.divergence_prediction(vec, challenge_mod.forecast_vector(features.iloc[due_idx], o.get("spec") or {}))
                    for o in obj["challengers"] if o["id"] != ch["id"]
                ]
                d_existing = [challenge_mod.divergence_prediction(vec, v) for v in existing_vectors]
            else:
                path_arr = cached_path(spec, due_idx)
                d_champ = challenge_mod.divergence_position(path_arr, champ_path)
                d_others = [
                    challenge_mod.divergence_position(path_arr, cached_path(o.get("spec") or {}, due_idx))
                    for o in obj["challengers"] if o["id"] != ch["id"]
                ]
                d_existing = [challenge_mod.divergence_position(path_arr, p) for p in existing_paths]
            if d_champ >= THRESHOLD and all(d >= THRESHOLD for d in d_others + d_existing):
                continue
            fixed = None
            for cand in mutation_candidates(spec):
                if validate_spec(cand):
                    continue
                if ctype == "prediction":
                    v = challenge_mod.forecast_vector(features.iloc[due_idx], cand)
                    dc = challenge_mod.divergence_prediction(v, champ_vec)
                    do = [
                        challenge_mod.divergence_prediction(v, challenge_mod.forecast_vector(features.iloc[due_idx], o.get("spec") or {}))
                        for o in obj["challengers"] if o["id"] != ch["id"]
                    ]
                    de = [challenge_mod.divergence_prediction(v, x) for x in existing_vectors]
                else:
                    p = challenge_mod.position_path(cand, features, due_idx, fwd_arrays=fwd_arrays)
                    dc = challenge_mod.divergence_position(p, champ_path)
                    do = [
                        challenge_mod.divergence_position(p, cached_path(o.get("spec") or {}, due_idx))
                        for o in obj["challengers"] if o["id"] != ch["id"]
                    ]
                    de = [challenge_mod.divergence_position(p, x) for x in existing_paths]
                if dc >= THRESHOLD and all(d >= THRESHOLD for d in do + de):
                    fixed = cand
                    break
            if fixed is None:
                print(f"{rid} {ch['id']}: 未能自动修正 (d_champ={d_champ:.4f}, min_others={min(d_others + d_existing):.4f})")
                continue
            ch["spec"] = fixed
            ch["change_description"] = ch.get("change_description", "") + "（precheck 修正：小幅调整权重/门以通过 divergence 筛查）"
            round_fixed += 1
            total_fixed += 1
        if round_fixed:
            obj["revision"] = int(obj.get("revision", 1)) + 1
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{rid}: 修正 {round_fixed} 个")
        # 登记（含修正后的候选）
        try:
            reg = challenge_mod.create_screening(
                obj, due_idx, features, usable, daily, ca,
                champion_spec, ctype,
                parent_champion_id=champion_id,
                existing_challengers=registered,
                path_cache=path_cache,
            )
            registered.extend(reg)
            processed.add(rid)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.add(rid)
            print(f"{rid}: 本轮处理失败，已跳过（登记 0 个）")
        save_checkpoint()
    if cp_path.exists():
        cp_path.unlink()
    print(f"failed rounds: {sorted(failed)}")
    print(f"total fixed: {total_fixed}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.etf))
