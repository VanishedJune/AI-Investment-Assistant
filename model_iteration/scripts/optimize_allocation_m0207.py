"""8 ETF 协同交易新模型搜索（M0207 同法：v2 稳健评分 + 全枚举顺序 + 真实账户）。

新硬约束：优先级前 3 锁定为 159915（创业板）→ 518600（黄金）→ 515220（煤炭）；
其余 5 支（159622/159941/512690/512800/516150）全枚举 120 种顺序。
阶段 A：120 顺序 × 基础参数；阶段 B：Top 50 顺序 × 每顺序 80 组细化参数。
可部署约束：rebalance=4（28 天）、floor=0、core+rem=100（与 production loader 一致）。
评价：70%×最近年 + 20%×中间年 + 10%×早期年；门槛 = 至少两年为正 且 近五年 ≥ 基线。
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import LADDERS, LADDERS2, LADDERS4, range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    OUT_DIR,
    PHASE_B_PER_ORDER,
    PHASE_B_SEED,
    TOP_ORDERS,
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    passes_gates,
    robust_score,
    run_windows,
    simulate_realistic,
)

TOP3 = ["159915", "518600", "515220"]
REST = [c for c in CODES if c not in TOP3]
BASE_ORDER = ["159915", "518600", "515220", "516150", "159622", "159941", "512800", "512690"]
BASE_PARAMS = {"core_budget": 80, "remainder_budget": 20, "floor": 0.0,
               "ladder": 3, "max_single": 100, "rebalance": 4}
PROD_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
               "ladder": 0, "max_single": 100, "rebalance": 4}


def all_orders_m0207() -> list[list[str]]:
    orders = []
    for rest in itertools.permutations(REST):
        orders.append(TOP3 + list(rest))
    return sorted(orders)


def deploy_ladder(params: dict, core_size: int = 3) -> dict:
    if core_size >= 4:
        return LADDERS4[int(params["ladder"])]
    if core_size <= 2:
        return LADDERS2[int(params["ladder"])]
    return LADDERS[int(params["ladder"])]


def main() -> int:
    etfs = {}
    for code in CODES:
        etfs[code] = load_etf(code)
        print("loaded", code, NAMES[code], "start", etfs[code]["start_date"])

    from rolling.ledger import set_workspace as sw
    from rolling.data import load_aligned_data as lad
    sw("159915")
    _, _, _, _, usable_m, _ = lad("159915")
    master_dates = [d.date().isoformat() for d in usable_m["signal"]]
    master_execs = [
        None if ex is None else str(np.datetime64(ex).astype("datetime64[D]"))
        for ex in usable_m["exec"]
    ]
    idx = {w: range_idx(master_dates, *b) for w, b in {
        "Y1": WINDOW_Y1, "Y2": WINDOW_Y2, "Y3": WINDOW_Y3, "Y5": WINDOW_5Y,
    }.items()}

    # 基线：新顺序 + 生产参数（B0207 结构）
    base = {"model": {"id": "BASE_M0207", "order": BASE_ORDER, "params": dict(PROD_PARAMS)}}
    base.update(run_windows(etfs, master_dates, master_execs, idx, BASE_ORDER, PROD_PARAMS))
    base["score"] = robust_score(base)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline:", {w: round(base[w]["cum_return"], 6) for w in ("Y1", "Y2", "Y3", "Y5")},
          "score", round(base["score"], 4))

    orders = all_orders_m0207()
    print("valid orders:", len(orders))
    phase_a: list[dict] = []
    for oi, order in enumerate(orders):
        r = run_windows(etfs, master_dates, master_execs, idx, order, BASE_PARAMS)
        rec = {"model": {"id": f"A{oi:04d}", "order": order, "params": dict(BASE_PARAMS)}}
        rec.update(r)
        rec["score"] = robust_score(rec)
        phase_a.append(rec)
    top_orders = sorted(
        (r for r in phase_a if passes_gates(r, baseline_5y)),
        key=lambda r: r["score"], reverse=True,
    )[:TOP_ORDERS]
    if not top_orders:
        top_orders = sorted(phase_a, key=lambda r: r["score"], reverse=True)[:TOP_ORDERS]
    print("phase A done; top order score:", round(top_orders[0]["score"], 4) if top_orders else None)

    rng = np.random.default_rng(PHASE_B_SEED)
    core_choices = [70, 75, 80, 85]
    floor_choices = [0.0]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    rebal_choices = [4]
    phase_b: list[dict] = []
    bid = 0
    for tr in top_orders:
        order = tr["model"]["order"]
        for _ in range(PHASE_B_PER_ORDER):
            core = int(rng.choice(core_choices))
            params = {
                "core_budget": core,
                "remainder_budget": 100 - core,
                "floor": float(rng.choice(floor_choices)),
                "ladder": int(rng.choice(ladder_choices)),
                "max_single": int(rng.choice(max_choices)),
                "rebalance": int(rng.choice(rebal_choices)),
            }
            rec = {"model": {"id": f"B{bid:04d}", "order": order, "params": params}}
            rec.update(run_windows(etfs, master_dates, master_execs, idx, order, params))
            rec["score"] = robust_score(rec)
            phase_b.append(rec)
            bid += 1
    print("phase B done:", len(phase_b))

    candidates = [r for r in phase_b if passes_gates(r, baseline_5y)]
    if not candidates:
        candidates = phase_b
    champion = max(candidates, key=lambda r: r["score"])
    in_sample_best = max(phase_b, key=lambda r: r["Y5"]["cum_return"])
    all_models = phase_a + phase_b

    # 生产口径复验：每周加仓 + 周度看空保护
    def run_deploy(order, params):
        rec = {}
        for w in ("Y1", "Y2", "Y3", "Y5"):
            rec[w] = simulate_realistic(
                etfs, master_dates, master_execs, *idx[w], order, params,
                weekly_deploy=True, bearish_guard=True,
            )
        rec["score"] = robust_score(rec)
        return rec

    champion_deploy = run_deploy(champion["model"]["order"], champion["model"]["params"])
    base_deploy = run_deploy(BASE_ORDER, PROD_PARAMS)

    lines = [
        "# 8 ETF 协同交易新模型搜索（M0207 同法；前三固定：创业板→黄金→煤炭）",
        "",
        "硬约束：159915 第一、518600 第二、515220 第三；其余 5 支全枚举 120 顺序。",
        "可部署约束：rebalance=4 周、floor=0、core+rem=100。",
        "评价：综合评分 70/20/10（近/中/早年度）；门槛=至少两年为正且近5年≥基线。",
        "",
        f"## 基线（新顺序 + B0207 生产参数）",
        f"- 顺序：{' → '.join(BASE_ORDER)}",
        f"- 近5年：{base['Y5']['cum_return']:.2%} 综合分 {base['score']:.4f}",
        "",
        f"## 新模型冠军：{champion['model']['id']}",
        "",
        f"- 顺序：{' → '.join(champion['model']['order'])}",
        f"- 参数：core={champion['model']['params']['core_budget']}% rem="
        f"{champion['model']['params']['remainder_budget']}% ladder="
        f"{champion['model']['params']['ladder']} max="
        f"{champion['model']['params']['max_single']}% rebalance="
        f"{champion['model']['params']['rebalance']}周",
        f"- Y1/Y2/Y3/近5年：{champion['Y1']['cum_return']:.2%} / {champion['Y2']['cum_return']:.2%} / "
        f"{champion['Y3']['cum_return']:.2%} / {champion['Y5']['cum_return']:.2%}",
        f"- 综合分：{champion['score']:.4f}（基线 {base['score']:.4f}）",
        "",
        "## 生产口径复验（每周加仓 + 周度看空保护）",
        "",
        "| 方案 | Y1 | Y2 | Y3 | 近5年 | 综合分 | 5年IRR | Sharpe | MDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in (("新模型", champion_deploy), ("基线 B0207(新顺序)", base_deploy)):
        y5 = r["Y5"]
        lines.append(
            f"| {label} | {r['Y1']['cum_return']:.1%} | {r['Y2']['cum_return']:.1%} | "
            f"{r['Y3']['cum_return']:.1%} | {y5['cum_return']:.1%} | {r['score']:.3f} | "
            f"{y5['annual_return_irr']:.1%} | {y5['sharpe']:.2f} | {y5['max_drawdown']:.1%} |"
        )
    lines += ["", "## Top 10（按综合评分）", "",
              "| 模型 | 顺序 | core/rem | ladder | max | Y1 | Y2 | Y3 | 近5年 | 综合分 | 门槛 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    top10 = sorted(all_models, key=lambda r: r["score"], reverse=True)[:10]
    for r in top10:
        p = r["model"]["params"]
        passed = passes_gates(r, baseline_5y)
        lines.append(
            f"| {r['model']['id']} | {'/'.join(r['model']['order'])} | {p['core_budget']}/{p['remainder_budget']} | "
            f"{p['ladder']} | {p['max_single']} | {r['Y1']['cum_return']:.1%} | {r['Y2']['cum_return']:.1%} | "
            f"{r['Y3']['cum_return']:.1%} | {r['Y5']['cum_return']:.1%} | {r['score']:.3f} | {'是' if passed else '否'} |"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "allocation_optimization_m0207.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": "allocation-optimization-m0207",
        "constraints": {"top3": TOP3, "orders": len(orders), "deployable": True},
        "baseline": {"order": BASE_ORDER, "params": PROD_PARAMS,
                     **{w: base[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": base["score"],
                     "deploy": {w: base_deploy[w] for w in ("Y1", "Y2", "Y3", "Y5")}},
        "champion": {"id": champion["model"]["id"], "order": champion["model"]["order"],
                     "params": champion["model"]["params"],
                     **{w: champion[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": champion["score"],
                     "deploy": {w: champion_deploy[w] for w in ("Y1", "Y2", "Y3", "Y5")}},
        "in_sample_best": {"id": in_sample_best["model"]["id"], "order": in_sample_best["model"]["order"],
                           "params": in_sample_best["model"]["params"],
                           **{w: in_sample_best[w] for w in ("Y1", "Y2", "Y3", "Y5")},
                           "score": in_sample_best["score"]},
        "top10": [{"id": r["model"]["id"], "order": r["model"]["order"], "params": r["model"]["params"],
                   **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]} for r in top10],
        "all_models": [{"id": r["model"]["id"], "order": r["model"]["order"], "params": r["model"]["params"],
                        **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5")}, "score": r["score"]} for r in all_models],
    }
    (OUT_DIR / "allocation_optimization_m0207.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", OUT_DIR / "allocation_optimization_m0207.md")
    print("champion:", champion["model"]["id"], "score", round(champion["score"], 4),
          "order", "/".join(champion["model"]["order"]))
    print("baseline score:", round(base["score"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
