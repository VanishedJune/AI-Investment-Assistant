"""按生产口径（每周加仓 + 周度看空保护）对 M0207 搜索候选重新排序并部署。"""

from __future__ import annotations

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
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    robust_score,
    simulate_realistic,
)

BASE_ORDER = ["159915", "518600", "515220", "516150", "159622", "159941", "512800", "512690"]
PROD_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
               "ladder": 0, "max_single": 100, "rebalance": 4}
CONFIG_PATH = ROOT / "configs" / "investment_priority.json"


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

    def run_deploy(order, params):
        rec = {}
        for w in ("Y1", "Y2", "Y3", "Y5"):
            rec[w] = simulate_realistic(
                etfs, master_dates, master_execs, *idx[w], order, params,
                weekly_deploy=True, bearish_guard=True,
            )
        rec["score"] = robust_score(rec)
        return rec

    base_deploy = run_deploy(BASE_ORDER, PROD_PARAMS)
    baseline_5y = base_deploy["Y5"]["cum_return"]
    print("baseline deploy:", {w: round(base_deploy[w]["cum_return"], 4) for w in ("Y1", "Y2", "Y3", "Y5")},
          "score", round(base_deploy["score"], 4))

    payload = json.loads((OUT_DIR / "allocation_optimization_m0207.json").read_text(encoding="utf-8"))
    models = payload["all_models"]
    pool = sorted(models, key=lambda r: r["score"], reverse=True)[:300]
    ranked = []
    for r in pool:
        d = run_deploy(r["order"], r["params"])
        ranked.append({
            "id": r["id"], "order": r["order"], "params": r["params"],
            "research_score": r["score"],
            "deploy": {w: d[w] for w in ("Y1", "Y2", "Y3", "Y5")},
            "deploy_score": d["score"],
        })
    ranked.sort(key=lambda r: r["deploy_score"], reverse=True)

    candidates = [r for r in ranked if r["deploy"]["Y5"]["cum_return"] >= baseline_5y
                  and sum(1 for w in ("Y1", "Y2", "Y3") if r["deploy"][w]["cum_return"] > 0) >= 2]
    champ = max(candidates or ranked, key=lambda r: r["deploy_score"])
    improved = champ["deploy_score"] > base_deploy["score"]

    # 部署到生产配置
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["priority"] = champ["order"]
    cfg["core_budget_pct"] = champ["params"]["core_budget"]
    cfg["remainder_budget_pct"] = champ["params"]["remainder_budget"]
    cfg["max_single_pct"] = champ["params"]["max_single"]
    cfg["core_ladder_pct"] = {"positions": deploy_ladder(champ["params"])}
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# M0207 生产口径排序结果（每周加仓 + 周度看空保护）",
        "",
        f"基线（B0207 新顺序）：5年 {base_deploy['Y5']['cum_return']:.2%} 综合分 {base_deploy['score']:.4f}",
        "",
        f"## 生产口径冠军：{champ['id']}（{'改进' if improved else '未改进，维持基线优先' if champ['id'] == 'B0119' else ''}）",
        f"- 顺序：{' → '.join(champ['order'])}",
        f"- 参数：core={champ['params']['core_budget']} rem={champ['params']['remainder_budget']} "
        f"ladder={champ['params']['ladder']} max={champ['params']['max_single']}",
        f"- 生产口径 Y1/Y2/Y3/5年：{champ['deploy']['Y1']['cum_return']:.2%} / "
        f"{champ['deploy']['Y2']['cum_return']:.2%} / {champ['deploy']['Y3']['cum_return']:.2%} / "
        f"{champ['deploy']['Y5']['cum_return']:.2%}",
        f"- 生产综合分：{champ['deploy_score']:.4f}（基线 {base_deploy['score']:.4f}）",
        "",
        "## 生产口径 Top 10",
        "",
        "| 模型 | 顺序 | core/rem | ladder | max | Y1 | Y2 | Y3 | 5年 | 综合分 | 门槛 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked[:10]:
        p = r["params"]
        passed = r["deploy"]["Y5"]["cum_return"] >= baseline_5y and sum(
            1 for w in ("Y1", "Y2", "Y3") if r["deploy"][w]["cum_return"] > 0) >= 2
        lines.append(
            f"| {r['id']} | {'/'.join(r['order'])} | {p['core_budget']}/{p['remainder_budget']} | "
            f"{p['ladder']} | {p['max_single']} | {r['deploy']['Y1']['cum_return']:.1%} | "
            f"{r['deploy']['Y2']['cum_return']:.1%} | {r['deploy']['Y3']['cum_return']:.1%} | "
            f"{r['deploy']['Y5']['cum_return']:.1%} | {r['deploy_score']:.3f} | {'是' if passed else '否'} |"
        )
    (OUT_DIR / "allocation_optimization_m0207_deploy.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "baseline_deploy": {w: base_deploy[w] for w in ("Y1", "Y2", "Y3", "Y5")},
        "baseline_score": base_deploy["score"],
        "champion": {"id": champ["id"], "order": champ["order"], "params": champ["params"],
                     "deploy": champ["deploy"], "deploy_score": champ["deploy_score"]},
        "improved": improved,
    }
    (OUT_DIR / "allocation_optimization_m0207_deploy.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("deploy champion:", champ["id"], "score", round(champ["deploy_score"], 4),
          "order", "/".join(champ["order"]))
    print("baseline score:", round(base_deploy["score"], 4), "improved:", improved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
