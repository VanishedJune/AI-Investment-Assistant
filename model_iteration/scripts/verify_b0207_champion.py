# -*- coding: utf-8 -*-
"""B0207 真冠军检验：更大模型池 + 更严格标准。

检验目标：B0207（核心3，85/15，原版无周度保护）在更大随机搜索下是否仍然
是真正的 Top1，而不是 5440 个模型里的幸运儿。

方法：
1. 模型池：8000 个随机 (顺序, 参数) 模型（与 B0207 同一搜索空间：
   159915 前二、159622 前四、516150 前四；核心3；2/3/4/6周再平衡；
   真实账户、无 weekly_deploy）+ 强制纳入 B0207 与基线；
2. 标准一（原有）：稳健年度评分 70/20/10 + 门槛（至少两年为正、近5年≥基线）；
3. 标准二（严格切分）：选择窗 2021-08-10~2024-08-09，验证窗 2024-08-10~
   2026-08-07；按选择窗收益取 Top100，再看验证窗表现；
4. 标准三（多指标聚合）：综合评分、近5年、IRR、Sharpe、MDD 五个指标
   的百分位均值。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.investment_strategy_vs_gem_dca import CODES, NAMES, load_etf  # noqa: E402
from scripts.optimize_allocation import range_idx  # noqa: E402
from scripts.optimize_allocation_v2 import (  # noqa: E402
    CURRENT_ORDER,
    DEFAULT_PARAMS,
    OUT_DIR,
    WINDOW_5Y,
    WINDOW_Y1,
    WINDOW_Y2,
    WINDOW_Y3,
    passes_gates,
    robust_score,
    simulate_realistic,
)

POOL_SIZE = 8000
SEED = 20260811
B0207_ORDER = ["159915", "518600", "516150", "159622", "159941", "512010", "512800", "512690"]
B0207_PARAMS = {"core_budget": 85, "remainder_budget": 15, "floor": 0.0,
                "ladder": 0, "max_single": 100, "rebalance": 4, "core_size": 3}
WINDOW_SEL = ("2021-08-10", "2024-08-09")
WINDOW_VAL = ("2024-08-10", "2026-08-07")
OUT = ROOT.parent / "champion_vs_dca"


def pct(x) -> str:
    return f"{float(x):.1%}"


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
        "SEL": WINDOW_SEL, "VAL": WINDOW_VAL,
    }.items()}

    def run(order, params) -> dict:
        rec = {w: simulate_realistic(etfs, master_dates, master_execs, *idx[w], order, params,
                                     weekly_deploy=False)
               for w in ("Y1", "Y2", "Y3", "Y5", "SEL", "VAL")}
        rec["score"] = robust_score(rec)
        return rec

    base = run(CURRENT_ORDER, DEFAULT_PARAMS)
    baseline_5y = base["Y5"]["cum_return"]
    print("baseline 5y:", round(baseline_5y, 4))

    rng = np.random.default_rng(SEED)
    orders_all = []
    fixed = {"159915", "159622", "516150"}
    rest = [c for c in CODES if c not in fixed]
    import itertools
    for pos915 in (0, 1):
        slots = [p for p in range(4) if p != pos915]
        for s1, s2 in itertools.permutations(slots, 2):
            base_list: list[str | None] = [None] * 8
            base_list[pos915] = "159915"
            base_list[s1] = "159622"
            base_list[s2] = "516150"
            free = [i for i in range(8) if base_list[i] is None]
            for p in itertools.permutations(rest):
                order = list(base_list)
                for idx_c, code in zip(free, p):
                    order[idx_c] = code
                orders_all.append(order)
    print("valid orders:", len(orders_all))

    core_choices = [70, 75, 80, 85]
    rem_choices = [15, 20, 25, 30]
    floor_choices = [0.0, 0.1, 0.2, 0.3, 0.4]
    ladder_choices = [0, 1, 2, 3]
    max_choices = [60, 70, 80, 100]
    rebal_choices = [2, 3, 4, 6]

    pool = []
    for k in range(POOL_SIZE):
        core = int(rng.choice(core_choices))
        rem_options = [r for r in rem_choices if r <= 100 - core]
        params = {
            "core_budget": core,
            "remainder_budget": int(rng.choice(rem_options)),
            "floor": float(rng.choice(floor_choices)),
            "ladder": int(rng.choice(ladder_choices)),
            "max_single": int(rng.choice(max_choices)),
            "rebalance": int(rng.choice(rebal_choices)),
            "core_size": 3,
        }
        order = list(orders_all[int(rng.integers(0, len(orders_all)))])
        pool.append({"id": f"P{k:05d}", "order": order, "params": params})

    pool.append({"id": "B0207", "order": B0207_ORDER, "params": B0207_PARAMS})
    results = []
    for m in pool:
        rec = {"model": m}
        rec.update(run(m["order"], m["params"]))
        rec["gate"] = passes_gates(rec, baseline_5y)
        results.append(rec)
        if (len(results) % 1000) == 0:
            print("simulated", len(results))

    gated = [r for r in results if r["gate"]]
    b0207 = next(r for r in results if r["model"]["id"] == "B0207")
    print("pool", len(results), "gated", len(gated), "b0207 gate", b0207["gate"])

    def rank_metrics(metric: str, higher_better: bool = True):
        def key(r):
            return r[metric] if higher_better else -r[metric]
        ordered = sorted(gated, key=key, reverse=True)
        rank = next(i for i, r in enumerate(ordered, 1) if r["model"]["id"] == "B0207")
        return ordered, rank

    metrics_def = [
        ("score", "综合评分", lambda r: r["score"]),
        ("Y5_cum", "近5年收益", lambda r: r["Y5"]["cum_return"]),
        ("irr", "5年IRR", lambda r: r["Y5"]["annual_return_irr"]),
        ("sharpe", "5年Sharpe", lambda r: r["Y5"]["sharpe"]),
        ("mdd", "5年MDD", lambda r: r["Y5"]["max_drawdown"]),
    ]
    lines = [
        "# B0207 真冠军检验报告",
        "",
        f"模型池：{len(results)} 个（{POOL_SIZE} 随机 + B0207 + 基线）；种子 {SEED}；"
        "与 B0207 同一搜索空间（159915 前二、159622/516150 前四、核心3、真实账户、无周度保护）。",
        f"门槛：至少两年为正 且 近5年 ≥ 基线（{pct(baseline_5y)}）；通过门槛 {len(gated)} 个。",
        "",
        "## 一、B0207 排名总览（在通过门槛的模型中）",
        "",
        "| 指标 | B0207 值 | 排名 | 百分位 | 超过它的模型数 | 该指标最优模型 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    ranking = {}
    for key, label, func in metrics_def:
        ordered = sorted(gated, key=lambda r: func(r), reverse=True)
        rank = next(i for i, r in enumerate(ordered, 1) if r["model"]["id"] == "B0207")
        value = func(b0207)
        beaters = rank - 1
        best = ordered[0]
        lines.append(
            f"| {label} | {pct(value) if key != 'score' else f'{value:.3f}'} | {rank}/{len(gated)} | "
            f"{(1 - rank / len(gated)):.1%} | {beaters} | {best['model']['id']} "
            f"({pct(func(best)) if key != 'score' else f'{func(best):.3f}'}) |"
        )
        ranking[key] = {"rank": rank, "percentile": 1 - rank / len(gated), "beaters": beaters}

    # 聚合排名：五个指标百分位均值（先排序再查表，避免 O(n²)）
    for r in gated:
        r["agg_list"] = []
    for key, label, func in metrics_def:
        vals = np.asarray([func(r) for r in gated], dtype=float)
        order = np.argsort(-vals, kind="stable")
        rank_map = {}
        for pos, i in enumerate(order):
            rank_map[gated[i]["model"]["id"]] = pos + 1
        for r in gated:
            r["agg_list"].append(1 - rank_map[r["model"]["id"]] / len(gated))
    for r in gated:
        r["agg"] = float(np.mean(r["agg_list"]))
    top_agg = sorted(gated, key=lambda r: -r["agg"])[:20]
    b_agg_rank = next(i for i, r in enumerate(sorted(gated, key=lambda x: -x["agg"]), 1)
                      if r["model"]["id"] == "B0207")
    lines += [
        "",
        f"B0207 聚合排名（5指标百分位均值）：第 {b_agg_rank}/{len(gated)} 名",
        "",
        "## 二、严格切分检验（选择窗 2021-08~2024-08 → 验证窗 2024-08~2026-08）",
        "",
    ]
    sel_ordered = sorted(gated, key=lambda r: r["SEL"]["cum_return"], reverse=True)
    b_sel_rank = next(i for i, r in enumerate(sel_ordered, 1) if r["model"]["id"] == "B0207")
    val_ordered = sorted(gated, key=lambda r: r["VAL"]["cum_return"], reverse=True)
    b_val_rank = next(i for i, r in enumerate(val_ordered, 1) if r["model"]["id"] == "B0207")
    top_sel = sel_ordered[:100]
    top_sel_val = sorted(top_sel, key=lambda r: r["VAL"]["cum_return"], reverse=True)[:10]
    b_val_rank_in_top_sel = next(
        (i for i, r in enumerate(top_sel, 1) if r["model"]["id"] == "B0207"), None
    )
    lines.append(
        f"B0207：选择窗收益 {pct(b0207['SEL']['cum_return'])}（排名 {b_sel_rank}/{len(gated)}），"
        f"验证窗收益 {pct(b0207['VAL']['cum_return'])}（排名 {b_val_rank}/{len(gated)}）"
        + (f"，在按选择窗取的 Top100 中排第 {b_val_rank_in_top_sel} 名" if b_val_rank_in_top_sel else "，未进入选择窗 Top100")
    )
    lines += ["", "按选择窗 Top100 → 验证窗 Top10：", "",
              "| 模型 | 选择窗 | 验证窗 | 综合分 | 近5年 |",
              "|---|---:|---:|---:|---:|"]
    for r in top_sel_val:
        lines.append(f"| {r['model']['id']} | {pct(r['SEL']['cum_return'])} | {pct(r['VAL']['cum_return'])} | "
                     f"{r['score']:.3f} | {pct(r['Y5']['cum_return'])} |")
    lines += ["", "## 三、聚合排名 Top 20", "",
              "| 模型 | 综合分 | 近5年 | IRR | Sharpe | MDD | 聚合百分位 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in top_agg:
        mark = " ← B0207" if r["model"]["id"] == "B0207" else ""
        lines.append(f"| {r['model']['id']}{mark} | {r['score']:.3f} | {pct(r['Y5']['cum_return'])} | "
                     f"{pct(r['Y5']['annual_return_irr'])} | {r['Y5']['sharpe']:.2f} | "
                     f"{pct(r['Y5']['max_drawdown'])} | {r['agg']:.1%} |")
    lines += [
        "",
        f"## 四、结论",
        "",
        f"B0207 综合评分排名：第 {ranking['score']['rank']}/{len(gated)}；"
        f"近5年收益排名：第 {ranking['Y5_cum']['rank']}/{len(gated)}；"
        f"严格切分下验证窗排名：第 {b_val_rank}/{len(gated)}；"
        f"聚合排名：第 {b_agg_rank}/{len(gated)}。",
        "",
        "判定规则：若 B0207 在 综合评分 / 近5年 / 聚合排名 中至少两项进入前 10%，"
        "且验证窗排名不显著落后（前 25%），判定为真冠军；否则判定为搜索幸运儿。",
    ]
    score_ok = ranking["score"]["percentile"] >= 0.90
    y5_ok = ranking["Y5_cum"]["percentile"] >= 0.90
    agg_ok = (1 - b_agg_rank / len(gated)) >= 0.90
    val_ok = (1 - b_val_rank / len(gated)) >= 0.75
    verdict = "真冠军（稳健）" if (score_ok + y5_ok + agg_ok) >= 2 and val_ok else "存在更强模型（非唯一最优）"
    lines.append(f"**判定：{verdict}**")

    OUT.mkdir(parents=True, exist_ok=True)
    md_path = OUT / "b0207_champion_verification.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with (OUT / "b0207_champion_verification.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["模型ID", "顺序", "核心", "剩余", "下限", "档位", "上限", "频率",
                         "Y1", "Y2", "Y3", "近5年", "综合分", "SEL", "VAL", "门槛",
                         "IRR", "Sharpe", "MDD", "聚合百分位"])
        for r in sorted(results, key=lambda x: -x.get("agg", -1)):
            p = r["model"]["params"]
            writer.writerow([r["model"]["id"], "/".join(r["model"]["order"]), p["core_budget"],
                             p["remainder_budget"], pct(p["floor"]), p["ladder"], p["max_single"],
                             p["rebalance"], pct(r["Y1"]["cum_return"]), pct(r["Y2"]["cum_return"]),
                             pct(r["Y3"]["cum_return"]), pct(r["Y5"]["cum_return"]),
                             f"{r['score']:.4f}", pct(r["SEL"]["cum_return"]), pct(r["VAL"]["cum_return"]),
                             "是" if r["gate"] else "否", pct(r["Y5"]["annual_return_irr"]),
                             f"{r['Y5']['sharpe']:.2f}", pct(r["Y5"]["max_drawdown"]),
                             f"{r.get('agg', -1):.4f}"])
    payload = {
        "schema_version": "b0207-verification-v1",
        "seed": SEED,
        "pool_size": len(results),
        "gated": len(gated),
        "baseline_5y": baseline_5y,
        "ranking": ranking,
        "b0207_agg_rank": b_agg_rank,
        "b0207_sel_rank": b_sel_rank,
        "b0207_val_rank": b_val_rank,
        "verdict": verdict,
        "b0207": {**b0207["model"], **{w: b0207[w] for w in ("Y1", "Y2", "Y3", "Y5", "SEL", "VAL")},
                  "score": b0207["score"], "gate": b0207["gate"]},
        "top20_agg": [
            {"id": r["model"]["id"], "order": r["model"]["order"], "params": r["model"]["params"],
             **{w: r[w] for w in ("Y1", "Y2", "Y3", "Y5", "SEL", "VAL")},
             "score": r["score"], "agg": r["agg"]}
            for r in top_agg
        ],
        "top_sel_val": [
            {"id": r["model"]["id"], "SEL": r["SEL"]["cum_return"], "VAL": r["VAL"]["cum_return"],
             "score": r["score"], "Y5": r["Y5"]["cum_return"]}
            for r in top_sel_val
        ],
    }
    (OUT / "b0207_champion_verification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved", md_path)
    print("B0207 ranks: score", ranking["score"]["rank"], "y5", ranking["Y5_cum"]["rank"],
          "agg", b_agg_rank, "val", b_val_rank, "of", len(gated))
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "LEGACY_BLOCKED: 本脚本固定使用518600/512010旧ETF池，只保留历史B0207研究复现；"
        "当前冻结B0207不得由此脚本重新验证、训练或覆盖。"
    )
