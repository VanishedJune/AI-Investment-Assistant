# -*- coding: utf-8 -*-
"""第一轮实验执行器：159915 全历史 Champion/CH_A/CH_B → 指标 → 门禁 → ITERATION 记录。

只写 model_iteration/ 目录；不修改网页线任何文件。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .account import metrics as account_metrics
from .account import passive_same_exposure, simulate
from .calendar import build_signal_samples, signal_start_end
from .features import FEATURE_COLUMNS, build_features
from .gates import evaluate_gates, path_difference
from .strategies import ch_a_position, ch_b_position, champion_position, ridge_predict_walkforward

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
ITERATIONS = ROOT / "iterations"


def _slice_metrics(pos, fwd1, label8, forecast, prob, indices, cost_bps=10):
    nets, eq, turnover = simulate(pos[indices], fwd1[indices], cost_bps)
    m = account_metrics(
        nets,
        pos[indices],
        fwd1[indices],
        label8[indices] if label8 is not None else None,
        forecast[indices] if forecast is not None else None,
        prob[indices] if prob is not None else None,
    )
    m["turnover"] = round(turnover, 4)
    return m


def main() -> int:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", default="159915")
    parser.add_argument("--iteration", default="ITERATION_001_159915")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    iteration_id = args.iteration
    config_path = args.config or (CONFIGS / f"{iteration_id}.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    ITERATIONS.mkdir(parents=True, exist_ok=True)

    samples = build_signal_samples(args.etf)
    feat = build_features(samples, args.etf)
    for col in FEATURE_COLUMNS:
        samples[col] = feat[col].to_numpy()
    usable = samples.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    n = len(usable)
    first, last = signal_start_end(usable)

    fwd1 = usable["fwd1"].to_numpy(dtype=float)
    label8 = usable["label8"].to_numpy(dtype=float)

    pos_c = champion_position(usable)
    pos_a = ch_a_position(usable)
    preds_b = ridge_predict_walkforward(
        usable,
        usable["label8"],
        alpha=float(cfg["ridge_alpha"]),
        embargo=int(cfg["embargo_weeks"]),
        min_train=int(cfg.get("min_train_samples", 40)),
    )
    pos_b = ch_b_position(preds_b)

    bh_pos = np.ones(n)
    nets_c, eq_c, _ = simulate(pos_c, fwd1, cfg["roundtrip_cost_bps"])
    nets_a, eq_a, _ = simulate(pos_a, fwd1, cfg["roundtrip_cost_bps"])
    nets_b, eq_b, _ = simulate(pos_b, fwd1, cfg["roundtrip_cost_bps"])
    nets_bh, eq_bh, _ = simulate(bh_pos, fwd1, cfg["roundtrip_cost_bps"])

    # 预测/概率（用于验证期 MAE/Brier）
    dev_end = int(n * float(cfg["dev_fraction"]))
    dev_mask = np.zeros(n, dtype=bool)
    dev_mask[:dev_end] = True
    state_long = usable["w_slope"] > 0
    state_rec = (usable["w_slope"] < 0) & (usable["d_slope"] > 0) & (usable["d_dif1"] > 0) & (usable["d_dif2"] >= 0)
    mean_long = float(label8[dev_mask & state_long.to_numpy() & ~np.isnan(label8)].mean()) if (dev_mask & state_long.to_numpy() & ~np.isnan(label8)).any() else 0.0
    mean_rec = float(label8[dev_mask & state_rec.to_numpy() & ~np.isnan(label8)].mean()) if (dev_mask & state_rec.to_numpy() & ~np.isnan(label8)).any() else 0.0

    fc_c = np.where(pos_c > 0, mean_long, 0.0)
    pr_c = pos_c.copy()
    fc_a = np.where(pos_a == 1.0, mean_long, np.where(pos_a == 0.5, mean_rec, 0.0))
    pr_a = pos_a.copy()
    fc_b = preds_b.copy()
    pr_b = (preds_b > 0).astype(float)

    windows = {
        "full_history": np.arange(n),
        "last_165": np.arange(max(0, n - 165), n),
        "last_104": np.arange(max(0, n - 104), n),
        "validation": np.arange(dev_end, n),
    }

    def table(pos, forecast, prob):
        out = {}
        for name, idx in windows.items():
            out[name] = _slice_metrics(pos, fwd1, label8, forecast, prob, idx, cfg["roundtrip_cost_bps"])
        return out

    results = {
        "champion": table(pos_c, fc_c, pr_c),
        "ch_a": table(pos_a, fc_a, pr_a),
        "ch_b": table(pos_b, fc_b, pr_b),
        "buy_hold": table(bh_pos, None, None),
        "prediction_first_index": int(np.argmax(~np.isnan(preds_b))),
    }

    # 验证期门禁
    val_idx = windows["validation"]
    gates_out = {}
    for name, key, pos_x, fc_x, pr_x in (
        ("CH_A", "ch_a", pos_a, fc_a, pr_a),
        ("CH_B", "ch_b", pos_b, fc_b, pr_b),
    ):
        nets_x, _, _ = simulate(pos_x[val_idx], fwd1[val_idx], cfg["roundtrip_cost_bps"])
        avg_exp = float(np.nanmean(pos_x[val_idx]))
        passive_nets, _ = passive_same_exposure(fwd1[val_idx], avg_exp)
        passive_cum = float(np.prod(1.0 + passive_nets[~np.isnan(passive_nets)])) - 1.0
        diff_frac = path_difference(pos_c[val_idx], pos_x[val_idx])
        gates_out[name] = evaluate_gates(
            results["champion"]["validation"],
            results[key]["validation"],
            passive_cum,
            diff_frac,
            cfg["gates"],
        )
        gates_out[name]["passive_same_exposure_cum"] = round(passive_cum, 6)

    # 写 CSV
    equity_path = OUTPUTS / f"{iteration_id}_equity.csv"
    with open(equity_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["signal", "champion_equity", "ch_a_equity", "ch_b_equity", "buy_hold_equity"])
        for i in range(n):
            writer.writerow([usable["signal"].iloc[i].date().isoformat(), eq_c[i], eq_a[i], eq_b[i], eq_bh[i]])

    metrics_path = OUTPUTS / f"{iteration_id}_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "window", "metric", "value"])
        for strat, table_data in results.items():
            if not isinstance(table_data, dict):
                continue
            for window, m in table_data.items():
                for k, v in m.items():
                    writer.writerow([strat, window, k, v])

    experiment = {
        "iteration_id": iteration_id,
        "etf": args.etf,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window_policy": cfg["window_policy"],
        "usable_signal_weeks": n,
        "first_signal": first,
        "last_signal": last,
        "dev_weeks": dev_end,
        "validation_weeks": int(n - dev_end),
        "embargo_weeks": cfg["embargo_weeks"],
        "horizon_weeks": cfg["horizon_weeks"],
        "roundtrip_cost_bps": cfg["roundtrip_cost_bps"],
        "ridge_alpha": cfg["ridge_alpha"],
        "results": results,
        "gates": gates_out,
        "diagnosis": cfg["diagnosis"],
        "next_step": cfg["next_step"],
        "outputs": {
            "equity_csv": str(equity_path.relative_to(ROOT)),
            "metrics_csv": str(metrics_path.relative_to(ROOT)),
        },
    }
    (OUTPUTS / f"{iteration_id}_experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ITERATION 记录
    iteration_record = {
        "ITERATION_ID": iteration_id,
        "etf": args.etf,
        "problem": cfg["diagnosis"]["problem"],
        "evidence": cfg["diagnosis"]["evidence"],
        "hypothesis": cfg["diagnosis"]["hypothesis"],
        "changes": cfg["diagnosis"]["changes"],
        "results": {
            "champion_vs_challengers": {
                name: {
                    "decision": gates_out[name]["decision"],
                    "passed": gates_out[name]["passed"],
                    "validation": results[key]["validation"],
                }
                for name, key in (("CH_A", "ch_a"), ("CH_B", "ch_b"))
            },
            "champion_validation": results["champion"]["validation"],
        },
        "conclusion": "、".join(f"{k}={v['decision']}" for k, v in gates_out.items()),
        "next_step": cfg["next_step"],
        "generated_at": experiment["generated_at"],
    }
    (ITERATIONS / f"{iteration_id}.json").write_text(
        json.dumps(iteration_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown 摘要
    lines = [
        f"# {iteration_id}",
        "",
        f"- ETF：{args.etf} ｜ 可用信号周：{n}（{first} ~ {last}）",
        f"- 开发 {dev_end} 周 / 验证 {n - dev_end} 周；embargo {cfg['embargo_weeks']} 周；horizon {cfg['horizon_weeks']} 周",
        "",
        "## 问题与假设",
        cfg["diagnosis"]["problem"],
        "",
        cfg["diagnosis"]["hypothesis"],
        "",
        "## 验证期结果与门禁",
        "",
        "| 策略 | 累计收益 | Sharpe | 最大回撤 | 胜率 | 参与率 | 门禁 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, key in (("CH_A", "ch_a"), ("CH_B", "ch_b")):
        g = gates_out[name]
        m = results[key]["validation"]
        c = results["champion"]["validation"]
        lines.append(
            f"| {name} | {m['cumulative_return']*100:.2f}% | {m['sharpe']:.3f} | {m['max_drawdown']*100:.2f}% "
            f"| {m['win_rate']*100:.1f}% | {m['participation_up_weeks']*100:.1f}% | {g['decision']} |"
        )
    lines.append(
        f"| Champion | {c['cumulative_return']*100:.2f}% | {c['sharpe']:.3f} | {c['max_drawdown']*100:.2f}% "
        f"| {c['win_rate']*100:.1f}% | {c['participation_up_weeks']*100:.1f}% | 基线 |"
    )
    lines += ["", "## 下一步", cfg["next_step"], ""]
    (ITERATIONS / f"{iteration_id}.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
