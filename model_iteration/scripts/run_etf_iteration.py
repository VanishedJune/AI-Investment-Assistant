"""单支 ETF 全自动迭代驱动（执行口径由ETF配置决定）。

流程：diagnose-all → 同步诊断 → sanitized_factory → freeze_all → replay 循环
（rc=4 自动 approve_pending 后续跑；rc=3 自动补跑 factory+freeze 后续跑）→
write_iteration_record → 写运行日志。

用法（model_iteration 目录）：
    python -m scripts.run_etf_iteration --etf <code> [--refresh-syntax] [--check]

--refresh-syntax：先重新导出 sanitized_workspace/spec_syntax.json（用于纳入
  新增多参数角色模板）。
--check：只做启动前检查（数据装配、多参数特征、角色模板与 spec 校验），
  不写台账、不生成提案、不运行回放。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [PY, "-m", *args]
    log(f"RUN: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def tail(out: str, n: int = 12) -> str:
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def check_ready(etf: str, refresh_syntax: bool) -> int:
    """启动前检查：数据装配、多参数特征、语法角色、角色 spec 校验。"""
    import json as _json

    from rolling.config import load_etf_config as _cfg
    from rolling.data import load_aligned_data as _lad
    from rolling.features import MULTI_PARAM_FEATURES
    from rolling.ledger import set_workspace as _sw
    from rolling.spec import validate_spec

    _sw(etf)
    ws = ROOT / f"etf_{etf}"
    ok = True
    try:
        _, _, _, _, usable, features = _lad(etf)
        print(f"CHECK data_assembly: usable={len(usable)} OK")
    except Exception as exc:  # noqa: BLE001
        print(f"CHECK data_assembly: FAIL {exc}")
        return 1
    missing = [c for c in MULTI_PARAM_FEATURES if c not in features.columns]
    if missing:
        print(f"CHECK multi_param_features: FAIL missing={missing}")
        return 1
    print(f"CHECK multi_param_features: OK ({len(MULTI_PARAM_FEATURES)} features)")
    syntax_path = ws / "sanitized_workspace" / "spec_syntax.json"
    if refresh_syntax or not syntax_path.exists():
        proc = run_cli(["scripts.export_syntax", "--etf", etf])
        if proc.returncode != 0:
            print(f"CHECK export_syntax: FAIL {tail(proc.stdout)}{tail(proc.stderr)}")
            return 1
        print("CHECK export_syntax: regenerated")
    syntax = _json.loads(syntax_path.read_text(encoding="utf-8"))
    pred_roles = (syntax.get("prediction_roles") or {})
    strat_roles = (syntax.get("strategy_roles") or {})
    if "multi_param_joint" not in pred_roles:
        ok = False
        print("CHECK syntax: 缺少 prediction role multi_param_joint")
    if "multi_param_defensive" not in strat_roles:
        ok = False
        print("CHECK syntax: 缺少 strategy role multi_param_defensive")
    if ok:
        print("CHECK syntax roles: OK")
    for name, spec in (
        ("multi_param_joint", pred_roles.get("multi_param_joint", {}).get("spec")),
        ("multi_param_defensive", strat_roles.get("multi_param_defensive", {}).get("spec")),
    ):
        if spec is None:
            continue
        errors = validate_spec(spec)
        if errors:
            ok = False
            print(f"CHECK spec {name}: FAIL {errors}")
        else:
            print(f"CHECK spec {name}: OK")
    print("CHECK overall:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(
    etf: str,
    log_file: str = "run_20260828.log",
    check_mode: bool = False,
    refresh_syntax: bool = False,
) -> int:
    config_path = ROOT / 'configs' / f'etf_{etf}.json'
    if config_path.is_file() and json.loads(config_path.read_text(encoding='utf-8')).get('training_disabled'):
        print(f'{etf}为冻结迁移模型，禁止训练/晋级/刷新语法；仅允许只读推理。')
        return 2
    ws = ROOT / f"etf_{etf}"
    run_log = ws / log_file
    summary: list[str] = []
    started = datetime.now().astimezone().isoformat(timespec="seconds")

    def record(step: str, proc: subprocess.CompletedProcess | None, note: str = "") -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {step} rc={proc.returncode if proc else 'n/a'} {note}"
        summary.append(line)
        log(line)
        if proc is not None and proc.returncode != 0:
            summary.append(tail(proc.stdout))
            summary.append(tail(proc.stderr))

    if etf is None:
        return 2
    try:
        if check_mode:
            return check_ready(etf, refresh_syntax)
        if refresh_syntax:
            proc = run_cli(["scripts.export_syntax", "--etf", etf])
            if proc.returncode != 0:
                raise RuntimeError(f"export_syntax 失败: {tail(proc.stdout)}{tail(proc.stderr)}")
        # 1) 诊断
        proc = run_cli(["rolling.cli", "diagnose-all", "--etf", etf])
        record("diagnose-all", proc)
        if proc.returncode != 0:
            raise RuntimeError(f"diagnose-all 失败: {tail(proc.stdout)}{tail(proc.stderr)}")

        # 2) 同步诊断到隔离工作区
        src_diag = ws / "weekly_rolling" / "diagnostics"
        dst_diag = ws / "sanitized_workspace" / "diagnostics"
        dst_diag.mkdir(parents=True, exist_ok=True)
        for p in src_diag.glob("ROUND_*.json"):
            shutil.copy2(p, dst_diag / p.name)
        n_diag = len(list(dst_diag.glob("ROUND_*.json")))
        log(f"diagnostics synced: {n_diag}")
        summary.append(f"diagnostics_synced={n_diag}")

        # 3) 工厂草稿
        proc = run_cli(["scripts.sanitized_factory", "--etf", etf])
        record("sanitized_factory", proc)
        if proc.returncode != 0:
            raise RuntimeError(f"sanitized_factory 失败: {tail(proc.stdout)}{tail(proc.stderr)}")

        # 4) 冻结
        proc = run_cli(["scripts.freeze_all_proposals", "--etf", etf])
        record("freeze_all_proposals", proc)
        if proc.returncode != 0:
            raise RuntimeError(f"freeze_all_proposals 失败: {tail(proc.stdout)}{tail(proc.stderr)}")

        # 5) 回放循环
        resumed = False
        rc3_fixes = 0
        rc4_cycles = 0
        while True:
            args = ["rolling.cli", "replay", "--etf", etf]
            if resumed:
                args.append("--resume")
            proc = run_cli(args)
            record("replay" + (" --resume" if resumed else ""), proc)
            if proc.returncode == 0:
                log("replay completed rc=0")
                break
            if proc.returncode == 4:
                rc4_cycles += 1
                if rc4_cycles > 40:
                    raise RuntimeError("AI 复核循环超过 40 次，停止")
                log("AI_REVIEW_REQUIRED -> approve_pending")
                ap = run_cli(["scripts.approve_pending", "--etf", etf])
                record("approve_pending", ap)
                resumed = True
                continue
            if proc.returncode == 3:
                rc3_fixes += 1
                if rc3_fixes > 5:
                    raise RuntimeError("AGENT_REVIEW_REQUIRED 修复超过 5 次，停止")
                log("AGENT_REVIEW_REQUIRED -> 补跑 factory+freeze")
                run_cli(["scripts.sanitized_factory", "--etf", etf])
                proc2 = run_cli(["scripts.freeze_all_proposals", "--etf", etf])
                record("freeze_all_proposals(repair)", proc2)
                resumed = True
                continue
            raise RuntimeError(
                f"replay 异常退出码 {proc.returncode}: {tail(proc.stdout)}{tail(proc.stderr)}"
            )

        # 6) 完成校验
        state = json.loads((ws / "weekly_rolling" / "state.json").read_text(encoding="utf-8"))
        n_proposals = len(list((ws / "weekly_rolling" / "proposals").glob("ROUND_*.json")))
        events_size = (ws / "weekly_rolling" / "events.jsonl").stat().st_size
        checks = {
            "pending_ai_reviews_empty": len(state.get("pending_ai_reviews", [])) == 0,
            "paused_at_none": state.get("paused_at") is None,
            "resume_consistent": state.get("resume_from") == state.get("latest_anchor_index", -1) + 1,
            "proposals_match_diag": n_proposals == n_diag,
            "events_nonempty": events_size > 0,
        }
        for name, ok in checks.items():
            log(f"CHECK {name}: {ok}")
            summary.append(f"CHECK_{name}={ok}")
        if not all(checks.values()):
            raise RuntimeError(f"完成校验未全部通过: {checks}")

        # 7) 迭代记录
        proc = run_cli(["scripts.write_iteration_record", "--etf", etf])
        record("write_iteration_record", proc)
        if proc.returncode != 0:
            raise RuntimeError(f"write_iteration_record 失败: {tail(proc.stdout)}{tail(proc.stderr)}")

        # 8) 摘要
        final = {
            "etf": etf,
            "status": "done",
            "started_at": started,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "diagnostics": n_diag,
            "proposals_frozen": n_proposals,
            "latest_anchor_index": state.get("latest_anchor_index"),
            "resume_from": state.get("resume_from"),
            "final_champion": state.get("champion", {}).get("id"),
            "promotions": len(state.get("promotion_history", [])),
            "pending_ai_reviews": len(state.get("pending_ai_reviews", [])),
            "checks": checks,
        }
        summary.append("FINAL=" + json.dumps(final, ensure_ascii=False))
        run_log.write_text("\n".join(summary) + "\n", encoding="utf-8")
        log(json.dumps(final, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        summary.append(f"FAILED: {exc}")
        run_log.write_text("\n".join(summary) + "\n", encoding="utf-8")
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    parser.add_argument("--refresh-syntax", action="store_true")
    parser.add_argument("--check", dest="check_mode", action="store_true")
    parser.add_argument("--log-file", default="run_20260828.log")
    args = parser.parse_args()
    raise SystemExit(main(args.etf, args.log_file, args.check_mode, args.refresh_syntax))
