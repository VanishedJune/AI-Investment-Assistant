"""8 只 ETF 多参数决策迭代批次启动器（每批最多并行 4 支，批门禁防漏跑）。

职责：
- 按批次并发启动 `run_etf_iteration`（每支一个独立进程）；
- 批内全部成功（rc=0）才进入下一批；失败自动重试（默认 1 次），仍失败则停批；
- 状态写入 `logs/迭代批次_<suffix>.json`，作为单一事实来源；
- `--check-only` 只跑启动前检查（数据/特征/角色/spec），不写台账、不迭代。

用法（model_iteration 目录）：
    python -m scripts.run_iteration_8etf [--etfs 159915,159611,...] [--batch-size 4]
        [--refresh-syntax] [--check-only] [--log-suffix multiparam_20260829]
        [--max-retries 1]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

DEFAULT_ETFS = json.loads((ROOT / "configs/investment_priority.json").read_text(encoding="utf-8"))["priority"]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etfs", default=",".join(DEFAULT_ETFS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--refresh-syntax", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--log-suffix", default="multiparam_20260829")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    etfs = [c.strip() for c in args.etfs.split(",") if c.strip()]
    for code in etfs:
        cfg = json.loads((ROOT / "configs" / f"etf_{code}.json").read_text(encoding="utf-8"))
        if cfg.get("training_disabled"):
            print(f"{code}仅允许冻结推理；整批训练未启动。需用户另行授权研究。")
            return 2
    if not etfs:
        print("etfs 为空")
        return 2
    if args.batch_size < 1:
        print("batch-size 必须 >= 1")
        return 2

    status_path = ROOT / "logs" / f"迭代批次_{args.log_suffix}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status: dict = {
        "schema_version": "iteration-batch-v1",
        "mode": "check" if args.check_only else "run",
        "started_at": now(),
        "etfs": {c: {"status": "pending", "rc": None, "retries": 0, "started_at": None, "finished_at": None, "log": None} for c in etfs},
        "gates": {},
    }

    def save_status() -> None:
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    batches = [etfs[i : i + args.batch_size] for i in range(0, len(etfs), args.batch_size)]
    all_ok = True
    for bi, batch in enumerate(batches, 1):
        print(f"=== 批次 {bi}/{len(batches)}：{','.join(batch)} ===", flush=True)
        batch_ok = True
        remaining = list(batch)
        for attempt in range(args.max_retries + 1):
            if not remaining:
                break
            procs = {}
            for code in remaining:
                status["etfs"][code]["status"] = "running"
                status["etfs"][code]["started_at"] = now()
                status["etfs"][code]["retries"] = attempt
                save_status()
                cmd = [PY, "-m", "scripts.run_etf_iteration", "--etf", code,
                       "--log-file", f"run_{args.log_suffix}.log"]
                if args.check_only:
                    cmd.append("--check")
                if args.refresh_syntax:
                    cmd.append("--refresh-syntax")
                print(f"[{now()}] 并行启动 {code} (attempt {attempt + 1}/{args.max_retries + 1})", flush=True)
                procs[code] = subprocess.Popen(cmd, cwd=str(ROOT))
            failed = []
            for code, proc in procs.items():
                rc = proc.wait()
                print(f"[{now()}] {code} rc={rc}", flush=True)
                if rc == 0:
                    status["etfs"][code]["status"] = "done"
                    status["etfs"][code]["rc"] = rc
                    status["etfs"][code]["finished_at"] = now()
                    status["etfs"][code]["log"] = f"run_{args.log_suffix}.log"
                    save_status()
                else:
                    failed.append(code)
                    print(f"[{now()}] {code} 失败 rc={rc}，等待重试", flush=True)
            if not failed:
                remaining = []
                break
            remaining = failed
            print(f"[{now()}] 批次 {bi} 重试：{','.join(remaining)}", flush=True)
        if remaining:
            for code in remaining:
                status["etfs"][code]["status"] = "failed"
                status["etfs"][code]["finished_at"] = now()
            save_status()
            batch_ok = False
        status["gates"][f"batch_{bi}"] = batch_ok
        save_status()
        if not batch_ok:
            all_ok = False
            print(f"=== 批次 {bi} 未全部通过，停止后续批次 ===", flush=True)
            break

    status["finished_at"] = now()
    status["all_ok"] = all_ok
    save_status()
    print(f"\n=== 汇总（{status_path}）===", flush=True)
    for code, rec in status["etfs"].items():
        print(f"{code}: {rec['status']} rc={rec['rc']} retries={rec['retries']}", flush=True)
    print("ALL_OK =", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
