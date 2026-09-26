"""当前冻结冠军 + 截止日已知行情 → B0207机械参考；不训练、不推进迭代。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != ROOT / 'scripts']
sys.path.insert(0, str(ROOT))
from rolling.inference import build_reference  # noqa: E402

OUT_DIR = ROOT.parent / "champion_vs_dca"


def _atomic_write_text(path: Path, content: str, *, correct_holiday_week: bool = False) -> None:
    if path.exists():
        previous_text = path.read_text(encoding="utf-8")
        if previous_text == content:
            return
        if not correct_holiday_week:
            raise RuntimeError(f"已有不同内容的机械参考，拒绝覆盖: {path.name}")
        previous, current = json.loads(previous_text), json.loads(content)
        if not (path.name == "monitoring_baseline_2026-09-24.json"
                and previous.get("data_run_id") == current.get("data_run_id")
                and previous.get("requested_as_of") == current.get("requested_as_of") == "2026-09-24"
                and previous.get("as_of") == "2026-09-18"
                and current.get("as_of") == "2026-09-24"):
            raise RuntimeError("只允许修正已核实的2026年中秋节周线截止日")
        archive = path.with_name("monitoring_baseline_2026-09-24.pre_holiday_correction.json")
        if archive.exists() and archive.read_text(encoding="utf-8") != previous_text:
            raise RuntimeError("历史监控归档目标已存在且内容不同")
        if not archive.exists():
            archive.write_text(previous_text, encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".monitoring-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows rename refuses an existing target, including a racing writer.
        if correct_holiday_week and path.exists():
            os.replace(temporary, path)
        else:
            os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="分析日；默认正式行情清单截止日，只使用完整周线")
    parser.add_argument("--check", action="store_true", help="仅计算并显示，不写任何文件")
    parser.add_argument("--correct-holiday-week", action="store_true", help="归档并修正9月24日中秋周的旧监控")
    args = parser.parse_args()
    payload = build_reference(args.as_of)
    print(f"行情截止 {payload['data_as_of']}；完整周线信号 {payload['as_of']}")
    for row in payload["instruments"]:
        print(f"{row['code']} {row['name']} {row['champion_id']} 冠军仓位={row['position']:.4f}")
    print("B0207:", payload["b0207_weights_pct"], "现金", payload["b0207_cash_pct"])
    for warning in payload["warnings"]:
        print("WARNING:", warning)
    if not args.check:
        out = OUT_DIR / f"monitoring_baseline_{payload['as_of']}.json"
        _atomic_write_text(out, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           correct_holiday_week=args.correct_holiday_week)
        print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
