# -*- coding: utf-8 -*-
"""Validate an objective feature snapshot without producing report content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_contract import load_and_validate


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="校验确定性日周参数文件")
    parser.add_argument("path", nargs="?", default="app/features/latest.json")
    args = parser.parse_args()
    path = (ROOT / args.path).resolve() if not Path(args.path).is_absolute() else Path(args.path)
    config = json.loads((ROOT / "config" / "instruments.json").read_text(encoding="utf-8"))
    configured = [item["code"] for item in config["instruments"]]
    priority_path = ROOT / "model_iteration" / "configs" / "investment_priority.json"
    expected = json.loads(priority_path.read_text(encoding="utf-8"))["priority"]
    if configured != list(expected):
        print("ERROR: investment priority and instrument configuration differ (order and membership must match)")
        return 1
    snapshot, errors = load_and_validate(path, expected)
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    if snapshot.get("data_run_id") != manifest.get("run_id"):
        errors.append("data_run_id does not match data_manifest.json")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"参数契约通过：{path.name} · run_id={snapshot['data_run_id']} · instruments={len(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
