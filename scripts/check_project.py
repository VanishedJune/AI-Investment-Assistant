# -*- coding: utf-8 -*-
"""Local project health check.

Run from the project root:

    python scripts/check_project.py

It only reads files and validates frozen contracts.  It never fetches market data
and never writes project state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _active_weights_view(weights: dict, priority: list[str]) -> dict:
    view = dict(weights)
    if "159611" in priority and "159611" not in view and "159941" in view:
        if view["159941"] != 0:
            raise ValueError("159941仍有非零真实持仓，不能随分析槽位自动转记")
        view["159611"] = 0.0
        view.pop("159941")
    return view


def _file_meta(path: Path) -> dict:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def check_manifest() -> list[str]:
    errors: list[str] = []
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        errors.append("data_manifest.json 缺少 run_id")
        return errors
    for record in manifest.get("files", []):
        rel = Path(str(record.get("path") or ""))
        path = (ROOT / rel).resolve()
        if ROOT.resolve() not in path.parents:
            errors.append(f"清单路径越界: {record.get('path')}")
            continue
        if not path.is_file():
            errors.append(f"清单文件缺失: {record.get('path')}")
            continue
        recorded_size = record.get("size_bytes")
        if recorded_size is not None and int(recorded_size) != path.stat().st_size:
            errors.append(f"清单文件大小不一致: {record.get('path')}")
        recorded_modified = record.get("modified_utc")
        if recorded_modified and _file_meta(path)["modified_utc"] != str(recorded_modified):
            errors.append(f"清单文件修改时间不一致: {record.get('path')}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(record.get("rows", -1)):
            errors.append(f"清单行数不一致: {record.get('path')}")
        if any(str(row.get("run_id")) != run_id for row in rows):
            errors.append(f"清单 run_id 不一致: {record.get('path')}")
    return errors


def check_feature_contract(*, non_model=False) -> list[str]:
    errors: list[str] = []
    snapshot_path = ROOT / "app" / "features" / "latest.json"
    if not snapshot_path.is_file():
        return ["缺少 app/features/latest.json"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    priority_path = ROOT / "model_iteration" / "configs" / "investment_priority.json"
    priority = ([x['code'] for x in json.loads((ROOT/'config/instruments.json').read_text(encoding='utf-8'))['instruments']]
                if non_model else list(json.loads(priority_path.read_text(encoding="utf-8"))["priority"]))
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    if snapshot.get("data_run_id") != manifest.get("run_id"):
        errors.append("feature snapshot run_id 与 data_manifest 不一致")
    if list(snapshot.get("instrument_order") or []) != priority:
        errors.append("feature snapshot instrument_order 与固定优先级不一致")
    try:
        import feature_contract
        errors.extend(feature_contract.validate_snapshot(snapshot, priority))
    except ImportError:
        errors.append("无法导入 feature_contract（请在项目根目录运行）")
    return errors


def check_configuration(*, non_model=False) -> list[str]:
    errors: list[str] = []
    instruments = [item.get("code") for item in json.loads((ROOT / "config" / "instruments.json").read_text(encoding="utf-8"))["instruments"]]
    priority = (instruments if non_model else list(json.loads((ROOT / "model_iteration" / "configs" / "investment_priority.json").read_text(encoding="utf-8"))["priority"]))
    if instruments != priority:
        errors.append("config/instruments.json 顺序与 B0207 固定优先级不一致")
    state = json.loads((ROOT / "portfolio_state.json").read_text(encoding="utf-8"))
    raw_weights = (state.get("last_confirmed_portfolio") or {}).get("weights_pct") or {}
    try:
        weights = _active_weights_view(raw_weights, priority)
    except ValueError as exc:
        errors.append(str(exc))
        weights = raw_weights
    if set(weights) != set(priority):
        errors.append("portfolio_state 的 8 只 ETF 集合不完整")
    values = list(weights.values()) + [(state.get('last_confirmed_portfolio') or {}).get('cash_pct')]
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=100 for v in values) or abs(sum(values)-100)>1e-8:
        errors.append("last_confirmed_portfolio 合计不等于 100%")
    monthly = state.get("current_monthly_report") or {}
    for field in ("monitoring_html_status", "formal_publication_status"):
        if not str(monthly.get(field) or "").strip():
            errors.append(f"current_monthly_report 缺少语义状态字段 {field}")
    return errors


def check_report_holdings() -> list[str]:
    errors = []
    if (ROOT/'.publication-journal.json').exists():
        errors.append('存在未完成发布事务，须先恢复')
    state = json.loads((ROOT/'portfolio_state.json').read_text(encoding='utf-8'))
    latest = ROOT/'app/reports/monthly/latest.json'
    report = json.loads(latest.read_text(encoding='utf-8'))
    dated = ROOT/'app/reports/monthly'/f"{report['generated_date']}.json"
    if not dated.exists() or dated.read_bytes()!=latest.read_bytes():
        errors.append('月报latest与同日JSON不一致')
    actual = state['last_confirmed_portfolio']
    history = json.loads((ROOT/'app/portfolio_history/actual_holdings.json').read_text(encoding='utf-8'))
    if history['entries'][-1]['current_snapshot'] != actual or report['last_confirmed_portfolio'] != actual:
        errors.append('实际持仓、历史和月报快照不一致')
    if report.get('user_final_target') != state.get('user_final_target'):
        errors.append('下周目标与月报不一致')
    target = state.get('user_final_target') or {}
    if target.get('status') == 'user_confirmed_next_week_target':
        values = list((target.get('weights_pct') or {}).values())+[target.get('cash_pct')]
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=100 for v in values) or abs(sum(values)-100)>1e-8:
            errors.append('用户目标比例无效')
        if state.get('user_decision',{}).get('fill_status')!='pending_user_fill_confirmation':
            errors.append('下周目标不应标成已经成交')
    return errors


def check_root_boundary() -> list[str]:
    errors: list[str] = []
    if any((ROOT / "scripts").glob("投资决策月报*.html")) and any(ROOT.glob("投资决策月报*.html")):
        errors.append("根目录不应存在月报 HTML，只应保留快捷入口")
    state = json.loads((ROOT / "portfolio_state.json").read_text(encoding="utf-8"))
    monthly = state.get("current_monthly_report") or {}
    expected_name = str(monthly.get("shortcut") or "")
    shortcuts = sorted(ROOT.glob("投资决策月报*.lnk"))
    if len(shortcuts) != 1:
        errors.append(f"根目录月报快捷入口必须恰好1个，当前为{len(shortcuts)}个")
    elif shortcuts[0].name != expected_name:
        errors.append(f"月报快捷入口与状态不一致：实际{shortcuts[0].name}，状态{expected_name or '缺失'}")
    html_relative = str(monthly.get("html") or "")
    if not html_relative or not (ROOT / html_relative).is_file():
        errors.append("current_monthly_report指向的HTML不存在")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Investment Assistant 项目体检")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument('--non-model', action='store_true', help='不检查模型或基线配置')
    args = parser.parse_args()
    checks = {
        "manifest": check_manifest(),
        "feature_contract": check_feature_contract(non_model=args.non_model),
        "configuration": check_configuration(non_model=args.non_model),
        "report_holdings": check_report_holdings(),
        "root_boundary": check_root_boundary(),
    }
    status = "PASSED" if not any(checks.values()) else "FAILED"
    if args.json:
        print(json.dumps({"status": status, "checks": checks}, ensure_ascii=False, indent=2))
    else:
        print(f"项目体检：{status}")
        for name, errors in checks.items():
            if errors:
                print(f"[FAIL] {name}")
                for error in errors:
                    print(f"  - {error}")
            else:
                print(f"[ OK ] {name}")
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
