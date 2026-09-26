"""Persistent decision-layer records for monthly analysis and reporting.

The ledger keeps four concepts separate:
1. the nominal-cycle B0207 baseline;
2. the B0207 observation produced for a specific analysis date;
3. the Agent recommendation for that analysis;
4. the reviewed, user-confirmed actual portfolio.

The module never trains a model, fetches market data, publishes a report or
changes a portfolio. Recording is explicit and uses an atomic replacement.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LEDGER_RELATIVE = Path("app/decision_history/decision_records.json")
ORDER = tuple(json.loads(
    (Path(__file__).resolve().parents[1] / "configs/investment_priority.json").read_text(encoding="utf-8")
)["priority"])
RECORD_TYPES = {
    "b0207_cycle_baseline",
    "b0207_weekly_observation",
    "agent_recommendation",
    "confirmed_portfolio",
}


class DecisionHistoryError(ValueError):
    pass


def _iso(value, field):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError) as exc:
        raise DecisionHistoryError(f"{field}必须是ISO日期") from exc


def _allocation(record, order):
    weights = record.get("weights_pct")
    if not isinstance(weights, dict) or list(weights) != list(order):
        raise DecisionHistoryError(record.get("record_id", "记录") + "的ETF顺序或范围错误")
    values = list(weights.values()) + [record.get("cash_pct")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 100
           for v in values):
        raise DecisionHistoryError(record.get("record_id", "记录") + "的账户比例非法")
    if abs(sum(values) - 100) > 1e-8:
        raise DecisionHistoryError(record.get("record_id", "记录") + "的ETF与现金合计不是100%")


def validate_record(record, order=ORDER):
    if not isinstance(record, dict) or record.get("record_type") not in RECORD_TYPES:
        raise DecisionHistoryError("未知记录类型")
    if not str(record.get("record_id", "")).strip():
        raise DecisionHistoryError("缺少record_id")
    _allocation(record, order)
    kind = record["record_type"]
    if kind == "b0207_cycle_baseline":
        start = _iso(record.get("decision_date"), "decision_date")
        cutoff = _iso(record.get("data_as_of"), "data_as_of")
        end = _iso(record.get("effective_until_exclusive"), "effective_until_exclusive")
        if cutoff > start or end <= start or record.get("status") != "FROZEN_CYCLE_REFERENCE":
            raise DecisionHistoryError("周期基线日期或状态错误")
    elif kind in ("b0207_weekly_observation", "agent_recommendation"):
        analysis = _iso(record.get("analysis_date"), "analysis_date")
        cutoff = _iso(record.get("data_as_of"), "data_as_of")
        if cutoff > analysis:
            raise DecisionHistoryError("分析使用了分析日之后的数据")
        expected = "NON_EXECUTION_MONITORING" if kind == "b0207_weekly_observation" else "ADVICE_ONLY"
        if record.get("status") != expected:
            raise DecisionHistoryError("非决策分析记录状态错误")
    else:
        _iso(record.get("confirmed_at"), "confirmed_at")
        if record.get("status") not in {
            "REVIEWED_USER_CONFIRMED",
            "REVIEWED_USER_CONFIRMED_COMPLETED",
        }:
            raise DecisionHistoryError("真实持仓尚未标记为审查通过且经用户确认")
    return record


def load_ledger(root=ROOT, *, required=True, expected_order=None):
    root = Path(root).resolve()
    path = (root / LEDGER_RELATIVE).resolve()
    if root not in path.parents:
        raise DecisionHistoryError("台账路径越界")
    if not path.is_file():
        if required:
            raise DecisionHistoryError("缺少决策历史台账")
        return None
    ledger = json.loads(path.read_text(encoding="utf-8"))
    order = tuple(expected_order or ORDER)
    if ledger.get("schema_version") != "decision-history-v1" or ledger.get("instrument_order") != list(order):
        raise DecisionHistoryError("决策历史台账版本或ETF顺序错误")
    seen = set()
    for record in ledger.get("records", []):
        validate_record(record, order)
        if record["record_id"] in seen:
            raise DecisionHistoryError("决策历史存在重复record_id")
        seen.add(record["record_id"])
    return ledger


def _latest(records, key):
    return max(records, key=lambda item: (str(item.get(key, "")), item["record_id"])) if records else None


def layers_for_analysis(root, analysis_date, data_as_of, *, expected_order=None, required=True):
    ledger = load_ledger(root, required=required, expected_order=expected_order)
    if ledger is None:
        return None
    analysis_day, cutoff = _iso(analysis_date, "analysis_date"), _iso(data_as_of, "data_as_of")
    records = ledger["records"]
    cycles = [r for r in records if r["record_type"] == "b0207_cycle_baseline"
              and _iso(r["decision_date"], "decision_date") <= analysis_day
              and analysis_day < _iso(r["effective_until_exclusive"], "effective_until_exclusive")]
    observations = [r for r in records if r["record_type"] == "b0207_weekly_observation"
                    and _iso(r["analysis_date"], "analysis_date") == analysis_day
                    and _iso(r["data_as_of"], "data_as_of") == cutoff]
    recommendations = [r for r in records if r["record_type"] == "agent_recommendation"
                       and _iso(r["analysis_date"], "analysis_date") == analysis_day
                       and _iso(r["data_as_of"], "data_as_of") == cutoff]
    portfolios = [r for r in records if r["record_type"] == "confirmed_portfolio"
                  and _iso(r["confirmed_at"], "confirmed_at") <= analysis_day]
    result = {
        "b0207_cycle_baseline": _latest(cycles, "decision_date"),
        "b0207_weekly_observation": _latest(observations, "recorded_at"),
        "agent_recommendation": _latest(recommendations, "recorded_at"),
        "confirmed_portfolio": _latest(portfolios, "confirmed_at"),
    }
    if required and any(value is None for value in result.values()):
        missing = [key for key, value in result.items() if value is None]
        raise DecisionHistoryError("本次分析缺少决策层记录: " + ", ".join(missing))
    return result


def append_record(record, root=ROOT, *, commit=False):
    root = Path(root).resolve()
    ledger = load_ledger(root, expected_order=ORDER)
    record = validate_record(record, ORDER)
    existing = next((r for r in ledger["records"] if r["record_id"] == record["record_id"]), None)
    if existing is not None:
        if existing != record:
            raise DecisionHistoryError("相同record_id对应不同内容，拒绝覆盖")
        return {"status": "UNCHANGED", "record_id": record["record_id"]}
    updated = {**ledger, "records": ledger["records"] + [record]}
    load_check = {**updated}
    for item in load_check["records"]:
        validate_record(item, ORDER)
    if not commit:
        return {"status": "CHECK_PASSED", "record_id": record["record_id"]}
    path = root / LEDGER_RELATIVE
    lock = root / ".decision-history.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    temporary = None
    try:
        current = load_ledger(root, expected_order=ORDER)
        if current != ledger:
            raise DecisionHistoryError("检查后台账发生变化")
        payload = (json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        stage_fd, stage_name = tempfile.mkstemp(prefix=".decision-history-", suffix=".tmp", dir=path.parent)
        temporary = Path(stage_name)
        with os.fdopen(stage_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return {"status": "RECORDED", "record_id": record["record_id"]}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(fd)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    listing = sub.add_parser("list")
    listing.add_argument("--type", choices=sorted(RECORD_TYPES))
    record = sub.add_parser("record")
    record.add_argument("--input", required=True, help="项目内单条记录JSON")
    record.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            ledger = load_ledger(ROOT)
            result = {"status": "OK", "record_count": len(ledger["records"])}
        elif args.command == "list":
            records = load_ledger(ROOT)["records"]
            result = [r for r in records if not args.type or r["record_type"] == args.type]
        else:
            source = (ROOT / args.input).resolve()
            if ROOT.resolve() not in source.parents or not source.is_file():
                raise DecisionHistoryError("输入记录必须位于项目内")
            result = append_record(json.loads(source.read_text(encoding="utf-8")), ROOT, commit=args.commit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (DecisionHistoryError, OSError, ValueError) as exc:
        print("BLOCKED: " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
