"""登记用户明确确认的当前持仓，并同步当前月报展示。

只记录完整账户比例；未提供的成交日期、金额、份数、价格和费用保持为空。
不改变B0207、冻结模型、Agent建议或行情数据。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from .monthly_details import DetailsError, ORDER, render_details, local_path
from app.publication_io import capture, commit_files, require_clean


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, text: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_weights(items: list[str], cash: float) -> dict[str, float]:
    weights = {code: 0.0 for code in ORDER}
    seen = set()
    for item in items:
        code, separator, raw = item.partition("=")
        if not separator or code not in weights or code in seen:
            raise DetailsError("持仓参数必须为唯一的项目内代码=比例")
        value = float(raw)
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise DetailsError("持仓比例必须在0至100之间")
        weights[code] = value
        seen.add(code)
    if not math.isfinite(cash) or not 0 <= cash <= 100 or abs(sum(weights.values()) + cash - 100) > 1e-8:
        raise DetailsError("8只ETF与现金必须合计100%")
    return weights


def _next_id(prefix: str, day: str, existing: list[str]) -> str:
    stem = f"{prefix}-{day.replace('-', '')}-"
    numbers = [int(value[len(stem):]) for value in existing
               if value.startswith(stem) and value[len(stem):].isdigit()]
    return stem + f"{(max(numbers, default=0) + 1):03d}"


def record(*, confirmed_date: str, weights: dict[str, float], cash: float,
           instruction: str, commit: bool = False) -> dict:
    date.fromisoformat(confirmed_date)
    weights = _parse_weights([f'{code}={value}' for code, value in weights.items()], cash)
    if not isinstance(instruction, str) or not instruction.strip():
        raise DetailsError('必须提供用户确认来源')
    require_clean(ROOT)
    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    state_path = ROOT / "portfolio_state.json"
    history_path = ROOT / "app" / "portfolio_history" / "actual_holdings.json"
    decision_path = ROOT / "app" / "decision_history" / "decision_records.json"
    report_path = ROOT / "app" / "reports" / "monthly" / "latest.json"
    paths = [state_path, history_path, decision_path, report_path,
             ROOT / 'app/reports/monthly' / f'{confirmed_date}.json',
             ROOT / 'data_manifest.json', ROOT / 'app/features/latest.json']
    expected = capture(ROOT, [p.relative_to(ROOT).as_posix() for p in paths])
    state, history, decision, report = [json.loads(expected[p.relative_to(ROOT).as_posix()])
                                      for p in (state_path, history_path, decision_path, report_path)]
    if report.get("generated_date") != confirmed_date:
        raise DetailsError("当前月报日期与持仓确认日期不同；请先生成当日月报")
    if expected[f'app/reports/monthly/{confirmed_date}.json'] != expected['app/reports/monthly/latest.json']:
        raise DetailsError('日期月报与latest不一致，禁止登记')
    previous = deepcopy(state['last_confirmed_portfolio'])
    matching = [entry for entry in history['entries'] if entry.get('event_id') == previous.get('source_event_id')]
    if len(matching) != 1 or matching[0].get('current_snapshot') != previous:
        raise DetailsError('当前持仓与源事件不一致，保留原始账本')
    if (matching[0].get('user_instruction') == instruction
            and matching[0].get('confirmed_at') == confirmed_date
            and previous.get('weights_pct') == weights and previous.get('cash_pct') == cash):
        if report.get('last_confirmed_portfolio') != previous:
            raise DetailsError('重复确认但月报持仓不同，需先修复展示')
        return {'status': 'UNCHANGED', 'event_id': previous['source_event_id']}
    allocation_text = '；'.join(f'{code} {value:g}%' for code, value in weights.items() if value > 0) + f'；现金{cash:g}%'

    event_id = _next_id("APH", confirmed_date, [entry["event_id"] for entry in history["entries"]])
    record_id = _next_id("PORTFOLIO", confirmed_date, [entry["record_id"] for entry in decision["records"]])
    previous = deepcopy(state["last_confirmed_portfolio"])
    held = [code for code in ORDER if weights[code] > 0]
    snapshot = {
        "source": instruction,
        "confirmed_at": confirmed_date,
        "confirmation_recorded_at": now,
        "trade_date": None,
        "valuation_status": "WEIGHTS_CONFIRMED_AMOUNTS_SHARES_AND_FILLS_PENDING",
        "held_codes": held,
        "weights_pct": weights,
        "cash_pct": cash,
        "market_values_cny": {code: (None if weights[code] > 0 else 0) for code in ORDER},
        "cash_amount_cny": None,
        "total_market_value_cny": None,
        "positions": {
            code: {
                "shares": None if weights[code] > 0 else 0,
                "status": "held_confirmed" if weights[code] > 0 else "not_held",
                "weight_pct": weights[code],
                "note": "用户确认当前账户比例；未提供交易日期、金额、价格、份数和费用。",
            }
            for code in ORDER
        },
        "source_event_id": event_id,
        "valuation_basis": "用户确认账户比例；金额、份数、成本、成交日期和费用未提供，不进行估算。",
    }
    history_entry = {
        "event_id": event_id,
        "confirmed_at": confirmed_date,
        "recorded_at": now,
        "source": "用户本轮对话明确确认",
        "change_type": "full_actual_holdings_weight_snapshot",
        "user_instruction": instruction,
        "review_status": "REVIEWED_USER_CONFIRMED",
        "previous_event_id": previous.get("source_event_id"),
        "previous_snapshot": previous,
        "current_snapshot": snapshot,
        "change_summary": [
            '当前真实持仓更新为' + allocation_text + '；其余为0%。',
            "本记录只确认账户比例，不补造成交日期、金额、价格、份数、成本或费用。",
            "B0207周期基线、当前周观测和原Agent建议保持不变。",
        ],
        "decision_history_record_id": record_id,
    }
    decision_record = {
        "record_id": record_id,
        "record_type": "confirmed_portfolio",
        "confirmed_at": confirmed_date,
        "recorded_at": now,
        "weights_pct": weights,
        "cash_pct": cash,
        "status": "REVIEWED_USER_CONFIRMED",
        "source": instruction,
        "source_event_id": event_id,
        "details_status": "WEIGHTS_CONFIRMED_AMOUNTS_SHARES_AND_FILLS_PENDING",
        "note": allocation_text + '；金额、份数、成本和成交明细未提供。',
    }
    history["entries"].append(history_entry)
    decision["records"].append(decision_record)

    state["last_updated"] = confirmed_date
    state["decision_revision"] = event_id + '-confirmed'
    state["last_confirmed_portfolio"] = snapshot
    completed = state.setdefault("execution_status", {}).setdefault("completed_steps", [])
    marker = event_id + '_user_confirmed_actual_portfolio'
    if marker not in completed:
        completed.append(marker)
    state["execution_status"]["status"] = "current_weights_user_confirmed_details_pending"
    state["execution_status"]["fill_assumption"] = '只确认' + allocation_text + '的账户比例；不编造逐笔成交。'
    pending = state["execution_status"].get("pending_target")
    if isinstance(pending, dict):
        pending["status"] = "superseded_by_current_holdings_confirmation"
    state["execution_status"]["target_status"] = "superseded_by_current_holdings_confirmation"
    if isinstance(state.get("user_final_target"), dict):
        state["user_final_target"]["status"] = "superseded_by_current_holdings_confirmation"
    if isinstance(state.get("user_decision"), dict):
        state["user_decision"]["fill_status"] = "superseded_by_current_holdings_confirmation"
        state["user_decision"]["current_weights_confirmation"] = instruction
        state["user_decision"]["actual_holdings_event_id"] = event_id
    state["actual_portfolio_history"] = {
        "path": "app/portfolio_history/actual_holdings.json",
        "latest_event_id": event_id,
        "record_count": len(history["entries"]),
        "recording_rule": "每次用户明确确认实际持仓后追加完整快照；历史记录不得覆盖。",
    }
    state.setdefault("current_monthly_report", {}).update({
        "status": "current_confirmed_portfolio_redesigned",
        "report_date": confirmed_date,
        "latest_actual_portfolio_event_id": event_id,
        "notice": '当前真实持仓为' + allocation_text + '。',
    })

    report["last_confirmed_portfolio"] = snapshot
    report["portfolio_history_event_id"] = event_id
    report["display_updated_at"] = now
    report["holdings_confirmation_instruction"] = instruction + "；未提供成交明细，不估算。"
    report["portfolio_display_mode"] = "current_confirmed"
    report.setdefault("report_redesign", {})["current_portfolio_display"] = allocation_text
    if isinstance(report.get("user_final_target"), dict):
        report["user_final_target"]["status"] = "superseded_by_current_holdings_confirmation"
    if isinstance(report.get("user_decision"), dict):
        report["user_decision"]["fill_status"] = "superseded_by_current_holdings_confirmation"
        report["user_decision"]["current_weights_confirmation"] = instruction
        report["user_decision"]["actual_holdings_event_id"] = event_id
    report["target_execution_status"] = "superseded_by_current_holdings_confirmation"
    if report.get("agent_recommendation", {}).get("status") == "historical_advice_before_user_holdings_update":
        report["agent_recommendation"]["reference_note"] = (
            'Agent保护后列保留调整前原建议，起点见published_reference_portfolio。'
            '当前真实持仓已更新为' + allocation_text + '；原建议不是当前订单。'
        )

    html_path = local_path(ROOT, report['html_path'])
    html_name = html_path.relative_to(ROOT).as_posix()
    expected[html_name] = html_path.read_bytes()
    rendered = render_details(expected[html_name].decode('utf-8'), ROOT, report,
                              portfolio_override=snapshot)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    result = {
        "status": "CHECK_PASSED",
        "event_id": event_id,
        "decision_record_id": record_id,
        "weights_pct": weights,
        "cash_pct": cash,
        "html": str(html_path),
    }
    if not commit:
        return result

    encode = lambda value: (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    outputs = {history_path.relative_to(ROOT).as_posix(): encode(history),
               decision_path.relative_to(ROOT).as_posix(): encode(decision),
               state_path.relative_to(ROOT).as_posix(): encode(state),
               f'app/reports/monthly/{confirmed_date}.json': report_text.encode('utf-8'),
               report_path.relative_to(ROOT).as_posix(): report_text.encode('utf-8'),
               html_name: rendered.encode('utf-8')}
    commit_files(ROOT, outputs, expected)
    result["status"] = "RECORDED_AND_REPORT_UPDATED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmed-date", required=True)
    parser.add_argument("--weight", action="append", default=[])
    parser.add_argument("--cash", required=True, type=float)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    try:
        weights = _parse_weights(args.weight, args.cash)
        print(json.dumps(record(confirmed_date=args.confirmed_date, weights=weights, cash=args.cash,
                                instruction=args.instruction, commit=args.commit),
                         ensure_ascii=False, indent=2))
        return 0
    except (DetailsError, OSError, ValueError, KeyError) as exc:
        print("BLOCKED: " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
