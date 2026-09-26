"""从已复核的同批次月报生成“当前真实持仓”展示版。

本入口不联网、不训练、不推理、不重算B0207周期基线，也不修改真实
持仓。它只把已经复核的ETF分析、021528独立净值分析和当前持仓台账
重新组合为新的月报快照。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo

from .monthly_details import DetailsError, ORDER, render_details, local_path
from .priority_slots import resolve_growth_slot
from .active_holdings import ActiveHoldingsError, active_portfolio_view
from app.publication_io import capture, commit_files, require_clean


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
ICON_PATH = ROOT / "app" / "assets" / "ai-research-owl-avatar-v1.ico"


def _shortcut_properties(link_path: Path) -> dict:
    env = os.environ.copy()
    env["MONTHLY_LINK"] = str(link_path)
    command = (
        "$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        "$w=New-Object -ComObject WScript.Shell;"
        "$s=$w.CreateShortcut($env:MONTHLY_LINK);"
        "[pscustomobject]@{TargetPath=$s.TargetPath;IconLocation=$s.IconLocation}|"
        "ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        [shutil.which("powershell.exe") or "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    if completed.returncode != 0:
        raise DetailsError("读取快捷入口失败：" + completed.stderr.strip())
    try:
        return json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise DetailsError("快捷入口属性无法解析") from exc


def create_shortcut(link_path: Path, target_path: Path) -> None:
    if not ICON_PATH.is_file():
        raise DetailsError("月报快捷入口图标缺失")
    env = os.environ.copy()
    env.update(MONTHLY_LINK=str(link_path), MONTHLY_TARGET=str(target_path),
               MONTHLY_ICON=str(ICON_PATH), MONTHLY_WORKDIR=str(SCRIPTS_DIR))
    command = (
        "$w=New-Object -ComObject WScript.Shell;"
        "$s=$w.CreateShortcut($env:MONTHLY_LINK);"
        "$s.TargetPath=$env:MONTHLY_TARGET;"
        "$s.WorkingDirectory=$env:MONTHLY_WORKDIR;"
        "$s.IconLocation=$env:MONTHLY_ICON+',0';$s.Save()"
    )
    completed = subprocess.run(
        [shutil.which("powershell.exe") or "powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    if completed.returncode != 0 or not link_path.is_file():
        raise DetailsError("创建快捷入口失败：" + completed.stderr.strip())


def verify_shortcut(link_path: Path, target_path: Path) -> None:
    props = _shortcut_properties(link_path)
    if Path(str(props.get("TargetPath") or "")).resolve() != target_path.resolve():
        raise DetailsError("快捷入口目标不正确")
    icon = str(props.get("IconLocation") or "").split(",", 1)[0]
    if Path(icon).resolve() != ICON_PATH.resolve():
        raise DetailsError("快捷入口图标不正确")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, text: str, *, refuse_existing: bool = False) -> None:
    if refuse_existing and path.exists():
        raise DetailsError(f"目标文件已存在，拒绝覆盖：{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _complete_weights(portfolio: dict) -> None:
    weights = portfolio.get("weights_pct") or {}
    values = [weights.get(code) for code in ORDER] + [portfolio.get("cash_pct")]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100
            for value in values) or abs(sum(values) - 100) > 1e-8):
        raise DetailsError("当前真实持仓必须完整覆盖8只ETF及现金并合计100%")


def _portfolio_display(portfolio: dict) -> str:
    names = {
        "159915": "创业板", "021528": "财通成长优选混合C", "517520": "黄金股", "516150": "稀土", "159622": "创新药",
        "159611": "电力", "515220": "煤炭", "512800": "银行", "512690": "酒",
    }
    weights = dict(portfolio["weights_pct"])
    weights.update(portfolio.get('growth_slot_member_weights_pct') or {})
    parts = [f"{names[code]}{weights[code]:g}%" for code in ('159915', '021528', *ORDER[1:]) if weights.get(code, 0) > 0]
    parts.append(f"现金{portfolio['cash_pct']:g}%")
    return "；".join(parts)


def build(report_date: str, source_date: str, *, commit: bool = False,
          replace_existing: bool = False) -> dict:
    require_clean(ROOT)
    generated = date.fromisoformat(report_date)
    date.fromisoformat(source_date)
    # Capture before reading/rendering. Capturing only immediately before commit
    # would accept a newer user holding and then overwrite it with stale state.
    expected = capture(ROOT, ['portfolio_state.json', 'data_manifest.json',
        'app/features/latest.json', 'app/decision_history/decision_records.json',
        'app/portfolio_history/actual_holdings.json',
        f'app/reports/monthly/{source_date}.json',
        f'app/reports/monthly/{report_date}.json', 'app/reports/monthly/latest.json',
        f'scripts/投资决策月报{report_date}.html', f'投资决策月报{report_date}.lnk'])
    if not replace_existing and any(expected[name] is not None for name in (
            f'app/reports/monthly/{report_date}.json', f'scripts/投资决策月报{report_date}.html')):
        raise DetailsError('目标报告已存在；未授权替换，不得覆盖历史报告')
    source_path = ROOT / "app" / "reports" / "monthly" / f"{source_date}.json"
    if not source_path.is_file():
        raise DetailsError(f"来源月报不存在：{source_path.name}")
    source_report = _read_json(source_path)
    if source_report.get("generated_date") != source_date:
        raise DetailsError("来源月报日期字段与文件名不一致")
    if generated < date.fromisoformat(source_report["data_as_of"]):
        raise DetailsError("报告生成日不能早于行情截止日")

    state_path = ROOT / "portfolio_state.json"
    state = _read_json(state_path)
    try:
        portfolio = active_portfolio_view(state["last_confirmed_portfolio"], ORDER)
    except ActiveHoldingsError as exc:
        raise DetailsError(str(exc)) from exc
    _complete_weights(portfolio)
    portfolio_display = _portfolio_display(portfolio)
    try:
        source_portfolio = active_portfolio_view(source_report.get("last_confirmed_portfolio") or {}, ORDER)
    except ActiveHoldingsError as exc:
        raise DetailsError(str(exc)) from exc
    if source_portfolio != portfolio:
        raise DetailsError("来源月报与当前真实持仓不同，须先完成持仓同步")

    report = deepcopy(source_report)
    report["generated_date"] = report_date
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    report["display_updated_at"] = report["generated_at"]
    report["html_path"] = f"scripts/投资决策月报{report_date}.html"
    report["previous_report_path"] = (
        source_report.get("previous_report_path")
        if report_date == source_date
        else f"app/reports/monthly/{source_date}.json"
    )
    report["analysis_reference_date"] = source_report.get("analysis_reference_date") or source_date
    report["analysis_reuse_mode"] = "same_batch_display_redesign"
    report["analysis_status"] = "reviewed_same_batch_with_growth_priority_slot"
    report["portfolio_display_mode"] = "current_confirmed"
    report["last_confirmed_portfolio"] = portfolio
    report["priority_slot_analysis"] = resolve_growth_slot(
        ROOT,
        source_report["agent_recommendation"]["instruments"][0]["quantitative_evidence"]["weekly"]["as_of"],
        source_report["mechanical_baseline"]["champion_positions"]["159915"],
        available_as_of=report_date,
    )
    report["report_redesign"] = {
        "version": "growth-priority-slot-v5-complete-holdings-summary",
        "first_rows": ["159915", "021528"],
        "shared_rank_rule": "两条分析行共用序号01，其他列保持独立",
        "instrument_cell_rule": "产品名称加代码与收盘价或基金净值，与其他ETF格式一致",
        "matrix_holdings_basis": "agent_recommendation.published_reference_portfolio",
        "page_holdings_basis": "portfolio_state.json.last_confirmed_portfolio",
        "summary_card_rule": "每只当前持仓ETF及现金分别显示账户占比",
        "allocation_rule": "159915与021528分别分析，合计只占第一优先级一个槽位",
        "portfolio_basis": "portfolio_state.json.last_confirmed_portfolio",
        "current_portfolio_display": portfolio_display,
        "unfilled_target_display": "retained_in_json_not_presented_as_actual_holdings",
    }
    report["holdings_confirmation_instruction"] = (
        f"本页使用当前已确认真实持仓：{portfolio_display}；金额、份数、成交价和费用未提供，不估算。"
    )

    source_html_path = local_path(ROOT, source_report["html_path"])
    source_name = str(source_html_path.relative_to(ROOT))
    expected.setdefault(source_name, source_html_path.read_bytes())
    source_html = expected[source_name].decode('utf-8')
    rendered = render_details(source_html, ROOT, report)
    rendered, count = re.subn(
        r"<title>[\s\S]*?</title>",
        f"<title>投资决策月报｜{report_date}｜当前真实持仓</title>",
        rendered,
        count=1,
    )
    if count != 1:
        raise DetailsError("月报缺少唯一标题")

    dated_json = ROOT / "app" / "reports" / "monthly" / f"{report_date}.json"
    latest_json = ROOT / "app" / "reports" / "monthly" / "latest.json"
    html_path = ROOT / report["html_path"]
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    result = {
        "status": "CHECK_PASSED",
        "report_date": report_date,
        "data_as_of": report["data_as_of"],
        "fund_nav_as_of": report["priority_slot_analysis"]["fund_analysis"]["nav_as_of"],
        "portfolio": {"weights_pct": portfolio["weights_pct"], "cash_pct": portfolio["cash_pct"]},
        "html": str(html_path),
        "json": str(dated_json),
    }
    if not commit:
        return result

    current_report = deepcopy(state.get("current_monthly_report") or {})
    current_report.update({
        "status": "current_confirmed_portfolio_redesigned",
        "report_date": report_date,
        "html": report["html_path"],
        "json": "app/reports/monthly/latest.json",
        "shortcut": f"投资决策月报{report_date}.lnk",
        "data_as_of": report["data_as_of"],
        "data_run_id": report["data_run_id"],
        "latest_actual_portfolio_event_id": portfolio.get("source_event_id"),
        "notice": f"第一优先级已拆为两条完整记录；页面按当前真实持仓{portfolio_display}展示。",
        "formal_publication_status": "non_cycle_monitoring_only",
    })
    state["current_monthly_report"] = current_report
    state["last_updated"] = report_date
    state_text = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    if replace_existing and dated_json.exists():
        existing = _read_json(dated_json)
        if (existing.get("generated_date") != report_date
                or (existing.get("report_redesign") or {}).get("version") not in {
                    "growth-priority-slot-v1", "growth-priority-slot-v2-split-rows",
                    "growth-priority-slot-v3-shared-rank", "growth-priority-slot-v4-analysis-holdings",
                    "growth-priority-slot-v5-complete-holdings-summary",
                }):
            raise DetailsError("只允许重绘同日同版式月报，拒绝覆盖其他报告")
    shortcut_name = f"投资决策月报{report_date}.lnk"
    shortcut_path = ROOT / shortcut_name
    with tempfile.TemporaryDirectory(prefix=".current-report-shortcut-", dir=ROOT) as temporary_dir:
        staged_shortcut = Path(temporary_dir) / shortcut_name
        create_shortcut(staged_shortcut, html_path)
        verify_shortcut(staged_shortcut, html_path)
        shortcut_bytes = staged_shortcut.read_bytes()

    outputs: dict[str, bytes | None] = {
        str(dated_json.relative_to(ROOT)): payload.encode("utf-8"),
        str(latest_json.relative_to(ROOT)): payload.encode("utf-8"),
        str(html_path.relative_to(ROOT)): rendered.encode("utf-8"),
        str(state_path.relative_to(ROOT)): state_text.encode("utf-8"),
        shortcut_name: shortcut_bytes,
    }
    archive_root = ROOT / "backup" / "monthly_shortcuts" / report_date
    for old_shortcut in sorted(ROOT.glob("投资决策月报*.lnk")):
        if old_shortcut.name == shortcut_name:
            continue
        archive_path = archive_root / old_shortcut.name
        old_bytes = old_shortcut.read_bytes()
        if archive_path.exists() and archive_path.read_bytes() != old_bytes:
            raise DetailsError(f"快捷入口归档目标已存在且内容不同：{archive_path.name}")
        if not archive_path.exists():
            outputs[str(archive_path.relative_to(ROOT))] = old_bytes
        outputs[str(old_shortcut.relative_to(ROOT))] = None

    # Additional archive destinations must also remain unchanged under lock.
    for name, value in capture(ROOT, outputs).items():
        expected.setdefault(name, value)
    commit_files(ROOT, outputs, expected)
    result["status"] = "GENERATED"
    result["shortcut"] = str(shortcut_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--source-date", required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.report_date, args.source_date, commit=args.commit,
                               replace_existing=args.replace_existing), ensure_ascii=False, indent=2))
        return 0
    except (DetailsError, OSError, ValueError, KeyError) as exc:
        print("BLOCKED: " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
