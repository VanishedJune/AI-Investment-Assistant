"""Explicitly refresh the current non-cycle report's eight ETF detail rows.

python model_iteration/scripts/refresh_monthly_details.py --check
python model_iteration/scripts/refresh_monthly_details.py --commit

Requires an already reviewed latest-batch report. Does not generate analysis,
infer models, fetch data, publish a new cycle, or change confirmed holdings.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

# The adjacent calendar.py is a model module, not Python's stdlib calendar.
# Pandas imported later by the fund slot must not see it as the stdlib module.
sys.path[:] = [entry for entry in sys.path
               if Path(entry or '.').resolve() != Path(__file__).resolve().parent]
import calendar  # noqa: E402
import pandas  # noqa: E402

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.monthly_details import DetailsError, local_path, render_details

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.publication_io import capture, commit_files, require_clean


def refresh(root=ROOT, *, commit=False):
    root = Path(root).resolve()
    require_clean(root)
    expected = capture(root, ['portfolio_state.json', 'app/reports/monthly/latest.json',
        'app/decision_history/decision_records.json', 'app/portfolio_history/actual_holdings.json',
        'data_manifest.json', 'app/features/latest.json'])
    report_path = local_path(root, 'app/reports/monthly/latest.json')
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get('analysis_status') == 'pending_after_instrument_replacement':
        raise DetailsError('旧报告尚未接入当前冻结模型预测及同批次Agent分析；请先发布已复核监控快照，不需要重新训练或推进迭代')
    if report.get('schema_version') != 'monthly-report-v1' or report.get('decision_type') != 'user_confirmed_portfolio_snapshot_non_cycle':
        raise DetailsError('此入口只更新当前已确认的非周期监控报告；正式周期使用原发布器')
    user = report.get('user_decision', {})
    if user.get('confirmation_source') != 'explicit_user_instruction' or user.get('status') not in ('modified_by_user', 'confirmed_agent_recommendation'):
        raise DetailsError('没有用户确认，禁止更新月报')
    html_path = local_path(root, report['html_path'])
    dated_path = local_path(root, f"app/reports/monthly/{report['generated_date']}.json")
    if dated_path.read_bytes() != report_bytes:
        raise DetailsError('当前日期JSON与latest不一致，停止更新')
    original = html_path.read_bytes()
    expected[str(html_path.relative_to(root))] = original
    expected[str(dated_path.relative_to(root))] = report_bytes
    rendered = render_details(original.decode('utf-8'), root, report).encode('utf-8')
    result = {'status': 'CHECK_PASSED', 'data_as_of': report['data_as_of'],
              'updated_rows': 8, 'html': str(html_path), 'changed': rendered != original,
              'note': '本地最新已收盘共同日；未联网、未重算基线、未训练、未修改持仓'}
    if not commit or rendered == original:
        return {**result, 'status': 'UNCHANGED' if commit else 'CHECK_PASSED'}
    commit_files(root, {str(html_path.relative_to(root)): rendered}, expected)
    return {**result, 'status': 'UPDATED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--commit', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(refresh(commit=args.commit), ensure_ascii=False, indent=2))
        return 0
    except (DetailsError, OSError, ValueError, KeyError) as exc:
        print('BLOCKED: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
