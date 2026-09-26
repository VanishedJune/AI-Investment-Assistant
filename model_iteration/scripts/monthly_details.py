"""Latest-session ETF detail contract and deterministic HTML rendering.

No downloads, model inference, training, portfolio mutation or Agent decisions.
An old analysis cannot be relabelled as current by updating the report date.
"""
from __future__ import annotations

import csv
from datetime import date
from html import escape
import json
import math
from pathlib import Path
import re

from .agent_multi_parameter_gate import validate_quantitative_evidence
from .agent_v70_protection import validate_plan, ProtectionError
from .decision_history import DecisionHistoryError, layers_for_analysis, load_ledger
from .priority_slots import PrioritySlotError, resolve_growth_slot
from .active_holdings import ActiveHoldingsError, active_portfolio_view

ORDER = tuple(json.loads((Path(__file__).resolve().parents[1] / 'configs/investment_priority.json').read_text(encoding='utf-8'))['priority'])


class DetailsError(ValueError):
    pass


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise DetailsError('来源不存在或路径越界: ' + str(relative))
    return path


def read_json(root, relative):
    return json.loads(local_path(root, relative).read_text(encoding='utf-8'))


def latest_context(root, report, historical_display=False):
    if report.get('analysis_status') == 'pending_after_instrument_replacement':
        raise DetailsError('这是标的替换阶段的旧候选；须先读取517520自身全周期迭代台账并完成同批次Agent复核')
    replacement = report.get('instrument_replacement') or {}
    if replacement.get('new_code') == '517520':
        transfer = replacement.get('model_transfer') or {}
        if (replacement.get('analysis_status') != 'independent_full_cycle_completed'
                or transfer.get('performance_status') != 'FULL_CYCLE_COMPLETED_NO_PROMOTION'
                or transfer.get('target_trained_anchor_index') is None):
            raise DetailsError('报告仍引用517520旧迁移模型状态，禁止发布或刷新')
    if historical_display:
        cutoff, run = report.get('data_as_of'), report.get('data_run_id')
        if not cutoff or not run:
            raise DetailsError('历史月报缺少原始截止日或批次')
        features = read_json(root, f'app/features/{run}.json')
        if features.get('as_of') != cutoff or features.get('data_run_id') != run:
            raise DetailsError('历史月报的冻结日周参数不存在或批次不一致')
        rows = report.get('agent_recommendation', {}).get('instruments', [])
        if [row.get('code') for row in rows] != list(ORDER):
            raise DetailsError('历史月报没有完整保留8只ETF分析')
        prices = {}
        for row in rows:
            daily = row.get('forward_evidence', {}).get('daily', [])
            if (not daily or daily[-1].get('date') != cutoff
                    or row.get('analysis_data_as_of') != cutoff
                    or row.get('analysis_data_run_id') != run):
                raise DetailsError(row.get('code', 'ETF') + '历史展示证据与月报批次不一致')
            price = float(daily[-1]['raw_close'])
            if not math.isfinite(price) or price <= 0:
                raise DetailsError(row['code'] + '历史展示价格非法')
            prices[row['code']] = price
        manifest = {
            'as_of': cutoff,
            'run_id': run,
            'market_status': 'closed',
            'quality_status': 'historical_report_snapshot',
            'latest_dates': {code: cutoff for code in ORDER},
        }
        return manifest, features['instruments'], prices
    manifest = read_json(root, 'data_manifest.json')
    features = read_json(root, 'app/features/latest.json')
    cutoff, run = manifest.get('as_of'), manifest.get('run_id')
    if not cutoff or date.fromisoformat(cutoff) > date.today() or not run:
        raise DetailsError('最新共同截止日/批次非法或包含未来日期')
    if manifest.get('market_status') != 'closed':
        raise DetailsError('行情尚未正式收盘，禁止刷新详情')
    if manifest.get('quality_status') not in ('passed', 'passed_with_warnings'):
        raise DetailsError('数据质量未通过，保留原报告')
    if any(manifest.get('latest_dates', {}).get(c) != cutoff for c in ORDER):
        raise DetailsError('8只ETF没有一致的最新共同截止日')
    for label, payload in (('日周参数', features), ('Agent报告', report)):
        actual = payload.get('as_of') if label == '日周参数' else payload.get('data_as_of')
        if actual != cutoff or payload.get('data_run_id') != run:
            raise DetailsError(label + '过期；需同批次最新参数及Agent复核，禁止沿用旧分析')
    instruments = features.get('instruments', {})
    if set(instruments) != set(ORDER):
        raise DetailsError('最新参数未完整覆盖8只ETF')
    prices = {}
    for code in ORDER:
        d, w = instruments[code]['daily'], instruments[code]['weekly']
        if d.get('as_of') != cutoff or d.get('is_partial') is not False:
            raise DetailsError(code + '日K尚未完成或过期')
        if not w.get('as_of') or w['as_of'] > cutoff or w.get('is_partial') is not False:
            raise DetailsError(code + '周K不是已完成且当时可得的完整周')
        sources = [x for x in manifest.get('files', [])
                   if Path(x['path']).name.startswith(code + '_') and x['path'].endswith('_日线.csv')]
        if len(sources) != 1:
            raise DetailsError(code + '缺少唯一正式日线源')
        source = sources[0]
        if source.get('end') != cutoff or source.get('run_id') != run:
            raise DetailsError(code + '日线清单截止日/批次不一致')
        with local_path(root, source['path']).open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != source['rows'] or not rows or rows[-1]['date'] != cutoff:
            raise DetailsError(code + '实际日线行数/截止日不一致')
        if any(x['code'] != code or x['run_id'] != run for x in rows):
            raise DetailsError(code + '实际日线代码/批次不一致')
        if any(a['date'] >= b['date'] for a, b in zip(rows, rows[1:])):
            raise DetailsError(code + '日线日期重复或乱序')
        prices[code] = float(rows[-1]['raw_close'])
        if not math.isfinite(prices[code]) or prices[code] <= 0:
            raise DetailsError(code + '原始收盘价非法')
        if abs(float(rows[-1]['adj_close']) - d['adj_close']) > 1e-8:
            raise DetailsError(code + '参数复权价与实际日线不一致')
    return manifest, instruments, prices


def checked_rows(root, report, baseline_path=None, portfolio_override=None,
                 historical_display=False):
    manifest, features, prices = latest_context(root, report, historical_display)
    formal = report.get('schema_version') == 'monthly-report-v2'
    rows = (report.get('decision_snapshot', {}).get('strategies', []) if formal
            else report.get('agent_recommendation', {}).get('instruments', []))
    if [r.get('code') for r in rows] != list(ORDER):
        raise DetailsError('详情必须覆盖8只ETF且保持固定顺序')
    if formal:
        baseline = read_json(root, baseline_path or report['policy_snapshot_path'])
        weights = baseline['mechanical_baseline_weights_pct']
        positions = {c: baseline['champions'][c]['position'] for c in ORDER}
    else:
        baseline = read_json(root, report['monitoring_baseline_path'])
        weights, positions = baseline['b0207_weights_pct'], baseline['champion_positions']
        if report.get('mechanical_baseline') != baseline:
            raise DetailsError('报告机械参考与已存在的源文件不同；不得重算或自动替换')
    if baseline.get('data_as_of') != manifest['as_of'] or baseline.get('data_run_id') != manifest['run_id']:
        raise DetailsError('机械参考不是最新共同日期/批次；禁止用旧参考发布')
    try:
        portfolio = active_portfolio_view(
            portfolio_override or read_json(root, 'portfolio_state.json')['last_confirmed_portfolio'],
            ORDER,
        )
        report_portfolio = active_portfolio_view(report.get('last_confirmed_portfolio') or {}, ORDER)
    except ActiveHoldingsError as exc:
        raise DetailsError(str(exc)) from exc
    if report_portfolio != portfolio:
        raise DetailsError('报告真实持仓过期；需同步用户确认记录')
    result = []
    for row in rows:
        c = row['code']
        evidence = row.get('forward_evidence', {})
        if evidence.get('data_as_of') != manifest['as_of'] or evidence.get('data_run_id') != manifest['run_id']:
            raise DetailsError(c + 'Agent连续证据过期；必须先重新分析')
        daily_points, weekly_points = evidence.get('daily', []), evidence.get('weekly', [])
        if not daily_points or not weekly_points or daily_points[-1].get('date') != manifest['as_of'] or weekly_points[-1].get('date') != features[c]['weekly']['as_of']:
            raise DetailsError(c + 'Agent连续证据没有覆盖当前日K/完整周K末端')
        if abs(float(daily_points[-1]['raw_close']) - prices[c]) > 1e-8:
            raise DetailsError(c + 'Agent参考价与最新实际原始收盘价不同，需重新复核')
        if not row.get('agent_direction'):
            raise DetailsError(c + '缺少Agent判断，脚本不能代填')
        if formal:
            validate_quantitative_evidence(row.get('quantitative_evidence'), features[c], c)
            reason = row.get('multi_parameter_conclusion', '')
        else:
            if row.get('quantitative_evidence') != features[c]:
                raise DetailsError(c + 'Agent分析所绑定的参数已过期或被替换')
            if row.get('analysis_data_as_of') != manifest['as_of'] or row.get('analysis_data_run_id') != manifest['run_id']:
                raise DetailsError(c + '缺少同批次Agent复核日期，不得自动续用旧结论')
            reason = row.get('agent_reason', '')
        if not str(reason).strip():
            raise DetailsError(c + '缺少Agent综合依据')
        result.append((row, features[c], prices[c], weights[c], positions[c], reason))
    return manifest, portfolio, result


def percentage_text(value):
    """Account weights only; unknown values must never become zero or share ratios."""
    if value is None:
        return '—'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise DetailsError('账户份额必须为0至100的有限数值或空值')
    return f'{value:g}%'


def trend_text(label, value):
    """Color the authored direction only; never infer or change its meaning."""
    direction, separator, detail = str(value).partition('｜')
    if direction.startswith('看多'):
        kind, color = 'bullish', '#c83c4b'
    elif direction.startswith('看空'):
        kind, color = 'bearish', '#087f6d'
    elif direction.startswith('震荡') and any(word in direction for word in ('减弱', '偏弱', '走弱')):
        kind, color = 'weakening', '#a65f00'
    elif direction.startswith('震荡') and any(word in direction for word in ('加强', '增强', '修复', '偏强', '走强')):
        kind, color = 'strengthening', '#2463ad'
    else:
        kind, color = 'neutral', '#64748b'
    return (escape(label) + f'<span data-trend="{kind}" style="color:{color};font-weight:750">'
            + escape(direction) + '</span>' + escape(separator + detail))


def allocation_cell(weight, trend, *, column=None, tooltip=''):
    marker = f' data-column="{escape(column)}"' if column else ''
    direction, _, _ = str(trend).partition('｜')
    return (f'<td class="decision"{marker} style="white-space:normal;vertical-align:middle" title="{escape(tooltip)}">'
             '<div data-allocation-headline="true" style="display:flex;align-items:baseline;gap:10px;white-space:nowrap;font-size:18px;line-height:1.5">'
             f'<b style="font-size:inherit">{percentage_text(weight)}</b>'
             + trend_text('', direction) + '</div></td>')


def refresh_quadrant_chart(source, rows, summary, cutoff, decision_date):
    """Rebind the fixed SVG point layer to the same batch as the matrix."""
    colors = {
        'positive': ('#f4a582', '#d6604d', '#b2182b'),
        'negative': ('#92c5de', '#4393c3', '#2166ac'),
    }

    def second_color(value):
        value = float(value)
        if abs(value) < 0.05:
            return '#d7dde5'
        level = 0 if abs(value) < 0.25 else 1 if abs(value) < 0.75 else 2
        return colors['positive' if value > 0 else 'negative'][level]

    offsets = {
        '159915': (-170, -25), '517520': (25, 42), '516150': (-170, 12),
        '159622': (-165, 42), '159611': (-165, 15), '515220': (25, -12),
        '512800': (22, -20), '512690': (25, 38), '518600': (25, 42),
        '159941': (-165, 15), '512010': (25, 15),
    }
    short_names = {
        '159915': '创业板', '517520': '黄金股', '516150': '稀土', '159622': '创新药',
        '159611': '电力', '515220': '煤炭', '512800': '银行', '512690': '酒',
        '518600': '黄金', '159941': '纳指', '512010': '医药',
    }
    leaders, points = [], []
    for row, feature, _raw_close, _weight, _position, _reason in rows:
        code = row['code']
        daily, weekly = feature['daily'], feature['weekly']
        x_value = max(-3.0, min(3.0, float(weekly['dif_first_normalized'])))
        y_value = max(-3.0, min(3.0, float(daily['dif_first_normalized'])))
        x = 520.0 + x_value * (400.0 / 3.0)
        y = 300.0 - y_value * (250.0 / 3.0)
        dx, dy = offsets.get(code, (25, 15))
        tx, ty = x + dx, y + dy
        leader_x = tx + (105 if dx < 0 else -10)
        leader_y = ty + (3 if dy >= 0 else 10)
        leaders.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{leader_x:.2f}" y2="{leader_y:.2f}"></line>')
        agent_label = ((summary.get('instruments') or {}).get(code) or {}).get('agent', row['agent_direction'])
        title = (f'{feature["name"]}｜周一 {weekly["dif_first_normalized"]:+.6f} / '
                 f'日一 {daily["dif_first_normalized"]:+.6f}｜日二 {daily["dif_second_normalized"]:+.6f} / '
                 f'周二 {weekly["dif_second_normalized"]:+.6f}｜截止{cutoff}｜Agent{agent_label}')
        label = f'{short_names.get(code, code)} {weekly["dif_first_normalized"]:+.2f} / {daily["dif_first_normalized"]:+.2f}'
        points.append(
            '<g><title>' + escape(title) + '</title>'
            f'<path d="M {x:.2f} {y-10:.2f} A 10 10 0 0 0 {x:.2f} {y+10:.2f} L {x:.2f} {y-10:.2f} Z" fill="{second_color(daily["dif_second_normalized"])}"></path>'
            f'<path d="M {x:.2f} {y-10:.2f} A 10 10 0 0 1 {x:.2f} {y+10:.2f} L {x:.2f} {y-10:.2f} Z" fill="{second_color(weekly["dif_second_normalized"])}"></path>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="10" fill="none" stroke="#fff" stroke-width="3"></circle>'
            f'<line x1="{x:.2f}" y1="{y-9.2:.2f}" x2="{x:.2f}" y2="{y+9.2:.2f}" stroke="#fff" stroke-width="1.5"></line>'
            f'<text x="{tx:.2f}" y="{ty:.2f}" fill="#172a46">{escape(label)}</text></g>'
        )
    replacement = ('<g stroke="#8190a5" stroke-width="1.5">' + ''.join(leaders) + '</g>\n'
                   '          <g font-family="Microsoft YaHei" font-size="15" font-weight="700">'
                   + ''.join(points) + '</g>\n        </svg>')
    pattern = (r'<g stroke="#8190a5" stroke-width="1\.5">[\s\S]*?'
               r'<g font-family="Microsoft YaHei" font-size="15" font-weight="700">[\s\S]*?</g>\s*</svg>')
    source, count = re.subn(pattern, lambda _: replacement, source, count=1)
    if count != 1:
        raise DetailsError('月报象限图点位层无法唯一刷新')
    source = re.sub(
        r'图、当前周观测及Agent分析统一使用[^<]*',
        f'图、当前周观测及Agent分析统一使用{cutoff}收盘；周期基线绑定{decision_date}名义决策日。',
        source,
        count=1,
    )
    return source


def protection_policy_text(analysis, *, historical=False):
    """Describe the policy attached to the displayed recommendation."""
    policy = str((analysis or {}).get('protection_policy') or '')
    if historical or policy != 'agent-v70-bearish-protection-with-holding-peak-stop-reinvest-prior-holdings':
        return ('本列保留2026-09-13前的历史保护建议：V70参考减持70%，'
                '当时卖出款转现金；仅作历史展示，不代表现行规则或当前订单。')
    return ('Agent V70看空参考减持70%；持有期高点回撤严格超过10%时建议清仓100%；'
            '保护卖出款按保护前其他仍持有ETF的市值比例回投；不回买被保护ETF，'
            '不新建此前未持有ETF；没有合格接收标的时才留现金；不自动执行。')


def holding_text(portfolio, code):
    weight = (portfolio.get('weights_pct') or {}).get(code)
    if weight is not None:
        return f'当前{weight:g}%'
    if code not in portfolio.get('held_codes', []):
        return '持仓待核实'
    conversion = portfolio.get('conversion', {})
    if code == conversion.get('from_code') and conversion.get('retain_fraction_of_previous_shares') is not None:
        return f"剩余原份额{conversion['retain_fraction_of_previous_shares'] * 100:g}% / 账户占比待补"
    if code == '518600' and portfolio.get('conversion', {}).get('gold_shares_unchanged'):
        return '份额不动'
    fraction = portfolio.get('conversion', {}).get('allocation_of_proceeds', {}).get(code)
    return f'转入资金{fraction * 100:g}%' if fraction is not None else '已持有/比例待补'


def with_agent_column(source):
    """User-authorized one-column upgrade; old historical HTML stays on disk."""
    marker = 'data-column="agent-direction"'
    if marker in source:
        if source.count(marker) not in (9, 10, 11):
            raise DetailsError('Agent列必须覆盖表头、全部产品分析行及可选现金格')
        return source
    if source.count('<th>B0207方向</th>') != 1:
        raise DetailsError('未找到唯一B0207方向表头，停止升级表格')
    source = source.replace('<th>B0207方向</th>',
                            '<th>B0207方向</th><th data-column="agent-direction">Agent判断</th>')
    body = re.search(r'<tbody>([\s\S]*?)</tbody>', source)
    if not body:
        raise DetailsError('缺少详情表体')
    rows = re.findall(r'<tr\b[^>]*>[\s\S]*?</tr>', body[1])
    if len(rows) != 8:
        raise DetailsError('详情必须恰好8行')
    upgraded = []
    for row in rows:
        cells = list(re.finditer(r'<td\b[^>]*>[\s\S]*?</td>', row))
        if len(cells) != 9:
            raise DetailsError('旧详情必须为9列，禁止扩大排版变更范围')
        offset = cells[4].end()
        upgraded.append(row[:offset] + '<td class="signal neutral" data-column="agent-direction"><b>待复核</b></td>' + row[offset:])
    return source[:body.start(1)] + ''.join(upgraded) + source[body.end(1):]


def with_four_decision_columns(source):
    """Keep the technical matrix and expand only its decision-layer columns."""
    source = with_agent_column(source)
    wanted = ('data-column="cycle-baseline"', 'data-column="weekly-observation"',
              'data-column="agent-direction"')
    if all(marker in source for marker in wanted):
        for old in ('Agent建议', 'Agent判断', 'Agent'):
            source = source.replace(
                f'<th data-column="agent-direction">{old}</th>',
                '<th data-column="agent-direction">Agent保护后</th>',
            )
        return source
    patterns = (
        '<th>B0207</th><th data-column="agent-direction">Agent</th><th>当前持仓</th>',
        '<th>B0207方向</th><th data-column="agent-direction">Agent判断</th><th>B0207参考 / 当前持仓</th>',
        '<th>B0207方向</th><th data-column="agent-direction">Agent判断</th><th>当前持仓</th>',
        '<th>B0207方向</th><th data-column="agent-direction">Agent判断</th><th>上一目标 / 当前持仓</th>',
        '<th>B0207方向</th><th data-column="agent-direction">Agent保护判断</th><th>当前持仓</th>',
    )
    replacement = ('<th data-column="cycle-baseline">B0207周期基线</th>'
                   '<th data-column="weekly-observation">B0207当前周观测</th>'
                   '<th>当前真实持仓</th>'
                   '<th data-column="agent-direction">Agent保护后</th>')
    matches = [pattern for pattern in patterns if pattern in source]
    if len(matches) != 1:
        raise DetailsError('无法唯一识别现有B0207、Agent和当前持仓表头')
    return source.replace(matches[0], replacement)


PRESENTATION_CSS = """
/* monthly-presentation:start — user-approved desktop layout */
body{background-image:none;font-variant-numeric:tabular-nums}
.head{gap:24px;padding:26px 28px}.brand{gap:18px}
h1{font-size:34px;letter-spacing:.025em}.subtitle{font-size:16px}
.avatar-shell{width:88px;height:88px;flex-basis:88px;border-radius:24px}
.avatar-shell img{border-radius:20px}
.stamp{min-width:0;flex-shrink:0;text-align:left;font-family:inherit;font-size:14px;line-height:1.85;padding:12px 18px}
.stamp b{display:inline;font-size:15px;font-weight:600;margin:0}
section{box-shadow:0 6px 22px rgba(19,38,64,.06);border-radius:14px}
.section-head{padding:18px 24px}.section-body{padding:24px}
h2{font-size:23px}.callout{font-size:18px;line-height:1.85}
.thesis{grid-template-columns:1.35fr 1fr}.big{font-size:29px}
.rulebar{gap:20px;font-size:16px;flex-wrap:wrap}
.map-grid{grid-template-columns:minmax(0,1.9fr) minmax(300px,1fr)}
.legend-item span{font-size:14px;line-height:1.7}
.matrix-wrap table{table-layout:fixed;font-size:15px}
.matrix-wrap th,.matrix-wrap td{padding:16px 10px;vertical-align:middle}
.matrix-wrap th{font-size:14px;line-height:1.65}
.matrix-wrap th:nth-child(1){width:3%}.matrix-wrap th:nth-child(2){width:14%}
.matrix-wrap th:nth-child(3),.matrix-wrap th:nth-child(4){width:10%}
.matrix-wrap th:nth-child(5){width:8%}.matrix-wrap th:nth-child(6){width:12%}
.matrix-wrap th:nth-child(7){width:8%}.matrix-wrap th:nth-child(8){width:13%}
.matrix-wrap th:nth-child(9){width:15%}.matrix-wrap th:nth-child(10){width:7%}
.matrix-wrap .code{font-size:13px;white-space:nowrap;margin-top:5px}
.matrix-wrap td:nth-child(5),.matrix-wrap td:nth-child(7){font-size:18px}
.matrix-wrap td.protection-status{font-size:13px;line-height:1.8;white-space:pre-line;color:var(--muted)}
.matrix-wrap tr[data-row='cash'] td{background:#edf3f8;border-top:2px solid #cad7e4}
.matrix-wrap tr[data-row='growth-slot'] td{background:#f8fbff}
.matrix-wrap tr[data-row='growth-slot-member'] td{background:#f8fbff;border-top:1px dashed #c8d7e8}
.matrix-wrap>.small{white-space:pre-line;font-size:13px;line-height:1.85;background:#f5f8fb;border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:18px 0 0}
.mark{white-space:nowrap;text-align:center;padding-left:5px!important;padding-right:10px!important}
.mark button{min-width:30px;margin:0 2px;padding:3px 6px;font-size:16px}
.trade-plan{gap:20px}.plan-card{font-size:16px;line-height:1.85}
footer{gap:24px;font-size:13px}
/* monthly-presentation:end */
"""


def polish_presentation(source, report):
    """Idempotent presentation only; never derive advice or change financial data."""
    source = re.sub(r'\n?/\* monthly-presentation:start[\s\S]*?/\* monthly-presentation:end \*/\n?', '', source)
    source, count = re.subn(r'</style>', lambda _: PRESENTATION_CSS + '</style>', source)
    if count != 1:
        raise DetailsError('月报必须含唯一内嵌样式块')
    source = re.sub(r'((?<![\w.-])(?:th|td)\s*\{[^}]*vertical-align\s*:)\s*(?:top|bottom)', r'\1middle', source)
    metadata = ' · '.join(str(report[k]) for k in ('generated_at', 'decision_date', 'execution_date') if report.get(k))
    stamp = (f'<div class="stamp"><b title="{escape(metadata)}">报告 {escape(report["generated_date"])} · 本地日周收盘 {escape(report["data_as_of"])}</b><br>'
             f'下次决策 {escape(report["next_decision_date"])} · 批次 {escape(report["data_run_id"])}</div>')
    source, count = re.subn(r'<div class="stamp">[\s\S]*?</div>', lambda _: stamp, source)
    if count != 1:
        raise DetailsError('月报必须含唯一日期栏')
    return source


def render_details(source, root, report, baseline_path=None, portfolio_override=None,
                   historical_display=False):
    source = polish_presentation(source, report)
    source = with_four_decision_columns(source)
    for old_header in ('机械基线 / Agent趋势', 'Agent依据 / 当前监控', '当前监控'):
        source = source.replace('<th>' + old_header + '</th>', '')
    # Explicit user request: vertically center headers and cells, no other CSS edits.
    source = re.sub(r'((?<![\w.-])(?:th|td)\s*\{[^}]*vertical-align\s*:)\s*(?:top|bottom)',
                    r'\1middle', source)
    manifest, portfolio, rows = checked_rows(
        root, report, baseline_path, portfolio_override, historical_display
    )
    formal = bool(report.get('decision_snapshot'))
    analysis = report.get('agent_analysis', {}) if formal else report.get('agent_recommendation', {})
    agent_weights = analysis.get('recommended_weights_pct' if formal else 'unconstrained_weights_pct') or {}
    agent_cash = analysis.get('recommended_cash_pct' if formal else 'cash_pct')
    observation_cash = (report.get('mechanical_baseline') or {}).get('b0207_cash_pct')
    if formal:
        observation_cash = read_json(root, baseline_path or report['policy_snapshot_path'])['mechanical_baseline_cash_pct']
    enforce_history = date.fromisoformat(report['generated_date']) >= date(2026, 9, 12)
    display_only_reuse = (
        report.get('analysis_reuse_mode') == 'same_batch_display_redesign'
        and report.get('analysis_reference_date')
        and report['analysis_reference_date'] < report['generated_date']
        and report.get('previous_report_path') == f"app/reports/monthly/{report['analysis_reference_date']}.json"
    )
    try:
        analysis_reference_date = report.get('analysis_reference_date') or report['generated_date']
        if (date.fromisoformat(analysis_reference_date) > date.fromisoformat(report['generated_date'])
                or date.fromisoformat(analysis_reference_date) < date.fromisoformat(manifest['as_of'])):
            raise DetailsError('分析引用日期必须位于数据截止日与报告生成日之间')
        layers = layers_for_analysis(root, analysis_reference_date, manifest['as_of'],
                                     expected_order=ORDER, required=enforce_history)
    except DecisionHistoryError as exc:
        raise DetailsError(str(exc)) from exc
    if layers:
        cycle = layers['b0207_cycle_baseline']
        cycle_weights, cycle_cash = cycle['weights_pct'], cycle['cash_pct']
        if enforce_history:
            observation = layers['b0207_weekly_observation']
            recommendation = layers['agent_recommendation']
            confirmed = layers['confirmed_portfolio']
            source_weights = {row[0]['code']: row[3] for row in rows}
            if observation['weights_pct'] != source_weights or observation['cash_pct'] != observation_cash:
                raise DetailsError('B0207当前周观测与本次报告机械观测不一致')
            if recommendation['weights_pct'] != agent_weights or recommendation['cash_pct'] != agent_cash:
                raise DetailsError('Agent保护后组合与本次分析台账不一致')
            portfolio_weights = {code: (portfolio.get('weights_pct') or {}).get(code) for code in ORDER}
            if (not display_only_reuse
                    and (confirmed['weights_pct'] != portfolio_weights
                         or confirmed['cash_pct'] != portfolio.get('cash_pct'))):
                raise DetailsError('当前真实持仓与审查通过台账不一致')
    else:
        cycle_weights = {row[0]['code']: row[3] for row in rows}
        cycle_cash = (report.get('mechanical_baseline') or {}).get('b0207_cash_pct')
        if cycle_cash is None and baseline_path:
            cycle_cash = read_json(root, baseline_path).get('mechanical_baseline_cash_pct')
        if cycle_cash is None:
            raise DetailsError('缺少B0207周期基线现金比例')
    if abs(sum(cycle_weights.values()) + cycle_cash - 100) > 1e-8:
        raise DetailsError('B0207周期基线合计不是100%')
    source = render_holdings_summary(source, portfolio, report, observation={
        'weights_pct': {row[0]['code']: row[3] for row in rows},
        'cash_pct': observation_cash,
        'data_as_of': manifest['as_of'],
    })
    if enforce_history and (set(agent_weights) != set(ORDER) or agent_cash is None
                            or abs(sum(agent_weights.values()) + agent_cash - 100) > 1e-8):
        raise DetailsError('Agent保护后组合必须完整覆盖8只ETF及现金并合计100%')
    if enforce_history:
        base_kind = analysis.get('protection_base')
        required_base = 'b0207_cycle_baseline' if formal else 'confirmed_portfolio'
        if base_kind != required_base:
            raise DetailsError('Agent保护起点错误：名义周期候选用周期基线，非决策周用实际持仓')
        protection_portfolio = portfolio
        reference_id = analysis.get('protection_reference_record_id')
        if reference_id:
            # Preserve an already published recommendation after the user explicitly
            # updates holdings. Never reinterpret the old advice as a new order.
            ledger = load_ledger(root, expected_order=ORDER)
            reference = next((r for r in ledger['records'] if r['record_id'] == reference_id
                              and r['record_type'] == 'confirmed_portfolio'), None)
            origin = analysis.get('published_reference_portfolio')
            if (formal or report.get('execution_status') != 'current_weights_user_confirmed_details_pending'
                    or analysis.get('status') != 'historical_advice_before_user_holdings_update'
                    or report.get('execution_allowed') is not False or reference is None
                    or not isinstance(origin, dict)
                    or origin.get('weights_pct') != reference['weights_pct']
                    or origin.get('cash_pct') != reference['cash_pct']
                    or date.fromisoformat(str(reference['confirmed_at'])[:10]) > date.fromisoformat(report['generated_date'])):
                raise DetailsError('原Agent建议必须绑定已确认的历史持仓，并明确不是新订单')
            protection_portfolio = origin
        if not display_only_reuse:
            try:
                validate_plan(
                    policy=analysis.get('protection_policy'), base_kind=base_kind,
                    base_weights=cycle_weights if formal else {code: (protection_portfolio.get('weights_pct') or {}).get(code) for code in ORDER},
                    base_cash=cycle_cash if formal else protection_portfolio.get('cash_pct'),
                    weights=agent_weights, cash=agent_cash, rows=[row[0] for row in rows], order=ORDER,
                    allow_legacy=str(report.get('analysis_reference_date') or report.get('generated_date') or '') < '2026-09-13',
                )
            except ProtectionError as exc:
                raise DetailsError(str(exc)) from exc
    matrix_portfolio = protection_portfolio if enforce_history else portfolio
    historical_origin = (
        enforce_history
        and analysis.get('status') == 'historical_advice_before_user_holdings_update'
        and analysis.get('protection_reference_record_id')
    )
    policy_description = protection_policy_text(analysis, historical=bool(historical_origin))
    holding_column_label = '上一周期持仓' if historical_origin else '当前真实持仓'
    for old_label in ('当前真实持仓', '上一周期持仓'):
        if old_label != holding_column_label:
            source = source.replace(f'<th>{old_label}</th>', f'<th>{holding_column_label}</th>')
    estimate = report.get('portfolio_carry_forward_estimate')
    if estimate:
        if (estimate.get('source_current_event_id') != portfolio.get('source_event_id')
                or estimate.get('generated_date') != report.get('generated_date')
                or estimate.get('data_as_of') != manifest['as_of']
                or estimate.get('status') != 'HISTORICAL_AMOUNT_CARRY_FORWARD_NOT_LIVE_VALUATION'):
            raise DetailsError('持仓结转估算日期或来源失效，需重新计算')
        if abs(sum(estimate['weights_pct'].values()) + estimate['cash_pct'] - 100) > 1e-6:
            raise DetailsError('持仓估算份额不合计100%')
    actual_weights = matrix_portfolio.get('weights_pct') or {}
    def displayed_actual(code):
        value = actual_weights.get(code)
        return value if value is not None else (estimate or {}).get('weights_pct', {}).get(code)
    rendered = []
    cutoff, run = manifest['as_of'], manifest['run_id']
    growth_slot = None
    embedded_slot = report.get('priority_slot_analysis')
    if embedded_slot is not None:
        try:
            growth_slot = resolve_growth_slot(
                Path(root), rows[0][1]['weekly']['as_of'], rows[0][4],
                available_as_of=report.get('generated_date'),
            )
        except PrioritySlotError as exc:
            raise DetailsError(str(exc)) from exc
        # Report-generation date and already-rendered allocation fields are not
        # model/data identity.  A frozen champion applied to the same NAV may be
        # read again on the next calendar day without becoming a different slot.
        ignored_slot_fields = {"fund_report_generated_on", "fund_daily_observation_as_of",
                               "slot_weight_pct", "member_weights_pct"}
        embedded_contract = {k: v for k, v in embedded_slot.items() if k not in ignored_slot_fields}
        current_contract = {k: v for k, v in growth_slot.items() if k not in ignored_slot_fields}
        if embedded_contract != current_contract:
            raise DetailsError('第一优先级共享槽位资料已变化，须重新生成月报')
    summary = report.get('matrix_trend_summary', {})
    if summary and (summary.get('data_as_of') != cutoff or summary.get('data_run_id') != run
                    or set(summary.get('instruments', {})) != set(ORDER)):
        raise DetailsError('矩阵短评日期、批次或ETF范围过期，不能沿用旧判断')
    source = refresh_quadrant_chart(
        source, rows, summary, cutoff, report.get('decision_date') or report['generated_date']
    )
    for index, (row, f, raw_close, weight, position, reason) in enumerate(rows, 1):
        code, d, w = row['code'], f['daily'], f['weekly']
        conditions = (d['dif_first_raw'] < 0, d['dif_second_raw'] < 0,
                      w['dif_first_raw'] < 0, w['dif_second_raw'] < 0,
                      f['relative_strength']['return_20d'] < -.05)
        protection = f'机械辅助{sum(conditions)}/5' + ('满足' if all(conditions) else '未全满足')
        if enforce_history:
            review = row['v70_protection_review']
            protection += '\nV70 ' + (review['bearish_rule'] or '无确认看空')
            interval_label = {'new': '待用户确认', 'already_applied': '本区间已执行',
                              'not_triggered': '未触发', 'unknown': '区间待核实'}[review['interval_status']]
            user = report.get('user_decision') or {}
            target_weight = (user.get('final_target_weights_pct') or {}).get(code)
            actual_weight = (portfolio.get('weights_pct') or {}).get(code)
            if (review['interval_status'] == 'new'
                    and user.get('confirmation_source') == 'explicit_user_instruction'
                    and user.get('status') in {'modified_by_user', 'confirmed_agent_recommendation'}
                    and user.get('fill_status') == 'pending_user_fill_confirmation'
                    and isinstance(target_weight, (int, float))
                    and isinstance(actual_weight, (int, float)) and target_weight < actual_weight):
                interval_label = '用户目标已确认 / 待成交'
            protection += ' / ' + interval_label
            if analysis.get('status') == 'historical_advice_before_user_holdings_update':
                protection = protection.replace('待用户确认', '原建议，非新订单')
                if portfolio.get('weights_pct', {}).get(code) == 0:
                    protection += ' / 当前无仓，不执行'
        if all(conditions) and portfolio.get('weights_pct', {}).get(code) == 0:
            protection += '（无仓）'
        tooltip = []
        for label, v in (('日K', d), ('周K', w)):
            tooltip.append(f"{label}截至{v['as_of']}：DIF {v['dif']:.6f}；离散一阶 {v['dif_first_raw']:+.6f}；二阶 {v['dif_second_raw']:+.6f}；MA5 {v['sma5']:.4f} / MA10 {v['sma10']:.4f} / MA20 {v['sma20']:.4f} / MA60 {v['sma60']:.4f}；RSI {v['rsi14']:.2f}")
        tooltip.append(f"成交量 {f['volume']['latest_volume']}；20日中位数比 {f['volume']['volume_to_20d_median']:.3f}；成交额缺失不推测补齐")
        # Preserve all previously required narrative fields without adding DOM.
        tooltip.extend(str(row[k]) for k in ('weekly_evidence', 'daily_evidence',
                       'timeframe_relationship', 'volume_confirmation', 'relative_advantage',
                       'evidence_for', 'evidence_against', 'agent_recommendation', 'entry_trigger',
                       'execution_time', 'cancel_or_exit', 'risk_note') if row.get(k))
        direction = {'bullish': '看多', 'bearish': '看空', 'neutral': '震荡'}.get(row['agent_direction'], row['agent_direction'])
        agent_color = 'pos' if direction.startswith('看多') else 'neg' if direction.startswith('看空') else 'neutral'
        display_reason = str(reason)
        # Avoid repeating the standalone direction at the start of the reasons.
        if display_reason.startswith(direction + '，') or display_reason.startswith(direction + '；'):
            display_reason = display_reason[len(direction) + 1:]
        tooltip.append('完整Agent保护依据：' + str(reason))
        short = summary.get('instruments', {}).get(code)
        if short:
            if not short.get('mechanical') or not short.get('agent'):
                raise DetailsError(code + '缺少机械或Agent短评')
            display_reason = '机械基线：' + short['mechanical'] + '\nAgent保护：' + short['agent']
        mechanical_trend = short['mechanical'] if short else ('看多' if position > 0 else '看空')
        agent_trend = short['agent'] if short else direction
        instrument_cell = f'<td><b>{escape(f["name"])}</b><div class="code">{code} · {raw_close:.3f}</div></td>'
        weekly_cell = f'<td class="signal {"neg" if w["dif_first_normalized"] < 0 else "neutral"}">{w["dif_first_normalized"]:+.2f} / {w["dif_second_normalized"]:+.2f}</td>'
        daily_cell = f'<td class="signal {"neg" if d["dif_first_normalized"] < 0 else "pos"}">{d["dif_first_normalized"]:+.2f} / {d["dif_second_normalized"]:+.2f}</td>'
        if code == '159915' and growth_slot:
            fund = growth_slot['fund_analysis']
            fund_daily, fund_weekly = fund['daily'], fund['weekly']
            selected_member = growth_slot.get('selected_member')

            def split_slot_weight(slot_weight, member, *, selected=selected_member):
                """Render one shared allocation once, on the member that carries the slot."""
                if slot_weight is None:
                    return None
                carrier = selected if selected in ('159915', '021528') else '159915'
                return float(slot_weight) if member == carrier else 0.0

            cycle_etf = split_slot_weight(cycle_weights[code], '159915', selected='159915')
            cycle_fund = split_slot_weight(cycle_weights[code], '021528', selected='159915')
            weekly_etf = split_slot_weight(weight, '159915')
            weekly_fund = split_slot_weight(weight, '021528')
            actual_members = matrix_portfolio.get('growth_slot_member_weights_pct')
            if actual_members is not None:
                if (set(actual_members) != {'159915', '021528'}
                        or abs(sum(actual_members.values()) - displayed_actual(code)) > 1e-8):
                    raise DetailsError('第一优先级真实持仓成员与共享槽位总仓位不一致')
                actual_etf = actual_members['159915']
                actual_fund = actual_members['021528']
            else:
                actual_etf = displayed_actual(code)
                actual_fund = (matrix_portfolio.get('weights_pct') or {}).get('021528', 0.0)
            agent_members = analysis.get('growth_slot_member_weights_pct')
            if agent_members is not None:
                if (set(agent_members) != {'159915', '021528'}
                        or abs(sum(agent_members.values()) - float(agent_weights.get(code, 0))) > 1e-8):
                    raise DetailsError('Agent保护后第一槽位成员权重与槽位总权重不一致')
                agent_etf, agent_fund = agent_members['159915'], agent_members['021528']
            elif agent_weights.get(code) == displayed_actual(code) and actual_members is not None:
                # No protection change: retain the actual fund/ETF carrier even
                # if the current mechanical slot selects neither instrument.
                agent_etf, agent_fund = actual_members['159915'], actual_members['021528']
            else:
                agent_etf = split_slot_weight(agent_weights.get(code), '159915')
                agent_fund = split_slot_weight(agent_weights.get(code), '021528')
            tooltip.append(
                f'021528净值截至{fund["nav_as_of"]}：日DIF {fund_daily["dif"]:+.6f}，一阶 {fund_daily["dif_first"]:+.6f}，二阶 {fund_daily["dif_second"]:+.6f}；'
                f'MA5/10/20/60 {fund_daily["ma5"]:.4f}/{fund_daily["ma10"]:.4f}/{fund_daily["ma20"]:.4f}/{fund_daily["ma60"]:.4f}；RSI {fund_daily["rsi14"]:.2f}。'
                f'完整周截至{fund["completed_week_as_of"]}：周DIF {fund_weekly["dif"]:+.6f}，一阶 {fund_weekly["dif_first"]:+.6f}，二阶 {fund_weekly["dif_second"]:+.6f}；'
                f'MA5/10/20 {fund_weekly["ma5"]:.4f}/{fund_weekly["ma10"]:.4f}/{fund_weekly["ma20"]:.4f}。成交量不适用。')
            shared_tooltip = '；'.join(tooltip)
            first_cells = (
                f'<td rowspan="2" class="rank">{index:02d}</td>'
                f'<td><b>{escape(f["name"])}</b><div class="code">159915 · {raw_close:.3f}</div></td>'
                + weekly_cell + daily_cell
                + f'<td class="decision" data-column="cycle-baseline" title="名义决策日冻结至下一决策日；第一优先级只占一个槽位">{percentage_text(cycle_etf)}</td>'
                + allocation_cell(weekly_etf, mechanical_trend, column='weekly-observation',
                                  tooltip='当前周共享槽位机械观测；非决策周不自动执行。')
                + f'<td class="decision">{percentage_text(actual_etf)}</td>'
                + allocation_cell(agent_etf, agent_trend, column='agent-direction', tooltip=shared_tooltip)
                + f'<td class="protection-status">{protection}\n日K {cutoff}\n完整周K {w["as_of"]}</td>'
                + f'<td class="mark" data-code="{code}"><button class="yes" data-v="✓">✓</button><button class="no" data-v="✗">✗</button></td>')
            second_cells = (
                f'<td><b>{escape(fund["name"])}</b><div class="code">021528 · {fund["unit_nav"]:.4f}</div></td>'
                '<td class="signal neutral" title="021528为净值DIF原始离散导数">'
                f'{fund_weekly["dif_first"]:+.4f} / {fund_weekly["dif_second"]:+.4f}</td>'
                '<td class="signal neutral" title="021528为净值DIF原始离散导数">'
                f'{fund_daily["dif_first"]:+.4f} / {fund_daily["dif_second"]:+.4f}</td>'
                f'<td class="decision" data-column="cycle-baseline">{percentage_text(cycle_fund)}</td>'
                + allocation_cell(weekly_fund, fund['direction'], column='weekly-observation',
                                  tooltip='021528自身净值分析；与159915共享第一优先级槽位。')
                + f'<td class="decision">{percentage_text(actual_fund)}</td>'
                + allocation_cell(agent_fund, fund['direction'], column='agent-direction',
                                  tooltip='021528自身净值分析；与159915共享第一优先级槽位。')
                + '<td class="protection-status" title="基金没有OHLCV和成交量；当前报告没有基金专属V70保护审查记录，不能把冠军方向当作保护触发。">'
                '机械辅助：不适用\nV70 未独立确认\n'
                f'日净值 {fund["nav_as_of"]}\n完整周净值 {fund["completed_week_as_of"]}</td>'
                '<td class="mark" data-code="021528"><button class="yes" data-v="✓">✓</button><button class="no" data-v="✗">✗</button></td>')
            rendered.append(f'<tr data-row="growth-slot" data-member="159915" data-as-of="{cutoff}" data-run-id="{run}">{first_cells}</tr>')
            rendered.append(f'<tr data-row="growth-slot-member" data-member="021528" data-as-of="{fund["nav_as_of"]}" data-run-id="fund-nav">{second_cells}</tr>')
            continue
        cells = (f'<td class="rank">{index:02d}</td>'
                 + instrument_cell
                 + weekly_cell
                 + daily_cell
                 + f'<td class="decision" data-column="cycle-baseline" title="名义决策日冻结至下一决策日">{percentage_text(cycle_weights[code])}</td>'
                 + allocation_cell(weight, mechanical_trend, column='weekly-observation', tooltip='当前周B0207机械观测；非决策周不自动执行。')
                 + f'<td class="decision">{percentage_text(displayed_actual(code))}</td>'
                 + allocation_cell(agent_weights.get(code), agent_trend, column='agent-direction',
                                   tooltip=policy_description + '；' + '；'.join(tooltip))
                 +
                 f'<td class="protection-status">{protection}\n日K {cutoff}\n完整周K {w["as_of"]}</td>'
                 f'<td class="mark" data-code="{code}"><button class="yes" data-v="✓">✓</button><button class="no" data-v="✗">✗</button></td>')
        rendered.append(f'<tr data-as-of="{cutoff}" data-run-id="{run}">{cells}</tr>')
    agent = report.get('agent_recommendation', {})
    baseline_cash = (report.get('mechanical_baseline') or {}).get('b0207_cash_pct')
    if baseline_cash is None and baseline_path:
        baseline_cash = read_json(root, baseline_path).get('mechanical_baseline_cash_pct')
    actual_cash = matrix_portfolio.get('cash_pct')
    if actual_cash is None:
        actual_cash = (estimate or {}).get('cash_pct')
    if historical_origin:
        cash_trend = '历史保护现金｜当时卖出款转现金。'
        cash_tooltip = agent.get('cash_reason', '历史Agent保护减持差额转现金')
    else:
        cash_trend = '保护现金｜无合格原有持仓承接时才保留。'
        cash_tooltip = agent.get('cash_reason', '保护卖出款先回投其他原有持仓；无合格接收标的时才留现金')
    cash_row = ('<tr data-row="cash"><td class="rank">09</td><td><b>现金</b></td>'
                '<td>—</td><td>—</td>'
                + f'<td class="decision" data-column="cycle-baseline">{percentage_text(cycle_cash)}</td>'
                + allocation_cell(baseline_cash, '现金配置｜当前周机械观测。', column='weekly-observation')
                + f'<td class="decision">{percentage_text(actual_cash)}</td>'
                + allocation_cell(agent_cash, cash_trend, column='agent-direction', tooltip=cash_tooltip)
                +
                 f'<td class="protection-status">{cutoff}</td><td>—</td></tr>')
    if 'data-row="cash"' not in source:
        source = source.replace('</tbody>', cash_row + '</tbody>')
    rendered.append(cash_row)
    result, n = re.subn(r'<tbody>[\s\S]*?</tbody>', lambda _: '<tbody>' + ''.join(rendered) + '</tbody>', source)
    if n != 1:
        raise DetailsError('锁定月报必须存在唯一8只ETF详情表')
    result = result.replace('<th>当前监控</th>', '<th>Agent依据 / 当前监控</th>')
    result = result.replace('<th>Agent依据 / 当前监控</th>', '<th>机械基线 / Agent趋势</th>')
    caption = (f'详情最新共同交易日：{cutoff}；批次：{run}。每次生成从正式日线、日周参数、同批次机械参考及Agent复核重新渲染，禁止续用旧表。'
               '页面为生成时快照，不是盘中行情；没有更新数据时不得冒称已联网获取最新交易日。'
               f'决策部分依次列出B0207周期基线、B0207当前周观测、{holding_column_label}和Agent保护后组合。B0207每28天完整调仓；周期基线冻结至下一名义决策日。{policy_description}B0207当前周观测只作非执行监控。两列格内仅显示全账户份额及彩色方向，不再重复日K、周K分析。—表示未提供目标或比例未知，不表示0%。转入资金比例及原份额保留比例不是全账户比例。悬停Agent保护格查看原始DIF、离散导数、均线、量能及持有期高点回撤。'
               '数据质量警告保留；达标标记仅保存在当前浏览器本地。')
    caption += ' 五参数仅为机械辅助观察，不等于Agent保护触发；Agent按V70审查R002/R003/R008，并执行R007防机械追空复核。'
    if summary:
        caption += ' 趋势观察期：' + summary['horizon'] + '。机械看多/看空表示冻结模型的配置方向，不代表独立预测的价格路径；Agent短评为有条件展望，完整依据保留在悬停提示。'
        caption += (' 趋势颜色：' + '；'.join(trend_text('', label) for label in
                    ('看多', '看空', '震荡加强／修复', '震荡减弱／偏弱', '震荡中性')) + '。')
    if formal:
        caption += ' Agent组合判断：' + str(analysis.get('portfolio_thesis') or '')
        caption += ' 不操作对比：' + str(analysis.get('no_action_comparison') or '')
        for note in analysis.get('risk_notes') or []:
            caption += ' 风险说明：' + str(note)
        confirmation = str((report.get('user_decision') or {}).get('confirmation_note') or '')
        if confirmation:
            caption += ' 用户确认：' + confirmation
    if growth_slot:
        fund = growth_slot['fund_analysis']
        caption += (f' 第一优先级为共享槽位：159915与021528分别分析但仓位只计算一次；当前承接成员为'
                    f'{growth_slot.get("selected_member") or "无"}。021528使用自身公布净值，最新净值日{fund["nav_as_of"]}，'
                     f'完整周截止{fund["completed_week_as_of"]}；基金日线只作为报告生成时已知的当前观察，不回写历史Agent建议，'
                     '槽位分配仍只使用对应完整周信号；基金没有OHLCV与成交量，相关字段显示不适用。')
    if historical_origin:
        caption += (' 矩阵“上一周期持仓”读取当次Agent分析绑定的已确认起点组合；'
                    '网页顶部、持仓卡片和底部持仓内容仍读取最新真实持仓，两者不得混写。')
        result = result.replace(
            '本周只做持仓保护，减持资金转现金，最终目标由用户决定。',
            '本列保留当时“减持资金转现金”的历史保护建议；现行正式保护资金改为回投其他原有持仓，没有合格接收标的时才留现金。',
        )
    if estimate:
        caption += (f' 当前持仓列采用用户授权的历史金额结转估算（非实际成交或实时市值），原始金额日{estimate["source_valuation_date"]}；'
                    f'黄金股{estimate["market_values_cny"]["517520"]:,.2f}元，酒{estimate["market_values_cny"]["512690"]:,.2f}元，'
                    f'现金{estimate["cash_amount_cny"]:,.2f}元。现金份额：B0207 {percentage_text(report["mechanical_baseline"]["b0207_cash_pct"])} / '
                    f'Agent保护后 {percentage_text(report["agent_recommendation"]["cash_pct"])} / 当前持仓 {percentage_text(estimate["cash_pct"])}。'
                    '结转未计期间涨跌、手续费或分红；不重放已被最新持仓纠正的银行中间路径。')
    if agent.get('target_basis') == 'pre_gold_sale_reassessment':
        caption += ' Agent保护后列以黄金股减持前组合为起点，是本次事后保护复核，不是独立全仓配置，也不是当时已给出的历史建议；当前持仓列为实际已执行30%减持后的金额结转估算。二者差额不构成补仓指令。'
    # Separate existing disclosures into readable groups without dropping caveats.
    for boundary in ('决策部分依次列出', ' 趋势观察期：', ' 趋势颜色：', ' 当前持仓列采用', ' Agent保护后列以黄金股'):
        caption = caption.replace(boundary, '\n\n' + boundary.strip())
    result, n = re.subn(r'</table><p class="small">[\s\S]*?</p>', lambda _: '</table><p class="small">' + caption + '</p>', result)
    if n != 1:
        raise DetailsError('详情截止日期说明位置不符合锁定模板')
    if analysis.get('status') == 'historical_advice_before_user_holdings_update':
        result = result.replace('决策部分依次列出',
            escape(str(analysis.get('reference_note') or 'Agent保护后列为调整前原建议，不是当前订单，不自动买回已退出标的。'))+'\n\n决策部分依次列出')
    # Explicitly authorized trend-label spans are the only new wrappers allowed.
    def tags(text):
        text = re.sub(r'<span data-trend="[a-z]+"[^>]*>([\s\S]*?)</span>', r'\1', text)
        # User-authorized table migration is checked separately below.
        text = re.sub(r'<table>[\s\S]*?</table>', '<table></table>', text)
        return re.findall(r'<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>', text)
    table_rows = re.findall(r'<tr\b[^>]*>([\s\S]*?)</tr>', re.search(r'<tbody>([\s\S]*?)</tbody>', result)[1])
    cell_counts = [len(re.findall(r'<td\b', row)) for row in table_rows]
    if growth_slot:
        if (len(table_rows) != 10 or cell_counts[:2] != [10, 9]
                or any(count != 10 for count in cell_counts[2:])):
            raise DetailsError('共享第一槽位必须拆成两条分析记录，并仅共用一个序号单元格')
        first_two = ''.join(table_rows[:2])
        if (len(re.findall(r'\browspan=', first_two)) != 1
                or not re.search(r'<td\s+rowspan="2"\s+class="rank">', table_rows[0])):
            raise DetailsError('共享第一槽位只允许序号01跨两行显示')
    elif len(table_rows) != 9 or any(count != 10 for count in cell_counts):
        raise DetailsError('决策矩阵必须为8只ETF及现金共9行，每行10列')
    if tags(source) != tags(result):
        raise DetailsError('详情更新改变了HTML标签结构')
    return result


def render_holdings_summary(source, portfolio, report=None, *, observation=None):
    """All headline/card figures come from confirmed weights, never old template text."""
    report = report or {}
    target = report.get('user_final_target') or {}
    force_current = report.get('portfolio_display_mode') == 'current_confirmed'
    next_week = not force_current and target.get('status') == 'user_confirmed_next_week_target'
    if next_week:
        user = report.get('user_decision') or {}
        if user.get('confirmation_source') != 'explicit_user_instruction':
            raise DetailsError('下周持仓缺少用户明确确认')
        values = [target.get('weights_pct', {}).get(c) for c in ORDER] + [target.get('cash_pct')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in values) or abs(sum(values)-100)>1e-8:
            raise DetailsError('下周持仓比例必须完整且合计100%')
        if user.get('final_target_weights_pct') != target['weights_pct'] or user.get('final_cash_pct') != target['cash_pct']:
            raise DetailsError('下周持仓与用户确认不一致')
        slot_members = target.get('growth_slot_member_weights_pct')
        if slot_members is None:
            slot_members = {'159915': target['weights_pct']['159915'], '021528': 0}
        if (set(slot_members) != {'159915', '021528'}
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100
                       for v in slot_members.values())
                or abs(sum(slot_members.values()) - target['weights_pct']['159915']) > 1e-8):
            raise DetailsError('第一优先级成员仓位必须由159915与021528组成，并等于共享槽位仓位')
        if user.get('growth_slot_member_weights_pct', slot_members) != slot_members:
            raise DetailsError('第一优先级成员仓位与用户确认不一致')
        period = target.get('effective_period') or {}
        try:
            start, end = date.fromisoformat(period['start']), date.fromisoformat(period['end'])
        except (KeyError, ValueError, TypeError) as exc:
            raise DetailsError('下周目标缺少有效日期范围') from exc
        if start > end or start <= date.fromisoformat(report['generated_date']):
            raise DetailsError('下周目标日期必须晚于报告日期，且起止有序')
        portfolio = target
    names = {'159915':'创业板','021528':'财通成长优选混合C','517520':'黄金股','516150':'稀土','159622':'创新药',
             '159611':'电力','515220':'煤炭','512800':'银行','512690':'酒'}
    weights = portfolio.get('weights_pct') or {}
    member_weights = portfolio.get('growth_slot_member_weights_pct')
    if next_week:
        member_weights = target.get('growth_slot_member_weights_pct') or {
            '159915': weights.get('159915', 0), '021528': 0}
        held = ([(names[c], member_weights[c]) for c in ('159915', '021528') if member_weights[c] > 0]
                + [(names[c], weights[c]) for c in ORDER if c != '159915'
                   and isinstance(weights.get(c), (int,float)) and weights[c] > 0])
    elif member_weights is not None:
        if (set(member_weights) != {'159915', '021528'}
                or abs(sum(member_weights.values()) - weights.get('159915', 0)) > 1e-8):
            raise DetailsError('第一优先级真实持仓成员与共享槽位总仓位不一致')
        held = ([(names[c], member_weights[c]) for c in ('159915', '021528') if member_weights[c] > 0]
                + [(names[c], weights[c]) for c in ORDER if c != '159915'
                   and isinstance(weights.get(c), (int,float)) and weights[c] > 0])
    else:
        held = [(names[c], weights[c]) for c in ORDER if isinstance(weights.get(c), (int,float)) and weights[c]>0]
    cash = portfolio.get('cash_pct')
    if cash is None or (not held and cash != 100):
        raise DetailsError('持仓或现金比例缺失，禁止保留旧持仓摘要冒充当前结果')
    labels = [name for name,_ in held] + (['现金'] if cash>0 else [])
    detail = '；'.join([name+percentage_text(value) for name,value in held] + ['现金'+percentage_text(cash)])
    heading = '下周持仓' if next_week else '当前真实持仓：用户确认比例'
    note = ('目标期间'+period['start']+'至'+period['end']+'；用户已确认配置，不代表已经成交，当前真实持仓见决策矩阵。') if next_week else '其余ETF为0%。成交日期、价格、份数和费用未补录，不估算。'
    panel = ('<div class="current"><div class="eyebrow">'+heading+'</div>'
             '<div class="big">'+escape(' · '.join(labels))+'</div><div>'+escape(detail)+'</div>'
             '<div class="small">'+note+'</div></div>')
    source = re.sub(r'<div class="current">[\s\S]*?</div>\s*</div>',lambda _:panel,source,count=1)
    holding_count = str(len(held)) + '只产品' + ('＋现金' if cash > 0 else '')
    facts = ([(percentage_text(value), name + '账户占比') for name, value in held]
             + [(percentage_text(cash), '现金账户占比')]
             + [(('下周配置已确认' if next_week else '当前比例已确认'),
                 holding_count + ' / ' + ('成交另行登记' if next_week else '成交明细待补'))])
    source = re.sub(r'<div class="facts">[\s\S]*?</div>\s*</div>',
        lambda _:'<div class="facts">'+''.join('<div class="fact"><b>'+escape(a)+'</b><span>'+escape(b)+'</span></div>' for a,b in facts)+'</div>',source,count=1)
    if next_week or force_current:
        from lxml import html
        tree = html.document_fromstring(source)
        def replace_text(node, text):
            for child in list(node):
                node.remove(child)
            node.text = text
        cards = tree.xpath('//div[@class="plan-card"]')
        if len(cards) >= 4:
            formal_plan = (report.get('decision_snapshot') or {}).get('portfolio_plan')
            if formal_plan:
                def plan_text(plan_weights, plan_cash):
                    parts = [names[code] + percentage_text(plan_weights[code])
                             for code in ORDER if isinstance(plan_weights.get(code), (int, float))
                             and plan_weights[code] > 0]
                    parts.append('现金' + percentage_text(plan_cash))
                    return '；'.join(parts)
                cycle_detail = plan_text(formal_plan['mechanical_baseline_weights_pct'],
                                         formal_plan['mechanical_baseline_cash_pct'])
                agent_detail = plan_text(formal_plan['agent_recommended_weights_pct'],
                                         formal_plan['agent_recommended_cash_pct'])
                cutoff = report.get('data_as_of')
                decision_date = report.get('decision_date')
                if not cutoff or not decision_date or not report.get('data_run_id'):
                    raise DetailsError('月报计划卡缺少截止日、决策日或批次')
                replace_text(cards[0].xpath('./h3')[0], 'B0207周期基线与当前周观测')
                weekly = observation or report.get('weekly_observation')
                if weekly is not None:
                    if weekly.get('data_as_of') != cutoff:
                        raise DetailsError('计划卡当前周观测截止日与报告不一致')
                    weekly_detail = plan_text(weekly['weights_pct'], weekly['cash_pct'])
                else:
                    weekly_detail = '详见同批次决策矩阵，摘要未提供独立周观测快照'
                steps = cards[0].xpath('./ol[@class="steps"]')[0]
                for child in list(steps):
                    steps.remove(child)
                for line in (
                    decision_date + '名义决策日B0207周期基线：' + cycle_detail + '。',
                    cutoff + '当前周观测：' + weekly_detail + '。',
                    'Agent保护后：' + agent_detail + '；保护依据见本批次已复核分析。',
                    '技术数据截止' + cutoff + '，批次' + report['data_run_id'] + '。',
                ):
                    steps.append(html.fragment_fromstring('<li>' + escape(line) + '</li>'))
            replace_text(cards[1].xpath('./h3')[0], '下周已确认持仓' if next_week else '当前真实持仓')
            allocation = cards[1].xpath('./div[@class="allocation"]')[0]
            for child in list(allocation): allocation.remove(child)
            for name, value in held + [('现金', cash)]:
                allocation.append(html.fragment_fromstring('<div class="alloc"><b>'+percentage_text(value)+'</b><span>'+escape(name)+'</span></div>'))
            replace_text(cards[1].xpath('./p')[-1], detail+('。其余ETF为0%；用户确认下周目标，成交后另行登记。' if next_week else '。其余产品为0%；金额、份数及成交价未提供，不估算。'))
            replace_text(cards[2].xpath('./h3')[0], '用户持仓与Agent保护建议分开记录')
            card_three_paragraphs = cards[2].xpath('./p')
            if card_three_paragraphs:
                replace_text(card_three_paragraphs[0],
                    (('用户已确认待执行目标：' if next_week else '用户已确认实际持仓：')+detail+'。')
                    + ((' Agent保护后：'+agent_detail+'。') if formal_plan else '')
                    + '用户决定不改写B0207周期基线或Agent保护判断。')
                replace_text(card_three_paragraphs[-1], '此目标尚未成交，真实持仓见决策矩阵。' if next_week else '按用户确认的真实持仓展示；成交价、份数和费用可后补。')
            replace_text(cards[3].xpath('./p')[0], ('目标期间'+period['start']+'至'+period['end']+'，按用户已确认目标安排：'+detail+'。黄金股为517520，不是旧黄金ETF518600；本次不重算模型、基线或Agent建议，不假设成交价格与份数。下一名义决策日'+str(report.get('next_decision_date') or '未提供')+'。') if next_week else '本期月报按当前真实持仓展示：'+detail+'。第一优先级159915与021528分别分析、共享一个槽位；本次不重算B0207周期基线，不假设任何未提供的成交明细。下一名义决策日'+str(report.get('next_decision_date') or '未提供')+'。')
        for node in tree.xpath('//div[@class="callout"] | //div[@class="warning"]'):
            replace_text(node, ('下周待执行目标：' if next_week else '当前真实持仓：')+detail+'；其余产品为0%。'+('尚未成交，真实持仓见决策矩阵。' if next_week else '已按用户确认登记，成交明细可后补。'))
        for node, value in zip(tree.xpath('//div[@class="rulebar"]/*')[:2], [('下周持仓已确认' if next_week else '当前真实持仓'), detail]):
            replace_text(node, value)
        # The plan execution date is not a confirmed trade date.
        bar = tree.xpath('//div[@class="rulebar"]/*')
        if len(bar) >= 4:
            replace_text(bar[2], '待执行' if next_week else '用户已确认')
            trade_date = portfolio.get('trade_date')
            replace_text(bar[3], '成交日期：' + trade_date if trade_date else '实际成交日期待补')
        source = html.tostring(tree, encoding='unicode', doctype='<!DOCTYPE html>').replace('viewbox=', 'viewBox=')
    return source
