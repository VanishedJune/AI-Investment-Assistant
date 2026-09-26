"""Legacy check-only historical-amount carry-forward, not a fills ledger.

Only this confirmed event and reviewed report are supported. Never infer dates,
prices, intervening bank trades, or current market returns from missing fills.
Production writes are permanently blocked; use the current monthly publication
pipeline for present holdings and reports.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from lxml import html

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.monthly_details import render_details, DetailsError, local_path
from scripts.publish_monitoring_snapshot import atomic, text

ROOT = Path(__file__).resolve().parents[2]


def _render_page(root, report, html_path, portfolio_override=None, historical_display=False):
    tree = html.document_fromstring(html_path.read_text(encoding='utf-8'))
    find = lambda c: tree.xpath('//*[@class="'+c+'"]')
    text(find('subtitle')[0], 'B0207四周主流程 · Agent周期中多参数看空保护 · 当前持仓独立列示')
    cards = find('plan-card')
    text(cards[0].xpath('.//li')[-1], 'Agent仅对减持前已有持仓进行多参数看空保护复核，只能保持、减仓或清仓，减持资金转现金；不新建、不加仓、不整体轮动。当前持仓按用户授权历史金额结转，非实时估值。')
    text(cards[1].xpath('./p')[-1], '实际已确认黄金股减持30%、煤炭全额转酒。按历史金额结转：黄金股80,657.50元、酒90,000元、现金34,567.50元；未计期间盈亏及费用，不冒充实际成交净额。')
    text(cards[2].xpath('./h3')[0], 'Agent保护复核：黄金股减20%，酒保持')
    text(cards[2].xpath('./p')[0], '起点为黄金股115,225元、酒90,000元、现金0元的历史金额结转组合。黄金股周一阶+0.015663、二阶−0.012236，日一二阶同负，建议减原份额20%并留现金，保留80%参与反弹。2.071未失守，不清仓；2.262未恢复，不追入。20%是主观风险取舍，不是已验证最优比例；不是历史事前建议，也不指示把已卖出的10%买回。')
    text(find('warning')[0], '报告生成2026-09-07，行情截至2026-09-04，未联网更新。Agent以黄金股减持前组合进行多参数看空保护复核，不作为独立全仓决策器；当前持仓独立展示实际30%减持后的结转估算，未更改真实成交。规则v70、记忆v78仅作保护证据，旧批次205/400低于上涨基准216/400，未证明独立决策优势。现金已纳入矩阵。')
    footer = tree.xpath('//footer')[0]
    text(footer.xpath('./span')[0], 'B0207：四周完整调仓｜Agent：周期中多参数看空保护｜最终执行：用户确认')
    return render_details(
        html.tostring(tree, encoding='unicode', doctype='<!DOCTYPE html>').replace('viewbox=', 'viewBox='),
        root,
        report,
        portfolio_override=portfolio_override,
        historical_display=historical_display,
    )


def build(root, display_only=False):
    read = lambda p: json.loads((root / p).read_text(encoding='utf-8-sig'))
    report = read('app/reports/monthly/latest.json')
    if display_only:
        if report['generated_date'] != '2026-09-07':
            raise DetailsError('展示迁移只允许处理已审核的2026-09-07历史月报')
        html_path = local_path(root, report['html_path'])
        page = _render_page(
            root, report, html_path, report['last_confirmed_portfolio'], historical_display=True
        )
        return {html_path: page.encode('utf-8')}, report.get('portfolio_carry_forward_estimate')
    portfolio = read('portfolio_state.json')['last_confirmed_portfolio']
    if (portfolio['source_event_id'] != 'APH-20260907-003'
            or report['generated_date'] != '2026-09-07'
            or report['last_confirmed_portfolio'] != portfolio):
        raise DetailsError('本次授权仅适用于9月7日报告及已确认持仓；新事件需重新审核')
    agent = report['agent_recommendation']
    for row in agent['instruments']:
        if row.get('analysis_data_as_of') != report['data_as_of']:
            raise DetailsError('不再是同日保持份额建议，禁止自动套用旧目标')
    entries = read('app/portfolio_history/actual_holdings.json')['entries']
    source = next(x for x in entries if x['event_id'] == 'APH-20260831-001')
    historical = source['current_snapshot']['market_values_cny']
    gold, coal = historical['518600'], historical['515220']
    conversion = portfolio['conversion']
    if (conversion['from_code'] != '517520' or conversion['to_code'] != 'CASH'
            or portfolio['prior_conversion']['allocation_of_proceeds'] != {'512690': 1}
            or set(portfolio['held_codes']) != {'517520', '512690'}):
        raise DetailsError('确认的资金结转路径已改变')
    fraction = conversion['sell_fraction_of_existing_shares']
    if not 0 <= fraction <= 1 or gold <= 0 or coal <= 0:
        raise DetailsError('金额或减持比例非法')
    amounts = {row['code']: 0.0 for row in agent['instruments']}
    amounts.update({'517520': gold * (1 - fraction), '512690': coal})
    cash, total = gold * fraction, gold + coal
    weights = {code: round(value / total * 100, 2) for code, value in amounts.items()}
    cash_pct = round(100 - sum(weights.values()), 2)
    estimate = dict(status='HISTORICAL_AMOUNT_CARRY_FORWARD_NOT_LIVE_VALUATION',
        generated_date=report['generated_date'], data_as_of=report['data_as_of'],
        source_valuation_date='2026-08-31', source_valuation_event_id=source['event_id'],
        source_current_event_id=portfolio['source_event_id'], market_values_cny=amounts,
        cash_amount_cny=cash, total_market_value_cny=total, weights_pct=weights, cash_pct=cash_pct,
        assumptions=['按历史金额等值换仓，不认为不同基金份数相等',
                     '采用最新完整持仓纠正后的资金路径，不重复转入银行',
                     '缺少真实成交明细，不计算期间盈亏、手续费、分红',
                     '只作报告估算，不覆盖真实成交及持仓事实'])
    report['portfolio_carry_forward_estimate'] = estimate
    # Agent-authored 9/7 reassessment, NOT a sizing formula or future-date default:
    # reduce 20% of PRE-SALE gold shares; keep wine; compare with the user's 30% sale.
    suggested_sale = .20
    target_amounts = dict(amounts)
    target_amounts['517520'] = gold * (1 - suggested_sale)
    targets = {c: round(v / total * 100, 2) for c, v in target_amounts.items()}
    target_cash_pct = round(100 - sum(targets.values()), 2)
    agent.update(status='pre_sale_reassessment_not_historical_advice',
        unconstrained_weights_pct=targets, cash_pct=target_cash_pct,
        target_generated_date=report['generated_date'], target_data_as_of=report['data_as_of'],
        target_basis='pre_gold_sale_reassessment',
        evaluation_start={'gold_value_cny':gold,'wine_value_cny':coal,'cash_cny':0,
                          'valuation_basis':'historical_amount_carry_forward', 'gold_sale_not_yet_applied':True},
        suggested_gold_sale_fraction_of_pre_sale_shares=suggested_sale,
        cash_reason='Agent保护现金来自减持前黄金股份额的20%；当前现金来自用户实际30%减持。均为历史金额结转口径，不是同一笔模拟成交。',
        note='以黄金股减持前组合进行多参数看空保护复核：保护建议为黄金股减持原份额20%、酒保持，所得留现金。Agent不作为独立全仓决策器，不新建、不加仓、不周度轮换。已知用户实际卖出30%，本次不冒称事前盲评或历史已冻结建议，也不指示买回差额。20%为Agent风险取舍，不是指标公式或已验证最优比例。')
    report['agent_target_confirmation_status'] = 'not_user_approved_for_execution'
    report['execution_allowed'] = False
    training = root.parent/'workspace-materials/ETF做题记录'
    memory = json.loads((training/'优化记忆.json').read_text(encoding='utf-8-sig'))
    training_state = json.loads((training/'训练状态.json').read_text(encoding='utf-8-sig'))
    if memory['memory_version'] != 78 or memory['current_policy']['policy_version'] != 70 or training_state['last_memory_version'] != 78:
        raise DetailsError('已读取的v70/v78记忆发生变化，本次建议须重新复核')
    report['iteration_model_reference'].update(rule_version=70, rules_version=70, policy_version=70,
        memory_version=78, state_memory_version=78, read=True,
        how_applied='已读取v70规则和v78记忆；R001/R003联合识别黄金股结构尚在但动量减速，R007禁止未破关键支撑就机械清仓。以减持前组合复核20%看空保护，不把Agent作为独立全仓决策器，也不把实际30%成交当成建议。候选及历史样本不证明预测优势。')
    report['iteration_model_reference']['files'] = [dict(path='../workspace-materials/ETF做题记录/'+name,
        size_bytes=(training/name).stat().st_size,
        modified_at_utc=datetime.fromtimestamp((training/name).stat().st_mtime,timezone.utc).isoformat())
        for name in ['训练规则.md','训练状态.json','优化记忆.json']]
    for row in agent['instruments']:
        row['agent_target_weight_pct'] = targets[row['code']]
        row['recommendation_reference'] = 'pre_gold_sale_reassessment'
        if row['code'] == '517520':
            reason = '以减持前黄金股为起点：周一阶+0.015663、二阶−0.012236，周上升结构仍在但连续减速；日一阶−0.004842、二阶−0.001522且MACD绿柱扩大，建议先减原份额20%、保留80%。2.071尚未失守且价格高于周主要均线，反对直接清仓；日动量未恢复，不支持原仓满额不动。20%是主观风险幅度，不是已验证最优值。'
            row.update(agent_reason=reason, multi_parameter_conclusion=reason,
                       agent_recommendation='多参数看空保护：黄金股减持原份额20%、保留80%；酒保持。仅减持已有仓位并转现金，不新建或轮换；不是按已减持30%的结果倒推，不指示当前回补。',
                       proposed_trade_fraction=-suggested_sale, trade_action='reduce',
                       proposed_trade_basis='gold_shares_before_APH-20260907-003',
                       position_size_reason='周结构未破而日周动量减速，选择先削减五分之一敞口、保留主要反弹参与；比例是本次判断而非模型输出。',
                       actual_followup='用户已完成30%减持，实际记录不改，本次目标差额不自动回补。')
            row['forward_analysis']['return_thesis'] = reason
            row['forward_analysis']['opportunity_cost'] = '相对减持前不操作，减少五分之一黄金股下跌敞口，也放弃同等反弹收益；酒不动，不为轮换强行建立银行新仓。'
    report['comparison']['rows'] = [dict(code=c, mechanical_weight_pct=report['mechanical_baseline']['b0207_weights_pct'][c],
        agent_target_pct=targets[c], current_carry_forward_pct=weights[c],
        difference_pct=round(targets[c]-report['mechanical_baseline']['b0207_weights_pct'][c],2)) for c in targets]
    report['comparison']['rows'].append(dict(code='CASH',mechanical_weight_pct=report['mechanical_baseline']['b0207_cash_pct'],
        agent_target_pct=target_cash_pct,current_carry_forward_pct=cash_pct,difference_pct=target_cash_pct-report['mechanical_baseline']['b0207_cash_pct']))
    report['comparison']['scope'] = 'pre_sale_agent_vs_baseline_vs_executed_carry_forward'
    report['comparison']['note'] = agent['note']
    html_path = local_path(root, report['html_path'])
    page = _render_page(root, report, html_path)
    encoded = (json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    return {root/'app/reports/monthly/latest.json': encoded,
            root/f'app/reports/monthly/{report["generated_date"]}.json': encoded,
            html_path: page.encode('utf-8')}, estimate


def run(root=ROOT, commit=False, display_only=False):
    if commit:
        raise DetailsError(
            'LEGACY_BLOCKED: refresh_portfolio_allocation只保留历史检查，不得写入当前月报或持仓'
        )
    outputs, estimate = build(root, display_only=display_only)
    original = {p:p.read_bytes() for p in outputs}
    if commit:
        lock = root / '.monthly-pipeline.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        written = []
        try:
            refreshed, _ = build(root, display_only=display_only)
            if refreshed != outputs or any(p.read_bytes() != original[p] for p in outputs):
                raise DetailsError('发布源发生变化')
            for p, data in outputs.items():
                if data != original[p]:
                    atomic(p, data)
                    written.append(p)
        except Exception:
            for p in reversed(written):
                atomic(p, original[p])
            raise
        finally:
            os.close(fd)
            lock.unlink()
    return {'status': 'PUBLISHED' if commit else 'CHECK_PASSED', 'estimate': estimate}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', action='store_true')
    parser.add_argument('--display-only', action='store_true',
                        help='仅迁移已审核历史月报的展示语义，不改报告JSON或当前持仓')
    args = parser.parse_args()
    print(json.dumps(run(commit=args.commit, display_only=args.display_only), ensure_ascii=False, indent=2))
