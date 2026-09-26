"""Publish a reviewed non-cycle snapshot without new targets or actual trades.

Consumes Agent-authored evidence and an existing frozen-model reference.
Uses monthly_details for every ETF row; does not compute or train any model.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from lxml import html

ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != ROOT / 'model_iteration/scripts']
sys.path.insert(0, str(ROOT / 'model_iteration'))
from scripts.monthly_details import render_details, ORDER, DetailsError
from scripts.agent_forward_evidence import load_forward_evidence


def read(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DetailsError('输入路径越界')
    return json.loads(path.read_text(encoding='utf-8'))


def text(node, value):
    for child in node.iter():
        child.text = None
        if child is not node:
            child.tail = None
    node.text = value


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def build(root, review_path, baseline_path):
    state = read(root, 'portfolio_state.json')
    current = read(root, 'app/reports/monthly/latest.json')
    review, baseline = read(root, review_path), read(root, baseline_path)
    features = read(root, 'app/features/latest.json')
    manifest = read(root, 'data_manifest.json')
    cutoff, run = features['as_of'], features['data_run_id']
    if current['generated_date'] != review['generated_date']:
        raise DetailsError('本入口只刷新当前日期报告；跨日期请使用正式发布流程')
    if review['generated_date'] != '2026-09-07' or review.get('portfolio_event_id') != 'APH-20260907-003':
        raise DetailsError('此发布修复仅适用于9月7日已确认持仓；不能跨期重放固定文案')
    if review.get('scope') != 'analysis_refresh_only_no_new_target_no_trades' or not review.get('publication_authorization'):
        raise DetailsError('缺少只更新分析的明确授权')
    if review['portfolio_event_id'] != state['actual_portfolio_history']['latest_event_id']:
        raise DetailsError('复核后实际持仓已变化')
    for value in (review, baseline):
        if value['data_as_of'] != cutoff or value['data_run_id'] != run:
            raise DetailsError('预测/分析与行情不属于同一批次')
    if manifest['as_of'] != cutoff or manifest['run_id'] != run:
        raise DetailsError('行情与参数不同批次')
    if [r['code'] for r in review['instruments']] != list(ORDER):
        raise DetailsError('Agent列表必须完整且固定排序')
    if baseline['fixed_priority_order'] != list(ORDER):
        raise DetailsError('机械列表固定排序错误')
    if abs(sum(baseline['b0207_weights_pct'].values()) + baseline['b0207_cash_pct'] - 100) > 1e-8:
        raise DetailsError('机械参考合计不是100%')
    evidence = load_forward_evidence(root, data_as_of=cutoff, data_run_id=run, instruments=features['instruments'])
    rows = copy.deepcopy(review['instruments'])
    p = state['last_confirmed_portfolio']
    for row in rows:
        code = row['code']
        f = features['instruments'][code]
        if row['quantitative_evidence'] != f or row['forward_evidence'] != evidence[code]:
            raise DetailsError(code + '复核证据与当前数据不一致')
        if not row.get('agent_direction') or any(word in row['agent_direction'] for word in ('待推理', '待复核')):
            raise DetailsError(code + '没有实际Agent判断')
        if row.get('new_position_eligible') or row.get('proposed_trade_fraction') != 0 or row.get('trade_action') not in ('hold', 'hold_with_exit', 'avoid_new'):
            raise DetailsError('本入口不得发布新的调仓目标或订单')
        if row.get('agent_unconstrained_weight_pct') is not None:
            raise DetailsError('本入口不能批准新的Agent目标比例')
        keys = ('weekly_first_positive', 'weekly_second_nonnegative', 'daily_first_positive', 'daily_second_nonnegative', 'daily_close_above_ma20', 'weekly_close_above_ma20', 'volume_confirmed')
        if any(not row.get('evidence_review', {}).get(key) for key in keys):
            raise DetailsError(code + '七项必查解释缺失')
        forward = row.get('forward_analysis', {})
        if (forward.get('information_cutoff') != cutoff or forward.get('reference_raw_close') != evidence[code]['daily'][-1]['raw_close']
                or not forward.get('horizon', {}).get('end')):
            raise DetailsError(code + '预测对象/期限错误')
        for key in ('current_state', 'observed_change', 'return_thesis', 'indicator_price_distinction', 'entry_quality', 'priced_in_risk', 'upside_basis', 'downside_basis', 'cost_and_execution', 'opportunity_cost', 'invalidation', 'review_date'):
            if not forward.get(key):
                raise DetailsError(code + '未来分析缺项:' + key)
        for scenario in ('up', 'sideways', 'down'):
            if any(not forward.get('scenarios', {}).get(scenario, {}).get(k) for k in ('condition', 'price_path', 'action')):
                raise DetailsError(code + '三路径缺项')
        d, w = f['daily'], f['weekly']
        severe = w['dif_second_normalized'] <= -1 and d['dif_first_raw'] < 0 and d['dif_second_raw'] < 0 and f['volume']['volume_to_20d_median'] < .6
        if severe and (row['agent_direction'].startswith('看多') or row['momentum_state'] != 'deteriorating'):
            raise DetailsError(code + '违反强恶化门禁')
        row.update(baseline_direction='看多' if baseline['champion_positions'][code] > 0 else '看空',
                   baseline_weight_pct=baseline['b0207_weights_pct'][code],
                   actual_holding=copy.deepcopy(p['positions'][code]), actual_weight_pct=p['weights_pct'][code],
                   agent_unconstrained_weight_pct=None, analysis_status='reviewed_same_batch',
                   analysis_source={'report': review_path, 'data_as_of': cutoff, 'data_run_id': run})
        conditions = [d['dif_first_raw'] < 0, d['dif_second_raw'] < 0, w['dif_first_raw'] < 0, w['dif_second_raw'] < 0, f['relative_strength']['return_20d'] < -.05]
        row['guard'] = {'triggered': all(conditions), 'condition_count': sum(conditions), 'automatic_execution': False}
    training_dir = root.parent / 'workspace-materials/ETF做题记录'
    memory = json.loads((training_dir / '优化记忆.json').read_text(encoding='utf-8'))
    training_state = json.loads((training_dir / '训练状态.json').read_text(encoding='utf-8'))
    iteration = copy.deepcopy(review['iteration_model_reference'])
    if not iteration.get('read') or not iteration.get('how_applied') or iteration['memory_version'] != memory['memory_version'] or iteration['state_memory_version'] != training_state['last_memory_version']:
        raise DetailsError('决策记忆版本或读取声明不一致')
    iteration['files'] = []
    for name in ('训练规则.md', '训练状态.json', '优化记忆.json'):
        file = training_dir / name
        stat = file.stat()
        iteration['files'].append({'path': '../workspace-materials/ETF做题记录/' + name, 'size_bytes': stat.st_size, 'modified_at_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
    already_reviewed = (current.get('analysis_status') == 'reviewed_same_batch'
                        and current.get('analysis_review_path') == review_path
                        and all(all(old.get(k) == value for k, value in new.items())
                                for old, new in zip(current.get('agent_recommendation', {}).get('instruments', []), review['instruments'])))
    report = copy.deepcopy(current)
    report.update(schema_version='monthly-report-v1', report_status='reviewed_monitoring_actual_holdings_unchanged',
        analysis_status='reviewed_same_batch', generated_at=current['generated_at'] if already_reviewed else datetime.now(timezone.utc).isoformat(),
        decision_type='user_confirmed_portfolio_snapshot_non_cycle', execution_allowed=False,
        execution_status='no_new_orders', monitoring_baseline_path=baseline_path, mechanical_baseline=baseline,
        agent_recommendation={'status':'reviewed_no_new_target', 'unconstrained_weights_pct':None, 'cash_pct':None,
                              'note':review['comparison_note'], 'instruments':rows},
        last_confirmed_portfolio=copy.deepcopy(p), user_decision=copy.deepcopy(state['user_decision']),
        user_final_target={'representation':'confirmed_constraints_not_estimated_weights', 'constraints':copy.deepcopy(p['conversion']),
                           'weights_pct':copy.deepcopy(p['weights_pct']), 'cash_pct':p['cash_pct']},
        portfolio_history_event_id=review['portfolio_event_id'], iteration_model_reference=iteration,
        gold_macro_review=review['gold_macro_review'], risk_notes=review['risk_notes'],
        analysis_review_path=review_path, publication_authorization=review['publication_authorization'],
        weekly_bearish_guard={r['code']:r['guard'] for r in rows},
        comparison={'scope':'monitoring_no_new_allocation','note':review['comparison_note'], 'rows':[
            {'code':r['code'],'mechanical_weight_pct':r['baseline_weight_pct'],'agent_action':r['agent_recommendation'],
             'actual_weight_pct':r['actual_weight_pct'],'agent_target_pct':None,'difference_pct':None,'reason':r['agent_reason']} for r in rows
        ] + [{'code':'CASH','mechanical_weight_pct':baseline['b0207_cash_pct'],'agent_action':'保留用户已确认卖出所得现金，不自动再投资',
              'actual_weight_pct':p['cash_pct'],'agent_target_pct':None,'difference_pct':None,'reason':'实际余额未知；不以机械现金0%覆盖用户决定'}]})
    report['monitoring_data_as_of'] = cutoff
    report['monitoring_data_run_id'] = run
    report['policy_data_as_of'] = cutoff
    report['policy_snapshot_path'] = None
    source = (root / current['html_path']).read_text(encoding='utf-8')
    tree = html.document_fromstring(source)
    by_class = lambda c: tree.xpath('//*[@class="' + c + '"]')
    text(tree.xpath('//title')[0], '投资决策月报｜2026-09-07｜已复核监控与真实持仓')
    text(by_class('subtitle')[0], '冻结模型直接预测 · 8只ETF同批次复核 · 实际持仓不变')
    text(by_class('stamp')[0], f'报告2026-09-07 · 本地日周收盘{cutoff} · 下次决策2026-09-19 · 批次{run}')
    text(by_class('callout')[0], review['summary'] + '模型80%是单标的档位，黄金股B0207组合参考为60%，都不是实际账户权重。')
    rule_text = ['同批次分析已复核','不训练、不推进模型',f'本地行情截至{cutoff}','不新增目标、不重复交易']
    for node, value in zip(list(by_class('rulebar')[0]), rule_text):
        text(node, value)
    cards = by_class('plan-card')
    text(cards[0].xpath('./h3')[0], 'B0207机械参考 · 非新调仓目标')
    lines = ['创业板0% · 黄金股60% · 稀土25% · 创新药0%。',
             '纳指0% · 煤炭10% · 银行0% · 酒5% · 现金0%。',
             '517520直接应用CH_S064_1冻结参数；仓位档位80%，核心三第2位不变。',
             'Agent与机械可不同；当前实际比例未知，不填造新100%目标。公司行动完整账本未核实只作当前预测风险披露。']
    # Allocation numbers come only from the existing verified reference.
    lines[0] = ' · '.join(f'{r["name"].split("ETF")[0]}{r["baseline_weight_pct"]:g}%' for r in rows[:4]) + '。'
    lines[1] = ' · '.join(f'{r["name"].split("ETF")[0]}{r["baseline_weight_pct"]:g}%' for r in rows[4:]) + f' · 现金{baseline["b0207_cash_pct"]:g}%。'
    for node, value in zip(cards[0].xpath('.//li'), lines): text(node, value)
    text(cards[2].xpath('./h3')[0], '黄金股：机械看多，Agent震荡偏弱')
    paragraphs = cards[2].xpath('./p')
    gold = review['gold_macro_review']
    text(paragraphs[0], gold['technical'] + ' 观察2.071支撑及2.262收复，周结构未全面失效；不自动回补已留现金。')
    text(paragraphs[1], gold['real_yield'] + ' 通胀补偿2.35%，1/5/20观测0/+4/+10bp；属A股收盘后背景。CPI日期待官方核对，FOMC9月15—16日。美元/汇率、国际金价、矿企与折溢价资料缺失，结论不作概率承诺。')
    paragraphs[1].set('title', '；'.join(str(v) for k,v in gold.items() if k not in ('technical',)))
    text(cards[3].xpath('./h3')[0], '本周路径、复核与执行边界')
    text(cards[3].xpath('./p')[0], '观察期9月7—11日：黄金股收复2.262并恢复日动量则复核偏弱判断；2.071—2.262震荡保持观察；跌破2.071重新提交减仓意见。酒收复0.438观察延续，失守0.418—0.417复核退出。只监控、不自动交易。现金可能错过反弹，也减少黄金股回撤敞口。9月19日正式决策需9月18日数据；目前不估算交易份数、费用或收益。')
    legend = by_class('legend-item')[-1].xpath('./span')[0]
    text(legend, f'图、机械参考与Agent均使用本地{cutoff}日周收盘；不是盘中行情。')
    for group, row in zip(tree.xpath('//svg/g/g[title]'), rows):
        node = group.xpath('./title')[0]
        node.text = re.sub(r'｜B0207.*$', '', node.text or '') + f'｜B0207{row["baseline_direction"]}｜Agent{row["agent_direction"]}'
    text(by_class('warning')[0], f'8只ETF当前分析已复核；本地行情截至2026-09-04，本次未联网更新。仅更新监控，不是新正式调仓。最新实际持仓事件APH-20260907-003；黄金股剩余原份额70%、酒与现金不变，金额待补。迭代记忆v{iteration["memory_version"]}/规则v51仅作证据，批量205/400低于基准216/400，不宣称预测已验证有效。')
    rendered = html.tostring(tree, encoding='unicode', doctype='<!DOCTYPE html>').replace('viewbox=', 'viewBox=')
    rendered = render_details(rendered, root, report)
    before = html.document_fromstring(render_details(source, root, report))
    after = html.document_fromstring(rendered)
    if [n.tag for n in before.iter()] != [n.tag for n in after.iter()]:
        raise DetailsError('改动了锁定HTML结构')
    for tag in ('style','script'):
        if [n.text for n in before.xpath('//' + tag)] != [n.text for n in after.xpath('//' + tag)]:
            raise DetailsError('改动了锁定样式或脚本')
    for tree_copy in (before, after):
        for node in tree_copy.xpath('//svg//title'): node.text = ''
    if [html.tostring(n) for n in before.xpath('//svg')] != [html.tostring(n) for n in after.xpath('//svg')]:
        raise DetailsError('同批次更新不应改变象限图几何和颜色')
    if any(word in rendered for word in ('待推理','待复核','机械推理当前阻断')):
        raise DetailsError('当前报告仍含过时占位状态')
    next_state = copy.deepcopy(state)
    next_state['current_monthly_report'].update(status='reviewed_monitoring_actual_holdings_unchanged',
        monitoring_html_status='reviewed_same_batch',decision_type=report['decision_type'],
        notice='已复核8只ETF机械参考及Agent；实际持仓不变，金额未知不估算。',analysis_review_path=review_path,
        monitoring_baseline_path=baseline_path)
    next_state['weekly_bearish_guard_state'] = {'status':'reviewed_same_batch', 'data_as_of':cutoff,
        'data_run_id':run,'evaluated_holdings':p['held_codes'], 'instruments':{c:report['weekly_bearish_guard'][c] for c in p['held_codes']},'automatic_execution':False}
    next_state['execution_status']['weekly_guard_status'] = 'reviewed_same_batch_no_automatic_trade'
    outputs = {'app/reports/monthly/latest.json':encode(report),
               f'app/reports/monthly/{review["generated_date"]}.json':encode(report),
               current['html_path']:rendered.encode('utf-8'),'portfolio_state.json':encode(next_state)}
    return outputs, report


def atomic(path, payload):
    fd, name = tempfile.mkstemp(prefix='.monitoring-publish-', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        if temp.read_bytes() != payload: raise DetailsError('暂存内容不一致')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def publish(root, review_path, baseline_path, *, commit=False):
    outputs, report = build(root, review_path, baseline_path)
    original = {p:(root/p).read_bytes() for p in outputs}
    if all(original[p] == data for p,data in outputs.items()): return {'status':'UNCHANGED'}
    result = {'status':'CHECK_PASSED','rows':8,'data_as_of':report['data_as_of'],'real_holdings_changed':False}
    if not commit: return result
    lock = root / '.monthly-pipeline.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    changed=[]
    try:
        checked, _ = build(root, review_path, baseline_path)
        # generated_at may differ; all content other than timestamp must agree.
        for key in outputs:
            if (root/key).read_bytes() != original[key]: raise DetailsError('报告检查后发生变化')
        outputs=checked
        archive=root/'archive/monthly_publications/2026-09-07_before_review'
        for relative,data in original.items():
            destination=archive/relative
            if destination.exists() and destination.read_bytes() != data:
                raise DetailsError('已存在不同的发布前存档，拒绝覆盖')
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists(): atomic(destination,data)
        for relative,data in outputs.items():
            atomic(root/relative,data); changed.append(relative)
        for relative,data in outputs.items():
            if (root/relative).read_bytes()!=data: raise DetailsError('发布后内容核对失败')
        actual=read(root,'portfolio_state.json')['last_confirmed_portfolio']
        if actual!=json.loads(original['portfolio_state.json'])['last_confirmed_portfolio']:
            raise DetailsError('实际持仓被改变')
        return {**result,'status':'PUBLISHED','html':report['html_path']}
    except Exception:
        for relative in reversed(changed): atomic(root/relative,original[relative])
        raise
    finally:
        os.close(fd); lock.unlink()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--commit',action='store_true')
    args=parser.parse_args()
    print(json.dumps(publish(ROOT,args.review,args.baseline,commit=args.commit),ensure_ascii=False))
