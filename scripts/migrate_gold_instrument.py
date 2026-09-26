"""Explicit 518600 -> 517520 migration. No training, baseline replay or trading.

--prepare-data fetches only 517520 into a private staging file. --commit requires
the user's completed conversion confirmation; preserves old identity/history.
Validation uses exact data/parameter comparisons, never file fingerprints.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
from datetime import date, datetime
from pathlib import Path

import update_data as updater

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'archive/instrument_replacements/518600_to_517520'
STAGE = ROOT / '.gold_replacement'
EVENT = 'APH-20260907-001'
NEW = {
    'code': '517520', 'name': '黄金股ETF永赢', 'exchange': 'SSE',
    'symbol': 'sh517520', 'inception_date': '2023-10-24',
    'sector': '沪深港黄金产业股票', 'risk_cluster': '黄金产业权益',
    'price_tick': 0.001, 'price_limit_pct': 10, 'qdii': False,
    'benchmark': {'code': None, 'name': '中证沪深港黄金产业股票指数',
                  'provider_symbol': None, 'currency': 'CNY', 'timezone': 'Asia/Shanghai',
                  'lag_sessions': 0, 'fx_symbol': None, 'mapping_status': 'numeric_code_unverified'},
    'official_source': 'https://www.maxwealthfund.com/',
    'identity_evidence': 'https://fundf10.eastmoney.com/jbgk_517520.html',
    'identity_note': '身份及指数名称已核对公开基金资料；指数代码及官方产品直达页待核实，不沿用上海金映射',
    'hong_kong_stock_connect': True,
}


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def csv_rows(relative):
    return list(csv.DictReader(io.StringIO((ROOT / relative).read_text(encoding='utf-8-sig'))))


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def safe(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise ValueError('迁移目标越界')
    return path


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.migration.tmp')
    with temp.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def prepare():
    priority = read_json('model_iteration/configs/investment_priority.json')['priority']
    if len(priority) > 1 and priority[1] == '517520':
        validate()
        return {'status': 'already_committed', 'event_id': EVENT,
                'analysis_status': 'independent_full_cycle_completed'}
    manifest = read_json('data_manifest.json')
    cutoff = date.fromisoformat(manifest['as_of'])
    if cutoff >= datetime.now(updater.SHANGHAI).date() or manifest['market_status'] != 'closed':
        raise ValueError('本次替换只接受最近已完成且在今日之前的共同交易日')
    if (STAGE / 'prepared.json').exists():
        cached = json.loads((STAGE / 'prepared.json').read_text(encoding='utf-8'))
        if cached['as_of'] == cutoff.isoformat():
            return {'status': 'already_prepared', 'rows': len(cached['daily']), 'as_of': cached['as_of']}
        raise ValueError('已有不同截止日暂存；请先核对，禁止覆盖')
    requests, _ = updater._dependencies()
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://gu.qq.com/'})
    now = datetime.now(updater.SHANGHAI)
    run = now.strftime('%Y%m%dT%H%M%S')
    daily, warnings = updater._build_daily(session, NEW, cutoff, run, now.isoformat(), force_full=True)
    warnings += updater._validate(daily, NEW, 'daily')
    if not daily or daily[-1]['date'] != cutoff.isoformat():
        raise ValueError('新标的未达到共同截止日')
    if daily[0]['date'] < NEW['inception_date'] or any(r['code'] != '517520' for r in daily):
        raise ValueError('新标的身份或上市前价格非法')
    payload = {'as_of': cutoff.isoformat(), 'run_id': run, 'generated_at': now.isoformat(),
               'daily': daily, 'warnings': warnings, 'instrument': NEW}
    write_atomic(STAGE / 'prepared.json', json_bytes(payload))
    return {'status': 'prepared', 'rows': len(daily), 'start': daily[0]['date'], 'end': daily[-1]['date']}


def build_plan(prepared):
    """Construct all active config/data/state bytes in memory before publication."""
    plan = {}
    add = lambda path, value: plan.__setitem__(path, json_bytes(value))
    manifest = read_json('data_manifest.json')
    instruments = read_json('config/instruments.json')
    priority = read_json('model_iteration/configs/investment_priority.json')
    if priority['priority'][1] != '518600' or instruments['instruments'][1]['code'] != '518600':
        raise ValueError('源标的已变化，拒绝重复或错位替换')
    if manifest['as_of'] != prepared['as_of'] or manifest['market_status'] != 'closed':
        raise ValueError('准备后源批次截止日变化，需重新核对')
    old_order = list(priority['priority'])
    priority['priority'][1] = '517520'
    instruments['instruments'][1] = copy.deepcopy(NEW)
    instruments['verified_on'] = '2026-09-07'
    add('config/instruments.json', instruments)
    add('model_iteration/configs/investment_priority.json', priority)
    run, cutoff = prepared['run_id'], prepared['as_of']
    files, daily_by_code = [], {}
    for record in manifest['files']:
        relative = record['path']
        if Path(relative).name.startswith('518600_'):
            plan[f'{ARCHIVE}/{relative}'] = safe(relative).read_bytes()
            continue
        rows = csv_rows(relative)
        if len(rows) != record['rows'] or not rows:
            raise ValueError(f'源文件行数不一致: {relative}')
        if Path(relative).name == 'fund_snapshot.csv':
            old = next(r for r in rows if r['code'] == '518600')
            new = {k: '' for k in old}
            new.update(code='517520', metadata_date=cutoff, metadata_status='degraded',
                       source='identity_only_metadata_unavailable', source_url=NEW['identity_evidence'])
            new['missing_fields'] = '|'.join(k for k, v in new.items() if v == '')
            rows = [new if r['code'] == '518600' else r for r in rows]
        else:
            code = rows[0]['code']
            if rows[-1]['date'] != record['end'] or any(r['run_id'] != manifest['run_id'] for r in rows):
                raise ValueError('保留行情批次或日期不一致')
            if relative.endswith('_日线.csv'):
                daily_by_code[code] = rows
        for r in rows:
            r['run_id'] = run
        plan[relative] = csv_bytes(rows)
        # Only batch metadata changes for the other seven ETFs; values stay exact.
        if not relative.endswith('fund_snapshot.csv'):
            original = csv_rows(relative)
            for before, after in zip(original, rows):
                if {k:v for k,v in before.items() if k != 'run_id'} != {k:v for k,v in after.items() if k != 'run_id'}:
                    raise ValueError('保留标的行情发生变化')
        files.append({k:v for k,v in record.items() if k in ('path','rows','start','end') } | {'run_id': run})
    daily = prepared['daily']
    for r in daily:
        r['run_id'] = run
    daily_by_code['517520'] = daily
    for timeframe, label in [('daily','日线'),('weekly','周线'),('monthly','月线')]:
        rows = daily if timeframe == 'daily' else updater._aggregate(daily, timeframe)
        updater._validate(rows, NEW, timeframe)
        relative = f'数据/517520_黄金股ETF永赢_{label}.csv'
        plan[relative] = csv_bytes(rows)
        files.append({'path': relative,'rows':len(rows),'start':rows[0]['date'],'end':rows[-1]['date'],'run_id':run})
    dates = {c:rows[-1]['date'] for c,rows in daily_by_code.items()}
    if set(dates) != set(priority['priority']) or set(dates.values()) != {cutoff}:
        raise ValueError('八只ETF共同截止日不一致')
    new_manifest = {k:v for k,v in manifest.items() if k in ('schema_version','market_status','quality_status','report_status')}
    new_manifest.update(run_id=run, as_of=cutoff, generated_at=prepared['generated_at'],
                        latest_dates=dates, files=sorted(files, key=lambda r:r['path']),
                        quality_status='passed_with_warnings', report_status='degraded',
                        warnings=[x for x in manifest.get('warnings',[]) if '518600' not in str(x)] + prepared['warnings'] + ['517520指数代码、官方产品直达页及公司行动完整性待核实'])
    add('data_manifest.json', new_manifest)
    plan['最新行情快照.csv'] = csv_bytes([{'code':c,'name':next(i['name'] for i in instruments['instruments'] if i['code']==c),
        'date':cutoff,'raw_close':daily_by_code[c][-1]['raw_close'],'adj_close':daily_by_code[c][-1]['adj_close'],
        'data_status':'closed','run_id':run} for c in priority['priority']])
    add('update_log.json', {'mode':'single_instrument_replacement','status':'degraded','run_id':run,'as_of':cutoff,'replacement':{'old_code':'518600','new_code':'517520'}})

    cfg = read_json('model_iteration/configs/etf_518600.json')
    state = read_json(f"model_iteration/{cfg['workspace']}/weekly_rolling/state.json")
    champion = {'id':state['champion']['id'], 'spec':copy.deepcopy(state['champion']['spec'])}
    provenance = {'source_code':'518600','source_workspace':cfg['workspace'], 'source_champion_id':champion['id'],
                  'source_trained_anchor_index':state['latest_anchor_index'], 'transfer_confirmed_on':'2026-09-07',
                  'mode':'initial_parameters_only','target_training_history':'independent_full_cycle_required',
                  'performance_status':'BOOTSTRAP_NOT_CURRENT'}
    target_cfg = {'etf':'517520','name':NEW['name'],'workspace':'etf_517520',
        'coverage':{'start':daily[0]['date'],'end':cutoff}, 'expected_anchors':None,
        'horizon':cfg['horizon'],'cost_bps':cfg['cost_bps'], 'champion':champion,
        'execution_policy':None, 'training_disabled':True,
        'model_status':'bootstrap_requires_independent_full_cycle',
        'initial_champion_provenance':provenance}
    add('model_iteration/configs/etf_517520.json', target_cfg)
    add('model_iteration/etf_517520/weekly_rolling/state.json', {'etf':'517520','latest_anchor_index':None,
        'champion':champion,'state_kind':'bootstrap_requires_independent_full_cycle',
        'training_disabled':True,'provenance':provenance})
    add('model_iteration/etf_517520/weekly_rolling/corporate_actions/517520_ca_events.json', {
        'schema_version':'ca-events-v1.0.0','etf':'517520', 'coverage':{'start':daily[0]['date'],'end':cutoff,
        'verified':False, 'source':['https://fundf10.eastmoney.com/fhsp_517520.html'],
        'note':'迁移引导阶段尚未独立核实公司行动；完成核实前不得训练或推理，也不得复制518600公司行动。'}, 'events':[]})
    migration = {'schema_version':'instrument-replacement-v1','status':'completed_user_confirmed_details_pending',
        'old_code':'518600','new_code':'517520','priority_position':2,'b0207_group':'core','data_as_of':cutoff,
        'data_run_id':run,'event_id':EVENT,'model_transfer':provenance,'old_priority':old_order,'new_priority':priority['priority'],
        'history_archive':ARCHIVE, 'analysis_status':'bootstrap_requires_independent_full_cycle',
        'limitations':['新标的是黄金产业股票，不是实物黄金','指数代码及部分基金元数据待核实','公司行动与自身全周期回放完成前，当前机械推理须失败关闭','本次只建立迁移引导状态，不生成Agent判断、不重算B0207']}
    add('config/gold_instrument_replacement.json', migration)

    p = read_json('portfolio_state.json')
    previous = copy.deepcopy(p['last_confirmed_portfolio'])
    current = copy.deepcopy(previous)
    current.update(confirmed_at='2026-09-07',trade_date=None,source='用户明确确认518600全部换为517520，成交明细后补')
    current['held_codes'] = ['517520' if x=='518600' else x for x in current['held_codes']]
    for key in ('weights_pct','market_values_cny'):
        current[key]['518600'] = 0
        current[key]['517520'] = None
    current['positions']['518600'] = {'shares':0,'status':'not_held','note':'用户确认全部换出；成交明细待补'}
    current['positions']['517520'] = {'shares':None,'status':'held_confirmed','note':'继承原黄金资金配置；不同基金份数不得一比一照抄，成交价/份数/金额待补'}
    current['prior_conversion'] = current.pop('conversion')
    current['prior_conversion']['scope_note'] = '仅记录9月5日煤炭转银行与酒；原黄金份额不动约束已被本次换仓取代'
    conversion = {'from_code':'518600','to_code':'517520','sell_fraction_of_existing_shares':1,
                  'allocation_of_proceeds':{'517520':1},'status':'completed_user_confirmed_details_pending',
                  'trade_date':None,'fills':None,'fees_cny':None,'net_proceeds_cny':None,'share_equivalence':False}
    current['conversion'] = conversion
    current['valuation_basis'] = '未提供换仓后市值，不能继承旧基金份数、价格或旧估值作为517520实际值'
    p.update(last_updated='2026-09-07',data_run_id=run,decision_revision='2026-09-07-gold-stock-completed-conversion')
    p['last_confirmed_portfolio'] = current
    p['execution_status']['completed_steps'].append('2026-09-07_518600_fully_converted_to_517520_user_confirmed')
    p['execution_status']['weekly_guard_status'] = '517520_pending_own_data_review'
    p['weekly_bearish_guard_state'] = {'status':'pending_after_instrument_replacement','evaluated_holdings':current['held_codes'],
        'notice':'原518600保护检查不适用于517520；待使用新标的客观参数复核，不产生订单'}
    p['user_decision'] = {'status':'modified_by_user','confirmed_at':'2026-09-07',
        'confirmation_source':'explicit_user_instruction','instruction':'将518600替换为黄金股ETF永赢517520，继承仓位、冻结冠军和排序',
        'fill_confirmation':'已完成换仓，成交明细后补','final_target_weights_pct':current['weights_pct'],
        'final_cash_pct':0,'target_representation':'confirmed_full_conversion_not_numeric_share_equivalence','final_target_constraints':conversion}
    p['current_monthly_report'].update(status='instrument_replacement_snapshot_analysis_pending',
        monitoring_html_status='pending_517520_analysis_not_old_gold_signals',report_date='2026-09-07',
        html='scripts/投资决策月报2026-09-07.html',json='app/reports/monthly/latest.json',
        shortcut='投资决策月报2026-09-07.lnk',monitoring_data_run_id=run,decision_date='2026-09-07',
        decision_type='user_confirmed_instrument_replacement',execution_rule='已换仓，明细待补；非新B0207决策')
    history = read_json('app/portfolio_history/actual_holdings.json')
    if any(x['event_id']==EVENT for x in history['entries']):
        raise ValueError('换仓记录已存在，禁止重复追加')
    history['entries'].append({'event_id':EVENT,'confirmed_at':'2026-09-07','source':'用户明确确认',
        'change_type':'full_instrument_conversion','user_instruction':p['user_decision']['instruction'],
        'fill_confirmation':p['user_decision']['fill_confirmation'],'previous_snapshot':previous,'current_snapshot':current,
        'change_summary':{'sold_code':'518600','bought_code':'517520','bank_and_wine_unchanged':True,'fills_pending':True},
        'report_binding':{'html':'scripts/投资决策月报2026-09-07.html','json':'app/reports/monthly/2026-09-07.json'}})
    p['actual_portfolio_history'].update(latest_event_id=EVENT, record_count=len(history['entries']))
    add('portfolio_state.json', p)
    add('app/portfolio_history/actual_holdings.json', history)

    gold = read_json('config/gold_analysis.json')
    # This is current configuration, not source evidence or a historical report.
    gold = json.loads(json.dumps(gold,ensure_ascii=False).replace('518600','517520').replace('广发黄金ETF','黄金股ETF永赢'))
    gold['schema_version'] = 'gold-analysis-v2-gold-equities'
    study = gold['historical_event_studies'][0]
    study.update(window_file='辅助数据/黄金宏观/黄金股ETF517520_FOMC事件窗口.csv',
                 summary_file='辅助数据/黄金宏观/黄金股ETF517520_FOMC事件窗口汇总.json')
    gold['legacy_event_studies'] = [{'instrument':'518600','role':'physical_gold_historical_context_only',
        'window_file':'辅助数据/黄金宏观/黄金ETF_FOMC事件窗口.csv','summary_file':'辅助数据/黄金宏观/黄金ETF_FOMC事件窗口汇总.json'}]
    gold['asset_distinction'] = {'asset_class':'gold_industry_equities','not_physical_gold':True,
        'required_additional_evidence':['黄金矿企利润与估值','开采成本、产量及经营风险','沪深港股风险偏好与交易日错位','持仓集中度、跟踪误差及折溢价'],
        'missing_rule':'逐项检查，缺失写unavailable；不以国际黄金支撑位替代517520自身支撑，不继承518600事件收益和看多结论',
        'historical_rule':'2023年加息4次均早于517520成立，517520自身对应收益不可得，不得拼接518600数据'}
    add('config/gold_analysis.json', gold)
    for relative in ('投资策略.md','AGENTS.md'):
        content = safe(relative).read_text(encoding='utf-8')
        # Current rule scope changes; dated prior holdings remain historical.
        content = content.replace('518600 广发黄金ETF','517520 黄金股ETF永赢').replace('黄金ETF（518600）','黄金股ETF（517520）')
        content = content.replace('同日518600','同日517520').replace('交易日518600','交易日517520').replace('固定使用518600复权收盘价','固定使用517520自身复权收盘价')
        content = content.replace('黄金ETF_FOMC事件窗口', '黄金股ETF517520_FOMC事件窗口')
        content = content.replace('创业板 → 黄金 →', '创业板 → 黄金股 →').replace('创业板、黄金、稀土','创业板、黄金股、稀土')
        content = content.replace('最新实际成交确认登记于2026-09-05', '上一笔实际成交确认登记于2026-09-05')
        content += '\n\n## 2026-09-07 已确认黄金标的替换（当前事实优先）\n\n' + (
            '用户确认518600已全部换仓为517520黄金股ETF永赢，成交日期、价格、份数、金额和费用待补。当前持有517520、512800、512690，银行和酒不变；此前“黄金原份额不动”仅是9月5日历史指令，已被本次换仓取代。不同基金不能继承数值份数或价格。\n\n'
            '517520保留第2优先级及核心三身份；B0207档位、85%/15%预算和28天周期不变。518600冠军只作为初始参数来源，不复制训练进度、晋级台账或历史收益。迁移完成后必须先核实517520公司行动并完成自身全周期回放；在此之前训练与当前推理均失败关闭。\n\n'
            '517520跟踪黄金产业股票而非实物黄金。保留实际利率、通胀补偿、美元汇率、FOMC、央行需求观察，同时必须检查矿企利润/成本/产量、估值、股市风险偏好、沪深港交易时段和基金折溢价。缺失项明确unavailable。旧518600支撑位、宏观反应系数、事件收益和多空判断不能直接继承。\n\n'
            '旧黄金ETF_FOMC事件窗口及旧月报只作518600历史背景，不能写为517520业绩；517520上市前/不足前后15日的窗口标记不可得。新行情只用517520真实日周数据，不拼接旧基金，月线不参与分析。当前换仓快照不是新的B0207决策；分析尚未复核时明确待分析而不是搬用旧方向。\n')
        plan[relative] = content.encode('utf-8')
    return plan, p, migration


def commit():
    if read_json('model_iteration/configs/investment_priority.json')['priority'][1] == '517520':
        # A completed migration remains complete after later portfolio events.
        try:
            validate()
        except (AssertionError, KeyError, ValueError, OSError) as exc:
            raise ValueError('存在不完整迁移，需核对，不能重复执行') from exc
        return {'status':'already_committed','event_id':EVENT}
    prepared = json.loads((STAGE / 'prepared.json').read_text(encoding='utf-8'))
    lock = ROOT / '.update.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    written, backups = [], {}
    try:
        plan, p, migration = build_plan(prepared)
        for relative in plan:
            path = safe(relative)
            if path.exists():
                backups[relative] = path.read_bytes()
                archive_path = safe(f'{ARCHIVE}/before/{relative}')
                if archive_path.exists():
                    if archive_path.read_bytes() != backups[relative]:
                        raise ValueError('已有不同来源备份，禁止覆盖')
                else:
                    write_atomic(archive_path, backups[relative])
        # Formal dated reports and legacy model state are deliberately untouched.
        for relative, content in plan.items():
            write_atomic(safe(relative), content)
            written.append(relative)
        import calculate_features
        feature_paths = ['app/features/latest.json', f"app/features/{prepared['run_id']}.json"]
        for relative in feature_paths:
            if safe(relative).exists():
                backups[relative] = safe(relative).read_bytes()
                archive_path = safe(f'{ARCHIVE}/before/{relative}')
                if not archive_path.exists():
                    write_atomic(archive_path, backups[relative])
            written.append(relative)
        features = calculate_features.calculate(ROOT/'data_manifest.json', ROOT/'app/features',
                    verify_integrity_digest=False, skip_research_metrics=True)
        from render_gold_replacement import render_snapshot
        report, html = render_snapshot(ROOT, p, migration, features)
        outputs = {'app/reports/monthly/latest.json':json_bytes(report),
                   'app/reports/monthly/2026-09-07.json':json_bytes(report),
                   'scripts/投资决策月报2026-09-07.html':html.encode('utf-8')}
        for relative, content in outputs.items():
            if safe(relative).exists():
                backups[relative] = safe(relative).read_bytes()
                archive_path = safe(f'{ARCHIVE}/before/{relative}')
                if not archive_path.exists():
                    write_atomic(archive_path, backups[relative])
            write_atomic(safe(relative), content)
            written.append(relative)
        validate(prepared)
        return {'status':'committed','event_id':EVENT,'data_as_of':prepared['as_of'],
                'daily_rows':len(prepared['daily']),'champion':migration['model_transfer']['source_champion_id'],
                'analysis':'independent_full_cycle_required_before_current_inference'}
    except Exception:
        for relative in reversed(written):
            if relative in backups:
                write_atomic(safe(relative), backups[relative])
            elif safe(relative).is_file():
                safe(relative).unlink()
        raise
    finally:
        os.close(fd)
        lock.unlink()


def validate(prepared=None):
    priority = read_json('model_iteration/configs/investment_priority.json')
    cfg = read_json('model_iteration/configs/etf_517520.json')
    target = read_json('model_iteration/etf_517520/weekly_rolling/state.json')
    assert target['etf'] == cfg['etf'] == '517520'
    assert target['champion']['spec'] == cfg['champion']['spec']
    assert target['champion']['id'] == cfg['champion']['id']
    replacement = read_json('config/gold_instrument_replacement.json')
    if replacement.get('analysis_status') == 'independent_full_cycle_completed':
        completed = cfg['full_cycle_iteration']
        record = read_json(completed['iteration_record'])
        ca = read_json('model_iteration/etf_517520/weekly_rolling/corporate_actions/517520_ca_events.json')
        assert completed['status'] == 'completed'
        assert record['etf'] == '517520'
        assert record['latest_anchor_index'] == target['latest_anchor_index'] == completed['latest_anchor_index']
        assert record['resume_from'] == target.get('resume_from')
        assert record['usable_anchors'] == completed['usable_anchors']
        assert record['final_champion'] == target['champion']['id'] == completed['final_champion_id']
        assert record['promotions'] == completed['promotions']
        assert not target.get('pending_ai_reviews')
        assert cfg['training_disabled'] is False
        assert cfg.get('execution_policy') is None
        assert 'frozen_transfer' not in cfg and 'prediction_price_source' not in cfg
        assert ca['coverage']['verified'] is True and ca['coverage']['end'] >= cfg['coverage']['end']
        assert replacement['model_transfer']['target_trained_anchor_index'] == target['latest_anchor_index']
        assert replacement['model_transfer']['target_final_champion_id'] == target['champion']['id']
    else:
        source = read_json('model_iteration/etf_518600/weekly_rolling/state.json')['champion']
        assert target['champion']['spec'] == source['spec']
        assert target['champion']['id'] == source['id']
        assert target['latest_anchor_index'] is None and cfg['training_disabled'] is True
    old_priority = read_json(f'{ARCHIVE}/before/model_iteration/configs/investment_priority.json')
    old_priority['priority'][1] = '517520'
    assert priority == old_priority
    manifest = read_json('data_manifest.json')
    assert set(manifest['latest_dates']) == set(priority['priority'])
    assert set(manifest['latest_dates'].values()) == {manifest['as_of']}
    for record in manifest['files']:
        rows = csv_rows(record['path'])
        assert len(rows) == record['rows']
        assert all(r['run_id'] == manifest['run_id'] for r in rows)
    history = read_json('app/portfolio_history/actual_holdings.json')
    old_history = read_json(f'{ARCHIVE}/before/app/portfolio_history/actual_holdings.json')
    # Later user-confirmed trades do not invalidate the earlier migration.
    event_indexes = [i for i, entry in enumerate(history['entries']) if entry['event_id'] == EVENT]
    assert len(event_indexes) == 1
    migration_index = event_indexes[0]
    assert history['entries'][:migration_index] == old_history['entries']
    migration_snapshot = history['entries'][migration_index]['current_snapshot']
    assert history['entries'][-1]['current_snapshot'] == read_json('portfolio_state.json')['last_confirmed_portfolio']
    assert set(migration_snapshot['held_codes']) == {'517520','512800','512690'}
    return {'status':'passed','event_id':EVENT}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare-data', action='store_true')
    group.add_argument('--commit', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = prepare() if args.prepare_data else commit() if args.commit else validate()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
