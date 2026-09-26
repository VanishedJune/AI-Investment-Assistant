"""2026 daily/weekly MA sensitivity of existing G5: 20/20, 20/10, 10/10.

Reuse frozen B0207, V70 replay logic and account arithmetic. Only substitute
the protective view's MA inputs; never alter champion inputs or B0207 targets.
No training, production writes, historical result overwrite or fingerprints.
"""
from __future__ import annotations

import json
import argparse
from collections import Counter
import math
import sys
from datetime import date, timedelta
from pathlib import Path

if __package__ in (None, ''):
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if str(Path(p or '.').resolve()) != script_dir]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scripts import four_way_2026_pk as pk
from rolling import ledger

GROUPS = {'MA20': 'V70日MA20＋周MA20', 'MA10': 'V70日MA20＋周MA10',
          'BOTH10': 'V70日MA10＋周MA10'}
AUDIT_STATUS = {'executed': '已保护减持', 'no_position': '无持仓',
                'episode_applied': '连续看空区间已处理，不重复减持',
                'nominal_rebalance': '名义调仓周，执行B0207完整调仓而非单独保护',
                'outside_window': '截止时尚无下一交易日成交价',
                'below_lot': '拟减持不足100份整手'}


def protection_status(*, nominal, shares, episode, executed):
    if nominal:
        return 'nominal_rebalance'
    if shares <= 0:
        return 'no_position'
    if episode:
        return 'episode_applied'
    if executed > 0:
        return 'executed'
    if int(shares * pk.GUARD_SELL_FRACTION // pk.LOT) * pk.LOT == 0:
        return 'below_lot'
    raise ValueError('Unexplained unexecuted protection signal')


class DividendAccount(pk.Account):
    def __post_init__(self):
        super().__post_init__()
        self.receivables = {}
        self.entitlements = {}
        self.ca_rows = []

    def value(self, prices):
        return super().value(prices) + sum(self.receivables.values())

    def corporate_actions(self, day, events, at_close=False):
        for code, event in events:
            key = (code, event['ex_date'])
            if event['type'] == 'split':
                if not at_close and day == event['ex_date']:
                    quantity = self.shares[code] * float(event['ratio'])
                    if quantity != int(quantity):
                        raise ValueError('Fractional split needs separate settlement')
                    self.shares[code] = int(quantity)
                continue
            if at_close:
                if day == event['entitlement_date']:
                    self.entitlements[key] = self.shares[code]
                continue
            if day == event['ex_date']:
                if key not in self.entitlements:
                    raise ValueError('Missing dividend record-date position')
                amount = self.entitlements[key] * float(event['dps'])
                self.receivables[key] = amount
                self.ca_rows.append(dict(code=code, ex_date=day, pay_date=event['pay_date'], amount=amount))
            if day == event['pay_date']:
                self.cash += self.receivables.pop(key, 0.)


def view(row, period, daily_period=20):
    if period not in (10, 20) or daily_period not in (10, 20):
        raise ValueError('Only MA10 and MA20 are authorized')
    alternative = row.copy(deep=True)
    for prefix in ('d', 'w'):
        selected_period = period if prefix == 'w' else daily_period
        value = row.get(f'{prefix}_ma{selected_period}')
        if value is None or not math.isfinite(float(value)) or value <= 0:
            raise ValueError('Missing protective MA input')
        alternative[f'{prefix}_ma20'] = value
        if selected_period == 10:
            alternative[f'{prefix}_ma20_slope'] = row[f'{prefix}_ma10_slope']
    return pk._agent_v70_view(alternative)


def load_market(codes):
    # Original set_workspace creates directories. Research switches only the
    # in-memory CA path, preserving every existing production directory.
    old_switch, old_ca = pk.set_workspace, ledger.CA_DIR
    def readonly_switch(code):
        cfg = json.loads((pk.MODEL_ROOT / 'configs' / f'etf_{code}.json').read_text(encoding='utf-8'))
        workspace = (pk.MODEL_ROOT / cfg.get('workspace', f'etf_{code}')).resolve()
        if not workspace.is_relative_to(pk.MODEL_ROOT.resolve()) or not workspace.is_dir():
            raise ValueError('Invalid model workspace')
        ledger.CA_DIR = workspace / 'weekly_rolling' / 'corporate_actions'
        return workspace
    try:
        pk.set_workspace = readonly_switch
        market = pk._load_market_bundle(codes)
    finally:
        pk.set_workspace, ledger.CA_DIR = old_switch, old_ca
    for code in codes:
        d, w = pk.analysis_frames(code, market[code]['ca'])
        ds = d['analysis_close'].rolling(10).mean().pct_change(5)
        ws = w['analysis_close'].rolling(10).mean().pct_change(1)
        ds.index, ws.index = pd.to_datetime(d['date']), pd.to_datetime(w['date'])
        for signal, entry in market[code]['by_signal'].items():
            entry['feature'] = entry['feature'].copy(deep=True)
            stamp = pd.Timestamp(signal)
            entry['feature']['d_ma10_slope'] = float(ds.loc[ds.index <= stamp].iloc[-1])
            entry['feature']['w_ma10_slope'] = float(ws.loc[ws.index <= stamp].iloc[-1])
    return market


def simulate(market, codes, signals, targets, nominal_signals, start, end, period, daily_period=20):
    gid = 'BOTH10' if daily_period == 10 else f'MA{period}'
    account = DividendAccount(gid, codes)
    account.protection_audit = []
    events = [(c,e) for c in codes for e in market[c]['ca'].get('events', []) if start <= e['ex_date'] <= end]
    executions = {targets[s]['exec_date']: s for s in signals
                  if targets[s]['exec_date'] and targets[s]['exec_date'] <= end}
    protective = {s: {'agent_views': {c: view(market[c]['by_signal'][s]['feature'], period, daily_period)
                                    for c in codes}} for s in signals}
    schedule = [day for day in market[codes[0]]['daily_by_date'] if start <= day <= end]
    for day in schedule:
        account.corporate_actions(day, events)
        close = {c: market[c]['daily_by_date'][day]['raw_close'] for c in codes}
        if day in executions and day != start:
            signal = executions[day]
            shares_before = account.shares.copy()
            episode_before = account.guard_episode.copy()
            trades_before = len(account.trade_rows)
            opens = {c: market[c]['daily_by_date'][day]['raw_open'] for c in codes}
            account.cash *= pk.CASH_WEEKLY_FACTOR
            if signal in nominal_signals:
                account.rebalance(signal=signal, execution=day, target=targets[signal]['b0207'],
                                  open_prices=opens, reason='nominal_28day_full_rebalance')
            else:
                account.apply_agent_guard(signal=signal, execution=day, targets=protective, open_prices=opens)
            for code in codes:
                v = protective[signal]['agent_views'][code]
                if v['direction'] != 'bearish':
                    continue
                sold = sum(t['shares'] for t in account.trade_rows[trades_before:]
                           if t['code']==code and 'agent_v70_bearish_guard' in t['reason'])
                status = protection_status(nominal=signal in nominal_signals,
                    shares=shares_before[code], episode=episode_before[code], executed=sold)
                account.protection_audit.append(dict(group=gid, signal_date=signal, execution_date=day,
                    code=code, name=market[code]['name'], rule=v['rule'],
                    shares_before=shares_before[code], episode_before=episode_before[code],
                    protection_shares_sold=sold, status=status, reason=AUDIT_STATUS[status]))
        nav = account.value(close)
        peak = max(nav, max((x['nav'] for x in account.nav_rows), default=pk.INITIAL_CAPITAL))
        account.corporate_actions(day, events, at_close=True)
        account.nav_rows.append(dict(group_id=gid, group_name=GROUPS[gid], date=day, nav=nav,
                                    cash=account.cash, dividend_receivable=sum(account.receivables.values()),
                                    exposure_pct=sum(account.shares[c]*close[c] for c in codes)/nav*100,
                                    drawdown_pct=(nav/peak-1)*100,
                                    shares=account.shares.copy()))
    for signal in signals:
        execution = targets[signal]['exec_date']
        if execution and execution <= end:
            continue
        point = next(r for r in account.nav_rows if r['date']==signal)
        for code in codes:
            v = protective[signal]['agent_views'][code]
            if v['direction']=='bearish':
                account.protection_audit.append(dict(group=gid, signal_date=signal, execution_date=execution,
                    code=code, name=market[code]['name'], rule=v['rule'], shares_before=point['shares'][code],
                    episode_before=account.guard_episode[code], protection_shares_sold=0,
                    status='outside_window', reason=AUDIT_STATUS['outside_window']
                    + ('；且同一看空区间已处理，不应重复减持' if account.guard_episode[code] else '')))
    if sum(r['status']=='executed' for r in account.protection_audit) != account.guard_count:
        raise ValueError('Protection audit does not reconcile with executed trades')
    expected_count = sum(v['direction']=='bearish' for s in signals for v in protective[s]['agent_views'].values())
    if len(account.protection_audit) != expected_count:
        raise ValueError('Missing bearish observations in audit')
    return account, protective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true', help='补充现有三组结果的全部看空信号，不改变收益或交易')
    args = parser.parse_args()
    config = pk.load_config()
    codes = list(config['priority'])
    market = load_market(codes)
    end = min(pk._iso(market[c]['daily']['date'].iloc[-1]) for c in codes)
    if end != '2026-09-11':
        raise ValueError('This dated experiment requires explicit review of a new cutoff')
    start = '2026-01-09'
    for code in codes:
        for e in market[code]['ca'].get('events', []):
            if start <= e['ex_date'] <= end:
                if e['type'] not in ('split','cash_dividend'):
                    raise ValueError('Unsupported corporate action')
                if e['type']=='cash_dividend':
                    if not start <= e['entitlement_date'] < e['ex_date'] <= e['pay_date']:
                        raise ValueError('Invalid dividend dates')
                    if any(day not in market[code]['daily_by_date'] for day in
                           (e['entitlement_date'], e['ex_date'], e['pay_date'])):
                        raise ValueError('Dividend settlement requires explicit trading-day handling')
    common_dates = set(market[codes[0]]['daily_by_date'])
    if any({d for d in market[c]['daily_by_date'] if start <= d <= end} !=
           {d for d in common_dates if start <= d <= end} for c in codes):
        raise ValueError('Trading date mismatch')
    signals = sorted(s for s in market[codes[0]]['by_signal'] if start <= s <= end)
    targets, _ = pk._signal_targets(market, codes, signals)
    first = pk.FIRST_NOMINAL_DECISION
    while first - timedelta(days=28) >= date.fromisoformat(start):
        first -= timedelta(days=28)
    nominal = pk.nominal_signal_dates(signals, end, first_nominal=first)
    pk.GROUPS.update(GROUPS)  # process-local labels only, no source mutation
    accounts, reviews = {}, {}
    for gid, period, daily_period in (('MA20',20,20), ('MA10',10,20), ('BOTH10',10,10)):
        accounts[gid], reviews[gid] = simulate(market, codes, signals, targets,
                                            set(nominal.values()), start, end, period, daily_period)
    differences = []
    for s in signals:
        for c in codes:
            a = reviews['MA20'][s]['agent_views'][c]
            for compared_group in ('MA10', 'BOTH10'):
                b = reviews[compared_group][s]['agent_views'][c]
                if (a['direction'] == 'bearish') == (b['direction'] == 'bearish'):
                    continue
                f = market[c]['by_signal'][s]['feature']
                differences.append(dict(compared_group=compared_group, signal_date=s, execution_date=targets[s]['exec_date'], code=c,
                    name=market[c]['name'], nominal_week=s in nominal.values(),
                    ma20_rule=a['rule'], ma20_direction=a['direction'], ma10_rule=b['rule'], ma10_direction=b['direction'],
                    close=float(f['close']), daily_ma10=float(f['d_ma10']), daily_ma20=float(f['d_ma20']),
                    weekly_ma10=float(f['w_ma10']), weekly_ma20=float(f['w_ma20'])))
    metrics = [pk._account_metrics(accounts[g], set(signals)) for g in GROUPS]
    payload = dict(start=start, end=end, first_nominal=min(nominal), nominal_dates=nominal,
        status='POST_HOC_FROZEN_MODEL_SENSITIVITY_NOT_REAL_AGENT_PERFORMANCE',
        description='三组分别为日MA20/周MA20、日MA20/周MA10、日MA10/周MA10。只替换R001–R008回放中的对应均线及斜率参考，其余条件、冠军输入和B0207权重不变。',
        initial_capital=200000, guard_share_fraction=.7, commission_rate=.00025, minimum_commission=5,
        cash_apr=.0175, execution='next_trading_day_raw_open', no_external_contributions=True,
        limitations=['现有V70回放是简化确定性近似，不是逐周人工综合判断。',
                    '当前冠军回放2026年历史存在样本内/选择偏差，不构成样本外证明。',
                    '保留原PK无滑点及周度现金计息；首次名义决策前保持现金。',
                    '同一保护区间只执行一次；按原PK在完整调仓后重置保护区间。',
                    '当前8只ETF固定回看，未模拟历史标的池替换。',
                    '公司行动覆盖登记末日之外为未独立核实区间；不得视为已验证无事件。'],
        metrics=metrics, signal_differences=differences,
        ca_coverage={c:market[c]['ca'].get('coverage') for c in codes},
        dividend_bookings={g:a.ca_rows for g,a in accounts.items()},
        champion_ids={c:market[c]['champion_id'] for c in codes},
        trades={g:a.trade_rows for g,a in accounts.items()}, nav={g:a.nav_rows for g,a in accounts.items()})
    previous = pk.PROJECT / 'champion_vs_dca' / '2026V70周MA10与周MA20保护PK_20260911' / '对比结果.json'
    old = json.loads(previous.read_text(encoding='utf-8'))
    for group in ('MA20', 'MA10'):
        if payload['nav'][group] != old['nav'][group] or payload['trades'][group] != old['trades'][group]:
            raise ValueError('Existing comparison changed; refusing mixed-scope results')
    output = pk.PROJECT / 'champion_vs_dca' / '2026V70日周均线三组保护PK_20260911'
    payload['protection_signal_audit'] = {g:a.protection_audit for g,a in accounts.items()}
    counts = {g:dict(Counter(r['status'] for r in a.protection_audit)) for g,a in accounts.items()}
    payload['protection_signal_counts'] = counts
    if args.audit_only:
        result_path, report_path = output/'对比结果.json', output/'对比报告.md'
        existing = json.loads(result_path.read_text(encoding='utf-8'))
        for key in ('metrics','nav','trades'):
            if existing[key] != payload[key]:
                raise ValueError('Audit would change existing results; refusing write')
        existing['protection_signal_audit'] = payload['protection_signal_audit']
        existing['protection_signal_counts'] = counts
        section = ['## 全部看空保护信号（含未执行）', '',
                   '按“ETF×信号日×组别”统计。连续看空每周都记录，因此信号数不等于独立区间数。已保护减持仅指模拟账户成交，不是用户真实交易。', '',
                   '| 类别 | 日20周20 | 日20周10 | 日10周10 |','|---|---:|---:|---:|']
        section.append('| 全部看空信号 | '+' | '.join(str(len(accounts[g].protection_audit)) for g in GROUPS)+' |')
        for status, label in AUDIT_STATUS.items():
            section.append('| '+label+' | '+' | '.join(str(counts[g].get(status,0)) for g in GROUPS)+' |')
        for g in GROUPS:
            section += ['', '### '+GROUPS[g], '', '| 信号日 | 执行日 | ETF | 规则 | 执行前份额 | 保护卖出份额 | 结果/原因 |',
                        '|---|---|---|---|---:|---:|---|']
            for r in accounts[g].protection_audit:
                section.append(f"| {r['signal_date']} | {r['execution_date'] or '窗口外/待数据'} | {r['code']} {r['name']} | {r['rule']} | {r['shares_before']} | {r['protection_shares_sold']} | {r['reason']} |")
        original = report_path.read_text(encoding='utf-8').split('\n## 全部看空保护信号（含未执行）')[0].rstrip()
        result_path.write_text(json.dumps(existing,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        report_path.write_text(original+'\n\n'+'\n'.join(section)+'\n',encoding='utf-8')
        print(json.dumps(dict(counts=counts, audit=payload['protection_signal_audit']),ensure_ascii=False,indent=2))
        return
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing research result')
    output.mkdir()
    (output / '对比结果.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# 2026年V70日周均线三组保护PK', '', f'观察区间：{start}至{end}；首次名义决策：{min(nominal)}。',
             '', payload['description'], '', '| 指标 | 日MA20＋周MA20 | 日MA20＋周MA10 | 日MA10＋周MA10 |', '|---|---:|---:|---:|']
    for title,key,fmt in [('期末资产','final_nav',',.2f'),('累计收益','cumulative_return','.2%'),
        ('年化收益','annualized_return','.2%'),('最大回撤','max_drawdown','.2%'),
        ('夏普','annualized_sharpe','.3f'),('周胜率','weekly_win_rate','.2%'),
        ('平均持仓','average_exposure','.2%'),('保护次数','guard_triggers','d'),('佣金','commission_total',',.2f')]:
        lines.append(f'| {title} | ' + ' | '.join(format(row[key],fmt) for row in metrics) + ' |')
    lines += ['', '## 看空信号差异', '']
    for difference in differences:
        s, code = difference['signal_date'], difference['code']
        held = [next(r for r in accounts[g].nav_rows if r['date']==s)['shares'][code] for g in ('MA20',difference['compared_group'])]
        lines.append(f"- {s} {code}：原版为{difference['ma20_direction']}，{GROUPS[difference['compared_group']]}为{difference['ma10_direction']}；对应两组持仓份额{held}。"
                     + ('信号没有窗口内下一交易日价格，不提前计入收益。' if not difference['execution_date'] else '无持仓不执行保护。' if not any(held) else '实际交易见逐笔明细。'))
    lines += ['', '## 口径与局限', '', *['- '+s for s in payload['limitations']], '',
              '未修改正式策略、模型、实际持仓和旧PK结果。逐日净值、持仓、逐笔交易与信号差异见同目录JSON。']
    (output / '对比报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), metrics=metrics, differences=differences),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
