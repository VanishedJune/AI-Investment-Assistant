"""Explicit closing-quote turnover enrichment; never rewrite frozen price/model/report files."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.publication_io import capture,commit_files
from app.turnover_evidence import checked_quote


def run(*,commit=False):
    destination='辅助数据/成交额补充.json'
    manifest=json.loads((ROOT/'data_manifest.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'config/instruments.json').read_text(encoding='utf-8'))
    daily=[r['path'] for r in manifest['files'] if r['path'].endswith('_日线.csv')]
    expected=capture(ROOT,[destination,'data_manifest.json','config/instruments.json',*daily])
    original=json.loads(expected[destination]) if expected[destination] else {'schema_version':'turnover-supplement-v1','records':[]}
    records={(r['code'],r['trade_date']):r for r in original['records']}
    updated=0;warnings=[]
    for instrument in config['instruments']:
        code=instrument['code'];symbol=instrument['symbol']
        path=next(p for p in daily if Path(p).name.startswith(code+'_'))
        with (ROOT/path).open(encoding='utf-8-sig',newline='') as handle:row=list(csv.DictReader(handle))[-1]
        url='https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?'+urllib.parse.urlencode({'param':f"{symbol},day,{row['date']},{row['date']},640,qfq"})
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'})
            with urllib.request.urlopen(request,timeout=20) as response:payload=json.load(response)
            quote=payload['data'][symbol]['qt'][symbol]
            point=checked_quote(quote,row,code,tick=float(instrument.get('price_tick',0.001)))
            point.update(source_url=url,verified_at=datetime.now(timezone.utc).isoformat())
            key=(code,row['date']); previous=records.get(key)
            if previous and previous['amount_cny']!=point['amount_cny']:raise ValueError('Previously recorded turnover was revised; review required')
            if not previous:records[key]=point;updated+=1
        except Exception as exc:warnings.append(code+': '+str(exc))
    result={'schema_version':'turnover-supplement-v1','checked_at':datetime.now(timezone.utc).isoformat(),
            'data_as_of':manifest['as_of'],'base_data_run_id':manifest['run_id'],
            'records':list(records.values()),'warnings':warnings,
            'scope':'独立收盘成交额补充；下一次显式行情更新只补已有空值；不回写冻结月报和模型输入',
            'history_status':'earlier_missing_amounts_remain_unavailable'}
    if commit:
        commit_files(ROOT,{destination:(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8')},expected)
    return {'status':'SAVED' if commit else 'CHECKED','new_records':updated,'total_records':len(records),'warnings':warnings}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',action='store_true')
    print(json.dumps(run(commit=parser.parse_args().commit),ensure_ascii=False,indent=2))
