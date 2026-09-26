"""Only accept an actual provider turnover amount matched to an unadjusted bar."""
import math


def checked_quote(quote, row, code, *, tick=0.001):
    if len(quote)<36 or quote[2]!=code or quote[30][:8]!=row['date'].replace('-',''):
        raise ValueError('Quote instrument/date differs from local closed bar')
    if len(quote[30])!=14 or quote[30][8:12]<'1500':
        raise ValueError('Quote is not a closing observation')
    price, lots, amount = map(float,quote[35].split('/'))
    expected={'raw_open':float(quote[5]),'raw_high':float(quote[33]),
              'raw_low':float(quote[34]),'raw_close':float(quote[3]),'volume':float(quote[6])*100}
    if not all(math.isfinite(v) for v in [price,lots,amount,*expected.values()]) or amount<0:
        raise ValueError('Invalid quote amount or price')
    if abs(price-expected['raw_close'])>tick or lots!=float(quote[6]):
        raise ValueError('Trade summary differs from quote')
    for key,value in expected.items():
        tolerance=0.001 if key=='volume' else tick+1e-9
        if abs(float(row[key])-value)>tolerance:
            raise ValueError('Quote/local OHLCV mismatch: '+key)
    if amount<=0 and expected['volume']>0:
        raise ValueError('Positive volume without usable turnover')
    return {'code':code,'trade_date':row['date'],'amount_cny':amount,
            'raw_ohlcv':{key:float(row[key]) for key in expected},
            'source_timestamp':quote[30]+' Asia/Shanghai','source_trade_summary':quote[35],
            'status':'verified_closed_quote','unit':'CNY','method':'provider_reported_amount_not_price_times_volume'}


def amounts_for_rows(records, rows, code):
    by_date={r['date']:r for r in rows}; result={}
    for item in records:
        if item.get('code')!=code or item.get('status')!='verified_closed_quote':continue
        row=by_date.get(item.get('trade_date'))
        if row is None:continue
        if any(float(row[k])!=v for k,v in item['raw_ohlcv'].items()):
            raise ValueError('Supplement no longer matches local OHLCV')
        amount=item['amount_cny']
        if type(amount) not in (int,float) or not math.isfinite(amount) or amount<0:
            raise ValueError('Invalid turnover supplement')
        result[row['date']]=amount
    return result
