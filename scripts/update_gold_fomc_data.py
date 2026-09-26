# -*- coding: utf-8 -*-
"""生成黄金ETF专属FOMC历史决议与前后15个自然日事件窗口。

政策决议使用美联储正式声明日期与目标利率区间；价格窗口只读取配置中标的的
本地日线的复权收盘价。该数据仅供黄金Agent证据层分析，不进入冠军模型或
B0207机械权重。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import io
import re
import sys
import urllib.request
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.publication_io import capture, commit_files
OUTPUT_DIR = ROOT / "辅助数据" / "黄金宏观"
GOLD_CONFIG = json.loads((ROOT / "config/gold_analysis.json").read_text(encoding="utf-8"))
GOLD_CODE = GOLD_CONFIG['instrument']['code']
PRICE_FILE = ROOT / "数据" / f"{GOLD_CODE}_{GOLD_CONFIG['instrument']['name']}_日线.csv"
FED_SOURCE = "Board of Governors of the Federal Reserve System"


def _decision(day: str, action: str, change_bp: int, lower: str, upper: str) -> dict[str, str | int]:
    compact = day.replace("-", "")
    return {
        "meeting_date": day,
        "action": action,
        "change_bp": change_bp,
        "target_lower_pct": lower,
        "target_upper_pct": upper,
        "source": FED_SOURCE,
        "source_url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{compact}a.htm",
    }


FOMC_DECISIONS = [
    _decision("2023-02-01", "hike", 25, "4.50", "4.75"),
    _decision("2023-03-22", "hike", 25, "4.75", "5.00"),
    _decision("2023-05-03", "hike", 25, "5.00", "5.25"),
    _decision("2023-06-14", "hold", 0, "5.00", "5.25"),
    _decision("2023-07-26", "hike", 25, "5.25", "5.50"),
    _decision("2023-09-20", "hold", 0, "5.25", "5.50"),
    _decision("2023-11-01", "hold", 0, "5.25", "5.50"),
    _decision("2023-12-13", "hold", 0, "5.25", "5.50"),
    _decision("2024-01-31", "hold", 0, "5.25", "5.50"),
    _decision("2024-03-20", "hold", 0, "5.25", "5.50"),
    _decision("2024-05-01", "hold", 0, "5.25", "5.50"),
    _decision("2024-06-12", "hold", 0, "5.25", "5.50"),
    _decision("2024-07-31", "hold", 0, "5.25", "5.50"),
    _decision("2024-09-18", "cut", -50, "4.75", "5.00"),
    _decision("2024-11-07", "cut", -25, "4.50", "4.75"),
    _decision("2024-12-18", "cut", -25, "4.25", "4.50"),
    _decision("2025-01-29", "hold", 0, "4.25", "4.50"),
    _decision("2025-03-19", "hold", 0, "4.25", "4.50"),
    _decision("2025-05-07", "hold", 0, "4.25", "4.50"),
    _decision("2025-06-18", "hold", 0, "4.25", "4.50"),
    _decision("2025-07-30", "hold", 0, "4.25", "4.50"),
    _decision("2025-09-17", "cut", -25, "4.00", "4.25"),
    _decision("2025-10-29", "cut", -25, "3.75", "4.00"),
    _decision("2025-12-10", "cut", -25, "3.50", "3.75"),
    _decision("2026-01-28", "hold", 0, "3.50", "3.75"),
    _decision("2026-03-18", "hold", 0, "3.50", "3.75"),
    _decision("2026-04-29", "hold", 0, "3.50", "3.75"),
    _decision("2026-06-17", "hold", 0, "3.50", "3.75"),
    _decision("2026-07-29", "hold", 0, "3.50", "3.75"),
]


class GoldFomcDataError(RuntimeError):
    pass


def load_adjusted_closes(path: Path = PRICE_FILE, *, as_of: date | None = None) -> dict[date, Decimal]:
    if not path.is_file():
        raise GoldFomcDataError(f"缺少黄金ETF日线：{path}")
    prices: dict[date, Decimal] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            expected = path.name.split('_', 1)[0]
            if expected not in ('518600', '517520') or str(row.get("code")) != expected:
                raise GoldFomcDataError("价格文件包含其他标的记录")
            trade_date = date.fromisoformat(str(row["date"]))
            if as_of is not None and trade_date > as_of:
                continue
            close = Decimal(str(row["adj_close"]))
            if not close.is_finite() or close <= 0:
                raise GoldFomcDataError(f"{trade_date}复权收盘价无效")
            if trade_date in prices:
                raise GoldFomcDataError(f"黄金ETF日线日期重复：{trade_date}")
            prices[trade_date] = close
    if not prices:
        raise GoldFomcDataError("黄金ETF日线为空")
    return dict(sorted(prices.items()))


def _nearest(candidates: list[date], target: date) -> date:
    if not candidates:
        raise GoldFomcDataError(f"{target}附近没有有效A股交易日")
    return min(candidates, key=lambda item: (abs((item - target).days), item))


def build_event_windows(
    prices: dict[date, Decimal], decisions: list[dict[str, str | int]] = FOMC_DECISIONS
) -> list[dict[str, str | int]]:
    dates = sorted(prices)
    rows: list[dict[str, str | int]] = []
    for decision in decisions:
        event_date = date.fromisoformat(str(decision["meeting_date"]))
        before_event = [item for item in dates if item <= event_date]
        after_event = [item for item in dates if item > event_date]
        if not before_event or not after_event:
            continue
        base_date = before_event[-1]
        availability_date = after_event[0]
        pre_target = event_date - timedelta(days=15)
        post_target = event_date + timedelta(days=15)
        if pre_target < dates[0] or post_target > dates[-1]:
            continue  # 上市前或尚未成熟的窗口不能用最近的远端数据冒充。
        pre_date = _nearest(before_event, pre_target)
        post_date = _nearest(after_event, post_target)
        base_close = prices[base_date]
        pre_return = (base_close / prices[pre_date] - Decimal("1")) * Decimal("100")
        post_return = (prices[post_date] / base_close - Decimal("1")) * Decimal("100")
        rows.append(
            {
                **decision,
                "base_date": base_date.isoformat(),
                "base_adj_close": f"{base_close:.6f}",
                "first_available_a_share_date": availability_date.isoformat(),
                "pre_15_target_date": pre_target.isoformat(),
                "pre_15_trade_date": pre_date.isoformat(),
                "pre_15_adj_close": f"{prices[pre_date]:.6f}",
                "return_pre15_to_event_pct": f"{pre_return:.6f}",
                "post_15_target_date": post_target.isoformat(),
                "post_15_trade_date": post_date.isoformat(),
                "post_15_adj_close": f"{prices[post_date]:.6f}",
                "return_event_to_post15_pct": f"{post_return:.6f}",
                "price_field": "adj_close",
                "window_basis": "15_calendar_days_nearest_valid_a_share_session",
            }
        )
    return rows


def summarize(rows: list[dict[str, str | int]], *, instrument: str = GOLD_CODE) -> dict:
    groups = {
        "2023_hikes": [row for row in rows if str(row["meeting_date"]).startswith("2023-") and row["action"] == "hike"],
        "2024_cuts": [row for row in rows if str(row["meeting_date"]).startswith("2024-") and row["action"] == "cut"],
        "2025_cuts": [row for row in rows if str(row["meeting_date"]).startswith("2025-") and row["action"] == "cut"],
        "all_cuts": [row for row in rows if row["action"] == "cut"],
        "all_holds": [row for row in rows if row["action"] == "hold"],
    }
    for year in sorted({str(r['meeting_date'])[:4] for r in rows}):
        for action in ('hike', 'cut', 'hold'):
            groups.setdefault(year+'_'+action, [r for r in rows if str(r['meeting_date']).startswith(year+'-') and r['action']==action])
    result: dict[str, dict[str, int | float]] = {}
    for name, members in groups.items():
        values = [float(row["return_event_to_post15_pct"]) for row in members]
        result[name] = {
            "events": len(values),
            "average_post15_return_pct": round(sum(values) / len(values), 4) if values else None,
            "positive_events": sum(value > 0 for value in values),
            "negative_events": sum(value < 0 for value in values),
        }
    return {
        "schema_version": "gold-fomc-window-v1",
        "instrument": instrument,
        "price_field": "adj_close",
        "event_count": len(rows),
        "method": {
            "window": "FOMC声明日前后15个自然日",
            "base": "FOMC声明前最后一个A股有效收盘",
            "availability": "美国声明后的首个A股交易日",
            "holiday": "目标日非交易日时使用距离目标日最近的有效A股交易日",
        },
        "groups": result,
    }


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise GoldFomcDataError(f"拒绝写入空CSV：{path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _download(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request,timeout=25) as response:
        return response.read().decode('utf-8')


def _rate(text):
    text=text.replace('‑','-').replace('–','-').strip()
    if '-' in text:
        whole, fraction=text.split('-',1); num,den=fraction.split('/')
        return Decimal(whole)+Decimal(num)/Decimal(den)
    return Decimal(text)


def refresh_policy(as_of, *, downloader=_download):
    """Explicit network operation. Parse only the Committee's decision paragraph."""
    from lxml import html
    calendar_url='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
    calendar=html.fromstring(downloader(calendar_url))
    links=sorted(set(calendar.xpath('//a/@href')))
    found=[]
    for href in links:
        match=re.fullmatch(r'/newsevents/pressreleases/monetary(\d{8})a.htm',href)
        if not match:continue
        day=datetime.strptime(match[1],'%Y%m%d').date()
        if day.year==as_of.year and day<=as_of:found.append((day,'https://www.federalreserve.gov'+href))
    if not found:raise GoldFomcDataError('官方日历中没有当年已公布声明；保留旧政策记录')
    decisions=[dict(r) for r in FOMC_DECISIONS if str(r['meeting_date'])[:4]<str(as_of.year)]
    evidence=[]
    for day,url in sorted(found):
        doc=html.fromstring(downloader(url))
        paragraphs=[' '.join(n.text_content().split()) for n in doc.xpath('//p')]
        chosen=next((p for p in paragraphs if 'Committee decided to' in p and 'target range' in p),None)
        if not chosen:raise GoldFomcDataError('未找到委员会正式决议段落: '+url)
        normalized=chosen.replace('‑','-').replace('–','-')
        match=re.search(r'target range for the federal funds rate (?:at|to) ([\d./-]+) to ([\d./-]+) percent',normalized)
        if not match:raise GoldFomcDataError('无法可靠解析利率区间: '+url)
        lower,upper=map(_rate,match.groups())
        previous=decisions[-1]
        change=(lower-Decimal(str(previous['target_lower_pct'])))*100
        if not change.is_finite() or change!=int(change) or upper-lower!=Decimal('0.25'):
            raise GoldFomcDataError('政策区间或变化不符合数据契约: '+url)
        action='hold' if change==0 else 'hike' if change>0 else 'cut'
        if action=='hold' and 'maintain' not in chosen:
            raise GoldFomcDataError('利率变化与声明动作不一致: '+url)
        decisions.append(_decision(day.isoformat(),action,int(change),f'{lower:.2f}',f'{upper:.2f}'))
        evidence.append({'meeting_date':day.isoformat(),'source_url':url,'decision_quote':chosen})
    return decisions,{'source_calendar':calendar_url,'checked_through':as_of.isoformat(),
        'verified_at':datetime.now(timezone.utc).isoformat(),'statements':evidence,
        'scope':'官方当年已公布声明；不把未来会议安排写成已发生决议'}


def _csv_bytes(rows):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成当前黄金标的的FOMC窗口，上市前和未成熟窗口披露为缺失")
    parser.add_argument("--as-of", help="价格截止日YYYY-MM-DD；默认使用本地日线最后日期")
    parser.add_argument("--check", action="store_true", help="只计算和校验，不写文件")
    parser.add_argument('--refresh-policy',action='store_true',help='显式联网核验当年美联储已公布声明')
    args = parser.parse_args(argv)
    study = GOLD_CONFIG['historical_event_studies'][0]
    output_names=[study['decision_file'],study['window_file'],study['summary_file'],'辅助数据/黄金宏观/FOMC官方核验.json']
    expected=capture(ROOT,output_names+[str(PRICE_FILE.relative_to(ROOT)), 'config/gold_analysis.json'])
    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    prices = load_adjusted_closes(as_of=as_of)
    cutoff=as_of or max(prices)
    decisions=[r for r in FOMC_DECISIONS if r['meeting_date']<=cutoff.isoformat()]
    evidence=None
    if args.refresh_policy:decisions,evidence=refresh_policy(cutoff)
    rows = build_event_windows(prices,decisions)
    summary = summarize(rows)
    available = {x['meeting_date'] for x in rows}
    summary['unavailable_events'] = [{**x, 'status': 'unavailable_incomplete_price_window'} for x in decisions if x['meeting_date'] not in available]
    summary['price_coverage'] = {'start': min(prices).isoformat(), 'end': max(prices).isoformat()}
    summary['policy_coverage']={'last_decision':decisions[-1]['meeting_date'],'decisions':len(decisions),
        'source_mode':'official_verified' if evidence else 'local_registered_history','checked_through':cutoff.isoformat() if evidence else None}
    if not args.check:
        study = GOLD_CONFIG['historical_event_studies'][0]
        outputs={study['decision_file']:_csv_bytes(decisions),study['window_file']:_csv_bytes(rows),
                 study['summary_file']:(json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode('utf-8')}
        if evidence:outputs['辅助数据/黄金宏观/FOMC官方核验.json']=(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        commit_files(ROOT,outputs,expected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
