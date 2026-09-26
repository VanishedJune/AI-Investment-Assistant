"""Known exchange closures used to decide whether a weekly bar is final.

The 2026 Mid-Autumn holiday is 25-27 September; the eight local ETF series
and the fund NAV all end on Thursday 24 September for that week.  Holiday
source: https://publicholidays.cn/mid-autumn-festival/ (checked 2026-09-26).
Do not infer an unlisted holiday merely because a data source is missing a day.
"""

from datetime import date, timedelta


KNOWN_EXCHANGE_CLOSURES = frozenset({date(2026, 9, 25)})


def completed_week(last_session: date, available_on: date) -> bool:
    """True only when no remaining weekday session can occur in that week."""
    if last_session > available_on or last_session.weekday() > 4:
        return False
    day = last_session + timedelta(days=1)
    friday = last_session + timedelta(days=4 - last_session.weekday())
    while day <= friday:
        if day.weekday() < 5 and day not in KNOWN_EXCHANGE_CLOSURES:
            return False
        day += timedelta(days=1)
    return True
