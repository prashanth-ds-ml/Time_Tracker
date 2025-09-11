# core/time_utils.py
from datetime import datetime, timedelta, timezone, date
from typing import Optional, List
import pytz

IST = pytz.timezone("Asia/Kolkata")

def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def now_ist() -> datetime:
    return datetime.now(IST)

def today_iso() -> str:
    return now_ist().date().isoformat()

def _to_utc_naive(dt: datetime) -> datetime:
    """Return UTC-naive datetime for Mongo 'date' type."""
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

def to_ist_display(dt: Optional[datetime]) -> datetime:
    """Make any Mongo datetime (usually UTC-naive) safely IST-aware for display."""
    if not isinstance(dt, datetime):
        return now_ist()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)

def week_key_from_datestr(datestr: str) -> str:
    y, m, d = map(int, datestr.split("-"))
    dt = datetime(y, m, d)
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_key_from_date(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def monday_from_week_key(week_key: str) -> datetime:
    year, wk = map(int, week_key.split("-"))
    return IST.localize(datetime.fromisocalendar(year, wk, 1))

def prev_week_key(week_key: str) -> str:
    mon = monday_from_week_key(week_key)
    prev_mon = mon - timedelta(days=7)
    iso = prev_mon.isocalendar()
    return f"{iso.year}-{iso.week:02d}"

def week_dates_list(week_key: str) -> List[str]:
    mon = monday_from_week_key(week_key).date()
    return [(mon + timedelta(days=i)).isoformat() for i in range(7)]

def pom_equiv(minutes: int) -> float:
    return round(float(minutes) / 25.0, 2)
