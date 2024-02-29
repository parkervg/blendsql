from typing import Dict, Tuple, Union, Optional
from fiscalyear import FiscalDateTime, FiscalQuarter
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import dateparser


def initialize_date_map(today_dt: datetime):
    """
    Optionally pass in a string in the DEFAULT_DATE_FORMAT to force the bot to run, given this date.
    """

    # To handle q1, q2, etc. type phrases
    fiscal_dt = FiscalDateTime.fromordinal(today_dt.toordinal())

    return {
        "today": {"start": today_dt, "end": today_dt},
        "yesterday": {"start": (today_dt - timedelta(days=1)), "end": today_dt},
        "this month": {
            "start": (datetime(today_dt.year, today_dt.month, 1)),
            "end": today_dt,
        },
        "last month": {
            "start": (
                datetime(today_dt.year, today_dt.month, 1) - relativedelta(months=1)
            ),
            "end": (datetime(today_dt.year, today_dt.month, 1)),
        },
        "last 2 months": {
            "start": (
                datetime(today_dt.year, today_dt.month, 1) - relativedelta(months=2)
            ),
            "end": (datetime(today_dt.year, today_dt.month, 1)),
        },
        "this year": {
            "start": (datetime(today_dt.year, 1, 1)),
            "end": today_dt,
        },
        "ytd": {
            "start": (datetime(today_dt.year, 1, 1)),
            "end": today_dt,
        },
        "last year": {
            "start": (datetime(today_dt.year, 1, 1) - timedelta(days=365)),
            "end": (datetime(today_dt.year, 12, 31) - timedelta(days=365)),
        },
        "this week": {
            "start": (today_dt - timedelta(days=12)),
            "end": today_dt,
        },
        "last week": {
            "start": (today_dt - timedelta(days=24)),
            "end": (today_dt - timedelta(days=12)),
        },
        "last 2 weeks": {
            "start": (today_dt - timedelta(days=36)),
            "end": (today_dt - timedelta(days=12)),
        },
        # Fiscal Quarters
        "this quarter": {
            "start": (fiscal_dt.prev_fiscal_quarter.end),
            "end": today_dt,
        },
        "last quarter": {
            "start": fiscal_dt.prev_fiscal_quarter.start,
            "end": fiscal_dt.prev_fiscal_quarter.end,
        },
        "q1": {
            "start": FiscalQuarter(today_dt.year, 2).start,
            "end": FiscalQuarter(today_dt.year, 2).end,
        },
        "q2": {
            "start": FiscalQuarter(today_dt.year, 3).start
            if today_dt.month >= 3
            else FiscalQuarter(today_dt.year - 1, 3).start,
            "end": FiscalQuarter(today_dt.year, 3).end
            if today_dt.month >= 3
            else FiscalQuarter(today_dt.year - 1, 3).end,
        },
        "q3": {
            "start": FiscalQuarter(today_dt.year, 4).start
            if today_dt.month >= 7
            else FiscalQuarter(today_dt.year - 1, 4).start,
            "end": FiscalQuarter(today_dt.year, 4).end
            if today_dt.month >= 7
            else FiscalQuarter(today_dt.year - 1, 4).end,
        },
        "q4": {
            "start": FiscalQuarter(today_dt.year, 1).start
            if today_dt.month >= 10
            else FiscalQuarter(today_dt.year - 1, 1).start,
            "end": FiscalQuarter(today_dt.year, 1).end
            if today_dt.month >= 10
            else FiscalQuarter(today_dt.year - 1, 1).end,
        },
    }


def _parse_date(
    date_map: Optional[Dict[str, Dict[str, datetime]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    today_dt: Optional[datetime] = None,
) -> Tuple[Union[datetime, None], Union[datetime, None]]:
    """A couple cases for dt logic
    - start and end are passed.
        - Access date_map for 'start' of start and 'start' of end
    - only start is passed
        - Access date_map for 'start' and 'end'
    """

    if all(x is None for x in [today_dt, date_map]):
        raise ValueError(
            "Either today_dt or date_map must be specified in _parse_date call!"
        )
    if date_map is None:
        date_map = initialize_date_map(today_dt)

    start_map, end_map, fallback_map = None, None, {}
    settings = {
        # "RELATIVE_BASE": self._today_dt,
        "PREFER_DATES_FROM": "past"
    }
    if start is not None and start in date_map:
        start_map = date_map[start]
    if end is not None and end in date_map:
        end_map = date_map[end]
    if start is not None and start_map is None:
        # Parse with dateparser
        inferred_end_dt: Union[datetime, None] = dateparser.parse(
            start, settings=settings
        )
        if inferred_end_dt is not None:
            fallback_map["start"] = inferred_end_dt
    if end is not None and end_map is None:
        # Parse with dateparser
        inferred_end_dt: Union[datetime, None] = dateparser.parse(
            end, settings=settings
        )
        if inferred_end_dt is not None:
            fallback_map["end"] = inferred_end_dt
    # Now make sense of our dates
    _start_map = start_map if start_map is not None else fallback_map
    _end_map = end_map if end_map is not None else fallback_map
    if end is None:
        start_dt, end_dt = _start_map["start"], _start_map["end"]
    else:  # We have an end passed
        start_dt, end_dt = _start_map["start"], _end_map["end"]
    return (start_dt, end_dt)
