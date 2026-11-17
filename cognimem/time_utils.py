"""
CogniMem - Time utilities

Central helpers for timezone-aware datetimes. Mixing naive (UTC) and
timezone-aware datetimes raises TypeError on comparison/subtraction, so all
timestamps in CogniMem are normalized through these helpers.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Timezone-aware UTC now (replacement for deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Return dt as a timezone-aware UTC datetime.

    Naive datetimes are assumed to be UTC (the historical CogniMem convention);
    aware datetimes are converted to UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
