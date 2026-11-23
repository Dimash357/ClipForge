"""Console log filters."""
import logging
import re

_POLL = re.compile(r'"GET /api/projects/\d+/status/ HTTP/[\d.]+" 200 ')


class QuietPolling(logging.Filter):
    """Drop successful status-poll request lines (the page asks every 1-2 s); errors still show."""

    def filter(self, record):
        try:
            return not _POLL.search(record.getMessage())
        except Exception:
            return True
