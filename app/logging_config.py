"""Logging configuration.

Replaces the previous inline ``basicConfig`` with a setup driven by
``LOG_LEVEL`` / ``LOG_FORMAT``. ``LOG_LEVEL`` is the primary verbosity dial
applied to the root logger; ``LOG_LEVEL_APP`` is an optional per-namespace
override that tunes the service's own loggers (``app.*``) without touching the
third-party loggers (so you can run verbose app logs without the httpx/boto3
flood).

The JSON formatter is a small in-repo class — no extra dependency, matching the
repo's lean-deps stance — so logs can ship to Loki / a JSON-aware aggregator
without a text-parsing stage. Caller-supplied ``extra=`` fields surface as
top-level keys, so structured fields (e.g. ``file_id``) are queryable.
"""

from __future__ import annotations

import json
import logging

from app.config import Settings

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Standard LogRecord attributes — anything NOT here is a caller-supplied
# ``extra=`` we want to promote to a top-level JSON field. Computed once from a
# blank record so it tracks the running Python version's record shape.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter: one JSON object per line.

    Emits the fields an aggregator wants as top-level keys (``ts``, ``level``,
    ``logger``, ``msg``) plus any ``extra=`` fields. ``exc_info`` is rendered
    into an ``exc`` string so tracebacks survive on a single line.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    """Configure the root logger from ``settings``. Idempotent.

    - ``LOG_LEVEL`` sets the root level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    - ``LOG_LEVEL_APP`` optionally overrides just the ``app`` namespace, so our
      own code can be verbose without flooding third-party loggers. Empty
      inherits the root level.
    - ``LOG_FORMAT`` picks the text (human) or json (aggregator) formatter.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handler = logging.StreamHandler()
    if settings.log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root = logging.getLogger()
    # Replace handlers so re-running (tests, reload) doesn't double-log.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Per-namespace override: make our own code (``app.*``) verbose while
    # leaving third-party loggers at the root level (no httpx/boto3 flood).
    # Empty LOG_LEVEL_APP → NOTSET → inherit the root level.
    app_logger = logging.getLogger("app")
    if settings.log_level_app:
        app_logger.setLevel(getattr(logging, settings.log_level_app, logging.NOTSET))
    else:
        app_logger.setLevel(logging.NOTSET)
