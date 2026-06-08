"""Logging configuration (app/logging_config.py)."""

import json
import logging

import pytest

from app.config import Settings
from app.logging_config import JsonFormatter, configure_logging


def _settings(**overrides) -> Settings:
    base = {"api_key": "a" * 32}
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture(autouse=True)
def _restore_logging():
    """configure_logging mutates the global root logger — snapshot and restore
    so these tests don't leak state into the rest of the suite."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_app_level = logging.getLogger("app").level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    logging.getLogger("app").setLevel(saved_app_level)


def test_log_level_sets_root_level():
    configure_logging(_settings(log_level="WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_debug_forces_app_namespace_to_debug():
    configure_logging(_settings(log_level="INFO", debug=True))
    assert logging.getLogger("app").level == logging.DEBUG


def test_debug_off_does_not_pin_app_namespace():
    configure_logging(_settings(log_level="INFO", debug=False))
    assert logging.getLogger("app").level == logging.NOTSET


def test_json_format_installs_json_formatter():
    configure_logging(_settings(log_format="json"))
    handlers = logging.getLogger().handlers
    assert any(isinstance(h.formatter, JsonFormatter) for h in handlers)


def test_json_formatter_emits_parseable_json_with_message():
    rec = logging.makeLogRecord(
        {
            "name": "app.test",
            "levelname": "INFO",
            "msg": "hello %s",
            "args": ("world",),
        }
    )
    obj = json.loads(JsonFormatter().format(rec))
    assert obj["level"] == "INFO"
    assert obj["logger"] == "app.test"
    assert obj["msg"] == "hello world"


def test_json_formatter_promotes_extra_fields():
    rec = logging.makeLogRecord(
        {"name": "app.test", "levelname": "INFO", "msg": "x", "file_id": "abc-123"}
    )
    obj = json.loads(JsonFormatter().format(rec))
    assert obj["file_id"] == "abc-123"
