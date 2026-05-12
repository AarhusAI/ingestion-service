"""Tests for the pipeline lifespan-orchestration layer.

These cover ``init_pipeline()`` itself, not the individual component
factories — those live in ``test_pipelines.py``. Hermetic by design: we
never let the real Haystack ``Pipeline`` construct (and therefore never
download HuggingFace or fastembed models in CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.pipelines import indexing


def test_init_pipeline_calls_warm_up(monkeypatch):
    """``init_pipeline()`` must call ``Pipeline.warm_up()`` so the sparse
    embedder downloads its model at startup, not on first request.

    Locks the contract: anyone removing the ``warm_up`` call by accident
    will trip this test. Mocks both the document store and the pipeline
    builder so no real Qdrant / Haystack construction happens.
    """
    mock_pipeline = MagicMock()
    monkeypatch.setattr(indexing, "_build_document_store", lambda _s: MagicMock())
    monkeypatch.setattr(indexing, "_build_pipeline", lambda _s, _ds: mock_pipeline)

    # Reset module-level state so the assertion is meaningful.
    monkeypatch.setattr(indexing, "_pipeline", None)
    monkeypatch.setattr(indexing, "_document_store", None)

    indexing.init_pipeline()

    mock_pipeline.warm_up.assert_called_once()


def test_is_pipeline_ready_false_before_init(monkeypatch):
    """Before ``init_pipeline()`` runs, readiness should report False so
    ``/health/ready`` returns 503."""
    monkeypatch.setattr(indexing, "_pipeline", None)
    monkeypatch.setattr(indexing, "_document_store", None)

    assert indexing.is_pipeline_ready() is False


def test_is_pipeline_ready_true_after_init(monkeypatch):
    """After init: both module-level slots non-None → ready."""
    monkeypatch.setattr(indexing, "_pipeline", MagicMock())
    monkeypatch.setattr(indexing, "_document_store", MagicMock())

    assert indexing.is_pipeline_ready() is True


def test_init_pipeline_logs_warm_up_duration(monkeypatch, caplog):
    """The "pipeline ready" log line carries elapsed warm-up time so
    operators can spot a slow model download in metrics / alerts."""
    import logging

    mock_pipeline = MagicMock()
    monkeypatch.setattr(indexing, "_build_document_store", lambda _s: MagicMock())
    monkeypatch.setattr(indexing, "_build_pipeline", lambda _s, _ds: mock_pipeline)
    monkeypatch.setattr(indexing, "_pipeline", None)
    monkeypatch.setattr(indexing, "_document_store", None)

    with caplog.at_level(logging.INFO, logger="app.pipelines.indexing"):
        indexing.init_pipeline()

    ready_lines = [r.message for r in caplog.records if "pipeline ready" in r.message]
    assert ready_lines, "expected a 'pipeline ready' log line"
    assert "warm_up=" in ready_lines[0]


def test_init_pipeline_warm_up_error_propagates(monkeypatch):
    """If warm-up fails (e.g. HuggingFace unreachable on a fresh deploy),
    the error must propagate — failing to start is better than silently
    serving traffic against an un-warmed pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.warm_up.side_effect = RuntimeError("HuggingFace unreachable")
    monkeypatch.setattr(indexing, "_build_document_store", lambda _s: MagicMock())
    monkeypatch.setattr(indexing, "_build_pipeline", lambda _s, _ds: mock_pipeline)
    monkeypatch.setattr(indexing, "_pipeline", None)
    monkeypatch.setattr(indexing, "_document_store", None)

    try:
        indexing.init_pipeline()
    except RuntimeError as exc:
        assert "HuggingFace" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")
