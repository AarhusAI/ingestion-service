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


# ---------------------------------------------------------------------------
# Per-file_id lock (sec.md Finding 7). Closes the delete-then-write race
# when two requests hit the same file_id concurrently.
# ---------------------------------------------------------------------------


def test_per_file_id_lock_serializes_same_file_id():
    """Two threads holding the same file_id lock execute sequentially."""
    import threading

    enter = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def first():
        with indexing._per_file_id_lock("file-x"):
            enter.set()
            order.append("first-enter")
            release.wait(timeout=2)
            order.append("first-exit")

    def second():
        enter.wait(timeout=2)  # only start after `first` is holding the lock
        with indexing._per_file_id_lock("file-x"):
            order.append("second-enter")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()

    enter.wait(timeout=2)
    # second is now blocked behind first — it must not have entered yet.
    assert order == ["first-enter"]
    release.set()

    t1.join(timeout=2)
    t2.join(timeout=2)
    assert order == ["first-enter", "first-exit", "second-enter"]


def test_per_file_id_lock_does_not_block_different_file_ids():
    """Different file_ids get independent locks — they don't serialize."""
    import threading

    inside_x = threading.Event()
    release_x = threading.Event()
    inside_y = threading.Event()

    def hold_x():
        with indexing._per_file_id_lock("file-x"):
            inside_x.set()
            release_x.wait(timeout=2)

    def acquire_y():
        inside_x.wait(timeout=2)  # only start after x is held
        with indexing._per_file_id_lock("file-y"):
            inside_y.set()

    t_x = threading.Thread(target=hold_x)
    t_y = threading.Thread(target=acquire_y)
    t_x.start()
    t_y.start()

    # If file-y blocked behind file-x, inside_y would never set within the timeout.
    assert inside_y.wait(timeout=2), "different file_ids should not serialize"
    release_x.set()
    t_x.join(timeout=2)
    t_y.join(timeout=2)


def test_per_file_id_lock_cleans_up_registry():
    """After the last waiter releases, the file_id's entry is removed."""
    # Snapshot keys before so we don't depend on test ordering.
    before = set(indexing._file_id_locks.keys())

    with indexing._per_file_id_lock("file-cleanup-test"):
        # While held, the entry exists.
        assert "file-cleanup-test" in indexing._file_id_locks

    # After release, registry returns to its prior state — no leaked entry.
    after = set(indexing._file_id_locks.keys())
    assert "file-cleanup-test" not in after
    assert after == before


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
