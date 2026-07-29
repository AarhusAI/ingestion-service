"""Tests for the pipeline lifespan-orchestration layer.

These cover ``init_pipeline()`` itself, not the individual component
factories — those live in ``test_pipelines.py``. Hermetic by design: we
never let the real Haystack ``Pipeline`` construct (and therefore never
download HuggingFace or fastembed models in CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from haystack import Document, component

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


def test_zero_chunk_ingest_warns_and_counts(monkeypatch, caplog):
    """A run that writes 0 chunks still reports success to the caller, so the
    warning log + ingest_empty_total counter are the only operator signal that
    extraction produced nothing (e.g. a sidecar response-shape drift)."""
    import logging

    from prometheus_client import REGISTRY

    _mock_raw_client(monkeypatch)
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "writer": {"documents_written": 0},
        "converter": {"documents": []},
    }
    monkeypatch.setattr(indexing, "_pipeline", mock_pipeline)

    before = REGISTRY.get_sample_value("ingest_empty_total") or 0.0

    with caplog.at_level(logging.WARNING, logger="app.pipelines.indexing"):
        result = indexing.run_indexing_pipeline(
            "/tmp/fake.pdf", {"file_id": "f-empty", "collection_name": "file-f-empty"}
        )

    assert result.chunks_count == 0
    assert any("0 chunks" in r.message for r in caplog.records)
    after = REGISTRY.get_sample_value("ingest_empty_total")
    assert after == before + 1


def test_nonzero_chunk_ingest_does_not_warn(monkeypatch, caplog):
    """The zero-chunk warning must not fire on a normal ingest."""
    import logging

    _mock_raw_client(monkeypatch)
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "writer": {"documents_written": 3},
        "converter": {"documents": []},
    }
    monkeypatch.setattr(indexing, "_pipeline", mock_pipeline)

    with caplog.at_level(logging.WARNING, logger="app.pipelines.indexing"):
        result = indexing.run_indexing_pipeline(
            "/tmp/fake.pdf", {"file_id": "f-ok", "collection_name": "file-f-ok"}
        )

    assert result.chunks_count == 3
    assert not any("0 chunks" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Versioned (blue/green) overwrite — the new version is written alongside the
# old one, stale versions are swept only after success, and a failed run tears
# down only its own points. See run_indexing_pipeline's docstring.
# ---------------------------------------------------------------------------


def _mock_pipeline_run(monkeypatch, *, chunks=3, run_side_effect=None):
    """Patch the cached pipeline with a mock whose run returns ``chunks``
    written documents (or raises ``run_side_effect``). Returns the mock."""
    mock_pipeline = MagicMock()
    if run_side_effect is not None:
        mock_pipeline.run.side_effect = run_side_effect
    else:
        mock_pipeline.run.return_value = {
            "writer": {"documents_written": chunks},
            "converter": {"documents": []},
        }
    monkeypatch.setattr(indexing, "_pipeline", mock_pipeline)
    return mock_pipeline


def _stamped_version(mock_pipeline) -> str:
    """The ingest_version run_indexing_pipeline stamped onto the pipeline meta."""
    (payload,), _ = mock_pipeline.run.call_args
    return payload["converter"]["meta"]["ingest_version"]


def test_overwrite_writes_new_version_before_sweeping_stale(monkeypatch):
    """No pre-delete: the pipeline writes first, then the stale-version sweep
    removes every point of the file EXCEPT the just-written version."""
    order: list[str] = []
    client = _mock_raw_client(monkeypatch)
    client.delete.side_effect = lambda **_: order.append("delete")
    mock_pipeline = _mock_pipeline_run(monkeypatch, chunks=3)
    mock_pipeline.run.side_effect = lambda *a, **k: (
        order.append("run")
        or {
            "writer": {"documents_written": 3},
            "converter": {"documents": []},
        }
    )

    indexing.run_indexing_pipeline(
        "/tmp/fake.pdf", {"file_id": "f-v", "collection_name": "file-f-v", "overwrite": True}
    )

    assert order == ["run", "delete"], "write must happen before the stale sweep"
    version = _stamped_version(mock_pipeline)
    _, kwargs = client.delete.call_args
    assert kwargs["points_selector"] == indexing._stale_version_filter("f-v", version)


def test_meta_stamped_with_version_and_overwrite_stripped(monkeypatch):
    """The pipeline meta carries ingest_version (payload contract) but not the
    overwrite control field."""
    _mock_raw_client(monkeypatch)
    mock_pipeline = _mock_pipeline_run(monkeypatch)

    indexing.run_indexing_pipeline(
        "/tmp/fake.pdf", {"file_id": "f-m", "collection_name": "file-f-m", "overwrite": True}
    )

    (payload,), _ = mock_pipeline.run.call_args
    meta = payload["converter"]["meta"]
    assert meta["ingest_version"]
    assert "overwrite" not in meta


def test_failed_ingest_tears_down_only_its_own_version(monkeypatch):
    """A pipeline failure deletes only the failed run's points — the filter
    must pin BOTH file_id and the run's ingest_version, so the previously
    indexed version survives (no more destroy-then-fail)."""
    client = _mock_raw_client(monkeypatch)
    mock_pipeline = _mock_pipeline_run(
        monkeypatch, run_side_effect=RuntimeError("extraction exploded")
    )

    try:
        indexing.run_indexing_pipeline(
            "/tmp/fake.pdf",
            {"file_id": "f-fail", "collection_name": "file-f-fail", "overwrite": True},
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the pipeline error to propagate")

    version = _stamped_version(mock_pipeline)
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    assert kwargs["points_selector"] == indexing._version_filter("f-fail", version)


def test_failed_append_never_touches_existing_points(monkeypatch):
    """overwrite=false + failure: teardown is still scoped to the run's own
    version — a failed append must not delete pre-existing data (the old
    teardown deleted ALL of the file's points regardless of overwrite)."""
    import pytest

    client = _mock_raw_client(monkeypatch)
    mock_pipeline = _mock_pipeline_run(monkeypatch, run_side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        indexing.run_indexing_pipeline(
            "/tmp/fake.pdf",
            {"file_id": "f-app", "collection_name": "file-f-app", "overwrite": False},
        )

    version = _stamped_version(mock_pipeline)
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    assert kwargs["points_selector"] == indexing._version_filter("f-app", version)


def test_zero_chunk_overwrite_keeps_previous_version(monkeypatch):
    """An overwrite that extracted nothing must NOT sweep the old version —
    an empty extraction never replaces a good index with nothing."""
    client = _mock_raw_client(monkeypatch)
    _mock_pipeline_run(monkeypatch, chunks=0)

    result = indexing.run_indexing_pipeline(
        "/tmp/fake.pdf", {"file_id": "f-z", "collection_name": "file-f-z", "overwrite": True}
    )

    assert result.chunks_count == 0
    client.delete.assert_not_called()


def test_append_success_does_no_cleanup(monkeypatch):
    """overwrite=false + success: nothing is deleted at all."""
    client = _mock_raw_client(monkeypatch)
    _mock_pipeline_run(monkeypatch, chunks=2)

    indexing.run_indexing_pipeline(
        "/tmp/fake.pdf", {"file_id": "f-a", "collection_name": "file-f-a", "overwrite": False}
    )

    client.delete.assert_not_called()


def test_stale_sweep_failure_is_swallowed(monkeypatch, caplog):
    """A failed stale-version sweep must not fail the (already successful)
    ingest — worst case is duplicates until the next overwrite, and the
    operator gets a WARNING."""
    import logging

    client = _mock_raw_client(monkeypatch, delete_side_effect=RuntimeError("qdrant unreachable"))
    _mock_pipeline_run(monkeypatch, chunks=3)

    with caplog.at_level(logging.WARNING, logger="app.pipelines.indexing"):
        result = indexing.run_indexing_pipeline(
            "/tmp/fake.pdf", {"file_id": "f-s", "collection_name": "file-f-s", "overwrite": True}
        )

    assert result.chunks_count == 3
    client.delete.assert_called_once()
    assert any("stale-version sweep failed" in r.message for r in caplog.records)


def test_stale_filter_shape_sweeps_unversioned_legacy_points():
    """The sweep filter must use must_not on ingest_version (matching points
    where the field is ABSENT too) so pre-versioning points get cleaned up by
    the first versioned overwrite."""
    f = indexing._stale_version_filter("f-1", "v-keep")
    assert [c.key for c in f.must] == ["meta.file_id"]
    assert [c.key for c in f.must_not] == ["meta.ingest_version"]
    assert f.must_not[0].match.value == "v-keep"


# ---------------------------------------------------------------------------
# delete_by_file_id — backs DELETE /api/v1/documents/{file_id}.
# ---------------------------------------------------------------------------


def _mock_raw_client(monkeypatch, *, count=0, delete_side_effect=None):
    """Patch the direct Qdrant client accessor and return the mock so tests can
    assert on it. Raw point ops (count/scroll/delete) go through this, NOT the
    QdrantDocumentStore, since qdrant-haystack 10.x hides its client."""
    from types import SimpleNamespace

    client = MagicMock()
    client.count.return_value = SimpleNamespace(count=count)
    if delete_side_effect is not None:
        client.delete.side_effect = delete_side_effect
    monkeypatch.setattr(indexing, "_raw_qdrant_client", lambda: client)
    # Non-None document store is the "pipeline initialized" signal.
    monkeypatch.setattr(indexing, "_document_store", MagicMock())
    return client


def test_delete_by_file_id_counts_then_deletes(monkeypatch):
    """Returns the pre-delete chunk count and issues a filtered client.delete."""
    client = _mock_raw_client(monkeypatch, count=3)

    count = indexing.delete_by_file_id("f-1")

    assert count == 3
    client.delete.assert_called_once()
    _, kwargs = client.delete.call_args
    assert kwargs["collection_name"] == indexing.global_settings.qdrant_index
    # points_selector is the same file_id filter used by count/scroll.
    assert kwargs["points_selector"] == indexing._file_id_filter("f-1")


def test_delete_by_file_id_takes_per_file_id_lock(monkeypatch):
    """The delete must serialize against a concurrent same-file ingest by
    entering the per-file-id lock (not just calling the raw client)."""
    _mock_raw_client(monkeypatch, count=0)

    seen: list[str] = []
    real_lock = indexing._per_file_id_lock

    def spy(file_id):
        seen.append(file_id)
        return real_lock(file_id)

    monkeypatch.setattr(indexing, "_per_file_id_lock", spy)

    indexing.delete_by_file_id("f-lock")

    assert seen == ["f-lock"]


def test_delete_by_file_id_raises_when_not_initialized(monkeypatch):
    """Before init_pipeline, there's no document store — surface it, don't no-op."""
    monkeypatch.setattr(indexing, "_document_store", None)

    try:
        indexing.delete_by_file_id("f-x")
    except RuntimeError as exc:
        assert "not initialized" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when document store is None")


def test_delete_by_file_id_propagates_qdrant_error(monkeypatch):
    """A genuine Qdrant failure must propagate (→ DELETE_FAILED at the route),
    not be swallowed the way the ingest-teardown _delete_ingest_version is."""
    _mock_raw_client(monkeypatch, count=2, delete_side_effect=RuntimeError("qdrant unreachable"))

    try:
        indexing.delete_by_file_id("f-err")
    except RuntimeError as exc:
        assert "qdrant" in str(exc).lower()
    else:
        raise AssertionError("expected the Qdrant error to propagate")


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


# -------------------- Pipeline wiring: the dense-vector invariant --------------------


@component
class _Passthrough:
    """Minimal Haystack component with documents in → documents out.

    Module-level (not nested in a helper) because Haystack resolves ``run``'s
    annotations with ``typing.get_type_hints``, which cannot see names from a
    function's local scope under ``from __future__ import annotations``.
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        return {"documents": documents}


def _passthrough_component():
    return _Passthrough()


@pytest.mark.parametrize("sparse_enabled", [True, False])
def test_build_pipeline_guards_writer_in_both_branches(monkeypatch, sparse_enabled):
    """The guard must be the last hop before the writer whether or not sparse
    embeddings are on — a chunk with no dense vector must never reach Qdrant."""
    from unittest.mock import MagicMock

    from app.pipelines import indexing

    monkeypatch.setattr(
        indexing, "_build_converter_for_pipeline", lambda _s: _passthrough_component()
    )
    monkeypatch.setattr(indexing, "build_splitter", lambda _s: _passthrough_component())
    monkeypatch.setattr(indexing, "build_dense_embedder", lambda _s: _passthrough_component())
    monkeypatch.setattr(
        indexing,
        "build_sparse_embedder",
        lambda _s: _passthrough_component() if sparse_enabled else None,
    )

    s = indexing.global_settings
    pipeline = indexing._build_pipeline(s, MagicMock())
    edges = {(a, b) for a, b, *_ in pipeline.graph.edges}

    assert ("embedding_guard", "writer") in edges
    # Nothing bypasses the guard on its way to the writer.
    assert [src for src, dst in edges if dst == "writer"] == ["embedding_guard"]
    if sparse_enabled:
        assert ("sparse_embedder", "embedding_guard") in edges
    else:
        assert ("dense_embedder", "embedding_guard") in edges


def test_dropped_dense_embedding_surfaces_as_embedding_failed():
    """End-to-end over the three layers: the guard raises inside a real Haystack
    pipeline, Haystack wraps it in ``PipelineRuntimeError``, and the route layer
    still classifies it as ``EMBEDDING_FAILED`` (not ``PIPELINE_FAILED``).

    Ties together the failure that silently wrote 28k vectorless points: a
    rejected embedding batch leaves documents with ``embedding=None``, and
    nothing downstream objected.
    """
    from haystack import Pipeline

    from app.pipelines.embedders import DenseEmbeddingGuard
    from app.routes.ingest import _classify_pipeline_error

    @component
    class _Source:
        @component.output_types(documents=list[Document])
        def run(self) -> dict:
            return {
                "documents": [
                    Document(content="ok", embedding=[0.1], meta={"split_id": 0}),
                    Document(content="dropped", meta={"split_id": 1}),
                ]
            }

    @component
    class _ExplodingWriter:
        @component.output_types(documents_written=int)
        def run(self, documents: list[Document]) -> dict:
            raise AssertionError("writer must never see a vectorless document")

    pipeline = Pipeline()
    pipeline.add_component("src", _Source())
    pipeline.add_component("embedding_guard", DenseEmbeddingGuard())
    pipeline.add_component("writer", _ExplodingWriter())
    pipeline.connect("src.documents", "embedding_guard.documents")
    pipeline.connect("embedding_guard.documents", "writer.documents")

    from haystack.core.errors import PipelineRuntimeError

    with pytest.raises(PipelineRuntimeError) as excinfo:
        pipeline.run({})

    assert _classify_pipeline_error(excinfo.value) == "EMBEDDING_FAILED"
