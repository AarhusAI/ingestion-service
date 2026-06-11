"""Haystack indexing pipeline.

Builds the pipeline once at lifespan startup and caches it as module-level
state — Haystack pipelines are non-trivial to construct so this matters for
per-request latency.

Idempotency: when ``meta.overwrite`` is true (default), existing Qdrant
points with matching ``meta.file_id`` are deleted before writing new chunks.
On any pipeline failure the same delete runs as teardown — partial writes
are not left behind.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import NamedTuple

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app import metrics
from app.config import Settings
from app.config import settings as global_settings
from app.log_utils import sanitize_for_log
from app.pipelines.converters import build_converter
from app.pipelines.embedders import build_dense_embedder, build_sparse_embedder
from app.pipelines.splitter import build_splitter

log = logging.getLogger(__name__)


_pipeline: Pipeline | None = None
_document_store: QdrantDocumentStore | None = None
# Settings used to build the cached pipeline — kept so request-time helpers
# (e.g. the extraction summary) can read the active engine without threading
# settings through every call.
_settings: Settings | None = None


class IndexingResult(NamedTuple):
    """Outcome of one ``run_indexing_pipeline`` call.

    ``extraction`` is ``{"engine": str, "route": dict | None}`` — ``route`` is
    the auto-router's signal+metrics when ``EXTRACTION_ENGINE=auto`` stamped
    them onto the documents, else ``None`` (pinned engine: no classification).
    """

    chunks_count: int
    extraction: dict | None


# Per-file_id locks serialize concurrent ingests of the same file. Without
# this, two requests racing on the same file_id can interleave the
# delete-then-write step and leave either duplicate vectors (both writes
# succeed) or no vectors (request A's teardown deletes request B's data
# after B already finished). See sec.md Finding 7.
#
# Process-local: FastAPI runs sync handlers in a thread pool inside one
# uvicorn worker process; multiple processes don't share state. Open WebUI's
# per-file state machine gates same-file_id calls across processes upstream,
# so a process-local lock is sufficient for the documented contract.
_file_id_locks_lock = threading.Lock()
_file_id_locks: dict[str, list] = {}  # file_id -> [Lock, refcount]


@contextmanager
def _per_file_id_lock(file_id: str):
    """Acquire a Lock keyed on file_id; release and clean up on exit.

    Entries are reference-counted so the registry doesn't grow unboundedly
    across the lifetime of the worker — file_ids are unique UUIDs, so
    without cleanup we'd accumulate one Lock per file ever ingested.
    """
    with _file_id_locks_lock:
        entry = _file_id_locks.get(file_id)
        if entry is None:
            entry = [threading.Lock(), 0]
            _file_id_locks[file_id] = entry
        entry[1] += 1
        lock = entry[0]

    try:
        with lock:
            yield
    finally:
        with _file_id_locks_lock:
            entry[1] -= 1
            if entry[1] == 0:
                # Last waiter released — drop the entry.
                _file_id_locks.pop(file_id, None)


def init_pipeline(settings: Settings | None = None) -> None:
    """Build (or rebuild) the pipeline + document store, then warm it up.

    Called from FastAPI lifespan. The warm-up step walks every component
    and calls each one's ``warm_up()`` if defined — Haystack's
    ``FastembedSparseDocumentEmbedder`` downloads the BM42 model there.
    Without this call, fastembed lazy-loads on the first ``.run()``, so
    the first user-facing ingest pays a ~20-60 s cold-download cost.
    Moving it here means startup takes longer but per-request latency
    is predictable.
    """
    global _pipeline, _document_store, _settings
    s = settings or global_settings
    _settings = s
    _document_store = _build_document_store(s)
    _pipeline = _build_pipeline(s, _document_store)

    started = time.monotonic()
    _pipeline.warm_up()
    warm_up_s = time.monotonic() - started

    log.info(
        "indexing pipeline ready (extraction=%s, embed_provider=%s, embed_model=%s, "
        "sparse=%s, qdrant_index=%s, warm_up=%.2fs)",
        s.extraction_engine,
        s.embedding_provider,
        s.embedding_model,
        s.enable_sparse_embeddings,
        s.qdrant_index,
        warm_up_s,
    )


def is_pipeline_ready() -> bool:
    """True iff ``init_pipeline()`` has finished and the pipeline is warm.

    Used by the ``/health/ready`` probe so Kubernetes / Docker readiness
    correctly waits for the model download before routing traffic.
    """
    return _pipeline is not None and _document_store is not None


def _build_document_store(s: Settings) -> QdrantDocumentStore:
    # QdrantDocumentStore.api_key must be a haystack Secret (or None);
    # passing a plain string raises 'str' has no attribute 'resolve_value'
    # at first write. Mirrors the pattern in embedders.py.
    api_key = Secret.from_token(s.qdrant_api_key) if s.qdrant_api_key else None
    return QdrantDocumentStore(
        url=s.qdrant_uri,
        api_key=api_key,
        index=s.qdrant_index,
        embedding_dim=s.embedding_dim,
        use_sparse_embeddings=s.enable_sparse_embeddings,
        # Multitenancy HNSW: per-tenant subgraphs keyed off the meta.collection_name
        # payload index (see app/services/qdrant_setup.py). Applies to the dense
        # vector; sparse vectors live in Qdrant's inverted index.
        hnsw_config={"m": 0, "payload_m": 16},
        recreate_index=False,
    )


def _build_converter_for_pipeline(s: Settings):
    """Pick the converter component for the cached pipeline.

    ``EXTRACTION_ENGINE=auto`` enables per-document routing — a single
    ``RoutingConverter`` wraps the routable engines and decides per source.
    Any other value pins one engine via ``build_converter`` (unchanged
    behaviour). Imported lazily so non-auto deployments never load the router.
    """
    if s.extraction_engine.lower() == "auto":
        from app.pipelines.routing_converter import RoutingConverter

        return RoutingConverter(s)
    return build_converter(s)


def _build_pipeline(s: Settings, document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    # Each component's run is wrapped to record per-stage latency
    # (pipeline_stage_duration_seconds{stage=...}); instrument_stage is a no-op
    # passthrough if wrapping ever fails, so it can't break the pipeline.
    pipeline.add_component(
        "converter", metrics.instrument_stage(_build_converter_for_pipeline(s), "converter")
    )
    # Token-aware splitter when chunk_split_by="token" (default), Haystack's
    # word/sentence/passage DocumentSplitter otherwise. See app/pipelines/splitter.py.
    pipeline.add_component("splitter", metrics.instrument_stage(build_splitter(s), "splitter"))
    pipeline.add_component(
        "dense_embedder", metrics.instrument_stage(build_dense_embedder(s), "dense_embedder")
    )

    pipeline.connect("converter.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "dense_embedder.documents")

    sparse = build_sparse_embedder(s)
    if sparse is not None:
        pipeline.add_component(
            "sparse_embedder", metrics.instrument_stage(sparse, "sparse_embedder")
        )
        pipeline.add_component(
            "writer",
            metrics.instrument_stage(DocumentWriter(document_store=document_store), "writer"),
        )
        pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
        pipeline.connect("sparse_embedder.documents", "writer.documents")
    else:
        pipeline.add_component(
            "writer",
            metrics.instrument_stage(DocumentWriter(document_store=document_store), "writer"),
        )
        pipeline.connect("dense_embedder.documents", "writer.documents")

    return pipeline


def run_indexing_pipeline(file_path: str, meta: dict) -> IndexingResult:
    """Run the configured pipeline against a single local file.

    Returns an :class:`IndexingResult` (chunk count + extraction summary).

    ``meta`` carries the document metadata (``file_id``, ``collection_name``,
    ``collection_type``, ``name``, ``source``, ``user_id``, ``overwrite``).
    Control fields (``overwrite``) are stripped before the meta hits Qdrant.

    Concurrent ingests of the same ``file_id`` are serialized via
    :func:`_per_file_id_lock` — see the comment on ``_file_id_locks`` for
    the race this closes (sec.md Finding 7).
    """
    if _pipeline is None or _document_store is None:
        raise RuntimeError("pipeline not initialized; call init_pipeline() first")

    file_id = meta["file_id"]
    overwrite = meta.get("overwrite", True)

    with _per_file_id_lock(file_id):
        if overwrite:
            _delete_existing_by_file_id(file_id)

        pipeline_meta = _strip_control_fields(meta)
        try:
            started = time.monotonic()
            # include_outputs_from exposes the converter's stamped meta (the
            # auto-router writes extraction_engine / extraction_route there) so
            # the decision surfaces in the response without an extra pass.
            result = _pipeline.run(
                {"converter": {"sources": [file_path], "meta": pipeline_meta}},
                include_outputs_from={"converter"},
            )
            metrics.ingest_duration_seconds.observe(time.monotonic() - started)

            chunks_count = result["writer"]["documents_written"]
            extraction = _extraction_summary(result)
            metrics.ingest_chunks.observe(chunks_count)
            if extraction:
                metrics.extraction_route_total.labels(
                    engine=extraction["engine"],
                    signal=(extraction["route"] or {}).get("signal", "default"),
                ).inc()

            log.info(
                "ingest ok: file_id=%s collection=%s chunks=%d engine=%s signal=%s",
                sanitize_for_log(file_id),
                sanitize_for_log(meta.get("collection_name")),
                chunks_count,
                extraction["engine"] if extraction else "-",
                (extraction["route"] or {}).get("signal", "-") if extraction else "-",
            )
            return IndexingResult(chunks_count=chunks_count, extraction=extraction)
        except Exception:
            log.exception(
                "ingest failed for file_id=%s; rolling back",
                sanitize_for_log(file_id),
            )
            _delete_existing_by_file_id(file_id)
            raise


def _extraction_summary(result: dict) -> dict | None:
    """Derive ``{"engine", "route"}`` from the pipeline result.

    Reads the converter's stamped meta (``extraction_engine`` /
    ``extraction_route``) captured via ``include_outputs_from``. For a pinned
    engine the router never ran, so nothing is stamped — fall back to the
    configured engine with no route.
    """
    docs = (result.get("converter") or {}).get("documents") or []
    meta0 = (docs[0].meta if docs else {}) or {}
    engine = meta0.get("extraction_engine")
    route = meta0.get("extraction_route")
    if engine is None:
        engine = _settings.extraction_engine if _settings is not None else "unknown"
    return {"engine": engine, "route": route}


def _delete_existing_by_file_id(file_id: str) -> None:
    if _document_store is None:
        return
    try:
        _document_store.client.delete(
            collection_name=_document_store.index,
            points_selector=_file_id_filter(file_id),
        )
    except Exception:
        # If the collection doesn't exist yet (first ever ingest), this is fine.
        log.debug(
            "delete-by-file_id skipped (collection likely empty): file_id=%s",
            sanitize_for_log(file_id),
        )


def _file_id_filter(file_id: str) -> Filter:
    """The Qdrant filter for every point belonging to one file — the single
    contract shared by delete, count, and the chunk-inspection scroll."""
    return Filter(must=[FieldCondition(key="meta.file_id", match=MatchValue(value=file_id))])


def count_chunks_by_file_id(file_id: str) -> int:
    """Exact count of stored chunks for ``file_id`` (read-only; used by the
    chunk-inspection endpoint). Raises if the pipeline isn't initialized."""
    if _document_store is None:
        raise RuntimeError("pipeline not initialized; call init_pipeline() first")
    result = _document_store.client.count(
        collection_name=_document_store.index,
        count_filter=_file_id_filter(file_id),
        exact=True,
    )
    return result.count


def scroll_chunks_by_file_id(file_id: str, limit: int, offset: str | None = None):
    """Page through stored chunks for ``file_id`` (read-only).

    Returns ``(points, next_offset)`` straight from Qdrant's cursor-based
    ``scroll``: ``points`` carry ``payload`` (content + meta), no vectors;
    ``next_offset`` is the cursor for the next page (``None`` when exhausted).
    Raises if the pipeline isn't initialized.
    """
    if _document_store is None:
        raise RuntimeError("pipeline not initialized; call init_pipeline() first")
    return _document_store.client.scroll(
        collection_name=_document_store.index,
        scroll_filter=_file_id_filter(file_id),
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )


def _strip_control_fields(meta: dict) -> dict:
    """Drop fields that are pipeline-control, not Qdrant payload."""
    return {k: v for k, v in meta.items() if k not in {"overwrite"}}
