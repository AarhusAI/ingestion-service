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

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.config import Settings
from app.config import settings as global_settings
from app.pipelines.converters import build_converter
from app.pipelines.embedders import build_dense_embedder, build_sparse_embedder
from app.pipelines.splitter import build_splitter

log = logging.getLogger(__name__)


_pipeline: Pipeline | None = None
_document_store: QdrantDocumentStore | None = None


def init_pipeline(settings: Settings | None = None) -> None:
    """Build (or rebuild) the pipeline + document store. Called from lifespan."""
    global _pipeline, _document_store
    s = settings or global_settings
    _document_store = _build_document_store(s)
    _pipeline = _build_pipeline(s, _document_store)
    log.info(
        "indexing pipeline ready (extraction=%s, embed_provider=%s, embed_model=%s, "
        "sparse=%s, qdrant_index=%s)",
        s.extraction_engine,
        s.embedding_provider,
        s.embedding_model,
        s.enable_sparse_embeddings,
        s.qdrant_index,
    )


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


def _build_pipeline(s: Settings, document_store: QdrantDocumentStore) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component("converter", build_converter(s))
    # Token-aware splitter when chunk_split_by="token" (default), Haystack's
    # word/sentence/passage DocumentSplitter otherwise. See app/pipelines/splitter.py.
    pipeline.add_component("splitter", build_splitter(s))
    pipeline.add_component("dense_embedder", build_dense_embedder(s))

    pipeline.connect("converter.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "dense_embedder.documents")

    sparse = build_sparse_embedder(s)
    if sparse is not None:
        pipeline.add_component("sparse_embedder", sparse)
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
        pipeline.connect("sparse_embedder.documents", "writer.documents")
    else:
        pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        pipeline.connect("dense_embedder.documents", "writer.documents")

    return pipeline


def run_indexing_pipeline(file_path: str, meta: dict) -> int:
    """Run the configured pipeline against a single local file. Returns chunk count.

    ``meta`` carries the document metadata (``file_id``, ``collection_name``,
    ``collection_type``, ``name``, ``source``, ``user_id``, ``overwrite``).
    Control fields (``overwrite``) are stripped before the meta hits Qdrant.
    """
    if _pipeline is None or _document_store is None:
        raise RuntimeError("pipeline not initialized; call init_pipeline() first")

    file_id = meta["file_id"]
    overwrite = meta.get("overwrite", True)

    if overwrite:
        _delete_existing_by_file_id(file_id)

    pipeline_meta = _strip_control_fields(meta)
    try:
        result = _pipeline.run({"converter": {"sources": [file_path], "meta": pipeline_meta}})
        chunks_count = result["writer"]["documents_written"]
        log.info(
            "ingest ok: file_id=%s collection=%s chunks=%d",
            file_id,
            meta.get("collection_name"),
            chunks_count,
        )
        return chunks_count
    except Exception:
        log.exception("ingest failed for file_id=%s; rolling back", file_id)
        _delete_existing_by_file_id(file_id)
        raise


def _delete_existing_by_file_id(file_id: str) -> None:
    if _document_store is None:
        return
    try:
        _document_store.client.delete(
            collection_name=_document_store.index,
            points_selector=Filter(
                must=[FieldCondition(key="meta.file_id", match=MatchValue(value=file_id))]
            ),
        )
    except Exception:
        # If the collection doesn't exist yet (first ever ingest), this is fine.
        log.debug("delete-by-file_id skipped (collection likely empty): file_id=%s", file_id)


def _strip_control_fields(meta: dict) -> dict:
    """Drop fields that are pipeline-control, not Qdrant payload."""
    return {k: v for k, v in meta.items() if k not in {"overwrite"}}
