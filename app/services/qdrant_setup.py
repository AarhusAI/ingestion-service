"""Qdrant startup hook.

``init_pipeline`` forces the collection into existence, but
``QdrantDocumentStore`` doesn't create payload indexes or migrate the sparse
IDF modifier on an existing collection. We do both explicitly here so
per-tenant HNSW and BM42 scoring work from day one.
"""

import logging

from haystack_integrations.document_stores.qdrant.document_store import SPARSE_VECTORS_NAME
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    KeywordIndexParams,
    KeywordIndexType,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
)

from app.pipelines.indexing import _active_settings, _raw_qdrant_client

log = logging.getLogger(__name__)


def ensure_payload_indexes() -> None:
    """Create the payload indexes: the tenant index on ``meta.collection_name``
    plus plain keyword indexes (collection_type, languages, file_id,
    ingest_version).

    Idempotent — Qdrant returns 409 if the index already exists, which we swallow.
    No-op if the collection is missing (only possible if it was deleted
    between ``init_pipeline`` and this call).
    """
    client = _raw_qdrant_client()
    index_name = _active_settings().qdrant_index

    try:
        client.get_collection(index_name)
    except (UnexpectedResponse, ValueError) as exc:
        log.warning("qdrant collection %r missing; skipping payload indexes (%s)", index_name, exc)
        return

    # - meta.collection_name: the tenant index the multitenancy HNSW keys off.
    # - meta.collection_type: admin queries; not used for retrieval filtering.
    # - meta.languages: ISO 639-1 codes from converter language detection
    #   (Kreuzberg's ``detected_languages``). KEYWORD on a list-valued field
    #   supports MatchAny — perfect for "any of these languages". No consumer
    #   yet on the retrieval-agent side, but adding the index now means new
    #   ingests are searchable as soon as the filter call site lands.
    # - meta.file_id: every point op filters on it — the versioned overwrite
    #   sweep/teardown and DELETE /api/v1/documents/{file_id}. Without the
    #   index those are full scans.
    # - meta.ingest_version: the versioned (blue/green) overwrite filters on it
    #   with must/must_not alongside file_id (see app/pipelines/indexing.py).
    tenant = KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True)
    keyword = PayloadSchemaType.KEYWORD
    for field_name, schema in (
        ("meta.collection_name", tenant),
        ("meta.collection_type", keyword),
        ("meta.languages", keyword),
        ("meta.file_id", keyword),
        ("meta.ingest_version", keyword),
    ):
        try:
            client.create_payload_index(
                collection_name=index_name, field_name=field_name, field_schema=schema
            )
            log.info("created payload index on %s.%s", index_name, field_name)
        except UnexpectedResponse as exc:
            # 409 conflict = already exists; anything else is real
            if exc.status_code != 409:
                raise
            log.debug("payload index on %s.%s already exists", index_name, field_name)


def ensure_sparse_idf() -> None:
    """Enable the IDF modifier on the sparse vector of an existing collection.

    New collections get it from ``sparse_idf=True`` at creation; collections
    created before that flag need this in-place update (no reindex — IDF is
    computed by Qdrant at query time). Idempotent.
    """
    s = _active_settings()
    if not s.enable_sparse_embeddings:
        return
    client = _raw_qdrant_client()
    try:
        info = client.get_collection(s.qdrant_index)
    except (UnexpectedResponse, ValueError) as exc:
        log.warning("qdrant collection %r missing; skipping IDF check (%s)", s.qdrant_index, exc)
        return
    sparse = (info.config.params.sparse_vectors or {}).get(SPARSE_VECTORS_NAME)
    if sparse is None or sparse.modifier == Modifier.IDF:
        return
    client.update_collection(
        collection_name=s.qdrant_index,
        sparse_vectors_config={SPARSE_VECTORS_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )
    log.info("enabled IDF modifier on %s.%s", s.qdrant_index, SPARSE_VECTORS_NAME)


def health_check() -> bool:
    """True iff Qdrant is reachable and answering. Used by /health/ready."""
    try:
        _raw_qdrant_client().get_collections()
        return True
    except Exception as exc:
        log.warning("qdrant health check failed: %s", exc)
        return False
