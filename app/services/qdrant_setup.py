"""Qdrant startup hook.

``QdrantDocumentStore`` creates the collection on first write but doesn't
create payload indexes. We do that explicitly here so per-tenant HNSW kicks
in from day one — important for retrieval performance once the agent is
rewritten in Phase 3.
"""

import logging

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import KeywordIndexParams, KeywordIndexType, PayloadSchemaType

from app.config import settings
from app.pipelines.indexing import _raw_qdrant_client

log = logging.getLogger(__name__)


def ensure_payload_indexes() -> None:
    """Create the payload indexes: the tenant index on ``meta.collection_name``
    plus plain keyword indexes (collection_type, languages, file_id,
    ingest_version).

    Idempotent — Qdrant returns 409 if the index already exists, which we swallow.
    No-op if the collection doesn't yet exist (first ingest creates it; we'll
    re-run this from the next startup).
    """
    client = _raw_qdrant_client()
    index_name = settings.qdrant_index

    try:
        client.get_collection(index_name)
    except (UnexpectedResponse, ValueError) as exc:
        log.info(
            "qdrant collection %r not yet created; payload index will be added on next startup "
            "after the first ingest creates it (%s)",
            index_name,
            exc,
        )
        return

    # - meta.collection_name: the tenant index the multitenancy HNSW keys off.
    # - meta.collection_type: admin queries; not used for retrieval filtering.
    # - meta.languages: ISO 639-1 codes from converter language detection
    #   (Kreuzberg's ``detected_languages``). KEYWORD on a list-valued field
    #   supports MatchAny — perfect for "any of these languages". No consumer
    #   yet on the retrieval-agent side, but adding the index now means new
    #   ingests are searchable as soon as the filter call site lands.
    # - meta.file_id: every point op filters on it — the versioned overwrite
    #   sweep/teardown, DELETE /api/v1/documents/{file_id}, and the
    #   chunk-inspection count/scroll. Without the index those are full scans.
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


def health_check() -> bool:
    """True iff Qdrant is reachable and answering. Used by /health/ready."""
    try:
        _raw_qdrant_client().get_collections()
        return True
    except Exception as exc:
        log.warning("qdrant health check failed: %s", exc)
        return False
