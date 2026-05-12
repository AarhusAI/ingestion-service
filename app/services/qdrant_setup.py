"""Qdrant startup hook.

``QdrantDocumentStore`` creates the collection on first write but doesn't
create payload indexes. We do that explicitly here so per-tenant HNSW kicks
in from day one — important for retrieval performance once the agent is
rewritten in Phase 3.
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import KeywordIndexParams, KeywordIndexType, PayloadSchemaType

from app.config import settings

log = logging.getLogger(__name__)


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_uri, api_key=settings.qdrant_api_key)


def ensure_payload_indexes() -> None:
    """Create the keyword + tenant payload index on ``meta.collection_name``.

    Idempotent — Qdrant returns 409 if the index already exists, which we swallow.
    No-op if the collection doesn't yet exist (first ingest creates it; we'll
    re-run this from the next startup).
    """
    client = _client()
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

    field_name = "meta.collection_name"
    try:
        client.create_payload_index(
            collection_name=index_name,
            field_name=field_name,
            field_schema=KeywordIndexParams(
                type=KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
        log.info("created payload index on %s.%s (tenant=True)", index_name, field_name)
    except UnexpectedResponse as exc:
        # 409 conflict = already exists; anything else is real
        if exc.status_code == 409:
            log.debug("payload index on %s.%s already exists", index_name, field_name)
        else:
            raise

    # Secondary index on collection_type for admin queries; not used for retrieval filtering.
    field_name_type = "meta.collection_type"
    try:
        client.create_payload_index(
            collection_name=index_name,
            field_name=field_name_type,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("created payload index on %s.%s", index_name, field_name_type)
    except UnexpectedResponse as exc:
        if exc.status_code == 409:
            log.debug("payload index on %s.%s already exists", index_name, field_name_type)
        else:
            raise

    # Multilingual filtering — ``meta.languages`` is a list of ISO 639-1
    # codes populated by the converter when language detection runs
    # (Kreuzberg's ``detected_languages``). KEYWORD on a list-valued field
    # supports MatchAny — perfect for "any of these languages". No consumer
    # yet on the retrieval-agent side, but adding the index now means new
    # ingests are searchable as soon as the filter call site lands.
    field_name_lang = "meta.languages"
    try:
        client.create_payload_index(
            collection_name=index_name,
            field_name=field_name_lang,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("created payload index on %s.%s", index_name, field_name_lang)
    except UnexpectedResponse as exc:
        if exc.status_code == 409:
            log.debug("payload index on %s.%s already exists", index_name, field_name_lang)
        else:
            raise


def health_check() -> bool:
    """True iff Qdrant is reachable and answering. Used by /health/ready."""
    try:
        _client().get_collections()
        return True
    except Exception as exc:
        log.warning("qdrant health check failed: %s", exc)
        return False
