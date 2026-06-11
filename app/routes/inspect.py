"""``GET /api/v1/documents/{file_id}/chunks`` — chunk inspection endpoint.

Read-only operator tool: returns the chunks stored for a document (by
``meta.file_id``) with their metadata, so you can answer "what did this
document get split into, and which extraction/classification path did it hit?"
without poking Qdrant directly. No write-path cost — it scrolls the existing
points.

Opt-in via ``ENABLE_INSPECTION_API`` (off by default — it returns chunk
*content*). Bearer-auth applies regardless; when disabled the endpoint 404s.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import verify_api_key
from app.config import settings
from app.log_utils import sanitize_for_log
from app.models import (
    ChunkView,
    DocumentChunksResponse,
    DocumentChunkStats,
    IngestError,
)
from app.pipelines.indexing import (
    count_chunks_by_file_id,
    is_pipeline_ready,
    scroll_chunks_by_file_id,
)
from app.routes.ingest import _safe_error_detail

log = logging.getLogger(__name__)

router = APIRouter()

# How much chunk text to return in ``preview`` mode.
_PREVIEW_CHARS = 200
# Hard cap on the page size so a single call can't pull an unbounded payload.
_MAX_LIMIT = 1000
_CONTENT_MODES = ("false", "preview", "full")


@router.get(
    "/api/v1/documents/{file_id}/chunks",
    response_model=DocumentChunksResponse,
    response_model_exclude_none=True,
    responses={
        404: {"model": IngestError},
        500: {"model": IngestError},
        503: {"model": IngestError},
    },
)
async def get_document_chunks(
    file_id: str,
    limit: int = Query(50, ge=1, le=_MAX_LIMIT, description="Max chunks to return."),
    offset: str | None = Query(
        None, description="Qdrant cursor from a previous response's next_offset."
    ),
    include_content: str = Query(
        "preview",
        description="How to return chunk text: false | preview | full.",
    ),
    _api_key: str = Depends(verify_api_key),
):
    if not settings.enable_inspection_api:
        # 404 (not 403) so a disabled endpoint is indistinguishable from one
        # that doesn't exist — no signal that the feature is merely off.
        raise HTTPException(
            status_code=404,
            detail=IngestError(
                error="chunk inspection API is disabled (set ENABLE_INSPECTION_API=true)",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    if include_content not in _CONTENT_MODES:
        raise HTTPException(
            status_code=400,
            detail=IngestError(
                error=f"include_content must be one of {', '.join(_CONTENT_MODES)}",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    if not is_pipeline_ready():
        raise HTTPException(
            status_code=503,
            detail=IngestError(
                error="pipeline not warmed up yet", code="PIPELINE_FAILED"
            ).model_dump(),
        )

    try:
        total = count_chunks_by_file_id(file_id)
        points, next_offset = scroll_chunks_by_file_id(file_id, limit=limit, offset=offset)
    except Exception as exc:
        log.exception("chunk inspection failed for file_id=%s", sanitize_for_log(file_id))
        raise HTTPException(
            status_code=500,
            detail=_safe_error_detail("PIPELINE_FAILED", exc),
        ) from exc

    chunks = [_to_chunk_view(p, include_content) for p in points]
    return DocumentChunksResponse(
        file_id=file_id,
        returned=len(chunks),
        limit=limit,
        next_offset=str(next_offset) if next_offset is not None else None,
        stats=_build_stats(total, points),
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_chunk_view(point, include_content: str) -> ChunkView:
    payload = getattr(point, "payload", None) or {}
    content = payload.get("content") or ""
    meta = payload.get("meta") or {}

    view_content: str | None = None
    if include_content == "full":
        view_content = content
    elif include_content == "preview":
        view_content = content[:_PREVIEW_CHARS]
        if len(content) > _PREVIEW_CHARS:
            view_content += "…"

    return ChunkView(
        split_id=meta.get("split_id"),
        content_length=len(content),
        content=view_content,
        meta=meta,
    )


def _build_stats(total: int, points) -> DocumentChunkStats:
    """Doc-level summary. The engine/route/languages/name fields are identical
    across a document's chunks, so they're read from the first returned chunk."""
    meta0 = ((getattr(points[0], "payload", None) or {}).get("meta") if points else {}) or {}
    return DocumentChunkStats(
        total_chunks=total,
        extraction_engine=meta0.get("extraction_engine"),
        extraction_route=meta0.get("extraction_route"),
        languages=meta0.get("languages"),
        name=meta0.get("name"),
        collection_name=meta0.get("collection_name"),
    )
