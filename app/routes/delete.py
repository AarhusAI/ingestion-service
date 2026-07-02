"""``DELETE /api/v1/documents/{file_id}`` — remove a file's chunks from Qdrant.

Symmetric counterpart to ``PUT /api/v1/ingest``: Open WebUI calls this when a
file is deleted so the file's vectors don't outlive the file (the vector store
now lives here, not inside Open WebUI, so OWUI's own cleanup no longer reaches
it). Idempotent — deleting an unknown / already-gone file_id returns 200 with
``chunks_deleted: 0``, so the caller can fire it unconditionally and retry.

Bearer-auth via the same ``API_KEY`` as ``/api/v1/ingest``.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException

from app import metrics
from app.auth import verify_api_key
from app.log_utils import sanitize_for_log
from app.models import DeleteResponse, IngestError
from app.pipelines.indexing import delete_by_file_id, is_pipeline_ready
from app.routes.ingest import _safe_error_detail

log = logging.getLogger(__name__)

router = APIRouter()


@router.delete(
    "/api/v1/documents/{file_id}",
    response_model=DeleteResponse,
    responses={
        500: {"model": IngestError},
        503: {"model": IngestError},
    },
)
async def delete_document(
    file_id: str,
    _api_key: str = Depends(verify_api_key),
):
    if not is_pipeline_ready():
        # Delete only needs the document store, but gating on full readiness
        # matches the inspection endpoint and keeps the collection guaranteed to
        # exist (qdrant_setup runs in the same lifespan as init_pipeline).
        metrics.delete_requests_total.labels(outcome="error", code="PIPELINE_FAILED").inc()
        raise HTTPException(
            status_code=503,
            detail=IngestError(
                error="pipeline not warmed up yet", code="PIPELINE_FAILED"
            ).model_dump(),
        )

    try:
        # delete_by_file_id acquires a threading.Lock, so it MUST run in a worker
        # thread — never block the event loop (and the /health probes) on it.
        chunks_deleted = await asyncio.to_thread(delete_by_file_id, file_id)
    except Exception as exc:
        log.exception("delete failed for file_id=%s", sanitize_for_log(file_id))
        metrics.delete_requests_total.labels(outcome="error", code="DELETE_FAILED").inc()
        raise HTTPException(
            status_code=500,
            detail=_safe_error_detail("DELETE_FAILED", exc),
        ) from exc

    metrics.delete_requests_total.labels(outcome="success", code="none").inc()
    # INFO so a deletion is visible in the logs (the /metrics counter and the
    # uvicorn access line don't record how many chunks were actually removed).
    # chunks_deleted=0 is a legitimate no-op (unknown / already-gone file_id).
    log.info(
        "deleted %d chunk(s) for file_id=%s",
        chunks_deleted,
        sanitize_for_log(file_id),
    )
    return DeleteResponse(file_id=file_id, chunks_deleted=chunks_deleted)
