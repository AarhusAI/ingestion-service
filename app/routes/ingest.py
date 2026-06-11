"""``PUT /api/v1/ingest`` endpoint.

Single handler that dispatches on Content-Type:
- ``application/json`` → S3-reference mode (preferred); body validated via
  :class:`IngestRequestJSON`.
- ``multipart/form-data`` → multipart fallback; file streamed to a tempfile.

Both modes converge on :func:`run_indexing_pipeline`. The handler owns the
local-file lifecycle (always ``os.unlink`` in ``finally``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from app import metrics
from app.auth import verify_api_key
from app.config import settings
from app.log_utils import sanitize_for_log
from app.models import ExtractionInfo, IngestError, IngestRequestJSON, IngestResponse
from app.pipelines.indexing import run_indexing_pipeline
from app.services.filenames import safe_suffix
from app.services.s3 import S3ObjectTooLarge, fetch_object_to_tempfile

log = logging.getLogger(__name__)

router = APIRouter()


@router.put(
    "/api/v1/ingest",
    response_model=IngestResponse,
    # extraction is None for pinned engines / mocked tests → omit it so the
    # historical {status, collection_name, chunks_count} body is unchanged.
    response_model_exclude_none=True,
    responses={
        400: {"model": IngestError},
        413: {"model": IngestError},
        415: {"model": IngestError},
        500: {"model": IngestError},
    },
)
async def ingest(
    request: Request,
    _api_key: str = Depends(verify_api_key),
):
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        try:
            body = IngestRequestJSON.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=IngestError(error=str(exc), code="INVALID_REQUEST").model_dump(),
            ) from exc
        _check_bucket_allowed(body.s3_bucket)
        local_path = _fetch_from_s3(body.s3_bucket, body.s3_key)
        meta = _meta_from_request(body)
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        local_path, meta = await _read_multipart(form)
    else:
        raise HTTPException(
            status_code=415,
            detail=IngestError(
                error=f"Unsupported Content-Type: {content_type!r}; "
                "use application/json or multipart/form-data",
                code="INVALID_REQUEST",
            ).model_dump(),
        )

    try:
        _validate_collection_binding(meta)
        metrics.ingest_document_bytes.observe(_safe_size(local_path))
        chunks, extraction = _run_pipeline_with_error_mapping(local_path, meta)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(local_path)

    metrics.ingest_requests_total.labels(outcome="success", code="none").inc()
    return IngestResponse(
        collection_name=meta["collection_name"],
        chunks_count=chunks,
        extraction=ExtractionInfo(**extraction) if extraction else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_bucket_allowed(bucket: str) -> None:
    """Refuse fetches against buckets outside ``S3_ALLOWED_BUCKETS``.

    Defense in depth: Open WebUI today derives ``s3_bucket`` server-side from
    ``file.path``, so end users can't directly choose what to read. A stolen
    API key or a future caller-side regression would re-open the path. The
    allow-list closes it at the service boundary.

    Empty allow-list = no enforcement (preserves behaviour for deployments
    that haven't configured the setting yet) — startup logs a warning so the
    gap is visible.
    """
    allowed = settings.allowed_buckets
    if not allowed:
        return
    if bucket not in allowed:
        raise HTTPException(
            status_code=403,
            detail=IngestError(
                error=f"S3 bucket {bucket!r} is not in S3_ALLOWED_BUCKETS",
                code="INVALID_REQUEST",
            ).model_dump(),
        )


def _fetch_from_s3(bucket: str, key: str) -> str:
    try:
        return fetch_object_to_tempfile(bucket=bucket, key=key)
    except S3ObjectTooLarge as exc:
        # Our own deterministic message — safe to reflect, doesn't leak internals.
        log.warning(
            "S3 fetch rejected (too large) for s3://%s/%s: %s",
            sanitize_for_log(bucket),
            sanitize_for_log(key),
            exc,
        )
        raise HTTPException(
            status_code=413,
            detail=IngestError(error=str(exc), code="INVALID_REQUEST").model_dump(),
        ) from exc
    except Exception as exc:
        log.exception(
            "S3 fetch failed for s3://%s/%s",
            sanitize_for_log(bucket),
            sanitize_for_log(key),
        )
        raise HTTPException(
            status_code=500,
            detail=_safe_error_detail("S3_FETCH_FAILED", exc),
        ) from exc


async def stream_upload_to_tempfile(upload) -> str:
    """Stream a Starlette ``UploadFile`` to a NamedTemporaryFile.

    Preserves the original file extension (some converters dispatch on it).
    The caller owns the returned path and must ``os.unlink`` it. Assumes
    ``upload`` is non-None — presence checks live in the route handlers so
    this helper stays HTTP-agnostic and reusable across routes (currently
    ``/api/v1/ingest`` and ``/api/v1/extract``).

    Enforces ``settings.max_upload_bytes`` — Starlette's ``max_part_size``
    only caps non-file form fields, so file uploads are otherwise unbounded
    and a single request can fill ``/tmp``.
    """
    # delete=False is intentional — caller (route handler) owns the file's lifetime
    # and unlinks it after the pipeline runs. Using a `with` block here would close
    # and delete the file before the pipeline can read it.
    fh = tempfile.NamedTemporaryFile(  # noqa: SIM115
        delete=False, suffix=safe_suffix(upload.filename or "")
    )
    written = 0
    max_bytes = settings.max_upload_bytes
    try:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                # Unlink the partial file before raising — finally only closes.
                fh.close()
                with contextlib.suppress(OSError):
                    os.unlink(fh.name)
                raise HTTPException(
                    status_code=413,
                    detail=IngestError(
                        error=(
                            f"upload exceeds max_upload_bytes={max_bytes} "
                            "(configure via MAX_UPLOAD_BYTES)"
                        ),
                        code="INVALID_REQUEST",
                    ).model_dump(),
                )
            fh.write(chunk)
        fh.flush()
    finally:
        if not fh.closed:
            fh.close()
    return fh.name


async def _read_multipart(form) -> tuple[str, dict[str, Any]]:
    upload = form.get("file")
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail=IngestError(
                error="multipart request missing 'file' field",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    path = await stream_upload_to_tempfile(upload)

    meta = {
        "file_id": _required_form(form, "file_id"),
        "filename": _required_form(form, "filename"),
        "collection_name": _required_form(form, "collection_name"),
        "collection_type": form.get("collection_type", "file"),
        "user_id": _required_form(form, "user_id"),
        "overwrite": _form_bool(form.get("overwrite", "true")),
        "name": _required_form(form, "filename"),
        "source": _required_form(form, "filename"),
    }
    return path, meta


def _meta_from_request(body: IngestRequestJSON) -> dict[str, Any]:
    return {
        "file_id": body.file_id,
        "filename": body.filename,
        "collection_name": body.collection_name,
        "collection_type": body.collection_type,
        "user_id": body.user_id,
        "overwrite": body.overwrite,
        "name": body.filename,
        "source": body.filename,
    }


def _required_form(form, key: str) -> str:
    value = form.get(key)
    if not value:
        raise HTTPException(
            status_code=400,
            detail=IngestError(
                error=f"multipart request missing required field {key!r}",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    return str(value)


def _form_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "y")


def _validate_collection_binding(meta: dict[str, Any]) -> None:
    """Defense-in-depth: refuse caller-supplied collection_name values that the
    caller demonstrably doesn't own.

    Open WebUI is supposed to gate this on the write path, but a single missing
    check there would let any authenticated user poison another tenant's
    collection (the ``_validate_collection_access`` helper in Open WebUI is
    only wired into the query routes). Re-asserting the binding here means a
    future regression in Open WebUI doesn't silently re-open the hole.

    Enforces two patterns the service can verify locally:
    - ``user-memory-{uid}`` must match ``meta.user_id``.
    - ``file-{fid}`` must match ``meta.file_id``.

    Other collection types (``knowledge``, ``web-search``, ``hash-based``)
    can't be verified without an Open WebUI API call and are passed through.
    """
    collection_name = meta.get("collection_name", "")
    user_id = meta.get("user_id", "")
    file_id = meta.get("file_id", "")

    if collection_name.startswith("user-memory-"):
        expected = f"user-memory-{user_id}"
        if collection_name != expected:
            raise HTTPException(
                status_code=403,
                detail=IngestError(
                    error=(
                        f"collection_name {collection_name!r} not authorized "
                        f"for user_id={user_id!r}"
                    ),
                    code="INVALID_REQUEST",
                ).model_dump(),
            )
    elif collection_name.startswith("file-"):
        expected = f"file-{file_id}"
        if collection_name != expected:
            raise HTTPException(
                status_code=403,
                detail=IngestError(
                    error=(
                        f"collection_name {collection_name!r} does not match file_id={file_id!r}"
                    ),
                    code="INVALID_REQUEST",
                ).model_dump(),
            )


def _safe_size(path: str) -> int:
    """Best-effort byte size of the local file for the size histogram."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _run_pipeline_with_error_mapping(
    file_path: str, meta: dict[str, Any]
) -> tuple[int, dict | None]:
    """Run the pipeline, mapping exceptions to ingestion error codes.

    Returns ``(chunks_count, extraction)`` from :func:`run_indexing_pipeline`.
    """
    try:
        return run_indexing_pipeline(file_path, meta)
    except HTTPException:
        raise
    except Exception as exc:
        code = _classify_pipeline_error(exc)
        metrics.ingest_requests_total.labels(outcome="error", code=code).inc()
        log.exception(
            "pipeline failure (%s) for file_id=%s",
            code,
            sanitize_for_log(meta.get("file_id")),
        )
        raise HTTPException(
            status_code=500,
            detail=_safe_error_detail(code, exc),
        ) from exc


# Generic per-code messages used when DEBUG is off. Mirrors the operational
# meaning of each ErrorCode in app/models.py — vague enough that we don't
# leak internal hostnames / paths / tokenizer model names back to clients,
# specific enough that the caller can decide whether to retry vs. surface
# the failure to the end user. The full exception is still in our logs via
# log.exception at every call site.
_SAFE_ERROR_MESSAGES: dict[str, str] = {
    "EXTRACTION_FAILED": "Document extraction failed.",
    "EMBEDDING_FAILED": "Embedding step failed.",
    "SPARSE_EMBEDDING_FAILED": "Sparse embedding step failed.",
    "QDRANT_WRITE_FAILED": "Vector store write failed.",
    "S3_FETCH_FAILED": "S3 object fetch failed.",
    "INVALID_REQUEST": "Invalid request.",
    "PIPELINE_FAILED": "Indexing pipeline failed.",
}


def _safe_error_detail(code: str, exc: Exception) -> dict:
    """Build the JSON body of an ``IngestError`` response without leaking
    internal details unless ``DEBUG`` is on.

    Production callers see a fixed per-code string (no hostnames, no paths,
    no AWS request IDs). Operators flipping ``DEBUG=true`` get ``str(exc)``
    for local triage. Logs always carry the full traceback regardless.
    """
    message = str(exc) if settings.debug else _SAFE_ERROR_MESSAGES.get(code, "Internal error.")
    return IngestError(error=message, code=code).model_dump()


def _classify_pipeline_error(exc: Exception) -> str:
    """Map a pipeline exception to an ``IngestError.code``.

    Dispatch order:
    1. Our own typed errors (``ExtractionError`` etc. from
       ``app.pipelines.errors``) — preferred, used by components we control.
    2. Known third-party exception classes via lazy ``isinstance`` checks —
       covers pypdf, openai, and qdrant_client failures accurately.
    3. Substring matching on the message — best-effort fallback for
       Haystack-internal failures that surface as generic ``Exception``.

    The previous implementation matched ``"PyPDFError" in name`` and
    ``"OpenAI" in name``, neither of which corresponds to a real exception
    class (pypdf uses ``PdfReadError`` / ``PyPdfError`` base; openai 1.x
    uses ``APIError`` and subclasses). Both branches silently fell through
    to ``PIPELINE_FAILED``, mis-labeling the two most common real-world
    failure modes (PDF parse errors, embedding-endpoint outages). See
    sec.md Finding 5.
    """
    from app.pipelines.errors import (
        EmbeddingError,
        ExtractionError,
        QdrantWriteError,
        SparseEmbeddingError,
    )

    # 1. Typed errors from components we control.
    if isinstance(exc, ExtractionError):
        return "EXTRACTION_FAILED"
    if isinstance(exc, SparseEmbeddingError):
        return "SPARSE_EMBEDDING_FAILED"
    if isinstance(exc, EmbeddingError):
        return "EMBEDDING_FAILED"
    if isinstance(exc, QdrantWriteError):
        return "QDRANT_WRITE_FAILED"

    # 2. Known library exception types (lazy-imported so a missing optional
    #    dep can't break the classifier).
    if _is_pypdf_error(exc):
        return "EXTRACTION_FAILED"
    if _is_qdrant_error(exc):
        return "QDRANT_WRITE_FAILED"
    if _is_openai_error(exc):
        return "EMBEDDING_FAILED"

    # 3. Substring fallback. Sparse must be checked before the generic
    #    embed branch, otherwise sparse failures get mis-labeled.
    msg = str(exc).lower()
    if "sparse" in msg and "embed" in msg:
        return "SPARSE_EMBEDDING_FAILED"
    if "qdrant" in msg or "vector store" in msg or ("vector" in msg and "write" in msg):
        return "QDRANT_WRITE_FAILED"
    if "embed" in msg:
        return "EMBEDDING_FAILED"
    if "tika" in msg or "extract" in msg or "converter" in msg:
        return "EXTRACTION_FAILED"
    return "PIPELINE_FAILED"


def _is_pypdf_error(exc: Exception) -> bool:
    try:
        from pypdf.errors import PyPdfError
    except ImportError:
        return False
    return isinstance(exc, PyPdfError)


def _is_openai_error(exc: Exception) -> bool:
    try:
        from openai import OpenAIError
    except ImportError:
        return False
    return isinstance(exc, OpenAIError)


def _is_qdrant_error(exc: Exception) -> bool:
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:
        return False
    return isinstance(exc, UnexpectedResponse)
