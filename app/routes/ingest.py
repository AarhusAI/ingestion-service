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

from app.auth import verify_api_key
from app.models import IngestError, IngestRequestJSON, IngestResponse
from app.pipelines.indexing import run_indexing_pipeline
from app.services.s3 import fetch_object_to_tempfile

log = logging.getLogger(__name__)

router = APIRouter()


@router.put(
    "/api/v1/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": IngestError},
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
        chunks = _run_pipeline_with_error_mapping(local_path, meta)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(local_path)

    return IngestResponse(
        collection_name=meta["collection_name"],
        chunks_count=chunks,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_from_s3(bucket: str, key: str) -> str:
    try:
        return fetch_object_to_tempfile(bucket=bucket, key=key)
    except Exception as exc:
        log.exception("S3 fetch failed for s3://%s/%s", bucket, key)
        raise HTTPException(
            status_code=500,
            detail=IngestError(error=str(exc), code="S3_FETCH_FAILED").model_dump(),
        ) from exc


async def stream_upload_to_tempfile(upload) -> str:
    """Stream a Starlette ``UploadFile`` to a NamedTemporaryFile.

    Preserves the original file extension (some converters dispatch on it).
    The caller owns the returned path and must ``os.unlink`` it. Assumes
    ``upload`` is non-None — presence checks live in the route handlers so
    this helper stays HTTP-agnostic and reusable across routes (currently
    ``/api/v1/ingest`` and ``/api/v1/extract``).
    """
    suffix = ""
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.rsplit(".", 1)[-1]

    # delete=False is intentional — caller (route handler) owns the file's lifetime
    # and unlinks it after the pipeline runs. Using a `with` block here would close
    # and delete the file before the pipeline can read it.
    fh = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # noqa: SIM115
    try:
        while chunk := await upload.read(1024 * 1024):
            fh.write(chunk)
        fh.flush()
    finally:
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
                        f"collection_name {collection_name!r} does not match "
                        f"file_id={file_id!r}"
                    ),
                    code="INVALID_REQUEST",
                ).model_dump(),
            )


def _run_pipeline_with_error_mapping(file_path: str, meta: dict[str, Any]) -> int:
    """Map exception types from the pipeline to ingestion error codes."""
    try:
        return run_indexing_pipeline(file_path, meta)
    except HTTPException:
        raise
    except Exception as exc:
        code = _classify_pipeline_error(exc)
        log.exception("pipeline failure (%s) for file_id=%s", code, meta.get("file_id"))
        raise HTTPException(
            status_code=500,
            detail=IngestError(error=str(exc), code=code).model_dump(),
        ) from exc


def _classify_pipeline_error(exc: Exception) -> str:
    """Best-effort classification. Falls back to PIPELINE_FAILED."""
    msg = str(exc).lower()
    name = type(exc).__name__
    if "tika" in msg or "extract" in msg or "converter" in msg or "PyPDFError" in name:
        return "EXTRACTION_FAILED"
    if "qdrant" in msg or "vector store" in msg or ("vector" in msg and "write" in msg):
        return "QDRANT_WRITE_FAILED"
    if "embed" in msg and "sparse" in msg:
        return "SPARSE_EMBEDDING_FAILED"
    if "embed" in msg or "OpenAI" in name:
        return "EMBEDDING_FAILED"
    return "PIPELINE_FAILED"
