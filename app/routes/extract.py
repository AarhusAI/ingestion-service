"""``POST /api/v1/extract`` endpoint.

Extraction-only sibling of ``/api/v1/ingest``. Runs the configured (or
per-request overridden) Haystack converter against an uploaded file and
returns the raw ``List[Document]`` output as JSON. No chunking, no
embedding, no Qdrant writes — intended for developers inspecting how a
given engine sees a document before committing to a full ingest.

Multipart-only by design: this is a curl-driven debugging tool, JSON/S3
mode would add no value here. Form fields are declared as typed ``File()``
/ ``Form()`` parameters so the endpoint is usable interactively from
``/docs`` (Swagger UI) — without typed params, FastAPI can't publish a
multipart request schema and the docs page has nothing to render.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.auth import verify_api_key
from app.config import settings as global_settings
from app.models import ExtractedDocument, ExtractResponse, IngestError
from app.pipelines.converters import build_converter
from app.pipelines.vision_profiles import KNOWN_PROFILES
from app.routes.ingest import _safe_error_detail, stream_upload_to_tempfile

log = logging.getLogger(__name__)

router = APIRouter()

_SUPPORTED_ENGINES = {
    "tika",
    "pypdf",
    "docling",
    "unstructured",
    "kreuzberg",
    "vision-llm",
    "hybrid-diagram",
}


@router.post(
    "/api/v1/extract",
    response_model=ExtractResponse,
    responses={
        400: {"model": IngestError},
        500: {"model": IngestError},
    },
)
async def extract(
    file: UploadFile | None = File(
        None,
        description="Document to extract (PDF, DOCX, plain text, etc.).",
    ),
    engine: str | None = Form(
        None,
        description=(
            "Optional override of EXTRACTION_ENGINE for this request. "
            "One of: tika, pypdf, docling, unstructured, kreuzberg, vision-llm, "
            "hybrid-diagram (diagram .docx: native text + vision Mermaid). "
            "When omitted, the configured engine is used. ('auto' is a routing "
            "mode for ingest, not a concrete engine — not accepted here.)"
        ),
        examples=["pypdf"],
    ),
    profile: str | None = Form(
        None,
        description=(
            "Optional vision-llm prompt profile for this request. One of: "
            f"{', '.join(sorted(KNOWN_PROFILES))}. Only valid with the vision-llm engine; "
            "when omitted, the configured VISION_LLM_PROFILE default is used."
        ),
        examples=["ocr"],
    ),
    _api_key: str = Depends(verify_api_key),
):
    # ``file``/``engine``/``profile`` are typed as Optional so FastAPI binds whatever is
    # in the form (even when fields are missing) and the existing 400 INVALID_REQUEST
    # contract is preserved — we'd lose it if we let FastAPI auto-422.
    engine = _validate_engine(engine)
    profile = _validate_profile(profile)
    if file is None:
        raise HTTPException(
            status_code=400,
            detail=IngestError(
                error="multipart request missing 'file' field",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    local_path = await stream_upload_to_tempfile(file)

    try:
        # The converter call is synchronous and can be slow (sidecar HTTP,
        # vision LLM) — offload so the event loop keeps serving other requests.
        documents = await asyncio.to_thread(_run_converter, local_path, engine, profile)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(local_path)

    # Reflect the profile actually used (the converter records it on each doc);
    # fall back to the requested value for empty extractions.
    used_profile = (documents[0].meta or {}).get("vision_profile") if documents else None
    return ExtractResponse(
        engine=engine or global_settings.extraction_engine,
        profile=used_profile or profile,
        documents=[
            ExtractedDocument(content=d.content or "", meta=d.meta or {}) for d in documents
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_engine(raw) -> str | None:
    """Return a normalized engine name, ``None`` (= use configured default), or 400."""
    if raw is None or raw == "":
        return None
    value = str(raw).lower()
    if value not in _SUPPORTED_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=IngestError(
                error=f"Unknown engine={raw!r} "
                f"(supported: {' | '.join(sorted(_SUPPORTED_ENGINES))})",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    return value


def _validate_profile(raw) -> str | None:
    """Return a normalized profile name, ``None`` (= engine default), or 400."""
    if raw is None or raw == "":
        return None
    value = str(raw).lower()
    if value not in KNOWN_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=IngestError(
                error=f"Unknown profile={raw!r} "
                f"(supported: {' | '.join(sorted(KNOWN_PROFILES))})",
                code="INVALID_REQUEST",
            ).model_dump(),
        )
    return value


def _run_converter(file_path: str, engine: str | None, profile: str | None = None) -> list:
    """Build the converter and call ``.run`` standalone (no pipeline).

    Mirrors how ``app/pipelines/indexing.py`` invokes the converter via the
    pipeline (``{"converter": {"sources": [...], "meta": {...}}}``) so the
    converters that work in the ingest path also work here. Note: the
    ``unstructured`` converter uses ``paths=`` rather than ``sources=`` — it
    has never worked in this codebase's ingest pipeline either; failures
    surface here as ``EXTRACTION_FAILED``.

    ``profile`` is forwarded only to profile-aware converters (vision-llm); a
    profile supplied for any other engine is a 400 rather than a silent no-op.
    """
    try:
        converter = build_converter(global_settings, engine_override=engine)
        run_kwargs: dict = {"sources": [file_path], "meta": {}}
        if profile is not None:
            if not getattr(converter, "accepts_profile", False):
                raise HTTPException(
                    status_code=400,
                    detail=IngestError(
                        error="profile is only valid with the vision-llm engine",
                        code="INVALID_REQUEST",
                    ).model_dump(),
                )
            run_kwargs["profile"] = profile
        result = converter.run(**run_kwargs)
        return result["documents"]
    except ImportError as exc:
        # Optional dep not installed (docling-haystack / unstructured-fileconverter-haystack).
        # The ImportError message names the missing package — that's user-actionable
        # configuration info, not an internal-state leak, so reflect it as-is.
        raise HTTPException(
            status_code=400,
            detail=IngestError(error=str(exc), code="INVALID_REQUEST").model_dump(),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception(
            "extraction failed (engine=%s)",
            engine or global_settings.extraction_engine,
        )
        raise HTTPException(
            status_code=500,
            detail=_safe_error_detail("EXTRACTION_FAILED", exc),
        ) from exc
