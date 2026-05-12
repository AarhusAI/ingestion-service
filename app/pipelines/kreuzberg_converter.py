"""HTTP client wrapper around the goldziher/kreuzberg API-server container.

Lives next to the in-process Haystack converters (``TikaDocumentConverter``,
``PyPDFToDocument`` etc.) but talks to a separate sidecar service over HTTP —
analogous to how ``TikaDocumentConverter`` talks to the ``tika`` container.
Keeps the ingestion-service image small and crash-isolates extraction.

The sidecar runs the Litestar-based API server bundled in the
``goldziher/kreuzberg`` image (``serve -H 0.0.0.0 -p 8000``). The
``POST /extract`` endpoint accepts one or more files in a repeated
multipart field named ``files`` and returns a JSON **array** — one
object per uploaded file, each with at least
``{"content": str, "mime_type": str, "metadata": {...}, "tables": [...]}``.
Verified against ``goldziher/kreuzberg:4.0.7-core`` (May 2026).
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

import httpx
from haystack import Document, component

log = logging.getLogger(__name__)


# Python's stdlib ``mimetypes`` only knows the half-dozen types in its own
# defaults table plus whatever ``/etc/mime.types`` ships on the host. The slim
# Python image used by this service is missing the OOXML and OpenDocument
# families, EPUB and RTF — exactly the long-tail formats Kreuzberg handles
# best. Register them here so ``mimetypes.guess_type`` returns the right
# Content-Type for the multipart upload (Kreuzberg dispatches on it).
_EXTRA_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".doc": "application/msword",
    ".xls": "application/vnd.ms-excel",
    ".ppt": "application/vnd.ms-powerpoint",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".epub": "application/epub+zip",
    ".rtf": "application/rtf",
}
for _ext, _ct in _EXTRA_TYPES.items():
    mimetypes.add_type(_ct, _ext)


@component
class KreuzbergRemoteConverter:
    """Posts each source file to the Kreuzberg ``/extract`` endpoint and turns
    the JSON response into a Haystack ``Document``.

    One Document per source — matches the Tika converter's output shape, so
    the rest of the pipeline (splitter → embedder → writer) is unchanged.
    Auto-extracted Kreuzberg metadata (title, author, page_count, …) is
    intentionally dropped for v1; the pipeline meta passed in by the caller
    is the source of truth for Qdrant payload.
    """

    def __init__(self, kreuzberg_url: str, timeout: float = 60.0):
        self._url = kreuzberg_url.rstrip("/") + "/extract"
        self._timeout = timeout

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str],
        meta: dict | list[dict] | None = None,
    ) -> dict:
        docs: list[Document] = []
        for i, source in enumerate(sources):
            path = Path(source)
            doc_meta = _meta_for(meta, i)
            # Kreuzberg dispatches on the multipart part's Content-Type rather
            # than sniffing the bytes. Sending ``application/octet-stream`` for
            # everything trips ``UnsupportedFormatError`` server-side, so guess
            # from the filename extension. Falls back to octet-stream only if
            # mimetypes can't decide — and Kreuzberg's resulting 500 will say
            # "Unsupported format" which is the right diagnostic.
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            try:
                with path.open("rb") as fh:
                    # Field name must be ``files`` — verified against the
                    # 4.0.x API server. ``file`` returns 400 "No files provided".
                    files = {"files": (path.name, fh, content_type)}
                    resp = httpx.post(self._url, files=files, timeout=self._timeout)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # Re-raised as plain RuntimeError so the route-layer heuristic
                # in ``_classify_pipeline_error`` catches "kreuzberg" / "extract"
                # and maps to EXTRACTION_FAILED.
                raise RuntimeError(f"kreuzberg extract failed for {path.name}: {exc}") from exc

            content = _content_from_payload(resp.json())
            docs.append(Document(content=content, meta=doc_meta))
        return {"documents": docs}


def _meta_for(meta: dict | list[dict] | None, i: int) -> dict:
    """Match Haystack convention: ``meta`` may be a single dict applied to all
    sources, a per-source list, or omitted entirely."""
    if meta is None:
        return {}
    if isinstance(meta, list):
        return dict(meta[i]) if i < len(meta) else {}
    return dict(meta)


def _content_from_payload(payload: object) -> str:
    """Extract the text from a Kreuzberg ``/extract`` response.

    The 4.0.x API returns a JSON **array** — one ``ExtractionResult`` per
    uploaded file, each with the text at the top-level ``content`` key. We
    upload one source per request so we take the first element. Also
    accepts a bare object (defensive against older builds / shape drift)
    and an unexpected payload (empty string → downstream all-or-nothing
    teardown handles the failure).
    """
    if isinstance(payload, list):
        if not payload:
            return ""
        first = payload[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]
        return ""
    if isinstance(payload, dict):
        if isinstance(payload.get("content"), str):
            return payload["content"]
        inner = payload.get("result")
        if isinstance(inner, dict) and isinstance(inner.get("content"), str):
            return inner["content"]
    return ""
