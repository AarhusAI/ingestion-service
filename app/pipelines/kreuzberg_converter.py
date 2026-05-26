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

from app.pipelines.errors import ExtractionError

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

    def __init__(
        self,
        kreuzberg_url: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        verify: bool = True,
    ):
        self._url = kreuzberg_url.rstrip("/") + "/extract"
        # Split timeout — connect fails fast so a stalled sidecar doesn't
        # tie up a worker for the full read window. Write/pool reuse the
        # connect value: nothing about the upload is read-shaped.
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        self._verify = verify

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str],
        meta: dict | list[dict] | None = None,
    ) -> dict:
        docs: list[Document] = []
        for i, source in enumerate(sources):
            path = Path(source)
            request_meta = _meta_for(meta, i)
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
                    resp = httpx.post(
                        self._url,
                        files=files,
                        timeout=self._timeout,
                        verify=self._verify,
                    )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # Typed so the route-layer classifier dispatches on isinstance
                # rather than substring-matching the message — substring matches
                # were fragile (the old classifier missed real pypdf/openai
                # class names entirely; see sec.md Finding 5).
                raise ExtractionError(f"kreuzberg extract failed for {path.name}: {exc}") from exc

            payload = resp.json()
            content = _content_from_payload(payload)
            doc_meta = _doc_meta_from_payload(payload)
            # Request meta is the contract with the route layer (file_id /
            # collection_name / user_id …) and must win on any collision.
            merged_meta = {**doc_meta, **request_meta}
            docs.append(Document(content=content, meta=merged_meta))
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
    """Build the chunk-source text from a Kreuzberg ``/extract`` response.

    The 4.0.x API returns a JSON **array** — one ``ExtractionResult`` per
    uploaded file. We upload one source per request, so we take the first
    element. From it we pull two fields:

    - ``content`` — the body text. In tabular PDFs, Kreuzberg flattens
      tables into this string without whitespace (``"Sales100120135"`` etc.),
      so the body alone is lossy for any document with tables.
    - ``tables`` — a separate array of structured tables, each with a
      pre-rendered ``markdown`` field and a ``page_number``. We append
      these as a ``## Tables`` Markdown section so they survive
      downstream chunking + embedding. With ``CHUNK_SPLIT_BY=markdown``
      each table lands in its own ``### Table N (page X)`` section.

    Defensive against shape drift: accepts a bare object as well as the
    canonical array, and returns the empty string if nothing usable is
    in the payload (downstream all-or-nothing teardown handles failure).
    """
    result = _first_result(payload)
    if result is None:
        return ""
    body = result.get("content") if isinstance(result.get("content"), str) else ""
    tables = result.get("tables") if isinstance(result.get("tables"), list) else []
    table_section = _render_tables_section(tables)
    if table_section:
        return f"{body}\n\n{table_section}" if body else table_section
    return body


def _first_result(payload: object) -> dict | None:
    """Pull the single ExtractionResult dict from a Kreuzberg response.

    The shipping shape is a one-element JSON array. The bare-object branch
    keeps us tolerant of older / variant builds that returned just the dict,
    matching the previous ``_content_from_payload`` fallback semantics.
    """
    if isinstance(payload, list):
        if not payload:
            return None
        first = payload[0]
        return first if isinstance(first, dict) else None
    if isinstance(payload, dict):
        if "content" in payload:
            return payload
        inner = payload.get("result")
        if isinstance(inner, dict):
            return inner
    return None


# Whitelist of document-level metadata fields we surface from Kreuzberg's
# response into Qdrant payload. Curated to retrieval-useful signals only —
# producer/pdf_version/dimensions etc. would just bloat the payload.
#
# ``languages`` is sourced from the top-level ``detected_languages`` (renamed
# to drop the "detected_" prefix that's now redundant in context); everything
# else is read from ``metadata.{key}``.
_DOC_META_FIELDS: tuple[str, ...] = ("title", "subject", "authors", "created_at")


def _doc_meta_from_payload(payload: object) -> dict:
    """Pick the retrieval-useful subset of document-level metadata.

    Empty / falsy values are dropped so chunks don't carry
    ``"subject": ""`` / ``"authors": []`` noise — the retrieval-agent
    preview whitelist also drops empty values, so emitting them here
    is just wasted bytes in Qdrant. Returns an empty dict for any
    payload that doesn't expose usable doc meta.
    """
    result = _first_result(payload)
    if result is None:
        return {}
    out: dict = {}
    inner = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    for key in _DOC_META_FIELDS:
        value = inner.get(key)
        if value in (None, "", [], {}):
            continue
        out[key] = value
    # ``detected_languages`` sits at the top level of the ExtractionResult,
    # not under ``metadata``. We surface it as ``languages`` so the Qdrant
    # payload key matches the keyword-index name in ``qdrant_setup.py``.
    langs = result.get("detected_languages")
    if isinstance(langs, list) and langs:
        out["languages"] = langs
    return out


def _render_tables_section(tables: list) -> str:
    """Render Kreuzberg's ``tables`` array as a Markdown section.

    Each table goes under a ``### Table N (page X)`` H3 inside a parent
    ``## Tables`` H2 so the structure-aware chunker (``CHUNK_SPLIT_BY=
    markdown``) gives each table its own chunk with a ``meta.headers``
    breadcrumb. Tables missing the pre-rendered ``markdown`` field are
    skipped (we don't render from ``cells`` ourselves — Kreuzberg already
    knows how, and falling back would risk shape drift). Numbering is
    monotonic across **successfully** rendered tables — a skipped table
    doesn't leave a gap in the visible count. Returns the empty string
    if nothing can be rendered, so we never emit an empty heading.
    """
    blocks: list[str] = []
    counter = 0
    for table in tables:
        if not isinstance(table, dict):
            continue
        md = table.get("markdown")
        if not isinstance(md, str) or not md.strip():
            continue
        counter += 1
        page = table.get("page_number")
        if isinstance(page, int) and not isinstance(page, bool):
            heading = f"### Table {counter} (page {page})"
        else:
            heading = f"### Table {counter}"
        blocks.append(f"{heading}\n\n{md.strip()}")
    if not blocks:
        return ""
    return "## Tables\n\n" + "\n\n".join(blocks)
