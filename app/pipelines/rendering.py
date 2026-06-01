"""Render a document's pages to PNG images for the vision-LLM converter.

Two stages, deliberately kept out of this service's container image:

- **office -> PDF** is delegated to a **Gotenberg** HTTP sidecar (headless
  LibreOffice behind a clean API), the same deployment model as the ``tika`` and
  ``kreuzberg`` sidecars. This keeps LibreOffice — and its cold start, user
  profile, and profile-lock concurrency traps — entirely out of this image.
- **PDF -> PNG** is done locally with **pypdfium2** (pip-only, ships its own
  pdfium, permissive-licensed). No ``apt`` packages, no ``subprocess``.

Every failure raises :class:`ExtractionError` so the route-layer classifier maps
it to ``EXTRACTION_FAILED`` (see ``app/routes/ingest.py``). Filenames appear in
error messages; secrets never do (there are none at this layer).
"""

from __future__ import annotations

import io
import logging
import mimetypes
from pathlib import Path

import httpx

from app.pipelines.errors import ExtractionError

log = logging.getLogger(__name__)


# Office formats LibreOffice (via Gotenberg) can turn into PDF. ``.pdf`` is
# handled separately (no conversion). Anything not in here is rejected before we
# bother the sidecar — the vision engine is only meaningful for paginated,
# renderable documents.
_OFFICE_EXTS = frozenset(
    {".docx", ".doc", ".odt", ".rtf", ".pptx", ".ppt", ".odp", ".xlsx", ".xls", ".ods"}
)

# The slim Python image's ``mimetypes`` table is missing the OOXML/OpenDocument
# families, so register them — Gotenberg dispatches partly on the part's
# Content-Type. Mirrors the registry in ``kreuzberg_converter.py``.
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
    ".rtf": "application/rtf",
}
for _ext, _ct in _EXTRA_TYPES.items():
    mimetypes.add_type(_ct, _ext)


def office_to_pdf(
    path: str,
    *,
    gotenberg_url: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 120.0,
    verify: bool = True,
) -> bytes:
    """Convert an office document to PDF bytes via the Gotenberg sidecar.

    POSTs the file to ``{gotenberg_url}/forms/libreoffice/convert`` as a
    multipart ``files`` part (mirrors ``KreuzbergRemoteConverter``'s upload).
    Split ``httpx.Timeout`` so a stalled sidecar fails fast on connect.
    """
    p = Path(path)
    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=connect_timeout,
        pool=connect_timeout,
    )
    url = gotenberg_url.rstrip("/") + "/forms/libreoffice/convert"
    content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    try:
        with p.open("rb") as fh:
            resp = httpx.post(
                url,
                files={"files": (p.name, fh, content_type)},
                timeout=timeout,
                verify=verify,
            )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ExtractionError(f"gotenberg convert failed for {p.name}: {exc}") from exc
    return resp.content


def pdf_to_pngs(pdf_bytes: bytes, *, dpi: int = 150, max_pages: int = 20) -> list[bytes]:
    """Rasterize the first ``max_pages`` pages of a PDF to PNG byte blobs.

    Pure/local via pypdfium2 — no network, trivially mockable. ``scale`` is
    ``dpi / 72`` because pdfium renders at 72 dpi when ``scale=1``.
    """
    import pypdfium2 as pdfium

    scale = dpi / 72.0
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except Exception as exc:  # pdfium raises its own error types on bad input
        raise ExtractionError(f"could not open PDF for rendering: {exc}") from exc

    pngs: list[bytes] = []
    try:
        page_count = min(len(pdf), max_pages)
        for i in range(page_count):
            page = pdf[i]
            try:
                pil_image = page.render(scale=scale).to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                pngs.append(buf.getvalue())
            finally:
                page.close()
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"PDF page render failed: {exc}") from exc
    finally:
        pdf.close()
    return pngs


def render_to_pngs(
    path: str,
    *,
    gotenberg_url: str,
    dpi: int = 150,
    max_pages: int = 20,
    connect_timeout: float = 5.0,
    read_timeout: float = 120.0,
    verify: bool = True,
) -> list[bytes]:
    """Render any supported document to a list of PNG byte blobs.

    ``.pdf`` is read straight off disk and rasterized; office formats go through
    Gotenberg first. Returns one PNG per rendered page (capped at ``max_pages``).
    Raises :class:`ExtractionError` for unsupported types or an empty render.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        try:
            pdf_bytes = Path(path).read_bytes()
        except OSError as exc:
            raise ExtractionError(f"could not read {Path(path).name}: {exc}") from exc
    elif ext in _OFFICE_EXTS:
        pdf_bytes = office_to_pdf(
            path,
            gotenberg_url=gotenberg_url,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            verify=verify,
        )
    else:
        raise ExtractionError(
            f"vision-llm rendering does not support {ext or 'this file'!r} "
            f"({Path(path).name})"
        )

    pngs = pdf_to_pngs(pdf_bytes, dpi=dpi, max_pages=max_pages)
    if not pngs:
        raise ExtractionError(f"rendering produced no pages for {Path(path).name}")
    return pngs
