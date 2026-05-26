"""Filename / S3-key extension hygiene.

Shared helper used by both the S3 fetcher (``app/services/s3.py``) and
the multipart ingest path (``app/routes/ingest.py``) to normalise the
``suffix=`` argument passed to ``tempfile.NamedTemporaryFile``.

Caller-supplied filenames carry arbitrary characters. Python's stdlib
``tempfile`` doesn't reject path separators in ``suffix`` (verified on
3.10 + 3.12 — see sec.md Finding 8), so an extension like ``.bar/baz``
flows through to ``os.path.join`` and surfaces as a ``FileNotFoundError``
whose message embeds the path. It isn't a traversal vuln (the random
middle component blocks landing in any attacker-chosen directory) but
it pollutes ``/tmp`` paths and error messages with unbounded garbage.

An allow-list keeps the suffix predictable and bounded.
"""

from __future__ import annotations

# Extensions the converters in this codebase actually dispatch on.
# Mirrors ``KreuzbergRemoteConverter._EXTRA_TYPES`` plus common plain-text /
# image cases. Lowercased for comparison; the returned suffix preserves
# the original casing so tempfile names look natural to a human grepping
# ``/tmp``.
_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        # Office — OOXML
        ".docx",
        ".xlsx",
        ".pptx",
        # Office — legacy
        ".doc",
        ".xls",
        ".ppt",
        # OpenDocument
        ".odt",
        ".ods",
        ".odp",
        # Plain text / markup
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".rtf",
        ".csv",
        ".tsv",
        ".html",
        ".htm",
        ".xml",
        ".json",
        # eBooks
        ".epub",
        # Images (OCR via Tika / Kreuzberg)
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".tif",
        ".gif",
        ".bmp",
        ".webp",
    }
)


def safe_suffix(name: str) -> str:
    """Return the extension of ``name`` iff it's in the allow-list, else ``""``.

    - Empty / no-dot input → ``""``.
    - Extension on the allow-list → ``"." + ext`` preserving original case.
    - Anything else (multi-segment, separators, control chars, unknown
      type) → ``""``. The tempfile is still created — converters dispatch
      on content sniffing rather than relying on the suffix for these
      paths.
    """
    if not name or "." not in name:
        return ""
    raw = "." + name.rsplit(".", 1)[-1]
    if raw.lower() in _ALLOWED_EXTENSIONS:
        return raw
    return ""
