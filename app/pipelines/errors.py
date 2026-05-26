"""Typed pipeline exceptions.

Raised by components we control (currently ``KreuzbergRemoteConverter``)
so the route layer's ``_classify_pipeline_error`` can dispatch on
``isinstance`` instead of substring-matching the exception message —
substring matching is fragile (the previous classifier's ``"PyPDFError"
in name`` and ``"OpenAI" in name`` branches never matched real exception
class names) and pollutes the error code when a filename happens to
contain a trigger word.

For library exceptions we can't replace at the source (pypdf, openai,
qdrant_client), the classifier still dispatches via lazy ``isinstance``
checks against the real classes — see ``_is_pypdf_error`` etc. in
``app/routes/ingest.py``.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base for typed pipeline errors raised by components we control."""


class ExtractionError(IngestionError):
    """Extraction stage failed (Tika sidecar, pypdf, Kreuzberg, etc.)."""


class EmbeddingError(IngestionError):
    """Dense embedder stage failed (OpenAI-compat endpoint, TEI, fastembed)."""


class SparseEmbeddingError(IngestionError):
    """Sparse embedder stage failed (fastembed sparse model)."""


class QdrantWriteError(IngestionError):
    """Qdrant writer / document store call failed."""
