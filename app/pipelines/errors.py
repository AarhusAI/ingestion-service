"""Typed pipeline exceptions.

Raised by components we control (the converters, ``DenseEmbeddingGuard``)
so the route layer's ``_classify_pipeline_error`` can dispatch on
``isinstance`` instead of substring-matching the exception message —
substring matching is fragile (the previous classifier's ``"PyPDFError"
in name`` and ``"OpenAI" in name`` branches never matched real exception
class names) and pollutes the error code when a filename happens to
contain a trigger word.

For library exceptions we can't replace at the source (pypdf, openai,
qdrant_client), the classifier dispatches via ``isinstance`` against the
real classes in ``app/routes/ingest.py``.
"""

from __future__ import annotations


class ExtractionError(Exception):
    """Extraction stage failed (Kreuzberg sidecar, pypdf, vision-llm, etc.)."""


class EmbeddingError(Exception):
    """Dense embedder stage failed (OpenAI-compat endpoint, TEI, fastembed)."""
