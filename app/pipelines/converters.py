"""Document converter factory.

Selects a Haystack converter based on ``EXTRACTION_ENGINE``. ``tika`` and
``pypdf`` ship in the day-one image. ``docling`` and ``unstructured`` are
wired but their (heavy) deps are deliberately not in pyproject.toml — they
will raise a clear ``ImportError`` at startup if selected without the dep
installed.
"""

from app.config import Settings


def build_converter(settings: Settings, engine_override: str | None = None):
    """Build the configured converter, or one identified by ``engine_override``.

    ``engine_override`` is used by ``POST /api/v1/extract`` so a caller can
    compare engines on the same file without restarting the container.
    """
    engine = (engine_override or settings.extraction_engine).lower()

    if engine == "tika":
        from haystack.components.converters import TikaDocumentConverter

        return TikaDocumentConverter(tika_url=settings.tika_url)

    if engine == "pypdf":
        from haystack.components.converters import PyPDFToDocument

        return PyPDFToDocument()

    if engine == "docling":
        # Optional dep: pip install docling-haystack
        try:
            from docling_haystack.converter import DoclingConverter
        except ImportError as exc:
            raise ImportError(
                "EXTRACTION_ENGINE=docling requires the 'docling-haystack' package. "
                "Add it to pyproject.toml dependencies and rebuild the image."
            ) from exc
        return DoclingConverter()

    if engine == "unstructured":
        # Optional dep: pip install unstructured-fileconverter-haystack
        try:
            from haystack_integrations.components.converters.unstructured import (
                UnstructuredFileConverter,
            )
        except ImportError as exc:
            raise ImportError(
                "EXTRACTION_ENGINE=unstructured requires the "
                "'unstructured-fileconverter-haystack' package. "
                "Add it to pyproject.toml dependencies and rebuild the image."
            ) from exc
        return UnstructuredFileConverter()

    raise ValueError(
        f"Unknown EXTRACTION_ENGINE={engine!r} (supported: tika | pypdf | docling | unstructured)"
    )
