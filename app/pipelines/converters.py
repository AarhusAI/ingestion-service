"""Document converter factory.

Selects a Haystack converter based on ``EXTRACTION_ENGINE``. ``tika``,
``pypdf`` and ``kreuzberg`` ship in the day-one image — ``tika`` and
``kreuzberg`` both run as external HTTP sidecars (the others are
in-process). ``docling`` and ``unstructured`` are wired but their (heavy)
deps are deliberately not in pyproject.toml — they will raise a clear
``ImportError`` at startup if selected without the dep installed.
``vision-llm`` renders pages (office->PDF via the Gotenberg sidecar,
PDF->PNG locally) and reconstructs structure via a multimodal LLM.

Not built here: ``"auto"`` is a routing *mode*, not an engine — see
``app/pipelines/routing_converter.py``, wired in ``indexing.py``.
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

    if engine == "kreuzberg":
        # Custom HTTP wrapper around the goldziher/kreuzberg sidecar. Lives in
        # our own code (no third-party haystack integration) so the dep is
        # just httpx, which is already required.
        from app.pipelines.kreuzberg_converter import KreuzbergRemoteConverter

        return KreuzbergRemoteConverter(
            kreuzberg_url=settings.kreuzberg_url,
            connect_timeout=settings.kreuzberg_connect_timeout,
            read_timeout=settings.kreuzberg_read_timeout,
            verify=settings.kreuzberg_tls_verify,
            min_table_columns=settings.kreuzberg_min_table_columns,
        )

    if engine == "vision-llm":
        # Renders pages and reconstructs structure (flowcharts, diagrams) via a
        # multimodal LLM. Office->PDF rendering goes through the Gotenberg
        # sidecar; PDF->PNG is local (pypdfium2). Our own code — dep surface is
        # httpx (already required) + pypdfium2/pillow.
        from app.pipelines.vision_llm_converter import VisionLLMConverter

        return VisionLLMConverter(
            api_base_url=settings.vision_llm_api_base_url,
            api_key=settings.vision_llm_api_key,
            model=settings.vision_llm_model,
            connect_timeout=settings.vision_llm_connect_timeout,
            read_timeout=settings.vision_llm_read_timeout,
            dpi=settings.vision_llm_dpi,
            max_pages=settings.vision_llm_max_pages,
            tls_verify=settings.vision_llm_tls_verify,
            language_hint=settings.vision_llm_language_hint,
            default_profile=settings.vision_llm_profile,
            gotenberg_url=settings.gotenberg_url,
            gotenberg_connect_timeout=settings.gotenberg_connect_timeout,
            gotenberg_read_timeout=settings.gotenberg_read_timeout,
            gotenberg_tls_verify=settings.gotenberg_tls_verify,
        )

    if engine == "hybrid-diagram":
        # Native docx text (authoritative labels) + a vision-inferred Mermaid
        # diagram. Wraps the vision-llm engine, so it carries the same config /
        # dep surface and no extra env vars. Used as the auto-router's diagram
        # engine; also selectable directly for forced use.
        from app.pipelines.hybrid_diagram_converter import HybridDiagramConverter

        return HybridDiagramConverter(settings)

    raise ValueError(
        f"Unknown EXTRACTION_ENGINE={engine!r} "
        "(supported: tika | pypdf | docling | unstructured | kreuzberg | "
        "vision-llm | hybrid-diagram)"
    )
