"""Embedder factories.

Decouple the pipeline from any single embedding model. Dense embedder is
required; sparse embedder is optional and gated by
``ENABLE_SPARSE_EMBEDDINGS``.
"""

from app.config import Settings


def _meta_fields_to_embed(settings: Settings) -> list[str]:
    """Meta fields the embedders prepend to the text they encode.

    The breadcrumb rides along as ``prefix + breadcrumb + separator +
    content`` at embed time only — stored chunk content is untouched, so
    nothing leaks into the answer context the retrieval agent builds.
    Chunks without the field (non-markdown modes, content outside any
    heading) are embedded unchanged.
    """
    if settings.embed_headers_breadcrumb:
        return ["headers_breadcrumb"]
    return []


def build_dense_embedder(settings: Settings):
    """Required dense embedder. Returns a Haystack DocumentEmbedder component."""
    provider = settings.embedding_provider.lower()

    if provider in ("openai-compat", "tei"):
        # TEI exposes an OpenAI-compatible endpoint, so it routes through the
        # same component. The distinction is purely documentation.
        from haystack.components.embedders import OpenAIDocumentEmbedder
        from haystack.utils import Secret

        return OpenAIDocumentEmbedder(
            api_base_url=settings.embedding_api_base_url,
            api_key=Secret.from_token(settings.embedding_api_key),
            model=settings.embedding_model,
            prefix=settings.embedding_prefix_doc,
            meta_fields_to_embed=_meta_fields_to_embed(settings),
        )

    if provider == "fastembed":
        from haystack_integrations.components.embedders.fastembed import (
            FastembedDocumentEmbedder,
        )

        return FastembedDocumentEmbedder(
            model=settings.embedding_model,
            # Cap the ONNX thread pool so in-process embedding leaves cores for
            # the event loop; parallel=1 forbids fastembed's batch multiprocessing
            # (each forked worker would re-grab cores and defeat the thread cap).
            threads=settings.resolved_embedding_threads(),
            parallel=1,
            meta_fields_to_embed=_meta_fields_to_embed(settings),
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r} "
        "(supported: openai-compat | fastembed | tei)"
    )


def build_sparse_embedder(settings: Settings):
    """Optional sparse embedder. Returns ``None`` when disabled."""
    if not settings.enable_sparse_embeddings:
        return None

    provider = settings.sparse_embedding_provider.lower()
    if provider == "none":
        return None

    if provider == "fastembed":
        from haystack_integrations.components.embedders.fastembed import (
            FastembedSparseDocumentEmbedder,
        )

        # See build_dense_embedder for why threads/parallel are pinned: the
        # sparse BM42 model is the CPU bottleneck that otherwise saturates all
        # cores and makes /health time out during ingestion.
        return FastembedSparseDocumentEmbedder(
            model=settings.sparse_embedding_model,
            threads=settings.resolved_embedding_threads(),
            parallel=1,
            # Heading terms boost BM42 keyword matching too.
            meta_fields_to_embed=_meta_fields_to_embed(settings),
        )

    raise ValueError(
        f"Unknown SPARSE_EMBEDDING_PROVIDER={settings.sparse_embedding_provider!r} "
        "(supported: fastembed | none)"
    )
