"""Embedder factories.

Decouple the pipeline from any single embedding model. Dense embedder is
required; sparse embedder is optional and gated by
``ENABLE_SPARSE_EMBEDDINGS``.
"""

from app.config import Settings


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
        )

    if provider == "fastembed":
        from haystack_integrations.components.embedders.fastembed import (
            FastembedDocumentEmbedder,
        )

        return FastembedDocumentEmbedder(model=settings.embedding_model)

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

        return FastembedSparseDocumentEmbedder(model=settings.sparse_embedding_model)

    raise ValueError(
        f"Unknown SPARSE_EMBEDDING_PROVIDER={settings.sparse_embedding_provider!r} "
        "(supported: fastembed | none)"
    )
