"""Embedder factories.

Decouple the pipeline from any single embedding model. Dense embedder is
required; sparse embedder is optional and gated by
``ENABLE_SPARSE_EMBEDDINGS``.

Also holds :class:`DenseEmbeddingGuard`, the last-hop check that no chunk
reaches Qdrant without a dense vector.
"""

import httpx
from haystack import Document, component

from app.config import Settings
from app.pipelines.errors import EmbeddingError


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
            # Passed explicitly because Haystack's fallback (timeout=30.0,
            # max_retries=5) is a library default, not a decision: it absorbed
            # only ~13s of downtime, so a ~2min blip on the embedding endpoint
            # turned into failed uploads for the user.
            #
            # A Timeout object is what buys the retries: the type hint says
            # float, but the value goes straight to OpenAI(...), which accepts
            # float | Timeout | None — and only the object form can fail fast on
            # connect while still waiting on read. Nothing serializes this
            # pipeline, so to_dict()'s inability to encode it is moot.
            timeout=httpx.Timeout(
                connect=settings.embedding_connect_timeout,
                read=settings.embedding_read_timeout,
                write=settings.embedding_connect_timeout,
                pool=settings.embedding_connect_timeout,
            ),
            max_retries=settings.embedding_max_retries,
            # Haystack defaults this to False, which logs a failed batch and
            # carries on — the documents come back with embedding=None and
            # Qdrant happily stores them with only their sparse vector, so the
            # run "succeeds" while those chunks are invisible to vector search
            # forever (and the blue/green sweep then deletes the good version).
            # Fail instead: _delete_ingest_version() tears down the partial
            # write and the previously indexed version stays live.
            raise_on_failure=True,
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


@component
class DenseEmbeddingGuard:
    """Fail the run if any chunk reached the writer without a dense embedding.

    The invariant this enforces is *"no point reaches Qdrant without a dense
    vector"*, and it is enforced here because Qdrant will not enforce it:
    ``convert_haystack_documents_to_qdrant_points`` omits the dense key when
    ``embedding is None`` (rather than erroring), so with sparse embeddings on,
    a point lands carrying only its sparse vector. Those chunks are invisible
    to dense and hybrid retrieval permanently, while the ingest reports success
    and the blue/green sweep deletes the previously good version.

    ``raise_on_failure=True`` on the OpenAI-compatible embedder covers the path
    that caused this in practice, but only that one: it catches ``APIError``
    from one provider. This guard is provider-agnostic — it also covers the
    in-process fastembed dense path and any embedder added later that drops
    documents instead of raising.
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        missing = [doc for doc in documents if getattr(doc, "embedding", None) is None]
        if missing:
            # split_id locates the chunks in the source document; chunk text is
            # deliberately left out of the message (it reaches logs and, with
            # DEBUG on, the API response).
            sample = [doc.meta.get("split_id") for doc in missing[:5]]
            raise EmbeddingError(
                f"{len(missing)} of {len(documents)} chunks have no dense embedding "
                f"(split_id sample: {sample}); refusing to write vectorless points"
            )
        return {"documents": documents}


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
