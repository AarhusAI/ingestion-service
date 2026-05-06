"""Factory selection: builders pick the right component for each config combo."""

import pytest

from app.config import Settings
from app.pipelines.converters import build_converter
from app.pipelines.embedders import build_dense_embedder, build_sparse_embedder


def _settings(**overrides) -> Settings:
    base = {
        "api_key": "k",
        "embedding_api_base_url": "http://fake:8080",
        "embedding_api_key": "k",
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_dim": 1024,
    }
    base.update(overrides)
    return Settings(**base)


# -------------------- Converters --------------------


def test_build_converter_tika():
    s = _settings(extraction_engine="tika")
    c = build_converter(s)
    assert type(c).__name__ == "TikaDocumentConverter"


def test_build_converter_pypdf():
    s = _settings(extraction_engine="pypdf")
    c = build_converter(s)
    assert type(c).__name__ == "PyPDFToDocument"


def test_build_converter_unknown():
    s = _settings(extraction_engine="banana")
    with pytest.raises(ValueError, match="Unknown EXTRACTION_ENGINE"):
        build_converter(s)


# -------------------- Dense embedders --------------------


def test_build_dense_openai_compat():
    s = _settings(embedding_provider="openai-compat")
    e = build_dense_embedder(s)
    assert type(e).__name__ == "OpenAIDocumentEmbedder"


def test_build_dense_tei_routes_to_openai_compat():
    s = _settings(embedding_provider="tei")
    e = build_dense_embedder(s)
    # TEI exposes an OpenAI-compatible endpoint, so we route through the same component.
    assert type(e).__name__ == "OpenAIDocumentEmbedder"


def test_build_dense_unknown():
    s = _settings(embedding_provider="banana")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        build_dense_embedder(s)


# -------------------- Sparse embedders --------------------


def test_build_sparse_disabled_returns_none():
    s = _settings(enable_sparse_embeddings=False)
    assert build_sparse_embedder(s) is None


def test_build_sparse_provider_none_returns_none():
    s = _settings(enable_sparse_embeddings=True, sparse_embedding_provider="none")
    assert build_sparse_embedder(s) is None


def test_build_sparse_unknown_provider():
    s = _settings(enable_sparse_embeddings=True, sparse_embedding_provider="banana")
    with pytest.raises(ValueError, match="Unknown SPARSE_EMBEDDING_PROVIDER"):
        build_sparse_embedder(s)
