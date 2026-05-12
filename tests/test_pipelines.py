"""Factory selection: builders pick the right component for each config combo."""

import pytest

from app.config import Settings
from app.pipelines.converters import build_converter
from app.pipelines.embedders import build_dense_embedder, build_sparse_embedder
from app.pipelines.splitter import build_splitter


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


def test_build_converter_override_takes_precedence():
    """``engine_override`` (used by /api/v1/extract) wins over the configured engine."""
    s = _settings(extraction_engine="tika")
    c = build_converter(s, engine_override="pypdf")
    assert type(c).__name__ == "PyPDFToDocument"


def test_build_converter_override_validates_unknown():
    s = _settings(extraction_engine="tika")
    with pytest.raises(ValueError, match="Unknown EXTRACTION_ENGINE='banana'"):
        build_converter(s, engine_override="banana")


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


# -------------------- Splitters --------------------


def test_build_splitter_word_returns_haystack_splitter():
    s = _settings(chunk_split_by="word")
    sp = build_splitter(s)
    assert type(sp).__name__ == "DocumentSplitter"


def test_build_splitter_token_requires_model():
    s = _settings(chunk_split_by="token", embedding_model="", tokenizer_model="")
    with pytest.raises(ValueError, match="TOKENIZER_MODEL or EMBEDDING_MODEL"):
        build_splitter(s)


def test_huggingface_tokenizer_splitter_run(monkeypatch):
    """``run()`` slices each doc, assigns ``split_id``, and skips empties.

    Covers the three branches in ``HuggingFaceTokenizerSplitter.run``:
    whitespace-only document skip, empty-piece skip, and the
    ``split_id`` numbering (which mirrors the enumerate index — a
    skipped empty piece leaves a gap, by design).
    """
    from haystack import Document

    from app.pipelines import splitter as splitter_mod

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m: object())

    class _FakeSplitter:
        @classmethod
        def from_huggingface_tokenizer(cls, *_args, **_kwargs):
            return cls()

        def split_text(self, text):
            # Three pieces; the empty middle exercises the skip branch.
            return [f"{text}::a", "", f"{text}::b"]

    monkeypatch.setattr(
        "langchain_text_splitters.RecursiveCharacterTextSplitter",
        _FakeSplitter,
    )

    sp = splitter_mod.HuggingFaceTokenizerSplitter(
        tokenizer_model="x", chunk_size=10, chunk_overlap=2
    )

    docs = [
        Document(content="hello", meta={"file_id": "f1"}),
        Document(content="   ", meta={"file_id": "f2"}),  # whitespace → skipped
        Document(content="", meta={"file_id": "f3"}),  # empty → skipped
    ]
    out = sp.run(documents=docs)["documents"]

    assert [d.content for d in out] == ["hello::a", "hello::b"]
    assert out[0].meta == {"file_id": "f1", "split_id": 0}
    # split_id 1 was the skipped empty piece — gap is intentional (the index
    # comes straight from enumerate, so downstream debugging can see the skip).
    assert out[1].meta == {"file_id": "f1", "split_id": 2}


def test_build_splitter_token_uses_hf_component(monkeypatch):
    # Avoid the real tokenizer download; the component only stores the
    # langchain splitter, so mock both layers.
    from app.pipelines import splitter as splitter_mod

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m: object())

    class _FakeSplitter:
        @classmethod
        def from_huggingface_tokenizer(cls, *_args, **_kwargs):
            return cls()

        def split_text(self, text):
            return [text]

    monkeypatch.setattr(
        "langchain_text_splitters.RecursiveCharacterTextSplitter",
        _FakeSplitter,
    )
    s = _settings(chunk_split_by="token", embedding_model="intfloat/multilingual-e5-large")
    sp = build_splitter(s)
    assert type(sp).__name__ == "HuggingFaceTokenizerSplitter"
