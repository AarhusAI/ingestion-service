"""Factory selection: builders pick the right component for each config combo."""

import pytest

from app.config import Settings
from app.pipelines.converters import build_converter
from app.pipelines.embedders import build_dense_embedder, build_sparse_embedder
from app.pipelines.splitter import build_splitter


def _settings(**overrides) -> Settings:
    base = {
        "api_key": "test-api-key-padded-to-min-length-x",
        "embedding_api_base_url": "http://fake:8080",
        "embedding_api_key": "k",
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_dim": 1024,
    }
    base.update(overrides)
    return Settings(**base)


# -------------------- Converters --------------------


def test_build_converter_pypdf():
    s = _settings(extraction_engine="pypdf")
    c = build_converter(s)
    assert type(c).__name__ == "PyPDFToDocument"


def test_build_converter_kreuzberg():
    s = _settings(extraction_engine="kreuzberg")
    c = build_converter(s)
    assert type(c).__name__ == "KreuzbergRemoteConverter"


def test_build_converter_override_kreuzberg():
    """``engine_override=kreuzberg`` wins over the configured engine."""
    s = _settings(extraction_engine="pypdf")
    c = build_converter(s, engine_override="kreuzberg")
    assert type(c).__name__ == "KreuzbergRemoteConverter"


def test_build_converter_vision_llm():
    s = _settings(extraction_engine="vision-llm", vision_llm_api_base_url="http://vlm:8080/v1")
    c = build_converter(s)
    assert type(c).__name__ == "VisionLLMConverter"


def test_build_converter_vision_llm_threads_profile():
    s = _settings(
        extraction_engine="vision-llm",
        vision_llm_api_base_url="http://vlm:8080/v1",
        vision_llm_profile="ocr",
    )
    c = build_converter(s)
    assert c._default_profile == "ocr"


def test_build_converter_vision_llm_invalid_profile_raises():
    s = _settings(
        extraction_engine="vision-llm",
        vision_llm_api_base_url="http://vlm:8080/v1",
        vision_llm_profile="banana",
    )
    with pytest.raises(ValueError, match="not a known profile"):
        build_converter(s)


def test_build_converter_unknown():
    s = _settings(extraction_engine="banana")
    with pytest.raises(ValueError, match="Unknown EXTRACTION_ENGINE"):
        build_converter(s)


def test_build_converter_for_pipeline_auto_returns_router():
    from app.pipelines.indexing import _build_converter_for_pipeline

    s = _settings(extraction_engine="auto", vision_llm_api_base_url="http://vlm:8080/v1")
    c = _build_converter_for_pipeline(s)
    assert type(c).__name__ == "RoutingConverter"


def test_build_converter_for_pipeline_concrete_engine():
    from app.pipelines.indexing import _build_converter_for_pipeline

    s = _settings(extraction_engine="kreuzberg")
    c = _build_converter_for_pipeline(s)
    assert type(c).__name__ == "KreuzbergRemoteConverter"


def test_build_converter_override_takes_precedence():
    """``engine_override`` (used by /api/v1/extract) wins over the configured engine."""
    s = _settings(extraction_engine="kreuzberg")
    c = build_converter(s, engine_override="pypdf")
    assert type(c).__name__ == "PyPDFToDocument"


def test_build_converter_override_validates_unknown():
    s = _settings(extraction_engine="kreuzberg")
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

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": object())

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

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": object())

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


# -------------------- Markdown chunker --------------------


def _patch_markdown_chunker(monkeypatch, *, header_sections, token_pieces=None):
    """Stub out MarkdownChunker's three external pieces.

    ``header_sections`` — what the fake ``MarkdownHeaderTextSplitter`` returns
    when ``split_text`` is called. Each item is a ``SimpleNamespace`` mimicking
    a langchain Document (``page_content`` + ``metadata``).

    ``token_pieces`` — what the fake ``RecursiveCharacterTextSplitter`` returns
    for any section that trips the stage-2 token-pack. Default ``None`` means
    "stage 2 shouldn't be reached in this test" — the test asserts that.

    The fake tokenizer's ``encode`` returns one element per whitespace-split
    token, so chunk_size + section-token-count is fully predictable from the
    test inputs.
    """
    from app.pipelines import splitter as splitter_mod

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": _FakeTokenizer())

    class _FakeMarkdownHeaderSplitter:
        def __init__(self, headers_to_split_on=None, strip_headers=False):
            pass

        def split_text(self, _text):
            return list(header_sections)

    class _FakeTokenSplitter:
        @classmethod
        def from_huggingface_tokenizer(cls, *_args, **_kwargs):
            return cls()

        def split_text(self, _text):
            if token_pieces is None:
                raise AssertionError("stage-2 token splitter was invoked unexpectedly")
            return list(token_pieces)

    monkeypatch.setattr(
        "langchain_text_splitters.MarkdownHeaderTextSplitter",
        _FakeMarkdownHeaderSplitter,
    )
    monkeypatch.setattr(
        "langchain_text_splitters.RecursiveCharacterTextSplitter",
        _FakeTokenSplitter,
    )


def _section(page_content, metadata=None):
    """Tiny stub of langchain's Document — only the attrs MarkdownChunker reads."""
    from types import SimpleNamespace

    return SimpleNamespace(page_content=page_content, metadata=metadata or {})


def test_build_splitter_markdown_returns_markdown_chunker(monkeypatch):
    _patch_markdown_chunker(monkeypatch, header_sections=[])
    s = _settings(chunk_split_by="markdown", embedding_model="intfloat/multilingual-e5-large")
    sp = build_splitter(s)
    assert type(sp).__name__ == "MarkdownChunker"


def test_build_splitter_markdown_requires_model():
    s = _settings(chunk_split_by="markdown", embedding_model="", tokenizer_model="")
    with pytest.raises(ValueError, match="TOKENIZER_MODEL or EMBEDDING_MODEL"):
        build_splitter(s)


def test_markdown_chunker_splits_on_h1_h2(monkeypatch):
    """Two header-bounded sections → two chunks, each carrying the breadcrumb."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("# Intro\nfirst", metadata={"h1": "Intro"}),
            _section("## Detail\nsecond", metadata={"h1": "Intro", "h2": "Detail"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10)
    out = ch.run(documents=[Document(content="(ignored, fake header splitter overrides)")])[
        "documents"
    ]

    assert [d.content for d in out] == ["# Intro\nfirst", "## Detail\nsecond"]
    assert [d.meta["headers"] for d in out] == [["Intro"], ["Intro", "Detail"]]
    assert [d.meta["split_id"] for d in out] == [0, 1]


def test_markdown_chunker_no_headers_passes_whole_text(monkeypatch):
    """Content without headings: MarkdownHeaderTextSplitter returns one section
    with empty metadata. The chunk passes through with ``headers=[]``."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[_section("plain text body", metadata={})],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10)
    out = ch.run(documents=[Document(content="plain text body")])["documents"]

    assert len(out) == 1
    assert out[0].content == "plain text body"
    assert out[0].meta["headers"] == []


def test_markdown_chunker_long_section_falls_back_to_token_split(monkeypatch):
    """Section longer than chunk_size tokens → handed to stage 2. Each
    sub-chunk inherits the same headers breadcrumb and gets a fresh
    monotonic split_id."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    # 6 whitespace tokens; chunk_size=3 → trips stage 2.
    long_section = "alpha beta gamma delta epsilon zeta"

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[_section(long_section, metadata={"h2": "Long"})],
        token_pieces=["alpha beta gamma", "delta epsilon zeta"],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=3, chunk_overlap=0)
    out = ch.run(documents=[Document(content=long_section)])["documents"]

    assert [d.content for d in out] == ["alpha beta gamma", "delta epsilon zeta"]
    # Same breadcrumb on both sub-chunks — the structural origin is preserved.
    assert [d.meta["headers"] for d in out] == [["Long"], ["Long"]]
    assert [d.meta["split_id"] for d in out] == [0, 1]


def test_markdown_chunker_preserves_source_meta(monkeypatch):
    """``file_id`` / ``source`` / etc. survive the two stages and don't get
    overwritten by ``headers`` / ``split_id``."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("# A\nshort", metadata={"h1": "A"}),
            _section("# B\nalso short", metadata={"h1": "B"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10)
    src_meta = {"file_id": "f-1", "source": "doc.md", "user_id": "u-7"}
    out = ch.run(documents=[Document(content="(stubbed)", meta=src_meta)])["documents"]

    for d in out:
        assert d.meta["file_id"] == "f-1"
        assert d.meta["source"] == "doc.md"
        assert d.meta["user_id"] == "u-7"
    assert out[0].meta["headers"] == ["A"]
    assert out[1].meta["headers"] == ["B"]


def test_markdown_chunker_skips_empty_input(monkeypatch):
    """Empty / whitespace-only Documents are skipped before the header
    splitter ever runs — matches HuggingFaceTokenizerSplitter behaviour."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    # The header splitter would error or return junk if called; the test
    # also serves as proof that it isn't called for empty docs.
    _patch_markdown_chunker(monkeypatch, header_sections=[])

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10)
    out = ch.run(
        documents=[
            Document(content="", meta={"file_id": "f-1"}),
            Document(content="   \n   ", meta={"file_id": "f-2"}),
        ]
    )["documents"]

    assert out == []


def test_markdown_chunker_split_id_monotonic(monkeypatch):
    """``split_id`` increments across sections AND across stage-2 sub-splits —
    so consumers can reconstruct document order by sorting on it."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("short", metadata={"h1": "S"}),
            _section("a b c d", metadata={"h1": "L"}),  # 4 tokens, chunk_size=2 → stage 2
        ],
        token_pieces=["a b", "c d"],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=2, chunk_overlap=0)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    # 1 short section + 2 stage-2 sub-chunks = 3 outputs, split_id 0..2.
    assert [d.meta["split_id"] for d in out] == [0, 1, 2]
    assert [d.meta["headers"] for d in out] == [["S"], ["L"], ["L"]]
