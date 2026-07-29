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


class _FakeTokenizer:
    """Stand-in for a HuggingFace tokenizer: one token per whitespace word.

    Makes every budget assertion below computable by hand. ``add_special_tokens``
    is accepted and ignored because the splitters always pass ``False`` and
    account for specials separately via ``num_special_tokens_to_add``.
    """

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def tokenize(self, text):
        return text.split()

    def num_special_tokens_to_add(self):
        return 2  # <s> ... </s>, as on XLM-R / e5


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


def test_build_dense_embeds_headers_breadcrumb_by_default():
    """EMBED_HEADERS_BREADCRUMB defaults on: the embedder prepends
    meta.headers_breadcrumb to the text it encodes (stored content untouched)."""
    e = build_dense_embedder(_settings(embedding_provider="openai-compat"))
    assert e.meta_fields_to_embed == ["headers_breadcrumb"]


def test_build_dense_breadcrumb_opt_out():
    e = build_dense_embedder(
        _settings(embedding_provider="openai-compat", embed_headers_breadcrumb=False)
    )
    assert e.meta_fields_to_embed == []


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


def test_build_sparse_embeds_headers_breadcrumb_by_default():
    """The sparse embedder gets the breadcrumb too — heading terms boost
    BM42 keyword matching."""
    e = build_sparse_embedder(_settings(enable_sparse_embeddings=True))
    assert e.meta_fields_to_embed == ["headers_breadcrumb"]


def test_build_sparse_breadcrumb_opt_out():
    e = build_sparse_embedder(
        _settings(enable_sparse_embeddings=True, embed_headers_breadcrumb=False)
    )
    assert e.meta_fields_to_embed == []


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

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": _FakeTokenizer())

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

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": _FakeTokenizer())

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
    assert [d.meta["headers_breadcrumb"] for d in out] == ["Intro", "Intro > Detail"]
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
    # No headings → no breadcrumb field at all (embedders skip absent keys).
    assert "headers_breadcrumb" not in out[0].meta


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
    assert [d.meta["headers_breadcrumb"] for d in out] == ["Long", "Long"]
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


def test_markdown_chunker_merges_tiny_adjacent_sections(monkeypatch):
    """Adjacent sections under chunk_min_size merge into one chunk, joined by
    a blank line (the canonical Markdown block separator)."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("## A\ntiny one", metadata={"h1": "Doc", "h2": "A"}),
            _section("## B\ntiny two", metadata={"h1": "Doc", "h2": "B"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10, chunk_min_size=10)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == ["## A\ntiny one\n\n## B\ntiny two"]
    assert out[0].meta["split_id"] == 0


def test_markdown_chunker_min_size_zero_disables_merging(monkeypatch):
    """chunk_min_size=0 (the component default) keeps every section separate —
    pre-existing behavior is unchanged unless the merge is wired in."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("tiny one", metadata={"h1": "A"}),
            _section("tiny two", metadata={"h1": "B"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == ["tiny one", "tiny two"]


def test_markdown_chunker_tiny_section_not_merged_past_chunk_size(monkeypatch):
    """A tiny section next to a huge one is emitted alone — merging would
    create a chunk stage 2 immediately re-splits. The huge neighbor still
    goes through stage 2 afterwards."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    huge = "a b c d e f g h i j k l"  # 12 tokens > chunk_size=10
    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("t1 t2", metadata={"h1": "Tiny"}),
            _section(huge, metadata={"h1": "Huge"}),
        ],
        token_pieces=["a b c d e f", "g h i j k l"],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=10, chunk_overlap=0, chunk_min_size=5)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == ["t1 t2", "a b c d e f", "g h i j k l"]
    assert [d.meta["headers"] for d in out] == [["Tiny"], ["Huge"], ["Huge"]]
    assert [d.meta["split_id"] for d in out] == [0, 1, 2]


def test_markdown_chunker_merge_stops_at_min_size(monkeypatch):
    """Accumulation stops once an entry reaches the minimum — sections merge
    up to chunk_min_size, not all the way to chunk_size, so normal-sized
    sections keep their natural boundaries."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("a1 a2 a3", metadata={"h1": "A"}),
            _section("b1 b2 b3", metadata={"h1": "B"}),
            _section("c1 c2 c3", metadata={"h1": "C"}),
            _section("d1 d2 d3", metadata={"h1": "D"}),
        ],
    )

    # min=5: A(3)+B(3)=6 >= min stops the first entry; C+D likewise.
    ch = MarkdownChunker(tokenizer_model="x", chunk_size=20, chunk_overlap=0, chunk_min_size=5)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == ["a1 a2 a3\n\nb1 b2 b3", "c1 c2 c3\n\nd1 d2 d3"]


def test_markdown_chunker_trailing_tiny_folds_backward(monkeypatch):
    """A trailing section under the minimum folds into the previous chunk
    when it fits — the 'tiny footer' case."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("w1 w2 w3 w4 w5 w6", metadata={"h1": "Body"}),
            _section("bye", metadata={"h1": "Footer"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=10, chunk_overlap=0, chunk_min_size=5)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == ["w1 w2 w3 w4 w5 w6\n\nbye"]


def test_markdown_chunker_trailing_tiny_emitted_when_no_room(monkeypatch):
    """When the backward fold would exceed chunk_size, the trailing tiny
    section is emitted on its own — never dropped."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("w1 w2 w3 w4 w5 w6 w7 w8 w9", metadata={"h1": "Body"}),
            _section("bye now", metadata={"h1": "Footer"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=10, chunk_overlap=0, chunk_min_size=5)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == ["w1 w2 w3 w4 w5 w6 w7 w8 w9", "bye now"]
    assert [d.meta["headers"] for d in out] == [["Body"], ["Footer"]]


def test_markdown_chunker_merged_headers_common_prefix(monkeypatch):
    """A merged chunk carries the longest common prefix of its sections'
    heading paths — the deepest heading true of the whole chunk (each
    section's own heading line survives in the content)."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("## A\ntiny", metadata={"h1": "Doc", "h2": "A"}),
            _section("## B\ntiny", metadata={"h1": "Doc", "h2": "B"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10, chunk_min_size=10)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert len(out) == 1
    assert out[0].meta["headers"] == ["Doc"]
    assert out[0].meta["headers_breadcrumb"] == "Doc"


def test_markdown_chunker_merged_disjoint_headers_drop_breadcrumb(monkeypatch):
    """Sections with no shared ancestor merge with headers=[] and no
    breadcrumb key — absence beats a wrong breadcrumb at embed time."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section("# X\ntiny", metadata={"h1": "X"}),
            _section("# Y\ntiny", metadata={"h1": "Y"}),
        ],
    )

    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10, chunk_min_size=10)
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert len(out) == 1
    assert out[0].meta["headers"] == []
    assert "headers_breadcrumb" not in out[0].meta


# -------------------- Embed-budget accounting --------------------
#
# Regression tests for the failure that motivated this budget: CHUNK_SIZE was
# enforced against chunk *content*, while the embedder additionally sends
# EMBEDDING_PREFIX_DOC + the heading breadcrumb and the tokenizer adds special
# tokens. 500-token chunks became 513-token requests, and the endpoint rejects
# the whole batch of 32 with HTTP 400.
#
# With _FakeTokenizer (1 token per word): specials=2, prefix "passage: "=1
# token, so the fixed overhead is 3 and _SAFETY_MARGIN adds 2.


def _recording_recursive_splitter(monkeypatch, pieces=None):
    """Fake stage-2 splitter that records the kwargs it was built with."""
    calls = []

    class _FakeSplitter:
        @classmethod
        def from_huggingface_tokenizer(cls, *_args, **kwargs):
            calls.append(kwargs)
            return cls()

        def split_text(self, text):
            return list(pieces) if pieces is not None else [text]

    monkeypatch.setattr(
        "langchain_text_splitters.RecursiveCharacterTextSplitter",
        _FakeSplitter,
    )
    return calls


def test_token_mode_clamps_chunk_size_to_model_limit(monkeypatch):
    """CHUNK_SIZE larger than the model's limit is clamped by the overhead."""
    from app.pipelines import splitter as splitter_mod

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": _FakeTokenizer())
    calls = _recording_recursive_splitter(monkeypatch)

    splitter_mod.HuggingFaceTokenizerSplitter(
        tokenizer_model="x",
        chunk_size=1000,
        chunk_overlap=100,
        embedding_prefix="passage: ",
        max_tokens=512,
    )

    # 512 - (2 specials + 1 prefix) - 2 margin = 507
    assert calls[0]["chunk_size"] == 507


def test_token_mode_leaves_fitting_chunk_size_alone(monkeypatch):
    """A CHUNK_SIZE that already fits is passed through untouched, so the
    documented default (400) keeps producing exactly the chunks it did."""
    from app.pipelines import splitter as splitter_mod

    monkeypatch.setattr(splitter_mod, "_load_tokenizer", lambda _m, _r="": _FakeTokenizer())
    calls = _recording_recursive_splitter(monkeypatch)

    splitter_mod.HuggingFaceTokenizerSplitter(
        tokenizer_model="x",
        chunk_size=400,
        chunk_overlap=80,
        embedding_prefix="passage: ",
        max_tokens=512,
    )

    assert calls[0]["chunk_size"] == 400
    assert calls[0]["chunk_overlap"] == 80


def test_markdown_breadcrumb_cost_forces_token_split(monkeypatch):
    """**The production bug.** Two sections of identical length, one with a
    breadcrumb and one without: the breadcrumb's tokens ride along at embed
    time, so only that section exceeds the model limit and must be re-split.

    Budget maths mirrors production (CHUNK_SIZE=500, EMBEDDING_MAX_TOKENS=512):
    overhead=3, margin=2 → 507 of room, so CHUNK_SIZE=500 binds when there is no
    breadcrumb. A 10-token breadcrumb drops the room to 497. Both sections are
    exactly 500 tokens, so only the one carrying the breadcrumb overflows.
    """
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    section = " ".join(f"w{i}" for i in range(500))
    breadcrumb = " ".join(f"H{i}" for i in range(10))  # 10 tokens
    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[
            _section(section, metadata={"h2": breadcrumb}),
            _section(section, metadata={}),
        ],
        token_pieces=["first half", "second half"],
    )

    ch = MarkdownChunker(
        tokenizer_model="x",
        chunk_size=500,
        chunk_overlap=0,
        embedding_prefix="passage: ",
        max_tokens=512,
        embed_breadcrumb=True,
    )
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    # Section 1 (breadcrumb) was token-packed into two pieces; section 2
    # (no breadcrumb) still fits and passes through whole.
    assert [d.content for d in out] == ["first half", "second half", section]
    assert [d.meta["headers"] for d in out] == [[breadcrumb], [breadcrumb], []]
    assert ch._content_budget(breadcrumb) == 497
    assert ch._content_budget("") == 500


def test_markdown_breadcrumb_free_when_not_embedded(monkeypatch):
    """With EMBED_HEADERS_BREADCRUMB=false the breadcrumb is never sent, so it
    must not be charged against the budget."""
    from haystack import Document

    from app.pipelines.splitter import MarkdownChunker

    section = " ".join(f"w{i}" for i in range(500))
    breadcrumb = " ".join(f"H{i}" for i in range(10))
    _patch_markdown_chunker(
        monkeypatch,
        header_sections=[_section(section, metadata={"h2": breadcrumb})],
        token_pieces=None,  # asserts stage 2 is NOT reached
    )

    ch = MarkdownChunker(
        tokenizer_model="x",
        chunk_size=500,
        chunk_overlap=0,
        embedding_prefix="passage: ",
        max_tokens=512,
        embed_breadcrumb=False,
    )
    out = ch.run(documents=[Document(content="(stubbed)")])["documents"]

    assert [d.content for d in out] == [section]


def test_markdown_budget_never_collapses_below_floor(monkeypatch):
    """A pathological heading path can't drive the budget to zero (or negative,
    which langchain rejects outright)."""
    from app.pipelines import splitter as splitter_mod
    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(monkeypatch, header_sections=[])

    ch = MarkdownChunker(
        tokenizer_model="x",
        chunk_size=500,
        chunk_overlap=100,
        embedding_prefix="passage: ",
        max_tokens=512,
        embed_breadcrumb=True,
    )
    huge = " ".join(f"h{i}" for i in range(600))
    assert ch._content_budget(huge) == splitter_mod._MIN_BUDGET
    # And the overlap handed to langchain stays legal for that budget.
    assert ch._splitter_for(splitter_mod._MIN_BUDGET) is not None


def test_markdown_budget_disabled_without_max_tokens(monkeypatch):
    """max_tokens=0 (feature off) leaves the old content-only behaviour."""
    from app.pipelines.splitter import MarkdownChunker

    _patch_markdown_chunker(monkeypatch, header_sections=[])
    ch = MarkdownChunker(tokenizer_model="x", chunk_size=100, chunk_overlap=10)
    assert ch._content_budget("a very long breadcrumb indeed") == 100


def test_build_splitter_passes_embed_budget(monkeypatch):
    """build_splitter wires EMBEDDING_MAX_TOKENS / prefix / breadcrumb flag
    through, so the ceiling isn't silently inert in the real pipeline."""
    _patch_markdown_chunker(monkeypatch, header_sections=[])
    s = _settings(chunk_split_by="markdown", chunk_size=500, embedding_max_tokens=512)
    ch = build_splitter(s)

    assert ch._max_tokens == 512
    assert ch._embed_breadcrumb is True
    assert ch._fixed_overhead == 3  # 2 specials + 1-token "passage: "
    # 512 - 3 - 2 = 507 of room, so CHUNK_SIZE=500 is the tighter bound — a
    # short breadcrumb costs nothing, and only a long one starts to bite.
    assert ch._content_budget("") == 500
    assert ch._content_budget("Setup > Docker") == 500
    assert ch._content_budget(" ".join(f"H{i}" for i in range(20))) == 487  # 507 - 20


# -------------------- Dense-embedding invariant --------------------


def test_dense_embedder_raises_on_failure():
    """A rejected batch must abort the run, not return unembedded documents."""
    e = build_dense_embedder(_settings(embedding_provider="openai-compat"))
    assert e.raise_on_failure is True


def test_embedding_guard_passes_embedded_documents():
    from haystack import Document

    from app.pipelines.embedders import DenseEmbeddingGuard

    docs = [Document(content="a", embedding=[0.1, 0.2]), Document(content="b", embedding=[0.3])]
    out = DenseEmbeddingGuard().run(documents=docs)["documents"]
    assert out == docs


def test_embedding_guard_rejects_missing_dense_vector():
    """Qdrant would store this point with only its sparse vector — invisible to
    vector search — under a 'successful' ingest. Fail instead."""
    from haystack import Document

    from app.pipelines.embedders import DenseEmbeddingGuard
    from app.pipelines.errors import EmbeddingError

    docs = [
        Document(content="ok", embedding=[0.1], meta={"split_id": 0}),
        Document(content="dropped", meta={"split_id": 1}),
    ]
    with pytest.raises(EmbeddingError, match="no dense embedding"):
        DenseEmbeddingGuard().run(documents=docs)


def test_embedding_guard_message_omits_chunk_text():
    """The message reaches logs and (with DEBUG) the API response."""
    from haystack import Document

    from app.pipelines.embedders import DenseEmbeddingGuard
    from app.pipelines.errors import EmbeddingError

    secret = "patient name and address"
    with pytest.raises(EmbeddingError) as excinfo:
        DenseEmbeddingGuard().run(documents=[Document(content=secret, meta={"split_id": 7})])
    assert secret not in str(excinfo.value)
    assert "7" in str(excinfo.value)
