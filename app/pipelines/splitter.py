"""Splitter factory.

Picks Haystack's built-in ``DocumentSplitter`` for word/sentence/passage
modes, a custom token-aware splitter that measures chunk lengths in the
embedding model's HuggingFace tokenizer (XLM-R for e5-large, SentencePiece
for bge-m3, etc.), or a structure-aware Markdown chunker. Token mode keeps
chunks under the model's context window without hand-tuning word counts —
a 440-word chunk in dense Danish prose can easily blow past e5-large's
512-token cap once the ``passage: `` prefix is prepended. Markdown mode
splits on heading boundaries first and token-packs each section, so the
embeddings line up with logical document structure when the converter
emits Markdown (Docling natively; Kreuzberg + table-rendering as a
follow-up). Heading hierarchy is preserved per chunk as ``meta.headers``.

Both HF-aware modes size chunks against what the *embedder sends*, not just
the chunk text: ``EMBEDDING_PREFIX_DOC``, the heading breadcrumb, and the
tokenizer's special tokens all count against ``EMBEDDING_MAX_TOKENS``. See
:func:`_fixed_embed_overhead` — budgeting content alone is what produced
513-token requests against a 512-token model and made the endpoint reject
whole batches.

Lazy imports: ``transformers`` and ``langchain_text_splitters`` are only
imported when token/markdown mode is selected, so word/sentence/passage
users don't pay the import cost.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import ClassVar

from haystack import Document, component
from haystack.components.preprocessors import DocumentSplitter

from app.config import Settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str, revision: str = ""):
    """Load and cache a HuggingFace tokenizer.

    ``AutoTokenizer.from_pretrained`` takes ~1-3s and downloads to the HF
    cache on first use. Cached so the pipeline doesn't reload per request.
    Thread-safe in CPython thanks to the GIL.

    ``revision`` pins a Hub branch, tag, or commit SHA. Empty string means
    "Hub HEAD at fetch time" — fine for dev, but production deployments
    should pin a SHA so an upstream model swap can't change tokenizer
    behaviour underneath the running service.
    """
    from transformers import AutoTokenizer

    log.info("loading HuggingFace tokenizer: %s (revision=%s)", model_name, revision or "HEAD")
    return AutoTokenizer.from_pretrained(model_name, revision=revision or None)


# Slack left under the model's hard limit to absorb sub-token drift at the
# seams: the budget is computed from parts (prefix / breadcrumb / content) but
# the endpoint tokenizes the concatenation, and SentencePiece can merge across a
# boundary. Measured 0 on the current corpus, so 2 is ample insurance.
_SAFETY_MARGIN = 2

# Never shrink a chunk below this, however expensive the breadcrumb: a heading
# path long enough to consume the whole context would otherwise drive the budget
# to zero (langchain rejects chunk_size <= 0 outright). Such a document is
# already unembeddable — its breadcrumb alone blows the limit — so overshooting
# and failing loudly downstream beats emitting a stream of 1-token fragments.
# Always applied *inside* the CHUNK_SIZE bound, so the floor can never raise the
# budget above what the operator configured.
_MIN_BUDGET = 64


def _fixed_embed_overhead(tokenizer, prefix: str) -> int:
    """Tokens the embedder adds around *every* chunk, which CHUNK_SIZE ignores.

    Two terms: the ``EMBEDDING_PREFIX_DOC`` prefix the embedder prepends, and
    the special tokens (``<s>``/``</s>``) the model's tokenizer wraps the
    sequence in. Both HF-aware splitters must subtract this — langchain's
    ``from_huggingface_tokenizer`` length function is
    ``len(tokenizer.tokenize(text))``, which counts *content only*, so its
    budget is no more aware of the overhead than ``_count_tokens`` is.

    Getting this wrong is what let 500-token chunks become 513-token requests
    and made the embedding endpoint reject whole batches with HTTP 400.
    """
    specials = tokenizer.num_special_tokens_to_add()
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False)) if prefix else 0
    return specials + prefix_tokens


@component
class HuggingFaceTokenizerSplitter:
    """Recursive char splitter that measures length in HuggingFace tokens.

    Wraps ``langchain_text_splitters.RecursiveCharacterTextSplitter
    .from_huggingface_tokenizer`` so chunk boundaries align with the
    embedding model's actual token boundaries. Each output chunk inherits
    the source document's meta and gets a ``split_id`` for ordering.

    ``chunk_size`` is clamped so the prefix and special tokens the embedder
    adds still fit under ``max_tokens`` (see :func:`_fixed_embed_overhead`).
    This mode never stamps ``headers_breadcrumb``, so the overhead is a
    constant here — unlike :class:`MarkdownChunker`, which budgets per section.
    """

    def __init__(
        self,
        tokenizer_model: str,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer_revision: str = "",
        embedding_prefix: str = "",
        max_tokens: int = 0,
    ):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        tokenizer = _load_tokenizer(tokenizer_model, tokenizer_revision)
        if max_tokens > 0:
            overhead = _fixed_embed_overhead(tokenizer, embedding_prefix)
            room = max_tokens - overhead - _SAFETY_MARGIN
            budget = min(chunk_size, max(_MIN_BUDGET, room))
            if budget < chunk_size:
                log.info(
                    "chunk budget clamped to %d tokens (CHUNK_SIZE=%d, "
                    "EMBEDDING_MAX_TOKENS=%d, embed overhead=%d)",
                    budget,
                    chunk_size,
                    max_tokens,
                    overhead,
                )
            chunk_size = budget
            chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self._splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        out: list[Document] = []
        for doc in documents:
            text = doc.content or ""
            if not text.strip():
                continue
            base_meta = dict(doc.meta or {})
            for i, piece in enumerate(self._splitter.split_text(text)):
                if not piece:
                    continue
                out.append(Document(content=piece, meta={**base_meta, "split_id": i}))
        return {"documents": out}


@component
class MarkdownChunker:
    """Two-stage chunker for Markdown content.

    Stage 1 splits on Markdown headers (``#`` / ``##`` / ``###``) using
    langchain's ``MarkdownHeaderTextSplitter`` so chunks align with
    section boundaries. Sections smaller than ``chunk_min_size`` tokens
    then merge into adjacent ones (never past ``chunk_size``; 0 disables)
    so heading-dense docs don't emit tiny chunks that embed poorly.
    Stage 2 token-packs each section through the same
    HF-tokenizer-aware recursive splitter ``HuggingFaceTokenizerSplitter``
    uses when a section exceeds ``chunk_size``. Heading hierarchy is
    preserved on each chunk as ``meta.headers`` (flat list, outermost
    first) plus ``meta.headers_breadcrumb`` (the same path joined with
    ``" > "``, omitted when empty — the string form the embedders inject
    via ``meta_fields_to_embed``); chunk ordering is preserved via
    ``meta.split_id`` counted across all output chunks.

    Content without headings degrades gracefully — the whole document
    flows through stage 2, producing the same chunks ``token`` mode
    would. So operators can flip the flag before all their content is
    actually Markdown without regressions on the plain-text long tail.
    """

    # H1-H3 covers the structural signal that matters for retrieval; deeper
    # headings stay in chunk body text. Trivial to extend if needed.
    _HEADERS: ClassVar[list[tuple[str, str]]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    def __init__(
        self,
        tokenizer_model: str,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer_revision: str = "",
        chunk_min_size: int = 0,
        embedding_prefix: str = "",
        max_tokens: int = 0,
        embed_breadcrumb: bool = False,
    ):
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )

        # ``strip_headers=False``: keep the ``## Section`` line in the chunk
        # text. The heading itself anchors the chunk's topic at embed time.
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._HEADERS,
            strip_headers=False,
        )
        tokenizer = _load_tokenizer(tokenizer_model, tokenizer_revision)
        self._tokenizer = tokenizer
        self._recursive_cls = RecursiveCharacterTextSplitter
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_size = chunk_min_size
        self._max_tokens = max_tokens
        self._embed_breadcrumb = embed_breadcrumb
        self._fixed_overhead = (
            _fixed_embed_overhead(tokenizer, embedding_prefix) if max_tokens else 0
        )
        # Stage-2 splitters, keyed by content budget. The budget varies per
        # section (the breadcrumb does), and building one is just a closure
        # around the tokenizer — cheap enough to memoize rather than share.
        self._splitters: dict[int, object] = {}
        # Stage-2 trigger uses the same tokenizer the embedding model uses,
        # so "would this section blow the model's context window?" is the
        # real question being answered.
        self._count_tokens = lambda txt: len(tokenizer.encode(txt, add_special_tokens=False))
        if max_tokens:
            clamped = self._content_budget("")
            if clamped < chunk_size:
                log.info(
                    "chunk budget clamped to %d tokens before breadcrumb "
                    "(CHUNK_SIZE=%d, EMBEDDING_MAX_TOKENS=%d, embed overhead=%d)",
                    clamped,
                    chunk_size,
                    max_tokens,
                    self._fixed_overhead,
                )

    def _content_budget(self, breadcrumb: str) -> int:
        """Tokens of *chunk content* that still fit under the model's limit.

        The embedder sends ``prefix + breadcrumb + separator + content`` and the
        tokenizer adds special tokens, so the room left for content shrinks as
        the heading path grows — measured up to 53 tokens on real documents,
        which is why this is per-section rather than one static subtraction that
        would penalise every chunk by the worst case.

        Returns ``chunk_size`` untouched when it already fits (the common case
        for a sanely configured CHUNK_SIZE) or when no limit is configured.
        """
        if not self._max_tokens:
            return self._chunk_size
        cost = 0
        if breadcrumb and self._embed_breadcrumb:
            # Include the separator the embedder joins with — it is a token too.
            cost = len(self._tokenizer.encode(breadcrumb + "\n", add_special_tokens=False))
        room = self._max_tokens - self._fixed_overhead - cost - _SAFETY_MARGIN
        return min(self._chunk_size, max(_MIN_BUDGET, room))

    def _splitter_for(self, budget: int):
        """Memoized stage-2 recursive splitter for a given content budget."""
        if budget not in self._splitters:
            self._splitters[budget] = self._recursive_cls.from_huggingface_tokenizer(
                self._tokenizer,
                # langchain's length function counts content only, so the
                # budget passes through unadjusted.
                chunk_size=budget,
                chunk_overlap=min(self._chunk_overlap, budget // 2),
            )
        return self._splitters[budget]

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        out: list[Document] = []
        global_idx = 0
        for doc in documents:
            text = doc.content or ""
            if not text.strip():
                continue
            base_meta = dict(doc.meta or {})

            # Stage 1: header split. Without headings this returns one
            # Document with the full text and empty metadata, which then
            # flows straight to stage 2 — same chunks as token mode.
            sections = self._header_splitter.split_text(text)
            pieces = [
                (section.page_content, _headers_breadcrumb(section.metadata or {}))
                for section in sections
                if section.page_content and section.page_content.strip()
            ]

            for piece_text, headers in self._merge_small_sections(pieces):
                section_meta = {**base_meta, "headers": headers}
                # Joined copy for the embedders: `meta_fields_to_embed`
                # stringifies values with str(), which would render the list
                # as "['Setup', 'Docker']". Omitted when there are no headings
                # so the field's absence keeps the embed-time prefix inert.
                breadcrumb = " > ".join(headers) if headers else ""
                if headers:
                    section_meta["headers_breadcrumb"] = breadcrumb

                # Budget is per section: this section's breadcrumb rides along
                # at embed time, so it eats into the room left for content.
                budget = self._content_budget(breadcrumb)

                # Stage 2: token-pack only when the section actually exceeds
                # the budget. Short sections keep their natural boundaries.
                if self._count_tokens(piece_text) <= budget:
                    out.append(
                        Document(content=piece_text, meta={**section_meta, "split_id": global_idx})
                    )
                    global_idx += 1
                    continue

                for piece in self._splitter_for(budget).split_text(piece_text):
                    if not piece:
                        continue
                    out.append(
                        Document(content=piece, meta={**section_meta, "split_id": global_idx})
                    )
                    global_idx += 1
        return {"documents": out}

    def _merge_small_sections(
        self, pieces: list[tuple[str, list[str]]]
    ) -> list[tuple[str, list[str]]]:
        """Greedy forward merge of adjacent sections smaller than chunk_min_size.

        Accumulation stops once an entry reaches the minimum — normal-sized
        sections keep their natural boundaries — and never crosses
        ``chunk_size``, so a tiny section next to a huge one is emitted alone
        rather than creating something stage 2 would re-split. A trailing
        remainder still under the minimum folds backward into the previous
        entry when it fits. Merged entries carry the longest common prefix of
        their sections' heading paths (each section's own heading line stays
        in the text via ``strip_headers=False``).

        Token budget is the sum of per-section counts; stage 2 re-counts the
        joined text, backstopping any drift from the ``"\\n\\n"`` seams.
        """
        if self._min_size <= 0 or len(pieces) < 2:
            return pieces

        merged: list[tuple[str, list[str], int]] = []  # (text, headers, tokens)

        def _fold(target: tuple[str, list[str], int], text, headers, tokens):
            prev_text, prev_headers, prev_tokens = target
            return (
                prev_text + "\n\n" + text,
                _headers_common_prefix(prev_headers, headers),
                prev_tokens + tokens,
            )

        for text, headers in pieces:
            tokens = self._count_tokens(text)
            if (
                merged
                and merged[-1][2] < self._min_size
                and merged[-1][2] + tokens <= self._chunk_size
            ):
                merged[-1] = _fold(merged[-1], text, headers, tokens)
            else:
                merged.append((text, headers, tokens))

        if (
            len(merged) >= 2
            and merged[-1][2] < self._min_size
            and merged[-2][2] + merged[-1][2] <= self._chunk_size
        ):
            tail = merged.pop()
            merged[-1] = _fold(merged[-1], *tail)

        return [(text, headers) for text, headers, _ in merged]


def _headers_breadcrumb(md_meta: dict) -> list[str]:
    """Flatten langchain's ``{"h1": ..., "h2": ...}`` into ``["...", "..."]``
    in heading order. Outermost first; missing levels collapse out so a
    chunk under ``## Section`` (no enclosing ``# Title``) returns
    ``["Section"]``, not ``["", "Section"]``."""
    return [md_meta[k] for k in ("h1", "h2", "h3") if md_meta.get(k)]


def _headers_common_prefix(a: list[str], b: list[str]) -> list[str]:
    """Longest common prefix of two heading paths — the deepest heading that
    is true of *both* merged sections. Empty when they share no ancestor,
    which downstream renders as "no breadcrumb" rather than a wrong one."""
    prefix: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix.append(x)
    return prefix


def build_splitter(s: Settings):
    """Pick the right splitter component for the configured chunk_split_by."""
    mode = s.chunk_split_by.lower()

    if mode == "token":
        model = (s.tokenizer_model or s.embedding_model).strip()
        if not model:
            raise ValueError(
                "CHUNK_SPLIT_BY=token requires TOKENIZER_MODEL or EMBEDDING_MODEL to be set"
            )
        return HuggingFaceTokenizerSplitter(
            tokenizer_model=model,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            tokenizer_revision=s.tokenizer_revision,
            embedding_prefix=s.embedding_prefix_doc,
            max_tokens=s.embedding_max_tokens,
        )

    if mode == "markdown":
        model = (s.tokenizer_model or s.embedding_model).strip()
        if not model:
            raise ValueError(
                "CHUNK_SPLIT_BY=markdown requires TOKENIZER_MODEL or EMBEDDING_MODEL to be set"
            )
        return MarkdownChunker(
            tokenizer_model=model,
            chunk_size=s.chunk_size,
            chunk_overlap=s.chunk_overlap,
            tokenizer_revision=s.tokenizer_revision,
            chunk_min_size=s.chunk_min_size,
            embedding_prefix=s.embedding_prefix_doc,
            max_tokens=s.embedding_max_tokens,
            # The breadcrumb only costs tokens when the embedders actually
            # prepend it (EMBED_HEADERS_BREADCRUMB) — see _meta_fields_to_embed.
            embed_breadcrumb=s.embed_headers_breadcrumb,
        )

    # Haystack handles word/sentence/passage natively; chunk_size is in those units.
    return DocumentSplitter(
        split_by=mode,
        split_length=s.chunk_size,
        split_overlap=s.chunk_overlap,
    )
