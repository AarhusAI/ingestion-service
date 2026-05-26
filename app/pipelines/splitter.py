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


@component
class HuggingFaceTokenizerSplitter:
    """Recursive char splitter that measures length in HuggingFace tokens.

    Wraps ``langchain_text_splitters.RecursiveCharacterTextSplitter
    .from_huggingface_tokenizer`` so chunk boundaries align with the
    embedding model's actual token boundaries. Each output chunk inherits
    the source document's meta and gets a ``split_id`` for ordering.
    """

    def __init__(
        self,
        tokenizer_model: str,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer_revision: str = "",
    ):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        tokenizer = _load_tokenizer(tokenizer_model, tokenizer_revision)
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
    section boundaries. Stage 2 token-packs each section through the same
    HF-tokenizer-aware recursive splitter ``HuggingFaceTokenizerSplitter``
    uses when a section exceeds ``chunk_size``. Heading hierarchy is
    preserved on each chunk as ``meta.headers`` (flat list, outermost
    first); chunk ordering is preserved via ``meta.split_id`` counted
    across all output chunks.

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
        self._token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._chunk_size = chunk_size
        # Stage-2 trigger uses the same tokenizer the embedding model uses,
        # so "would this section blow the model's context window?" is the
        # real question being answered.
        self._count_tokens = lambda txt: len(tokenizer.encode(txt, add_special_tokens=False))

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

            for section in sections:
                piece_text = section.page_content
                if not piece_text or not piece_text.strip():
                    continue
                headers = _headers_breadcrumb(section.metadata or {})
                section_meta = {**base_meta, "headers": headers}

                # Stage 2: token-pack only when the section actually exceeds
                # the budget. Short sections keep their natural boundaries.
                if self._count_tokens(piece_text) <= self._chunk_size:
                    out.append(
                        Document(content=piece_text, meta={**section_meta, "split_id": global_idx})
                    )
                    global_idx += 1
                    continue

                for piece in self._token_splitter.split_text(piece_text):
                    if not piece:
                        continue
                    out.append(
                        Document(content=piece, meta={**section_meta, "split_id": global_idx})
                    )
                    global_idx += 1
        return {"documents": out}


def _headers_breadcrumb(md_meta: dict) -> list[str]:
    """Flatten langchain's ``{"h1": ..., "h2": ...}`` into ``["...", "..."]``
    in heading order. Outermost first; missing levels collapse out so a
    chunk under ``## Section`` (no enclosing ``# Title``) returns
    ``["Section"]``, not ``["", "Section"]``."""
    return [md_meta[k] for k in ("h1", "h2", "h3") if md_meta.get(k)]


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
        )

    # Haystack handles word/sentence/passage natively; chunk_size is in those units.
    return DocumentSplitter(
        split_by=mode,
        split_length=s.chunk_size,
        split_overlap=s.chunk_overlap,
    )
