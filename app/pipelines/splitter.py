"""Splitter factory.

Picks Haystack's built-in ``DocumentSplitter`` for word/sentence/passage
modes, or a custom token-aware splitter that measures chunk lengths in the
embedding model's HuggingFace tokenizer (XLM-R for e5-large, SentencePiece
for bge-m3, etc.). Token mode keeps chunks under the model's context
window without hand-tuning word counts — a 440-word chunk in dense Danish
prose can easily blow past e5-large's 512-token cap once the
``passage: `` prefix is prepended.

Lazy imports: ``transformers`` and ``langchain_text_splitters`` are only
imported when token mode is selected, so word/sentence/passage users
don't pay the import cost.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from haystack import Document, component
from haystack.components.preprocessors import DocumentSplitter

from app.config import Settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    """Load and cache a HuggingFace tokenizer.

    ``AutoTokenizer.from_pretrained`` takes ~1-3s and downloads to the HF
    cache on first use. Cached so the pipeline doesn't reload per request.
    Thread-safe in CPython thanks to the GIL.
    """
    from transformers import AutoTokenizer

    log.info("loading HuggingFace tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


@component
class HuggingFaceTokenizerSplitter:
    """Recursive char splitter that measures length in HuggingFace tokens.

    Wraps ``langchain_text_splitters.RecursiveCharacterTextSplitter
    .from_huggingface_tokenizer`` so chunk boundaries align with the
    embedding model's actual token boundaries. Each output chunk inherits
    the source document's meta and gets a ``split_id`` for ordering.
    """

    def __init__(self, tokenizer_model: str, chunk_size: int, chunk_overlap: int):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        tokenizer = _load_tokenizer(tokenizer_model)
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
        )

    # Haystack handles word/sentence/passage natively; chunk_size is in those units.
    return DocumentSplitter(
        split_by=mode,
        split_length=s.chunk_size,
        split_overlap=s.chunk_overlap,
    )
