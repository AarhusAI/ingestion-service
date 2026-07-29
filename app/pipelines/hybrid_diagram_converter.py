"""Hybrid diagram converter: native text + vision-inferred topology.

A swim-lane flowchart drawn as a Word ``.docx`` is a worst case for both engine
families. Plain-text extractors drop the labels (they live in drawing text
boxes) and the vision-LLM, transcribing dense fragmented text from page images,
both garbles words (``"eget ønske"`` → ``"egetamske"``) and summarizes whole
branches away. Neither is faithful on its own.

This converter splits the job along each engine's strength:

- **Native text is the authoritative body.** ``app/pipelines/docx_text.py``
  pulls every label straight from ``word/document.xml`` — complete and verbatim,
  so nothing is dropped and no word is invented. (A degenerate doc can fragment a
  word across positioned shapes; that text is still real, just occasionally
  split across lines — see ``docx_text`` for the trade-off.)
- **The vision model supplies only the diagram.** Which profile it runs under is
  chosen per document by ``docx_diagram_profile``:
  - a *vector* flowchart (labels live in Word shapes, so they're already in the
    native text) → ``diagram-topology``: one Mermaid graph of lanes/arrows/branches,
    with the native text as *grounding* so labels are copied verbatim.
  - a *raster* figure (a flattened PNG, so the labels are pixels not text) →
    ``figure``: render ONLY the embedded figure (the model reads its labels from
    the image), ignoring the surrounding prose that the native text already covers.

The two are concatenated into one ``Document``: the native body, then the diagram
section (a ``## Procesdiagram (Mermaid)`` block for the topology profile, or the
whole figure block for the figure profile). Non-``.docx`` sources (e.g. scanned
PDFs with no native text) have no authoritative text to anchor on, so they fall
through to the plain vision path unchanged.

Drop-in for the ``"converter"`` slot like ``VisionLLMConverter`` /
``RoutingConverter``: a ``@component`` with ``run(sources, meta) ->
{"documents": [...]}`` and ``accepts_profile`` so a caller may pin the fallback
profile. ``ExtractionError`` propagates from the inner vision converter so the
route layer still maps failures to ``EXTRACTION_FAILED``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from haystack import Document, component

from app.config import Settings
from app.log_utils import sanitize_for_log
from app.pipelines.converters import build_converter
from app.pipelines.detectors import docx_diagram_profile
from app.pipelines.docx_images import extract_docx_figure_images
from app.pipelines.docx_text import extract_docx_lines

log = logging.getLogger(__name__)

# Below this many native lines a docx isn't worth treating as a text-bearing
# diagram (an empty/near-empty body would just be noise) — fall back to pure
# vision, which can at least OCR whatever is on the page.
_MIN_NATIVE_LINES = 3

# Vision profile for a docx whose diagram is drawn as Word shapes (labels are in
# native text): graph only, no prose (the prose comes from native text).
_TOPOLOGY_PROFILE = "diagram-topology"
# Vision profile for a docx whose diagram is a flattened raster image (labels are
# pixels): render only that embedded figure, ignoring the prose (also native).
_FIGURE_PROFILE = "figure"
# Fallback profile for non-docx sources / empty native text.
_DEFAULT_FALLBACK_PROFILE = "diagram"

_MERMAID_RE = re.compile(r"```mermaid\b.*?```", re.DOTALL)
_MERMAID_HEADING = "## Procesdiagram (Mermaid)"


@component
class HybridDiagramConverter:
    """Merge native docx text with a vision-inferred Mermaid diagram."""

    # Probed by RoutingConverter / the extract route so they only pass a
    # ``profile=`` to converters that accept one. Here it pins the *fallback*
    # profile used for non-docx sources; the docx path always uses the topology
    # profile internally.
    accepts_profile = True

    def __init__(self, settings: Settings):
        # The vision converter does the rendering + multimodal call. Building it
        # here (rather than re-implementing) keeps the dep surface and the
        # Gotenberg/VLM config wiring in one place (converters.build_converter).
        self._vision = build_converter(settings, engine_override="vision-llm")
        # Kept for the per-document vector-vs-raster profile choice in the docx
        # path (docx_diagram_profile reads the textbox threshold from settings).
        self._settings = settings

    def warm_up(self):
        """Fan warm-up to the inner vision converter, if it defines one."""
        warm = getattr(self._vision, "warm_up", None)
        if callable(warm):
            warm()

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str],
        meta: dict | list[dict] | None = None,
        profile: str | None = None,
    ) -> dict:
        fallback_profile = profile or _DEFAULT_FALLBACK_PROFILE
        docs: list[Document] = []
        for i, source in enumerate(sources):
            name = sanitize_for_log(Path(source).name)
            source_meta = _meta_for(meta, i)
            is_docx = Path(source).suffix.lower() == ".docx"
            lines = extract_docx_lines(source) if is_docx else []
            if len(lines) >= _MIN_NATIVE_LINES:
                # Vector flowchart (labels in native text) → topology profile;
                # raster figure (labels in pixels) → figure profile.
                pass_profile = docx_diagram_profile(source, self._settings)
                log.debug(
                    "hybrid %s: native_lines=%d -> grounded %s",
                    name,
                    len(lines),
                    pass_profile,
                )
                docs.append(self._hybrid_document(source, lines, source_meta, pass_profile))
            else:
                # No authoritative native text — let the vision model do the
                # whole job (full transcription, not just topology).
                reason = (
                    f"native_lines={len(lines)} < {_MIN_NATIVE_LINES}" if is_docx else "non-docx"
                )
                log.debug(
                    "hybrid %s: %s -> delegate full-vision (%s)", name, reason, fallback_profile
                )
                result = self._vision.run(
                    sources=[source], meta=source_meta, profile=fallback_profile
                )
                docs.extend(result.get("documents", []))
        return {"documents": docs}

    def _hybrid_document(
        self, source: str, lines: list[str], source_meta: dict, pass_profile: str
    ) -> Document:
        """Native body (authoritative) + a vision-inferred diagram → one Document."""
        native_body = "\n".join(lines)

        # The two raster/vector profiles want opposite things from the vision pass:
        #
        # - topology (vector flowchart): the labels ARE the native text, so ground
        #   the model on it ("use ONLY these strings as node labels") and let it
        #   render the page to infer structure.
        # - figure (raster diagram): the labels are pixels NOT in the native text,
        #   so grounding on the prose makes the model fabricate a diagram from prose
        #   headings. Instead, hand it the embedded figure at native resolution and
        #   read the labels from the image — no grounding. If extraction finds no
        #   figure (odd package, vector-only), fall back to the full-page render.
        if pass_profile == _FIGURE_PROFILE:
            grounding = None
            images_override = extract_docx_figure_images(source, self._settings) or None
        else:
            grounding = native_body
            images_override = None

        result = self._vision.run(
            sources=[source],
            meta=source_meta,
            profile=pass_profile,
            grounding=grounding,
            images_override=images_override,
        )
        vision_docs = result.get("documents", [])
        vision_doc = vision_docs[0] if vision_docs else None

        content = native_body
        section = _diagram_section(vision_doc.content if vision_doc else "", pass_profile)
        if section:
            content += "\n\n" + section

        doc_meta = {
            "extractor": "hybrid-diagram",
            "native_source": "docx-xml",
            "vision_profile": pass_profile,
        }
        if vision_doc and "page_count" in vision_doc.meta:
            doc_meta["page_count"] = vision_doc.meta["page_count"]
        # Request meta is the contract with the route layer (file_id /
        # collection_name / user_id …) and must win on any collision.
        merged_meta = {**doc_meta, **source_meta}
        return Document(content=content, meta=merged_meta)


def _diagram_section(vision_content: str, pass_profile: str) -> str:
    """Present the vision model's diagram output, keyed to which profile produced it.

    Returns ``''`` when the model produced nothing usable so the caller can ship
    the native body alone.
    """
    text = vision_content.strip()
    if not text:
        return ""
    if pass_profile == _FIGURE_PROFILE:
        # The figure profile already returns ONLY the embedded figure (its own
        # Markdown + an optional Mermaid block), so ship it whole — extracting
        # just the fence would drop the figure's title/labels around the graph.
        return text
    # Topology profile: asked for the fence only, but a chatty model may wrap it
    # in prose — keep just the ```mermaid block under a stable heading.
    match = _MERMAID_RE.search(text)
    if not match:
        log.debug("vision topology output had no mermaid fence; omitting diagram section")
        return ""
    return f"{_MERMAID_HEADING}\n\n{match.group(0)}"


def _meta_for(meta: dict | list[dict] | None, i: int) -> dict:
    """Match Haystack convention: ``meta`` may be a single dict applied to all
    sources, a per-source list, or omitted entirely."""
    if meta is None:
        return {}
    if isinstance(meta, list):
        return dict(meta[i]) if i < len(meta) else {}
    return dict(meta)
