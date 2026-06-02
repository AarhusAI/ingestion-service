"""Prompt profiles for the vision-LLM converter.

A profile is a (system, user) prompt pair that steers the multimodal model for a
class of document. The engine mechanics (render → chat-completions → Markdown)
are identical across profiles; only the instructions differ.

Shipped profiles:

- ``diagram`` — swim-lane flowcharts / process diagrams. Reconstructs lanes →
  ``##``, phases → ``###``, steps → bullet lists, plus one Mermaid ``flowchart``
  block. This is what the engine was originally built for.
- ``general`` — faithful full-page transcription to clean GitHub-Flavored
  Markdown (headings, paragraphs, lists, tables) with no flowchart assumptions.
  The catch-all for arbitrary visual documents.
- ``figure`` — scoped figure/diagram extraction for a doc that is *mostly prose*
  with one embedded raster image (a flattened PNG flowchart/chart). The running
  prose is supplied separately (verbatim from the docx XML by the hybrid
  converter), so this profile renders ONLY the embedded figure and ignores body
  paragraphs — the raster counterpart of ``diagram-topology`` (which assumes the
  whole page IS the diagram).
- ``ocr`` — plain-text transcription of scanned pages: reading order and line
  breaks preserved, minimal structure, no commentary.

All system prompts take the ``language_hint`` so the model keeps the document's
source language verbatim instead of translating.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """A named (system, user) prompt pair. ``system`` is parameterised by the
    configured language hint; ``user`` is static."""

    name: str
    system: Callable[[str], str]
    user: str


# --------------------------------------------------------------------------
# diagram (default behaviour of the original engine)
# --------------------------------------------------------------------------


def _diagram_system(language_hint: str) -> str:
    return (
        f"You are a document-structure extraction engine. You receive page images of a "
        f"{language_hint}-language document — typically a swim-lane flowchart or process "
        f"diagram where most text lives in drawing text boxes. Reconstruct the document's "
        f"structure as GitHub-Flavored Markdown. Preserve ALL {language_hint} text exactly as "
        f"written; never translate, summarize, or invent steps. Mark unreadable text "
        f"[unreadable]. Output Markdown only — no preamble, no commentary, and do not wrap the "
        f"whole answer in a code fence."
    )


# Terse, concrete scaffold with one worked example — a 4-bit model drifts from
# long multi-part schemas, so the example pins the node/edge syntax.
_DIAGRAM_USER = (
    "Reconstruct this document as Markdown, following this structure exactly:\n"
    "1. Start with `# <title>` if a title is visible.\n"
    "2. Each swim-lane (role/actor column or row) becomes `## <lane name>`.\n"
    "3. Each phase or stage within a lane becomes `### <phase name>`.\n"
    "4. Under each phase, list the steps as a bullet list in reading order (follow the "
    "arrows). Render a decision as `- <question>? -> Yes: ... / No: ...`.\n"
    "5. After the prose, output exactly one fenced mermaid block under "
    "`## Procesdiagram (Mermaid)`, like this:\n\n"
    "```mermaid\n"
    "flowchart TD\n"
    '  n1["Modtag ansøgning"] --> n2["Opret sag"]\n'
    '  n2 -->|Ja| n3["Bevilling"]\n'
    '  n2 -->|Nej| n4["Afslag"]\n'
    "```\n\n"
    "Mermaid rules: one node per box with a stable id (n1, n2, …) and its label in double "
    "quotes; one edge per arrow; decision branches use `-->|label|`; group each swim-lane with "
    "`subgraph \"Lane name\" ... end`. Output Markdown only."
)


# --------------------------------------------------------------------------
# diagram-topology (Mermaid graph only — used by the hybrid converter)
# --------------------------------------------------------------------------


def _diagram_topology_system(language_hint: str) -> str:
    return (
        f"You are a diagram-topology extraction engine. You receive page images of a "
        f"{language_hint}-language swim-lane flowchart or process diagram. Your only job is to "
        f"capture its STRUCTURE — the boxes, which lane each belongs to, the arrows between them, "
        f"and any branch conditions — as a single Mermaid flowchart. Preserve all {language_hint} "
        f"node labels exactly as written; never translate, summarize, or invent steps. Mark "
        f"unreadable text [unreadable]. Output only the Mermaid block — no prose, no preamble, no "
        f"commentary, and do not wrap it in anything other than the ```mermaid fence."
    )


# Mirrors the Mermaid rules of _DIAGRAM_USER but suppresses the prose section —
# the hybrid converter supplies the prose from native text and wants only the graph.
_DIAGRAM_TOPOLOGY_USER = (
    "Output exactly one fenced mermaid block and nothing else, like this:\n\n"
    "```mermaid\n"
    "flowchart TD\n"
    '  n1["Modtag ansøgning"] --> n2["Opret sag"]\n'
    '  n2 -->|Ja| n3["Bevilling"]\n'
    '  n2 -->|Nej| n4["Afslag"]\n'
    "```\n\n"
    "Mermaid rules: one node per box with a stable id (n1, n2, …) and its label in double "
    "quotes; one edge per arrow following its direction; decision branches use `-->|label|`; "
    "group each swim-lane with `subgraph \"Lane name\" ... end`. Output only the mermaid block."
)


# --------------------------------------------------------------------------
# general (faithful full-page transcription)
# --------------------------------------------------------------------------


def _general_system(language_hint: str) -> str:
    return (
        f"You are a document transcription engine. You receive page images of a "
        f"{language_hint}-language document. Transcribe each page faithfully to GitHub-Flavored "
        f"Markdown, preserving the document's structure: headings, paragraphs, bullet/numbered "
        f"lists, and tables (as Markdown tables). Preserve ALL {language_hint} text exactly as "
        f"written; never translate, summarize, paraphrase, or invent content. Keep reading order. "
        f"Mark unreadable text [unreadable]. Output Markdown only — no preamble, no commentary."
    )


_GENERAL_USER = (
    "Transcribe the document to clean Markdown. Use `#`/`##`/`###` for headings as they appear, "
    "bullet or numbered lists for lists, and Markdown tables for tabular content. Preserve "
    "reading order across pages. Do not add, summarize, or omit content. Output Markdown only."
)


# --------------------------------------------------------------------------
# figure (embedded figure only — prose comes from native docx text)
# --------------------------------------------------------------------------


def _figure_system(language_hint: str) -> str:
    return (
        f"You are a figure-extraction engine. You receive page images of a "
        f"{language_hint}-language document that is MOSTLY running prose — and that prose has "
        f"ALREADY been transcribed separately, so you must ignore it. Your ONLY job is to find "
        f"any embedded figure — a flowchart, process diagram, cycle, org chart, or chart — and "
        f"render ONLY that figure as GitHub-Flavored Markdown, plus, if it has flow or "
        f"structure, exactly one fenced Mermaid block under `## Procesdiagram (Mermaid)`. Do "
        f"NOT transcribe body paragraphs, headings, or lists that are ordinary prose. Preserve "
        f"all {language_hint} labels inside the figure exactly as written; never translate, "
        f"summarize, or invent. Mark unreadable text [unreadable]. If the pages contain no such "
        f"figure, output nothing at all."
    )


# Mirrors the Mermaid scaffold of _DIAGRAM_USER, but scoped to the embedded figure
# and explicitly suppressing the surrounding prose (supplied from native text).
_FIGURE_USER = (
    "Render only the embedded figure/diagram, ignoring all running body prose. If the figure "
    "shows flow or a cycle, give it a short `## <figure title>` heading (if one is visible) "
    "followed by exactly one fenced mermaid block under `## Procesdiagram (Mermaid)`, like this:"
    "\n\n"
    "```mermaid\n"
    "flowchart TD\n"
    '  n1["Modtag ansøgning"] --> n2["Opret sag"]\n'
    '  n2 -->|Ja| n3["Bevilling"]\n'
    '  n2 -->|Nej| n4["Afslag"]\n'
    "```\n\n"
    "Mermaid rules: one node per box with a stable id (n1, n2, …) and its label in double "
    "quotes; one edge per arrow following its direction; decision branches use `-->|label|`; "
    "group each lane/actor region with `subgraph \"Lane name\" ... end`. If there is no figure, "
    "output nothing."
)


# --------------------------------------------------------------------------
# ocr (plain-text transcription of scans)
# --------------------------------------------------------------------------


def _ocr_system(language_hint: str) -> str:
    return (
        f"You are an OCR engine. You receive page images of a scanned {language_hint}-language "
        f"document. Transcribe the visible text exactly, preserving reading order, line breaks, "
        f"and paragraph breaks. Do not translate, correct, summarize, or reformat; do not add "
        f"headings or commentary. Mark unreadable text [unreadable]. Output the transcribed text "
        f"only."
    )


_OCR_USER = (
    "Output the text content of each page in reading order. Preserve line breaks and paragraph "
    "breaks. Separate pages with a blank line. No commentary, no Markdown formatting beyond the "
    "text itself."
)


_PROFILES: dict[str, Profile] = {
    "diagram": Profile("diagram", _diagram_system, _DIAGRAM_USER),
    "diagram-topology": Profile(
        "diagram-topology", _diagram_topology_system, _DIAGRAM_TOPOLOGY_USER
    ),
    "general": Profile("general", _general_system, _GENERAL_USER),
    "figure": Profile("figure", _figure_system, _FIGURE_USER),
    "ocr": Profile("ocr", _ocr_system, _OCR_USER),
}

KNOWN_PROFILES = frozenset(_PROFILES)


def get_profile(name: str) -> Profile:
    """Return the named profile, or raise ``ValueError`` for an unknown name."""
    try:
        return _PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown vision-llm profile {name!r} (one of: {' | '.join(sorted(KNOWN_PROFILES))})"
        ) from None
