"""Native text extraction from a ``.docx`` package.

Pulls the document's text straight from ``word/document.xml`` without a full
Office parse — the motivating case is a swim-lane flowchart whose labels live in
hundreds of drawing text boxes that plain-text extractors (Tika/POI) never
reach. We read only that one zip member (never ``extractall`` — zip-bomb safe),
mirroring the safe single-member read in ``app/pipelines/detectors.py``.

Word stores text as ``<w:t>`` runs grouped into ``<w:p>`` paragraphs; both body
paragraphs and drawing-text-box paragraphs (nested inside ``<w:txbxContent>`` /
``<wps:txbx>`` / ``<v:textbox>``) carry real content, so we keep both in
document order. We split on paragraph/break *opening* tags rather than matching
``</w:p>`` closers because a text box nests a ``<w:p>`` inside an outer body
``<w:p>``, which defeats naive non-greedy closing-tag matching.

Caveat: a degenerate document can scatter a single word across several
positioned shapes (``"udarbejde"`` + ``"s"``), and document order is not
guaranteed to be visual reading order for floating shapes. That is acceptable
here — this text only needs to be *complete* and *verbatim*; the vision model
supplies the true flow order and rejoins spatially-adjacent fragments (see
``app/pipelines/hybrid_diagram_converter.py``).
"""

from __future__ import annotations

import html
import logging
import re
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

# Legacy VML mirror of each DrawingML shape — Word emits both inside
# <mc:AlternateContent>. Drop the fallback so shape text isn't counted twice.
_FALLBACK_RE = re.compile(r"<mc:Fallback>.*?</mc:Fallback>", re.DOTALL)
# Paragraph / line-break starts. Splitting on the *opening* tag (not the
# closer) sidesteps the text-box-nested-<w:p> problem. The trailing character
# class matches the char after the tag name (`<w:p>`, `<w:p …>`, `<w:br/>`)
# while never matching <w:pPr>, <w:proofErr>, etc.
_PARA_SPLIT_RE = re.compile(r"<w:(?:p|br|cr)[ />]")
# Inner text of a run. Runs hold plain text (no child elements), so a single
# capturing group is enough; namespace-stable on Word's conventional prefix.
_WT_RE = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)

# Glyphs that, alone on a line, are a bullet marker shape (the item text lives in
# the *next* shape). Normalised to "-" when rejoined. The non-ASCII markers
# (en dash, em dash, bullet, middle dot) are built from code points so no
# ambiguous-unicode glyph appears in the source (ruff RUF001).
_BULLET_GLYPHS = frozenset({"-", "*"} | {chr(c) for c in (0x2013, 0x2014, 0x2022, 0x00B7)})


def extract_docx_lines(source: str) -> list[str]:
    """Ordered, de-duplicated text lines from a ``.docx`` — body + text boxes.

    Each line is one Word paragraph (its ``<w:t>`` runs joined verbatim).
    Returns ``[]`` on any structural surprise (not a real docx, missing member)
    so the caller can fall back rather than crash — same defensive contract as
    ``detectors._detect_docx``.
    """
    try:
        with zipfile.ZipFile(source) as zf:
            # Read ONLY the one member we need — never extractall (zip-bomb safe).
            document_xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as exc:
        log.debug("docx text extraction failed for %s: %s", Path(source).name, exc)
        return []

    document_xml = _FALLBACK_RE.sub("", document_xml)
    lines: list[str] = []
    for segment in _PARA_SPLIT_RE.split(document_xml):
        text = "".join(html.unescape(t) for t in _WT_RE.findall(segment)).strip()
        # Collapse only *consecutive* repeats (a shape and its render artifact);
        # a label that legitimately recurs in a different sub-process is kept.
        if text and (not lines or lines[-1] != text):
            lines.append(text)
    return _tidy_lines(lines)


def _tidy_lines(lines: list[str]) -> list[str]:
    """Rejoin lone bullet-marker shapes with the item text that follows them.

    A bullet item is stored as a marker shape ("-") followed by a separate text
    shape, so the raw lines read ``["-", "Statusattest", "-", "Journaloplysninger"]``.
    We fold each lone glyph into the next line as ``"- <Item>"``.

    Guard: a standalone ``-`` is *also* used as a word-internal hyphen split
    across shapes (``"f.eks. 3", "-", "hjulet el"`` → ``3-hjulet el``). To avoid
    minting fake bullets from those, we only rejoin when the next line starts with
    an uppercase letter — true list items do (``Statusattest``), hyphen
    continuations don't (``hjulet el``). When unsure we leave the line untouched,
    so this never invents misleading structure or drops content.
    """
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        is_marker = len(line) == 1 and line in _BULLET_GLYPHS
        nxt = lines[i + 1] if i + 1 < n else None
        if is_marker and nxt and nxt[:1].isupper():
            out.append(f"- {nxt}")
            i += 2  # consume the marker and its item text
            continue
        # A trailing marker with no following item is dropped (noise, no content).
        if is_marker and nxt is None:
            break
        out.append(line)
        i += 1
    return out


def extract_docx_text(source: str) -> str:
    """The ``.docx`` text as a single newline-joined block (see ``extract_docx_lines``)."""
    return "\n".join(extract_docx_lines(source))
