"""Content-based extraction-engine detection.

Pure, dependency-free heuristics (stdlib ``zipfile`` + ``re``) used by
``RoutingConverter`` when ``EXTRACTION_ENGINE=auto`` to decide, per document,
whether the cheap default engine will do or whether the document needs the
expensive vision-LLM engine.

The motivating case is a Word ``.docx`` that is really a swim-lane *flowchart*:
its body holds a handful of words while hundreds of drawing text boxes carry the
real content. Plain-text extractors drop that text, so such documents must route
to the diagram engine. The signal is "drawing/textbox text vastly outweighs body
text", measured cheaply from the package XML without a full parse.

``detect_engine`` returns an engine name to route to, or ``None`` to mean "use
the router's default". It never raises for routing reasons — on any structural
surprise it returns ``None`` so the caller falls back safely.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

from app.config import Settings
from app.log_utils import sanitize_for_log

log = logging.getLogger(__name__)

# Drawing/text-box markers. Substring ``.count()`` on the conventional prefixed
# tag names is namespace-stable (Word always emits these prefixes) and far
# cheaper than a namespaced XML walk.
_TEXTBOX_TAGS = ("<w:txbxContent", "<wps:txbx", "<v:textbox")

_WORDS_RE = re.compile(r"<Words>\s*(\d+)\s*</Words>")
_WT_RE = re.compile(r"<w:t[ >].*?</w:t>", re.DOTALL)
# Non-greedy strip of every text-box region so the body-word fallback counts
# only paragraph text. Order-of-magnitude accuracy is all the ratio needs.
_TEXTBOX_REGION_RE = re.compile(
    r"<w:txbxContent\b.*?</w:txbxContent>"
    r"|<wps:txbx\b.*?</wps:txbx>"
    r"|<v:textbox\b.*?</v:textbox>",
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def detect_engine(source: str, settings: Settings) -> str | None:
    """Return an engine name to route ``source`` to, or ``None`` for the default."""
    ext = Path(source).suffix.lower()
    if ext == ".docx":
        return _detect_docx(source, settings)
    # PDF hook (future: scanned/image-ratio detection). Everything else passes
    # through to the router default.
    return None


def _detect_docx(source: str, settings: Settings) -> str | None:
    try:
        with zipfile.ZipFile(source) as zf:
            # Read ONLY the two members we need — never extractall (zip-bomb safe).
            document_xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
            try:
                app_xml = zf.read("docProps/app.xml").decode("utf-8", errors="replace")
            except KeyError:
                app_xml = ""
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as exc:
        log.debug("docx routing analysis failed for %s: %s", Path(source).name, exc)
        return None

    name = Path(source).name
    drawing_text_units = sum(document_xml.count(tag) for tag in _TEXTBOX_TAGS)
    if drawing_text_units < settings.extraction_router_min_textboxes:
        log.debug(
            "docx routing %s: textboxes=%d < min=%d -> default",
            sanitize_for_log(name),
            drawing_text_units,
            settings.extraction_router_min_textboxes,
        )
        return None

    body_words = _body_word_count(document_xml, app_xml)
    ratio = drawing_text_units / (body_words + 1)
    decision = (
        settings.extraction_router_diagram_engine
        if ratio >= settings.extraction_router_drawing_ratio
        else "default"
    )
    log.debug(
        "docx routing %s: textboxes=%d body_words=%d ratio=%.2f (min=%d drawing_ratio=%.2f) -> %s",
        sanitize_for_log(name),
        drawing_text_units,
        body_words,
        ratio,
        settings.extraction_router_min_textboxes,
        settings.extraction_router_drawing_ratio,
        decision,
    )
    if ratio >= settings.extraction_router_drawing_ratio:
        return settings.extraction_router_diagram_engine
    return None


def _body_word_count(document_xml: str, app_xml: str) -> int:
    """Best-effort count of *body* (non-textbox) words.

    Prefers Word's own ``<Words>`` figure in ``docProps/app.xml`` — it counts
    body paragraphs and excludes drawing/text-box content, which is exactly the
    denominator we want. Falls back to counting words in ``<w:t>`` runs after
    stripping every text-box region when app.xml is absent or has no ``<Words>``.
    """
    match = _WORDS_RE.search(app_xml)
    if match:
        return int(match.group(1))

    body_only = _TEXTBOX_REGION_RE.sub("", document_xml)
    text = " ".join(_TAG_RE.sub("", run) for run in _WT_RE.findall(body_only))
    return len(text.split())
