"""Content-based extraction-engine detection.

Pure, dependency-free heuristics (stdlib ``zipfile`` + ``re``) used by
``RoutingConverter`` when ``EXTRACTION_ENGINE=auto`` to decide, per document,
whether the cheap default engine will do or whether the document needs the
expensive vision-LLM engine.

Two motivating cases, both routed to the diagram engine:

- A Word ``.docx`` that is really a swim-lane *flowchart*: its body holds a
  handful of words while hundreds of drawing text boxes carry the real content.
  The signal is "drawing/textbox text vastly outweighs body text".
- A ``.docx`` whose key content is a flattened *raster* image (a PNG flowchart
  or chart embedded via DrawingML ``<a:blip>``): it has zero textboxes, so
  plain-text extractors drop the figure entirely (its labels are pixels). The
  signal is "a body image large enough to be content, not a logo/icon" —
  measured from the picture's ``<wp:extent>`` display area. Header/footer logos
  are excluded for free because we read only ``word/document.xml``.

Both signals are measured cheaply from the package XML without a full parse.

``detect_engine`` returns an engine name to route to, or ``None`` to mean "use
the router's default". It never raises for routing reasons — on any structural
surprise it returns ``None`` so the caller falls back safely.

``docx_diagram_profile`` is the companion the diagram converter uses to pick its
vision-pass profile per document: ``diagram-topology`` when the labels are in
native text (vector flowchart) or ``figure`` when they're in a raster image.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.log_utils import sanitize_for_log

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """The full per-document routing decision, not just the chosen engine.

    ``engine`` is the engine to route to, or ``None`` to mean "use the router's
    default" (detectors don't know the router's default engine). ``signal`` is
    which heuristic fired — ``"textbox"`` (vector flowchart), ``"raster"``
    (large embedded figure), or ``"default"`` (neither). ``metrics`` carries the
    measured numbers behind the decision (textbox/body-word counts, ratio,
    image area …) so "why this engine" is inspectable downstream.
    """

    engine: str | None
    signal: str
    metrics: dict = field(default_factory=dict)


# Drawing/text-box markers. Substring ``.count()`` on the conventional prefixed
# tag names is namespace-stable (Word always emits these prefixes) and far
# cheaper than a namespaced XML walk.
_TEXTBOX_TAGS = ("<w:txbxContent", "<wps:txbx", "<v:textbox")

# Embedded raster-image marker in the body. ``<a:blip`` is the DrawingML
# picture-fill reference (inline or floating) — what Word 2007+ emits, and the
# only form that carries a measurable ``<wp:extent>`` (legacy VML images size
# via CSS, so the area gate below couldn't see them anyway). Same namespace-
# stable substring trick as the textbox tags.
_IMAGE_TAGS = ("<a:blip",)

# Picture display extent in EMU. The trailing space after ``wp:extent`` matters:
# it keeps this from also matching the sibling ``<wp:effectExtent>`` (whose
# attributes are l/t/r/b, not cx/cy). Attribute order (cx then cy) is fixed by
# the OOXML schema, so a fixed-order capture is safe for Word output.
_EXTENT_RE = re.compile(r'<wp:extent\s+cx="(\d+)"\s+cy="(\d+)"')

# 914400 EMU = 1 inch, so one square inch is 914400**2 EMU². Used only to render
# a human-readable area in the debug log.
_EMU_PER_SQ_INCH = 914400 * 914400

# Vision-pass profile names returned by ``docx_diagram_profile`` (kept as literals
# here rather than imported from the converter to avoid a circular import; they
# must stay in sync with ``vision_profiles.KNOWN_PROFILES``).
_PROFILE_VECTOR = "diagram-topology"
_PROFILE_RASTER = "figure"

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


def classify_engine(source: str, settings: Settings) -> RoutingDecision:
    """Full routing decision for ``source`` (engine + signal + metrics).

    Never raises for routing reasons — on any structural surprise it returns a
    ``"default"`` decision so the caller falls back safely.
    """
    ext = Path(source).suffix.lower()
    if ext == ".docx":
        return _classify_docx(source, settings)
    # PDF hook (future: scanned/image-ratio detection). Everything else passes
    # through to the router default.
    return RoutingDecision(engine=None, signal="default")


def detect_engine(source: str, settings: Settings) -> str | None:
    """Engine name to route ``source`` to, or ``None`` for the default.

    Thin back-compat wrapper over :func:`classify_engine` — kept so callers that
    only need the engine name don't have to unpack the decision.
    """
    return classify_engine(source, settings).engine


def _classify_docx(source: str, settings: Settings) -> RoutingDecision:
    parts = _read_docx_xml(source)
    if parts is None:
        return RoutingDecision(engine=None, signal="default", metrics={"docx_read": "failed"})
    document_xml, app_xml = parts
    name = sanitize_for_log(Path(source).name)
    # Vector flowchart first (higher fidelity — labels are real text), then the
    # raster fallback. Both route to the same diagram engine; the converter picks
    # the right vision profile via ``docx_diagram_profile``.
    textbox = _detect_textbox(document_xml, app_xml, settings, name)
    if textbox and textbox.engine:
        return textbox
    raster = _detect_raster(document_xml, app_xml, settings, name)
    if raster and raster.engine:
        return raster
    # Neither signal fired → default, but surface whatever was measured so the
    # "why default" is visible to the inspection endpoint / logs.
    merged = {**(textbox.metrics if textbox else {}), **(raster.metrics if raster else {})}
    return RoutingDecision(engine=None, signal="default", metrics=merged)


def _read_docx_xml(source: str) -> tuple[str, str] | None:
    """Read ``word/document.xml`` (+ optional ``docProps/app.xml``) from a docx.

    Returns ``(document_xml, app_xml)`` or ``None`` on any structural failure.
    Reads ONLY the members it needs — never extractall (zip-bomb safe).
    """
    try:
        with zipfile.ZipFile(source) as zf:
            document_xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
            try:
                app_xml = zf.read("docProps/app.xml").decode("utf-8", errors="replace")
            except KeyError:
                app_xml = ""
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as exc:
        log.debug("docx xml read failed for %s: %s", sanitize_for_log(Path(source).name), exc)
        return None
    return document_xml, app_xml


def _detect_textbox(
    document_xml: str, app_xml: str, settings: Settings, name: str
) -> RoutingDecision | None:
    """Vector-flowchart signal: drawing/textbox text vastly outweighs body text.

    Returns ``None`` when there aren't enough drawing shapes to even consider
    the signal, a fired ``"textbox"`` decision when the ratio clears the gate,
    or a ``"default"`` decision (carrying the measured metrics) otherwise.
    """
    drawing_text_units = sum(document_xml.count(tag) for tag in _TEXTBOX_TAGS)
    if drawing_text_units < settings.extraction_router_min_textboxes:
        return None
    body_words = _body_word_count(document_xml, app_xml)
    ratio = drawing_text_units / (body_words + 1)
    fired = ratio >= settings.extraction_router_drawing_ratio
    decision = settings.extraction_router_diagram_engine if fired else "default"
    log.debug(
        "docx routing %s: textboxes=%d body_words=%d ratio=%.2f (min=%d drawing_ratio=%.2f) -> %s",
        name,
        drawing_text_units,
        body_words,
        ratio,
        settings.extraction_router_min_textboxes,
        settings.extraction_router_drawing_ratio,
        decision,
    )
    metrics = {
        "textboxes": drawing_text_units,
        "body_words": body_words,
        "ratio": round(ratio, 3),
    }
    if fired:
        return RoutingDecision(
            engine=settings.extraction_router_diagram_engine, signal="textbox", metrics=metrics
        )
    return RoutingDecision(engine=None, signal="default", metrics=metrics)


def _detect_raster(
    document_xml: str, app_xml: str, settings: Settings, name: str
) -> RoutingDecision | None:
    """Raster-figure signal: a body image rendered large enough to be content.

    Counts embedded body images and thresholds on the LARGEST image's display
    area (``<wp:extent>``, EMU²) — max not sum, so many small inline icons can't
    add up to a false trigger. Header/footer logos never reach here (we read only
    ``word/document.xml``).

    Returns ``None`` when there are too few body images to consider, a fired
    ``"raster"`` decision when the area (and optional word-ratio gate) clears,
    or a ``"default"`` decision carrying the measured metrics otherwise.
    """
    image_count = sum(document_xml.count(tag) for tag in _IMAGE_TAGS)
    if image_count < settings.extraction_router_min_body_images:
        return None
    max_area = max(
        (int(cx) * int(cy) for cx, cy in _EXTENT_RE.findall(document_xml)),
        default=0,
    )
    sq_in = max_area / _EMU_PER_SQ_INCH
    metrics = {"image_count": image_count, "max_image_emu": max_area, "sq_in": round(sq_in, 3)}
    if max_area < settings.extraction_router_min_image_emu:
        log.debug(
            "docx routing %s: images=%d max_image_emu=%d (%.2f in²) < min=%d -> default",
            name,
            image_count,
            max_area,
            sq_in,
            settings.extraction_router_min_image_emu,
        )
        return RoutingDecision(engine=None, signal="default", metrics=metrics)
    # Optional guard against a lone large decorative photo in a prose-heavy doc.
    if settings.extraction_router_min_image_word_ratio > 0:
        body_words = _body_word_count(document_xml, app_xml)
        word_ratio = max_area / (body_words + 1)
        metrics["word_ratio"] = round(word_ratio, 1)
        if word_ratio < settings.extraction_router_min_image_word_ratio:
            log.debug(
                "docx routing %s: images=%d max_image_emu=%d word_ratio=%.0f < min_ratio=%.0f "
                "-> default",
                name,
                image_count,
                max_area,
                word_ratio,
                settings.extraction_router_min_image_word_ratio,
            )
            return RoutingDecision(engine=None, signal="default", metrics=metrics)
    log.debug(
        "docx routing %s: images=%d max_image_emu=%d (%.2f in²) >= min=%d -> %s",
        name,
        image_count,
        max_area,
        sq_in,
        settings.extraction_router_min_image_emu,
        settings.extraction_router_diagram_engine,
    )
    return RoutingDecision(
        engine=settings.extraction_router_diagram_engine, signal="raster", metrics=metrics
    )


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


def docx_diagram_profile(source: str, settings: Settings) -> str:
    """Vision-pass profile for a diagram ``.docx``.

    ``diagram-topology`` when the labels live in native text (a vector/textbox
    flowchart, so the model only needs to infer structure) or ``figure`` when
    they're baked into a raster image (the model must read them from pixels).
    Defensive: returns ``diagram-topology`` on any read failure, so a missing or
    odd file keeps the original behaviour.
    """
    parts = _read_docx_xml(source)
    if parts is None:
        return _PROFILE_VECTOR
    document_xml, _ = parts
    textboxes = sum(document_xml.count(tag) for tag in _TEXTBOX_TAGS)
    profile = (
        _PROFILE_VECTOR
        if textboxes >= settings.extraction_router_min_textboxes
        else _PROFILE_RASTER
    )
    log.debug(
        "docx diagram profile %s: textboxes=%d (min=%d) -> %s",
        sanitize_for_log(Path(source).name),
        textboxes,
        settings.extraction_router_min_textboxes,
        profile,
    )
    return profile
