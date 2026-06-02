"""Native-resolution figure extraction from a ``.docx`` package.

The ``figure`` vision profile handles a docx that is mostly prose with one embedded
raster diagram (a flattened PNG flowchart/chart). It used to render the whole A4
page to a PNG and ask the model to find the ~2-inch figure within it — starving the
vision encoder of figure pixels, so labels got misread and structure garbled. But
the figure already lives in the package at native resolution
(``word/media/imageN.png``), so we read it straight from the zip and hand THAT to the
model instead — the encoder budget is spent on the figure, independent of any
DPI/encoder ceiling.

We read only the two members we need (``word/document.xml`` and
``word/_rels/document.xml.rels``) — never ``extractall`` (zip-bomb safe), the same
defensive contract as ``app/pipelines/docx_text.py`` and ``app/pipelines/detectors.py``.
Resolving relationships from ``document.xml.rels`` (not the footer/header ``.rels``)
keeps this to *body* images, so header/footer logos are excluded for free.

Returns ``[]`` on any structural surprise (not a real docx, missing member, no
resolvable image) so the caller falls back to the full-page render rather than
crashing — same defensive contract as ``extract_docx_lines``.
"""

from __future__ import annotations

import logging
import posixpath
import re
import zipfile
from pathlib import Path

from app.config import Settings
from app.log_utils import sanitize_for_log
from app.pipelines.detectors import _EXTENT_RE

log = logging.getLogger(__name__)

# Cap on figures returned (largest-first). Bounds image-token cost the same way
# ``vision_llm_max_pages`` bounds the page render; truncation is logged, never silent.
_MAX_FIGURES = 8

# Legacy VML mirror of each DrawingML shape lives in <mc:Fallback>; drop it so a
# raster figure isn't considered twice (mirrors ``docx_text._FALLBACK_RE``).
_FALLBACK_RE = re.compile(r"<mc:Fallback>.*?</mc:Fallback>", re.DOTALL)
# One DrawingML drawing block (inline or anchored). Drawings don't nest drawings,
# so a non-greedy match is safe.
_DRAWING_RE = re.compile(r"<w:drawing\b.*?</w:drawing>", re.DOTALL)
# The embedded picture's relationship id. ``r:embed`` is an internal package image;
# ``r:link`` (external) carries no bytes, so we only take ``r:embed``.
_EMBED_RE = re.compile(r'<a:blip\b[^>]*\br:embed="([^"]+)"')

# document.xml.rels relationships — parsed per tag so attribute order doesn't matter.
_RELATIONSHIP_RE = re.compile(r"<Relationship\b[^>]*?/?>")
_ID_ATTR_RE = re.compile(r'\bId="([^"]+)"')
_TARGET_ATTR_RE = re.compile(r'\bTarget="([^"]+)"')
_TARGETMODE_ATTR_RE = re.compile(r'\bTargetMode="([^"]+)"')

# Raster formats we can hand to a multimodal endpoint as image bytes. Vector parts
# (emf/wmf/svg) are the topology case, not a figure-profile raster, so skip them.
_RASTER_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"})


def extract_docx_figure_images(source: str, settings: Settings) -> list[bytes]:
    """Native-resolution body figure image bytes from a ``.docx``, largest first.

    Picks every embedded body picture whose rendered display area
    (``<wp:extent>``, EMU²) clears ``settings.extraction_router_min_image_emu`` — the
    same floor the raster detector uses — sorted largest-area first and capped at
    ``_MAX_FIGURES``. Returns ``[]`` on any structural surprise so the caller can
    fall back to rendering the page.
    """
    name = sanitize_for_log(Path(source).name)
    try:
        with zipfile.ZipFile(source) as zf:
            try:
                document_xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
                rels_xml = zf.read("word/_rels/document.xml.rels").decode(
                    "utf-8", errors="replace"
                )
            except KeyError:
                # Not a Word document, or no body relationships → nothing to extract.
                return []

            rel_targets = _rel_targets(rels_xml)
            members = set(zf.namelist())

            # Resolve each above-floor picture to a package member, dedup by rId
            # (the same image reused in two drawings is one figure). ``candidates``
            # is largest-area first, so dedup keeps the largest occurrence.
            resolved: list[str] = []
            seen_rids: set[str] = set()
            for _area, rid in _figure_candidates(
                document_xml, settings.extraction_router_min_image_emu
            ):
                if rid in seen_rids:
                    continue
                seen_rids.add(rid)
                member = _resolve_member(rel_targets.get(rid))
                if member is None or member in resolved:
                    continue
                if member not in members:
                    log.debug(
                        "docx figure %s: rel %s -> %s missing from package", name, rid, member
                    )
                    continue
                resolved.append(member)

            if len(resolved) > _MAX_FIGURES:
                log.debug(
                    "docx figure %s: %d figures over cap, keeping largest %d",
                    name,
                    len(resolved),
                    _MAX_FIGURES,
                )
                resolved = resolved[:_MAX_FIGURES]

            images = [zf.read(member) for member in resolved]
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        log.debug("docx figure extraction failed for %s: %s", name, exc)
        return []

    log.debug("docx figure %s: extracted %d image(s)", name, len(images))
    return images


def _figure_candidates(document_xml: str, min_emu: int) -> list[tuple[int, str]]:
    """``(area_emu, rId)`` for each body picture clearing ``min_emu``, largest first.

    One drawing block carries one picture: its own ``<wp:extent>`` (the first extent
    in the block — group/effect extents come later) paired with its ``<a:blip>``
    relationship id.
    """
    body = _FALLBACK_RE.sub("", document_xml)
    candidates: list[tuple[int, str]] = []
    for block in _DRAWING_RE.findall(body):
        embed = _EMBED_RE.search(block)
        if not embed:
            continue  # a shape/textbox drawing with no embedded image
        extents = _EXTENT_RE.findall(block)
        if not extents:
            continue
        cx, cy = extents[0]
        area = int(cx) * int(cy)
        if area < min_emu:
            continue
        candidates.append((area, embed.group(1)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _rel_targets(rels_xml: str) -> dict[str, str]:
    """Map relationship ``Id`` -> ``Target`` for internal (non-external) targets."""
    targets: dict[str, str] = {}
    for tag in _RELATIONSHIP_RE.findall(rels_xml):
        mode = _TARGETMODE_ATTR_RE.search(tag)
        if mode and mode.group(1).lower() == "external":
            continue  # external link — no bytes in the package
        rid = _ID_ATTR_RE.search(tag)
        target = _TARGET_ATTR_RE.search(tag)
        if rid and target:
            targets[rid.group(1)] = target.group(1)
    return targets


def _resolve_member(target: str | None) -> str | None:
    """Resolve a relationship ``Target`` to a zip member path, raster formats only.

    Targets are relative to ``word/`` (where ``document.xml`` lives); a rare
    leading-``/`` target is package-root absolute. Non-raster parts (emf/wmf/svg
    vector, unknown) return ``None`` so the caller skips them.
    """
    if not target:
        return None
    if target.startswith("/"):
        member = target.lstrip("/")
    else:
        member = posixpath.normpath(posixpath.join("word", target))
    ext = posixpath.splitext(member)[1].lower()
    if ext not in _RASTER_EXTS:
        return None
    return member
