"""Vision-LLM document converter.

For documents whose meaning lives in their *layout* — swim-lane flowcharts,
process diagrams, scanned forms — plain-text extractors either drop the text
(it sits in drawing text boxes Tika/POI never reach) or lose the flow. This
converter renders the pages to images (see ``app/pipelines/rendering.py``) and
asks an OpenAI-compatible multimodal endpoint to reconstruct the structure as
Markdown. The instructions vary by **profile** (``app/pipelines/vision_profiles.py``):
``diagram`` (lanes/phases/steps + a Mermaid ``flowchart``), ``general`` (faithful
full-page transcription), or ``ocr`` (plain scanned-text). The profile is the
engine default (``VISION_LLM_PROFILE``) unless ``run(profile=…)`` overrides it.

Lives next to ``KreuzbergRemoteConverter`` and follows the same shape: a
``@component`` with ``run(sources, meta) -> {"documents": [...]}``, a split
``httpx.Timeout``, ``ExtractionError`` on every failure (so the route-layer
classifier maps it to ``EXTRACTION_FAILED``), and the
``{**doc_meta, **request_meta}`` merge so caller meta wins. Uses plain ``httpx``
— no ``openai`` dependency. One ``Document`` per source: all pages go in a single
chat-completions call so a flowchart spanning pages is reconstructed as one
coherent graph rather than fragmented per page.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx
from haystack import Document, component

from app.log_utils import sanitize_for_log
from app.pipelines.errors import ExtractionError
from app.pipelines.rendering import render_to_pngs
from app.pipelines.vision_profiles import KNOWN_PROFILES, get_profile

log = logging.getLogger(__name__)


@component
class VisionLLMConverter:
    """Render pages -> multimodal LLM -> Markdown ``Document``."""

    # Capability flag probed by RoutingConverter / the extract route so they
    # only pass a ``profile=`` to converters that accept one.
    accepts_profile = True

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 180.0,
        dpi: int = 150,
        max_pages: int = 20,
        max_tokens: int = 16384,
        tls_verify: bool = True,
        language_hint: str = "Danish",
        default_profile: str = "general",
        gotenberg_url: str = "",
        gotenberg_connect_timeout: float = 5.0,
        gotenberg_read_timeout: float = 120.0,
        gotenberg_tls_verify: bool = True,
    ):
        if default_profile not in KNOWN_PROFILES:
            raise ValueError(
                f"VISION_LLM_PROFILE={default_profile!r} is not a known profile "
                f"(one of: {' | '.join(sorted(KNOWN_PROFILES))})"
            )
        self._default_profile = default_profile
        self._url = api_base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        # Split timeout — connect fails fast so a stalled endpoint doesn't tie up
        # a worker for the full read window; read is long because a quantized VLM
        # reasoning over several page images is slow.
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )
        self._verify = tls_verify
        self._dpi = dpi
        self._max_pages = max_pages
        self._max_tokens = max_tokens
        self._language_hint = language_hint
        # Forwarded verbatim to render_to_pngs for the office->PDF leg.
        self._gotenberg_url = gotenberg_url
        self._gotenberg_connect_timeout = gotenberg_connect_timeout
        self._gotenberg_read_timeout = gotenberg_read_timeout
        self._gotenberg_tls_verify = gotenberg_tls_verify

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str],
        meta: dict | list[dict] | None = None,
        profile: str | None = None,
        grounding: str | None = None,
        images_override: list[bytes] | None = None,
    ) -> dict:
        # ``images_override``, when given, replaces the page render: the caller has
        # already produced the exact image bytes to send (the hybrid converter
        # extracts the native-resolution figure straight from the docx). It applies
        # to a single-source call (how the hybrid converter uses it) and, because
        # there is no figure to "miss", an empty model response is then treated as
        # "no figure found" rather than a fatal error.
        # Defensive: an unknown profile (e.g. a future detector returning a name
        # not in the registry) falls back to the default rather than crashing.
        profile_name = profile or self._default_profile
        if profile_name not in KNOWN_PROFILES:
            log.warning(
                "unknown vision-llm profile %r; falling back to %s",
                profile_name,
                self._default_profile,
            )
            profile_name = self._default_profile

        docs: list[Document] = []
        for i, source in enumerate(sources):
            path = Path(source)
            request_meta = _meta_for(meta, i)

            if images_override:
                # Caller supplied the images (e.g. the docx figure, extracted at
                # native resolution) — skip the render entirely.
                images = images_override
            else:
                # render_to_pngs raises ExtractionError (with the filename) on any
                # rendering failure — let it propagate unchanged.
                images = render_to_pngs(
                    source,
                    gotenberg_url=self._gotenberg_url,
                    dpi=self._dpi,
                    max_pages=self._max_pages,
                    connect_timeout=self._gotenberg_connect_timeout,
                    read_timeout=self._gotenberg_read_timeout,
                    verify=self._gotenberg_tls_verify,
                )

            log.debug(
                "vision %s: profile=%s images=%d grounded=%s override=%s",
                sanitize_for_log(path.name),
                profile_name,
                len(images),
                bool(grounding and grounding.strip()),
                bool(images_override),
            )
            content = self._reconstruct(
                images, path.name, profile_name, grounding, allow_empty=bool(images_override)
            )
            doc_meta = {
                "extractor": "vision-llm",
                "vision_profile": profile_name,
                "page_count": len(images),
            }
            # Request meta is the contract with the route layer (file_id /
            # collection_name / user_id …) and must win on any collision.
            merged_meta = {**doc_meta, **request_meta}
            docs.append(Document(content=content, meta=merged_meta))
        return {"documents": docs}

    def _reconstruct(
        self,
        images: list[bytes],
        filename: str,
        profile_name: str,
        grounding: str | None = None,
        allow_empty: bool = False,
    ) -> str:
        """One chat-completions call over all page images → Markdown string.

        ``grounding``, when given, is the document's authoritative verbatim text
        (e.g. extracted natively from a docx). It is injected as a user text part
        *before* the images so the model copies labels from it instead of
        OCR-guessing — used by the hybrid diagram converter.

        ``allow_empty`` softens an empty model response into ``""`` instead of an
        ``ExtractionError``. Used on the figure path: the ``figure`` profile is told
        to "output nothing" when there is no figure (e.g. a purely decorative
        image), and the caller then ships the native body alone — that must not fail
        the whole ingest.
        """
        prof = get_profile(profile_name)
        image_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                },
            }
            for png in images
        ]
        user_parts: list[dict] = [{"type": "text", "text": prof.user}]
        if grounding and grounding.strip():
            user_parts.append({"type": "text", "text": _grounding_prompt(grounding)})
        user_parts.extend(image_parts)
        payload = {
            "model": self._model,
            # Determinism: the same diagram should reconstruct the same way.
            "temperature": 0,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": prof.system(self._language_hint)},
                {"role": "user", "content": user_parts},
            ],
        }
        # Never interpolate the Authorization header into any exception message.
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(
                self._url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
                verify=self._verify,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExtractionError(f"vision-llm request failed for {filename}: {exc}") from exc

        try:
            body = resp.json()
        except ValueError as exc:
            raise ExtractionError(f"vision-llm returned non-JSON for {filename}: {exc}") from exc

        # Truncation is a hard failure (distinct from the empty/no-figure case below):
        # the model hit the output cap and the markdown is cut off mid-document, so we
        # must not ship a half-transcribed page set into the index. Fires even on the
        # allow_empty/figure path.
        if _finish_reason_from_response(body) == "length":
            raise ExtractionError(
                f"vision-llm output truncated for {filename} at max_tokens={self._max_tokens} "
                f"(finish_reason=length); raise VISION_LLM_MAX_TOKENS or use a text engine"
            )

        content = _content_from_response(body)
        if not content.strip():
            if allow_empty:
                log.debug("vision-llm empty content for %s — no figure found", filename)
                return ""
            raise ExtractionError(f"vision-llm returned empty content for {filename}")
        return content


def _grounding_prompt(grounding: str) -> str:
    """Wrap the authoritative native text in instructions that pin it as the
    verbatim source of truth for node/step labels."""
    return (
        "AUTHORITATIVE TEXT — the exact verbatim text labels present in this document, one "
        "per line. Use ONLY these strings for node and step labels; copy them "
        "character-for-character; never invent, translate, merge, or split words. If a box's "
        "text is not in this list, mark it [unreadable]. Determine the structure (lanes, "
        "phases, arrows, branch conditions) from the images.\n\n" + grounding
    )


def _meta_for(meta: dict | list[dict] | None, i: int) -> dict:
    """Match Haystack convention: ``meta`` may be a single dict applied to all
    sources, a per-source list, or omitted entirely."""
    if meta is None:
        return {}
    if isinstance(meta, list):
        return dict(meta[i]) if i < len(meta) else {}
    return dict(meta)


def _content_from_response(body: object) -> str:
    """Pull ``choices[0].message.content`` defensively from an OpenAI-shaped body.

    Returns the empty string for any unexpected shape — the caller treats empty
    content as a failed extraction. Never raises on shape drift.
    """
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _finish_reason_from_response(body: object) -> str | None:
    """``choices[0].finish_reason`` from an OpenAI-shaped body, or ``None`` on drift.

    A missing/None finish_reason yields ``None`` (no truncation raised), so servers
    that omit the field stay tolerated.
    """
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None
