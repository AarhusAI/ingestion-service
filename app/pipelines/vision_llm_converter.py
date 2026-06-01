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
    ) -> dict:
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

            content = self._reconstruct(images, path.name, profile_name, grounding)
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
    ) -> str:
        """One chat-completions call over all page images → Markdown string.

        ``grounding``, when given, is the document's authoritative verbatim text
        (e.g. extracted natively from a docx). It is injected as a user text part
        *before* the images so the model copies labels from it instead of
        OCR-guessing — used by the hybrid diagram converter.
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
            "max_tokens": 4096,
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

        content = _content_from_response(body)
        if not content.strip():
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
