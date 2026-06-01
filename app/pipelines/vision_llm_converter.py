"""Vision-LLM document converter.

For documents whose meaning lives in their *layout* — swim-lane flowcharts,
process diagrams, scanned forms — plain-text extractors either drop the text
(it sits in drawing text boxes Tika/POI never reach) or lose the flow. This
converter renders the pages to images (see ``app/pipelines/rendering.py``) and
asks an OpenAI-compatible multimodal endpoint to reconstruct the structure as
Markdown plus a Mermaid ``flowchart`` block.

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

log = logging.getLogger(__name__)


@component
class VisionLLMConverter:
    """Render pages -> multimodal LLM -> Markdown ``Document``."""

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
        gotenberg_url: str = "",
        gotenberg_connect_timeout: float = 5.0,
        gotenberg_read_timeout: float = 120.0,
        gotenberg_tls_verify: bool = True,
    ):
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
    ) -> dict:
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

            content = self._reconstruct(images, path.name)
            doc_meta = {"extractor": "vision-llm", "page_count": len(images)}
            # Request meta is the contract with the route layer (file_id /
            # collection_name / user_id …) and must win on any collision.
            merged_meta = {**doc_meta, **request_meta}
            docs.append(Document(content=content, meta=merged_meta))
        return {"documents": docs}

    def _reconstruct(self, images: list[bytes], filename: str) -> str:
        """One chat-completions call over all page images → Markdown string."""
        image_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")
                },
            }
            for png in images
        ]
        payload = {
            "model": self._model,
            # Determinism: the same diagram should reconstruct the same way.
            "temperature": 0,
            "max_tokens": 4096,
            "messages": [
                {"role": "system", "content": _system_prompt(self._language_hint)},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": _USER_PROMPT}, *image_parts],
                },
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


def _system_prompt(language_hint: str) -> str:
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
_USER_PROMPT = (
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
