"""Per-document extraction-engine router.

The indexing pipeline is built once at startup and cached with a single fixed
``"converter"`` component (``app/pipelines/indexing.py``), so a per-request
engine decision can only live *inside* a component. ``RoutingConverter`` is that
component: it builds one inner converter per routable engine up front, inspects
each source at ``run()`` time via ``detect_engine``, and delegates to the right
inner converter — drawing-heavy ``.docx`` to the diagram engine, everything else
to the default. It is a drop-in for the ``"converter"`` slot, so the rest of the
pipeline wiring is unchanged.

Active only when ``EXTRACTION_ENGINE=auto``; otherwise the pipeline uses
``build_converter`` directly and this module is never instantiated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from haystack import Document, component

from app.config import Settings
from app.log_utils import sanitize_for_log
from app.pipelines.converters import build_converter
from app.pipelines.detectors import RoutingDecision, classify_engine
from app.pipelines.vision_profiles import KNOWN_PROFILES

log = logging.getLogger(__name__)


@component
class RoutingConverter:
    """Inspect each source and delegate to the appropriate inner converter."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._default_engine = settings.extraction_router_default.lower()
        self._diagram_engine = settings.extraction_router_diagram_engine.lower()
        # Profile the diagram route uses — pinned independently of the engine's
        # own default (VISION_LLM_PROFILE) so an operator who sets that to
        # ocr/general for forced use can't corrupt the auto-diagram route.
        # NOTE: the default diagram engine (hybrid-diagram) ignores this on its
        # main docx path — it pins `diagram-topology` itself. This value only
        # bites for hybrid's empty-docx fallback or when the diagram engine is
        # set to vision-llm. We still pass it; the converter decides what to do.
        self._diagram_profile = settings.extraction_router_diagram_profile
        if self._diagram_profile not in KNOWN_PROFILES:
            raise ValueError(
                f"EXTRACTION_ROUTER_DIAGRAM_PROFILE={self._diagram_profile!r} is not a known "
                f"profile (one of: {' | '.join(sorted(KNOWN_PROFILES))})"
            )
        # Build every engine we might route to up front. A misconfigured /
        # undeployable engine (e.g. an optional dep missing) therefore surfaces
        # at startup, not at the first matching document.
        routable = {self._default_engine, self._diagram_engine}
        self._converters = {
            name: build_converter(settings, engine_override=name) for name in routable
        }

    def warm_up(self):
        """Fan warm-up out to the inner converters.

        ``Pipeline.warm_up()`` only walks the pipeline's own top-level
        components; the inner converters live in a dict on this object, so we
        must warm them ourselves. Guarded because not every converter defines
        ``warm_up`` (Tika/Kreuzberg don't).
        """
        for conv in self._converters.values():
            warm = getattr(conv, "warm_up", None)
            if callable(warm):
                warm()

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str],
        meta: dict | list[dict] | None = None,
    ) -> dict:
        docs: list[Document] = []
        for i, source in enumerate(sources):
            decision = self._classify(source)
            engine = decision.engine
            converter = self._converters.get(engine) or self._converters[self._default_engine]
            source_meta = _meta_for(meta, i)
            # The diagram route pins the diagram profile; pass it only to a
            # profile-aware converter (a non-vision default engine never gets it).
            pinned_profile = engine == self._diagram_engine and getattr(
                converter, "accepts_profile", False
            )
            # INFO (not DEBUG): the per-document routing decision is high-value
            # operational signal, visible at the default level. The detailed
            # metrics behind it stay on the decision (stamped into chunk meta).
            log.info(
                "routing %s -> engine=%s signal=%s profile=%s",
                sanitize_for_log(Path(source).name),
                engine,
                decision.signal,
                self._diagram_profile if pinned_profile else "-",
            )
            if pinned_profile:
                result = converter.run(
                    sources=[source], meta=source_meta, profile=self._diagram_profile
                )
            else:
                result = converter.run(sources=[source], meta=source_meta)
            # Stamp the decision onto every output document so it flows through
            # chunk -> embed -> Qdrant (the splitter copies base meta), making
            # "which engine, and why" answerable per document after the fact.
            route_meta = {"signal": decision.signal, **decision.metrics}
            for doc in result.get("documents", []):
                doc.meta["extraction_engine"] = engine
                doc.meta["extraction_route"] = route_meta
                docs.append(doc)
        return {"documents": docs}

    def _classify(self, source: str) -> RoutingDecision:
        """Resolve the final routing decision for ``source`` — never raises.

        Wraps :func:`classify_engine` (which is itself exception-safe for
        routing reasons) and resolves the detector's engine (``None`` = default,
        or a diagram-engine name) to a concrete engine we actually built. The
        returned decision's ``engine`` is the one that will run.
        """
        try:
            decision = classify_engine(source, self._settings)
        except Exception:
            log.warning(
                "routing detection failed for %s; using default engine %s",
                sanitize_for_log(source),
                self._default_engine,
                exc_info=True,
            )
            return RoutingDecision(
                engine=self._default_engine,
                signal="default",
                metrics={"error": "detection_failed"},
            )
        engine = (decision.engine or self._default_engine).lower()
        # Defensive: a detected engine we didn't pre-build falls back to default.
        if engine not in self._converters:
            engine = self._default_engine
        return RoutingDecision(engine=engine, signal=decision.signal, metrics=decision.metrics)


def _meta_for(meta: dict | list[dict] | None, i: int) -> dict:
    """Match Haystack convention: ``meta`` may be a single dict applied to all
    sources, a per-source list, or omitted entirely."""
    if meta is None:
        return {}
    if isinstance(meta, list):
        return dict(meta[i]) if i < len(meta) else {}
    return dict(meta)
