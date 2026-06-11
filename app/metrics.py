"""Prometheus metrics for the ingestion service.

Collectors are module-level singletons on the default registry. They are
always live — incrementing/observing is cheap and unconditional. The
``METRICS_ENABLED`` flag only gates whether the ``/metrics`` endpoint *exposes*
them, so call-sites stay guard-free and the numbers are correct the moment an
operator turns the endpoint on.

``prometheus_client`` collectors are thread-safe, so they're safe to touch from
the per-request worker threads that run the shared cached pipeline.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)

# Ingest pipeline attempts by outcome. ``code`` is "none" on success, else the
# IngestError.code the route classified (EXTRACTION_FAILED, EMBEDDING_FAILED,
# QDRANT_WRITE_FAILED, …). Request-validation 4xx rejects (bad content-type,
# disallowed bucket) are NOT counted here — this tracks pipeline outcomes.
ingest_requests_total = Counter(
    "ingest_requests_total",
    "Ingest pipeline attempts by outcome and classified error code.",
    ["outcome", "code"],
)

# Wall-clock of the Haystack pipeline run (extract -> chunk -> embed -> write).
ingest_duration_seconds = Histogram(
    "ingest_duration_seconds",
    "Indexing pipeline wall-clock duration in seconds.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

# Chunks written per ingested document.
ingest_chunks = Histogram(
    "ingest_chunks",
    "Number of chunks written per ingested document.",
    buckets=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
)

# Size of the ingested document in bytes.
ingest_document_bytes = Histogram(
    "ingest_document_bytes",
    "Size of the ingested document in bytes.",
    buckets=(1024, 10_240, 102_400, 1_048_576, 10_485_760, 52_428_800, 104_857_600),
)

# Auto-routing decisions: which engine each document was routed to, and the
# signal that triggered it (textbox | raster | default).
extraction_route_total = Counter(
    "extraction_route_total",
    "Auto-routing engine decisions by chosen engine and trigger signal.",
    ["engine", "signal"],
)

# Per-component latency within the indexing pipeline. Stage label is the
# pipeline component name (converter | splitter | dense_embedder |
# sparse_embedder | writer) — the dense_embedder stage is the documented
# bottleneck under load.
pipeline_stage_duration_seconds = Histogram(
    "pipeline_stage_duration_seconds",
    "Per-component wall-clock duration within the indexing pipeline, in seconds.",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)


def instrument_stage(component, stage: str):
    """Wrap ``component.run`` to record per-stage latency, then return it.

    Best-effort: Haystack reads a component's input/output sockets from the
    *instance* (set at class-decoration time), not from the ``run`` callable,
    so overriding the bound method here doesn't disturb pipeline wiring. If
    wrapping fails for any reason we return the component untouched — metrics
    are never worth breaking the ingest path.
    """
    try:
        original = component.run

        def timed(*args, **kwargs):
            with pipeline_stage_duration_seconds.labels(stage=stage).time():
                return original(*args, **kwargs)

        component.run = timed
    except Exception:  # pragma: no cover - defensive; never break the pipeline
        log.debug("could not instrument stage %r for timing", stage, exc_info=True)
    return component
