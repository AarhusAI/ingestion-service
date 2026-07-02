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

# Ingest requests by outcome. ``code`` is "none" on success, else the
# IngestError.code from the failure response (EXTRACTION_FAILED,
# S3_FETCH_FAILED, INVALID_REQUEST, …). Counted centrally in the ingest route
# handler, so every failure class shows up — including S3 fetch errors and
# request-validation rejects that never reach the pipeline. Auth failures
# (401) happen in the dependency before the handler and are not counted.
ingest_requests_total = Counter(
    "ingest_requests_total",
    "Ingest requests by outcome and classified error code.",
    ["outcome", "code"],
)

# Delete requests by outcome. ``code`` is "none" on success, else the
# IngestError.code from the failure response (DELETE_FAILED, PIPELINE_FAILED for
# a not-ready pipeline, …). Counted in the delete route handler, mirroring
# ``ingest_requests_total``. Auth failures (401) happen in the dependency before
# the handler and are not counted.
delete_requests_total = Counter(
    "delete_requests_total",
    "Delete requests by outcome and classified error code.",
    ["outcome", "code"],
)

# Wall-clock of the Haystack pipeline run (extract -> chunk -> embed -> write).
ingest_duration_seconds = Histogram(
    "ingest_duration_seconds",
    "Indexing pipeline wall-clock duration in seconds.",
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)

# Ingests that completed successfully but wrote zero chunks. The caller sees
# status=true / chunks_count=0, so without this counter (and the paired
# warning log) an extraction engine silently yielding nothing — e.g. a
# sidecar response-shape drift — looks healthy on every dashboard.
ingest_empty_total = Counter(
    "ingest_empty_total",
    "Successful ingests that wrote zero chunks (extraction produced no content).",
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
