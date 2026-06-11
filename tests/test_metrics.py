"""Metrics instrumentation (app/metrics.py) + the /metrics endpoint."""

from haystack import Pipeline, component
from prometheus_client import REGISTRY

from app import metrics


@component
class _Echo:
    """Trivial real Haystack component, so we test stage-wrapping against the
    real Pipeline without downloading any models."""

    @component.output_types(value=int)
    def run(self, value: int) -> dict:
        return {"value": value + 1}


def test_instrument_stage_preserves_output_and_records_timing():
    """Wrapping a component's run for timing must not change its output, and
    must record an observation under the stage label."""
    comp = metrics.instrument_stage(_Echo(), "echo_stage")
    pipe = Pipeline()
    pipe.add_component("echo", comp)

    out = pipe.run({"echo": {"value": 41}})

    assert out["echo"]["value"] == 42
    count = REGISTRY.get_sample_value(
        "pipeline_stage_duration_seconds_count", {"stage": "echo_stage"}
    )
    assert count == 1


def test_ingest_requests_counter_increments():
    label = {"outcome": "success", "code": "none"}
    before = REGISTRY.get_sample_value("ingest_requests_total", label) or 0.0
    metrics.ingest_requests_total.labels(outcome="success", code="none").inc()
    after = REGISTRY.get_sample_value("ingest_requests_total", label)
    assert after == before + 1


async def test_metrics_endpoint_exposes_collectors(client, api_headers):
    response = await client.get("/metrics", headers=api_headers)
    assert response.status_code == 200
    body = response.text
    assert "ingest_requests_total" in body
    assert "pipeline_stage_duration_seconds" in body


async def test_metrics_endpoint_requires_auth(client):
    """Bearer-protected like /api/v1/ingest: no token → rejected, wrong → 401."""
    no_token = await client.get("/metrics")
    assert no_token.status_code in (401, 403)

    wrong_token = await client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
    assert wrong_token.status_code == 401


async def test_metrics_endpoint_404_when_disabled(client, api_headers, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "metrics_enabled", False)
    response = await client.get("/metrics", headers=api_headers)
    assert response.status_code == 404
