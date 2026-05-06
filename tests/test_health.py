from unittest.mock import patch


async def test_health_always_200(client):
    """``/health`` is a liveness probe — should be 200 even if Qdrant is down."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_ready_qdrant_ok(client):
    """``/health/ready`` returns 200 when Qdrant is reachable."""
    with patch("app.services.qdrant_setup.health_check", return_value=True):
        response = await client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


async def test_health_ready_qdrant_down(client):
    """``/health/ready`` returns 503 when Qdrant is unreachable."""
    with patch("app.services.qdrant_setup.health_check", return_value=False):
        response = await client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
