"""Settings-level validators (sec.md Findings 6 + 11).

Constructs ``Settings`` directly with explicit kwargs rather than going
through the environment so the validator behaviour is observable in
isolation. ``_env_file=None`` disables the default ``.env`` lookup.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings

_BASE = {
    "api_key": "a" * 32,
    "embedding_api_base_url": "http://embed.local",
    "embedding_api_key": "x",
}


@pytest.mark.parametrize(
    "field",
    [
        "kreuzberg_url",
        "vision_llm_api_base_url",
        "gotenberg_url",
        "qdrant_uri",
        "embedding_api_base_url",
        "s3_endpoint_url",
    ],
)
def test_url_validator_rejects_bad_scheme(field):
    """Every URL-typed setting must refuse non-http(s) schemes at startup."""
    bad = "file:///etc/passwd"
    with pytest.raises(ValidationError, match="http://"):
        Settings(_env_file=None, **{**_BASE, field: bad})


@pytest.mark.parametrize(
    "field",
    [
        "kreuzberg_url",
        "vision_llm_api_base_url",
        "gotenberg_url",
        "qdrant_uri",
        "embedding_api_base_url",
    ],
)
def test_url_validator_accepts_http_and_https(field):
    """http:// and https:// both pass."""
    for ok in ("http://kreuzberg:8000", "https://api.example.com/v1"):
        Settings(_env_file=None, **{**_BASE, field: ok})


def test_url_validator_allows_empty_for_optional_endpoints():
    """s3_endpoint_url and embedding_api_base_url use '' to mean
    'SDK default endpoint resolution' — must be accepted."""
    Settings(_env_file=None, **{**_BASE, "s3_endpoint_url": ""})
    Settings(_env_file=None, **{**_BASE, "embedding_api_base_url": ""})


def test_url_validator_rejects_missing_scheme():
    """Bare hostnames without a scheme are rejected (no implicit http://)."""
    with pytest.raises(ValidationError, match="http://"):
        Settings(_env_file=None, **{**_BASE, "kreuzberg_url": "kreuzberg:8000"})


# ----- Routing engine-name validator (EXTRACTION_ENGINE=auto) -----


def test_auto_mode_accepts_known_router_engines():
    """auto mode with the default kreuzberg/hybrid-diagram engines validates fine."""
    Settings(_env_file=None, **{**_BASE, "extraction_engine": "auto"})


def test_auto_mode_rejects_unknown_diagram_engine():
    with pytest.raises(ValidationError, match="not a known extraction engine"):
        Settings(
            _env_file=None,
            **{**_BASE, "extraction_engine": "auto", "extraction_router_diagram_engine": "banana"},
        )


def test_auto_mode_rejects_unknown_default_engine():
    with pytest.raises(ValidationError, match="not a known extraction engine"):
        Settings(
            _env_file=None,
            **{**_BASE, "extraction_engine": "auto", "extraction_router_default": "banana"},
        )


def test_non_auto_mode_skips_router_engine_validation():
    """Router engine names are only validated in auto mode — a stale/unknown
    value is tolerated when routing is off (the field is simply unused)."""
    Settings(
        _env_file=None,
        **{**_BASE, "extraction_engine": "pypdf", "extraction_router_diagram_engine": "banana"},
    )


def test_raster_signal_threshold_defaults():
    """The raster-image routing signal ships disabled-by-default-ratio with an
    area floor that clears a real figure but not a logo (see config docstrings)."""
    s = Settings(_env_file=None, **_BASE)
    assert s.extraction_router_min_body_images == 1
    assert s.extraction_router_min_image_emu == 1_500_000_000_000
    assert s.extraction_router_min_image_word_ratio == 0.0


def test_chunk_min_size_rejects_negative():
    with pytest.raises(ValidationError, match="CHUNK_MIN_SIZE must be >= 0"):
        Settings(_env_file=None, **{**_BASE, "chunk_min_size": -1})


def test_chunk_min_size_rejects_above_chunk_size():
    """A minimum above the maximum is nonsensical — fail fast at startup."""
    with pytest.raises(ValidationError, match="must not exceed"):
        Settings(_env_file=None, **{**_BASE, "chunk_min_size": 500, "chunk_size": 400})
