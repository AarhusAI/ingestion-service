"""RoutingConverter delegation (app/pipelines/routing_converter.py).

``build_converter`` and ``detect_engine`` are patched at the module symbol so no
real converter is constructed and the routing decision is forced per test.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from app.config import Settings
from app.pipelines import routing_converter as rc


def _settings(**overrides) -> Settings:
    # Pin the routable engines explicitly so the test is independent of any
    # EXTRACTION_ROUTER_* leaking in from the container environment (pydantic
    # reads os.environ even with _env_file=None).
    base = {
        "api_key": "a" * 32,
        "extraction_engine": "auto",
        "extraction_router_default": "tika",
        "extraction_router_diagram_engine": "vision-llm",
        # Pin the profile too — the diagram-profile assertions must not inherit a
        # deployment's EXTRACTION_ROUTER_DIAGRAM_PROFILE (e.g. diagram-topology).
        "extraction_router_diagram_profile": "diagram",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _fakes():
    """Return (tika, vision) converter mocks + a build_converter side_effect."""
    tika = MagicMock(name="tika")
    vision = MagicMock(name="vision")
    # tika has no profile concept; vision does. Set explicitly so the router's
    # `getattr(conv, "accepts_profile", False)` guard is deterministic (a bare
    # MagicMock would auto-return a truthy attribute).
    tika.accepts_profile = False
    vision.accepts_profile = True
    tika.run.return_value = {"documents": [Document(content="tika out")]}
    vision.run.return_value = {"documents": [Document(content="vision out")]}

    def build(_s, engine_override=None):
        return {"tika": tika, "vision-llm": vision}[engine_override]

    return tika, vision, build


def test_routes_diagram_doc_to_diagram_engine():
    tika, vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value="vision-llm"),
    ):
        conv = rc.RoutingConverter(_settings())
        out = conv.run(sources=["x.docx"], meta={"file_id": "f1"})["documents"]

    assert vision.run.called
    assert not tika.run.called
    assert out[0].content == "vision out"
    # request meta is forwarded to the inner converter
    _, kwargs = vision.run.call_args
    assert kwargs["meta"] == {"file_id": "f1"}


def test_routes_to_hybrid_diagram_engine_and_forwards_profile():
    default = MagicMock(name="tika")
    hybrid = MagicMock(name="hybrid")
    default.accepts_profile = False
    hybrid.accepts_profile = True
    default.run.return_value = {"documents": [Document(content="d")]}
    hybrid.run.return_value = {"documents": [Document(content="hybrid out")]}

    def build(_s, engine_override=None):
        return {"tika": default, "hybrid-diagram": hybrid}[engine_override]

    s = _settings(extraction_router_diagram_engine="hybrid-diagram")
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value="hybrid-diagram"),
    ):
        out = rc.RoutingConverter(s).run(sources=["x.docx"], meta={})["documents"]

    assert hybrid.run.called
    assert not default.run.called
    assert out[0].content == "hybrid out"
    _, kwargs = hybrid.run.call_args
    assert kwargs.get("profile") == "diagram"


def test_routes_default_when_detection_returns_none():
    tika, vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value=None),
    ):
        conv = rc.RoutingConverter(_settings())
        out = conv.run(sources=["x.docx"], meta={})["documents"]

    assert tika.run.called
    assert not vision.run.called
    assert out[0].content == "tika out"


def test_detection_error_falls_back_to_default():
    tika, _vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", side_effect=ValueError("boom")),
    ):
        conv = rc.RoutingConverter(_settings())
        # Must not raise — the bad detector degrades to the default engine.
        out = conv.run(sources=["x.docx"], meta={})["documents"]

    assert tika.run.called
    assert out[0].content == "tika out"


def test_warm_up_fans_out_to_inner_converters():
    tika, vision, build = _fakes()
    with patch.object(rc, "build_converter", side_effect=build):
        conv = rc.RoutingConverter(_settings())
        conv.warm_up()

    assert tika.warm_up.called
    assert vision.warm_up.called


def test_warm_up_skips_converters_without_warm_up():
    vision = MagicMock(name="vision")
    plain = object()  # no warm_up attribute

    def build(_s, engine_override=None):
        return {"tika": plain, "vision-llm": vision}[engine_override]

    with patch.object(rc, "build_converter", side_effect=build):
        conv = rc.RoutingConverter(_settings())
        conv.warm_up()  # must not raise on the plain object

    assert vision.warm_up.called


def test_meta_per_source_list_form():
    tika, _vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value=None),
    ):
        conv = rc.RoutingConverter(_settings())
        conv.run(sources=["a.docx", "b.docx"], meta=[{"file_id": "1"}, {"file_id": "2"}])

    metas = [kwargs["meta"] for _, kwargs in tika.run.call_args_list]
    assert metas == [{"file_id": "1"}, {"file_id": "2"}]


def test_documents_are_concatenated_across_sources():
    _tika, _vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value=None),
    ):
        conv = rc.RoutingConverter(_settings())
        out = conv.run(sources=["a.docx", "b.docx"], meta={})["documents"]

    assert len(out) == 2


def test_diagram_route_forwards_diagram_profile():
    _tika, vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value="vision-llm"),
    ):
        conv = rc.RoutingConverter(_settings())
        conv.run(sources=["x.docx"], meta={})

    _, kwargs = vision.run.call_args
    assert kwargs.get("profile") == "diagram"


def test_default_route_forwards_no_profile():
    tika, _vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value=None),
    ):
        conv = rc.RoutingConverter(_settings())
        conv.run(sources=["x.docx"], meta={})

    _, kwargs = tika.run.call_args
    assert "profile" not in kwargs


def test_custom_diagram_profile_is_honored():
    _tika, vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value="vision-llm"),
    ):
        conv = rc.RoutingConverter(_settings(extraction_router_diagram_profile="general"))
        conv.run(sources=["x.docx"], meta={})

    _, kwargs = vision.run.call_args
    assert kwargs.get("profile") == "general"


def test_profile_not_passed_to_non_profile_aware_diagram_engine():
    # Diagram engine pointed at a converter that doesn't accept a profile.
    plain = MagicMock(name="plain")
    plain.accepts_profile = False
    plain.run.return_value = {"documents": [Document(content="x")]}
    default = MagicMock(name="default")
    default.accepts_profile = False
    default.run.return_value = {"documents": [Document(content="d")]}

    def build(_s, engine_override=None):
        return {"kreuzberg": default, "docling": plain}[engine_override]

    s = _settings(
        extraction_router_default="kreuzberg",
        extraction_router_diagram_engine="docling",
    )
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value="docling"),
    ):
        conv = rc.RoutingConverter(s)
        conv.run(sources=["x.docx"], meta={})

    _, kwargs = plain.run.call_args
    assert "profile" not in kwargs


def test_debug_logs_routing_decision(caplog):
    _tika, _vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        patch.object(rc, "detect_engine", return_value="vision-llm"),
        caplog.at_level(logging.DEBUG, logger="app.pipelines.routing_converter"),
    ):
        rc.RoutingConverter(_settings()).run(sources=["Diagram.docx"], meta={})

    assert "routing Diagram.docx -> engine=vision-llm profile=diagram" in caplog.text


def test_invalid_diagram_profile_raises_at_construction():
    _tika, _vision, build = _fakes()
    with (
        patch.object(rc, "build_converter", side_effect=build),
        pytest.raises(ValueError, match="not a known"),
    ):
        rc.RoutingConverter(_settings(extraction_router_diagram_profile="banana"))
