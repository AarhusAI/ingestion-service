"""RoutingConverter delegation (app/pipelines/routing_converter.py).

``build_converter`` and ``detect_engine`` are patched at the module symbol so no
real converter is constructed and the routing decision is forced per test.
"""

from unittest.mock import MagicMock, patch

from haystack import Document

from app.config import Settings
from app.pipelines import routing_converter as rc


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, api_key="a" * 32, extraction_engine="auto", **overrides)


def _fakes():
    """Return (tika, vision) converter mocks + a build_converter side_effect."""
    tika = MagicMock(name="tika")
    vision = MagicMock(name="vision")
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
