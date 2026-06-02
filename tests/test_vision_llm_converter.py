"""VisionLLMConverter (app/pipelines/vision_llm_converter.py).

The renderer and the chat-completions HTTP call are patched at the module symbol
so no Gotenberg / pypdfium2 / VLM endpoint is touched.
"""

import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.pipelines.errors import ExtractionError
from app.pipelines.vision_llm_converter import VisionLLMConverter

_RENDER = "app.pipelines.vision_llm_converter.render_to_pngs"
_POST = "app.pipelines.vision_llm_converter.httpx.post"


def _conv(**overrides):
    base = dict(api_base_url="http://vlm/v1", api_key="SUPERSECRETKEY", model="m")
    base.update(overrides)
    return VisionLLMConverter(**base)


def _ok_response(content="## Konsulent\n- step", finish_reason="stop"):
    r = MagicMock()
    r.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    }
    r.raise_for_status.return_value = None
    return r


def test_happy_path_returns_one_markdown_document():
    with (
        patch(_RENDER, return_value=[b"png1", b"png2"]),
        patch(_POST, return_value=_ok_response("## Lane\n- x")) as post,
    ):
        out = _conv().run(sources=["f.docx"], meta={"file_id": "f1"})["documents"]

    assert len(out) == 1
    assert out[0].content == "## Lane\n- x"
    assert out[0].meta["file_id"] == "f1"
    assert out[0].meta["extractor"] == "vision-llm"
    assert out[0].meta["page_count"] == 2
    # Bearer auth set; both page images sent.
    _, kwargs = post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer SUPERSECRETKEY"
    image_parts = [p for p in kwargs["json"]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 2


def test_request_meta_wins_over_converter_meta():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()),
    ):
        out = _conv().run(sources=["f.docx"], meta={"extractor": "override"})["documents"]
    assert out[0].meta["extractor"] == "override"


def test_meta_none_yields_only_converter_meta():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()),
    ):
        out = _conv().run(sources=["f.docx"], meta=None)["documents"]
    assert out[0].meta == {
        "extractor": "vision-llm",
        "vision_profile": "general",
        "page_count": 1,
    }


def _system_text(post_mock) -> str:
    _, kwargs = post_mock.call_args
    return kwargs["json"]["messages"][0]["content"]


def test_default_profile_used_when_none_given():
    # _conv() defaults to "general" — its system prompt is transcription-flavoured.
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        out = _conv().run(sources=["f.docx"], meta=None)["documents"]
    assert out[0].meta["vision_profile"] == "general"
    assert "transcri" in _system_text(post).lower()


def test_explicit_profile_selects_that_prompt():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        out = _conv().run(sources=["f.docx"], meta=None, profile="diagram")["documents"]
    assert out[0].meta["vision_profile"] == "diagram"
    assert "swim-lane" in _system_text(post).lower()


def test_unknown_default_profile_raises_at_construction():
    with pytest.raises(ValueError, match="not a known profile"):
        _conv(default_profile="banana")


def test_unknown_run_profile_falls_back_to_default():
    # Defensive: an unknown profile at run() must not crash; it uses the default.
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        out = _conv(default_profile="ocr").run(
            sources=["f.docx"], meta=None, profile="banana"
        )["documents"]
    assert out[0].meta["vision_profile"] == "ocr"
    assert "ocr" in _system_text(post).lower() or "scan" in _system_text(post).lower()


def _user_text_parts(post_mock) -> list[str]:
    _, kwargs = post_mock.call_args
    content = kwargs["json"]["messages"][1]["content"]
    return [p["text"] for p in content if p["type"] == "text"]


def test_grounding_adds_one_user_text_part_before_images():
    with (
        patch(_RENDER, return_value=[b"png1", b"png2"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        _conv().run(
            sources=["f.docx"],
            meta=None,
            profile="diagram-topology",
            grounding="Sagsvurdering\n- Statusattest",
        )
    _, kwargs = post.call_args
    content = kwargs["json"]["messages"][1]["content"]
    types = [p["type"] for p in content]
    # prompt text, then the grounding text, then the images — grounding precedes images.
    assert types == ["text", "text", "image_url", "image_url"]
    grounding_part = content[1]["text"]
    assert "AUTHORITATIVE TEXT" in grounding_part
    assert "Sagsvurdering\n- Statusattest" in grounding_part


def test_no_grounding_leaves_payload_unchanged():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        _conv().run(sources=["f.docx"], meta=None)
    # Exactly one text part (the profile prompt) and one image — no grounding part.
    assert len(_user_text_parts(post)) == 1


def test_blank_grounding_is_ignored():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        _conv().run(sources=["f.docx"], meta=None, grounding="   ")
    assert len(_user_text_parts(post)) == 1


def test_http_error_raises_extraction_error():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, side_effect=httpx.HTTPError("endpoint down")),
        pytest.raises(ExtractionError),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_renderer_error_propagates():
    with (
        patch(_RENDER, side_effect=ExtractionError("gotenberg convert failed for f.docx")),
        pytest.raises(ExtractionError),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_malformed_response_no_choices_raises():
    r = MagicMock()
    r.json.return_value = {"unexpected": "shape"}
    r.raise_for_status.return_value = None
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=r),
        pytest.raises(ExtractionError, match="empty content"),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_non_json_body_raises():
    r = MagicMock()
    r.json.side_effect = ValueError("no json")
    r.raise_for_status.return_value = None
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=r),
        pytest.raises(ExtractionError, match="non-JSON"),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_api_key_never_appears_in_error_message():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, side_effect=httpx.HTTPError("connect failed")),
        pytest.raises(ExtractionError) as excinfo,
    ):
        _conv(api_key="SUPERSECRETKEY").run(sources=["f.docx"], meta=None)
    assert "SUPERSECRETKEY" not in str(excinfo.value)


def test_images_override_skips_render_and_sends_those_bytes():
    with (
        patch(_RENDER) as render,
        patch(_POST, return_value=_ok_response("## Fig")) as post,
    ):
        out = _conv().run(
            sources=["f.docx"],
            meta=None,
            profile="figure",
            images_override=[b"BLIP1", b"BLIP2"],
        )["documents"]
    render.assert_not_called()
    assert out[0].meta["page_count"] == 2  # page_count == figure-image count here
    _, kwargs = post.call_args
    image_parts = [p for p in kwargs["json"]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 2
    expected = "data:image/png;base64," + base64.b64encode(b"BLIP1").decode("ascii")
    assert image_parts[0]["image_url"]["url"] == expected


def test_empty_content_with_override_returns_empty_not_error():
    # The figure profile is told to output nothing when there is no figure; on the
    # override (figure) path that must yield "" so the caller ships native body.
    with (
        patch(_RENDER) as render,
        patch(_POST, return_value=_ok_response("   ")),
    ):
        out = _conv().run(
            sources=["f.docx"], meta=None, profile="figure", images_override=[b"PHOTO"]
        )["documents"]
    render.assert_not_called()
    assert out[0].content == ""


def test_empty_content_without_override_still_raises():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response("   ")),
        pytest.raises(ExtractionError, match="empty content"),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_truncated_output_raises():
    # finish_reason=length → hard failure even though partial content is present.
    truncated = _ok_response("partial markdown cut off", finish_reason="length")
    with (
        patch(_RENDER, return_value=[b"png1", b"png2"]),
        patch(_POST, return_value=truncated),
        pytest.raises(ExtractionError, match="truncated"),
    ):
        _conv().run(sources=["f.docx"], meta=None)


def test_truncated_output_raises_even_on_override_path():
    # Truncation is distinct from allow_empty: a cut-off figure response is a failure.
    with (
        patch(_RENDER) as render,
        patch(_POST, return_value=_ok_response("## Fig partial", finish_reason="length")),
        pytest.raises(ExtractionError, match="truncated"),
    ):
        _conv().run(sources=["f.docx"], meta=None, profile="figure", images_override=[b"BLIP"])
    render.assert_not_called()


def test_max_tokens_default_and_configurable():
    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post,
    ):
        _conv().run(sources=["f.docx"], meta=None)
    assert post.call_args.kwargs["json"]["max_tokens"] == 16384

    with (
        patch(_RENDER, return_value=[b"png"]),
        patch(_POST, return_value=_ok_response()) as post2,
    ):
        _conv(max_tokens=2048).run(sources=["f.docx"], meta=None)
    assert post2.call_args.kwargs["json"]["max_tokens"] == 2048
