"""Prompt-profile registry (app/pipelines/vision_profiles.py)."""

import pytest

from app.pipelines.vision_profiles import KNOWN_PROFILES, get_profile

_SHIPPED = ("diagram", "diagram-topology", "general", "figure", "ocr")


def test_known_profiles_are_exactly_the_shipped_set():
    assert set(KNOWN_PROFILES) == set(_SHIPPED)


def test_get_profile_returns_each_shipped_profile():
    for name in _SHIPPED:
        prof = get_profile(name)
        assert prof.name == name
        # system is parameterised by language_hint; user is a non-empty string.
        assert callable(prof.system)
        assert isinstance(prof.user, str) and prof.user.strip()


def test_get_profile_unknown_raises():
    with pytest.raises(ValueError, match="Unknown vision-llm profile"):
        get_profile("banana")


def test_language_hint_is_injected_into_system_prompt():
    for name in _SHIPPED:
        assert "Klingon" in get_profile(name).system("Klingon")


def test_diagram_topology_profile_is_mermaid_only():
    prof = get_profile("diagram-topology")
    text = (prof.system("Danish") + prof.user).lower()
    assert "mermaid" in text
    # Graph only: the prose-scaffold instructions of the `diagram` profile
    # (lanes → ##, phases → ###) must be absent so the body stays native.
    assert "swim-lane" in prof.system("Danish").lower()
    assert "## <lane name>" not in prof.user


def test_diagram_profile_keeps_flowchart_anchors():
    prof = get_profile("diagram")
    assert "mermaid" in prof.user.lower()
    assert "swim-lane" in prof.system("Danish").lower()


def test_general_profile_is_faithful_transcription():
    prof = get_profile("general")
    text = (prof.system("Danish") + prof.user).lower()
    assert "transcri" in text  # transcribe / transcription
    assert "mermaid" not in text  # no flowchart scaffold


def test_figure_profile_is_scoped_to_the_embedded_figure():
    prof = get_profile("figure")
    sys = prof.system("Danish").lower()
    text = (prof.system("Danish") + prof.user).lower()
    # Reads pixels into a Mermaid graph like the diagram profiles …
    assert "mermaid" in text
    # … but explicitly ignores the surrounding prose (it comes from native text)
    # and emits nothing when there's no figure.
    assert "prose" in sys
    assert "nothing" in text
    prof = get_profile("ocr")
    text = (prof.system("Danish") + prof.user).lower()
    assert "ocr" in text or "scan" in text
    assert "mermaid" not in text
