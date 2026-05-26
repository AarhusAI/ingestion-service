"""Tests for the filename-suffix allow-list (sec.md Finding 8)."""

import pytest

from app.services.filenames import safe_suffix


@pytest.mark.parametrize(
    "name, expected",
    [
        # Allow-listed — casing preserved.
        ("report.pdf", ".pdf"),
        ("Report.PDF", ".PDF"),
        ("notes.md", ".md"),
        ("table.xlsx", ".xlsx"),
        ("image.JPEG", ".JPEG"),
        # Empty / missing extension → empty.
        ("", ""),
        ("nodot", ""),
        # Not on allow-list → empty (no leak of garbage into the tempfile path).
        ("dangerous.exe", ""),
        ("file.zip", ""),
        ("script.sh", ""),
        # Path separator embedded → falls outside the allow-list → empty.
        # Closes the FileNotFoundError-on-tempfile-creation path documented
        # in sec.md Finding 8.
        ("file.bar/baz", ""),
        ("file.foo/../etc/passwd", ""),
        # Very long unknown extension → empty (no unbounded /tmp suffix).
        ("file." + "A" * 200, ""),
        # Multi-dot — only the final segment is checked.
        ("archive.tar.gz", ""),
        ("backup.pdf.bak", ""),
        ("notes.foo.md", ".md"),
    ],
)
def test_safe_suffix(name, expected):
    assert safe_suffix(name) == expected
