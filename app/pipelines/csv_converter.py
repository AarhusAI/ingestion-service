"""In-process CSV/TSV converter.

Serializes each data row as ``column: value`` lines (rows separated by a
blank line) so every chunk keeps its column context — flat-text extraction
loses the header once the splitter cuts past the first rows. The splitter
recognises ``meta.extraction_engine == "csv"`` and cuts only at row
boundaries with zero overlap.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from haystack import Document, component

from app.pipelines.converters import meta_for
from app.pipelines.errors import ExtractionError


def _norm(cell: str) -> str:
    return " ".join(cell.split())


def csv_to_text(text: str) -> str:
    first_line = text.split("\n", 1)[0]
    delimiter = max((";", ",", "\t"), key=first_line.count)
    if not first_line.count(delimiter):
        delimiter = ","
    rows = csv.reader(io.StringIO(text), delimiter=delimiter)
    # ponytail: row 1 is always the header — add a header-row detector if
    # title-line exports appear.
    header = [_norm(c) or f"column {i + 1}" for i, c in enumerate(next(rows, []))]
    blocks = []
    for row in rows:
        cells = [_norm(c) for c in row]
        if any(cells):
            pairs = zip(header, cells, strict=False)  # cells past the header are dropped
            blocks.append("\n".join(f"{key}: {val}" for key, val in pairs if val))
    return "\n\n".join(b for b in blocks if b)


@component
class CsvConverter:
    """Turns ``.csv``/``.tsv`` files into one row-serialized Document each."""

    @component.output_types(documents=list[Document])
    def run(self, sources: list[str], meta: dict | list[dict] | None = None) -> dict:
        docs: list[Document] = []
        for i, source in enumerate(sources):
            path = Path(source)
            try:
                # ponytail: whole file read into memory — stream if
                # multi-hundred-MB CSVs show up.
                raw = path.read_bytes()
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw.decode("cp1252")
                content = csv_to_text(text)
            except (OSError, UnicodeDecodeError, csv.Error) as e:
                raise ExtractionError(f"CSV extraction failed for {path.name}: {e}") from e
            docs.append(
                Document(content=content, meta={**meta_for(meta, i), "extraction_engine": "csv"})
            )
        return {"documents": docs}
