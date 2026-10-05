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
from itertools import islice
from pathlib import Path

from haystack import Document, component

from app.pipelines.converters import meta_for
from app.pipelines.errors import ExtractionError


def _norm(cell: str) -> str:
    return " ".join(cell.split())


# Tie order: Danish Excel exports use ";" with decimal commas, so ";" must win
# when "," is equally consistent (csv.Sniffer prefers "," on ties).
_CANDIDATES = (";", "\t", ",")


def _delimiter(text: str, tsv: bool) -> str:
    if tsv:
        return "\t"
    for d in _CANDIDATES:
        rows = islice(csv.reader(io.StringIO(text), delimiter=d), 20)
        widths = {len(r) for r in rows if any(r)}
        if len(widths) == 1 and widths.pop() > 1:
            return d
    first_line = text.split("\n", 1)[0]  # single column / ragged rows: header-line count
    delimiter = max(_CANDIDATES, key=first_line.count)
    return delimiter if first_line.count(delimiter) else ","


def csv_to_text(text: str, tsv: bool = False) -> str:
    first_line, _, rest = text.partition("\n")
    if first_line.strip().lower().startswith("sep=") and len(first_line.strip()) == 5:
        delimiter, text = first_line.strip()[4], rest  # Excel "sep=;" hint line
    else:
        delimiter = _delimiter(text, tsv)
    rows = csv.reader(io.StringIO(text), delimiter=delimiter)
    # Assumes row 1 is the header (no title-line detection).
    header = [_norm(c) or f"column {i + 1}" for i, c in enumerate(next(rows, []))]
    blocks = []
    for row in rows:
        cells = [_norm(c) for c in row]
        if any(cells):
            keys = header + [f"column {j + 1}" for j in range(len(header), len(cells))]
            pairs = zip(keys, cells, strict=False)  # short rows: missing cells are empty
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
                raw = path.read_bytes()
                if raw.startswith((b"\xff\xfe", b"\xfe\xff")):  # Excel "Unicode Text"
                    text = raw.decode("utf-16")
                else:
                    try:
                        text = raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text = raw.decode("cp1252", errors="replace")
                content = csv_to_text(text, tsv=path.suffix.lower() == ".tsv")
            except (OSError, UnicodeDecodeError, csv.Error) as e:
                raise ExtractionError(f"CSV extraction failed for {path.name}: {e}") from e
            docs.append(
                Document(content=content, meta={**meta_for(meta, i), "extraction_engine": "csv"})
            )
        return {"documents": docs}
