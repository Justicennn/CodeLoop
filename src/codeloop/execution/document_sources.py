"""Bounded local PDF and DOCX text-source adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError

DocumentType = Literal["pdf", "docx"]
DocumentUnitKind = Literal["page", "paragraph", "table"]


class DocumentSourceError(Exception):
    """A safe, structured document extraction failure."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


@dataclass(frozen=True)
class DocumentUnit:
    kind: DocumentUnitKind
    index: int
    text: str


@dataclass(frozen=True)
class DocumentUnitSpan:
    kind: DocumentUnitKind
    index: int
    start: int
    end: int

    def locator(self) -> dict[str, str | int]:
        return {"kind": self.kind, "index": self.index}


@dataclass(frozen=True)
class ExtractedDocument:
    document_type: DocumentType
    text: str
    spans: tuple[DocumentUnitSpan, ...]

    def locator_at(self, cursor: int) -> dict[str, str | int] | None:
        for span in self.spans:
            if span.start <= cursor < span.end:
                return span.locator()
        return None


class DocumentAdapter(Protocol):
    document_type: DocumentType
    separator: str

    def extract_units(self, path: Path) -> tuple[DocumentUnit, ...]: ...


class PdfDocumentAdapter:
    document_type: DocumentType = "pdf"
    separator = "\n\n"

    def extract_units(self, path: Path) -> tuple[DocumentUnit, ...]:
        try:
            reader = PdfReader(path)
            units = tuple(
                DocumentUnit(
                    kind="page",
                    index=index,
                    text=_normalize_unit(page.extract_text() or ""),
                )
                for index, page in enumerate(reader.pages, start=1)
            )
        except FileNotDecryptedError as exc:
            raise DocumentSourceError(
                "document_text_unavailable",
                f"Encrypted PDF text is unavailable: {path.name}",
            ) from exc
        except Exception as exc:
            raise DocumentSourceError(
                "malformed_document",
                f"PDF document could not be parsed: {path.name}",
            ) from exc
        return units


class DocxDocumentAdapter:
    document_type: DocumentType = "docx"
    separator = "\n"

    def extract_units(self, path: Path) -> tuple[DocumentUnit, ...]:
        try:
            document = Document(path)
            units: list[DocumentUnit] = []
            for index, block in enumerate(document.iter_inner_content(), start=1):
                if isinstance(block, Paragraph):
                    kind: DocumentUnitKind = "paragraph"
                    text = _normalize_unit(block.text)
                elif isinstance(block, Table):
                    kind = "table"
                    text = _table_text(block)
                else:  # pragma: no cover - python-docx documents the two variants.
                    continue
                units.append(DocumentUnit(kind=kind, index=index, text=text))
        except Exception as exc:
            raise DocumentSourceError(
                "malformed_document",
                f"DOCX document could not be parsed: {path.name}",
            ) from exc
        return tuple(units)


DOCUMENT_ADAPTERS: dict[str, DocumentAdapter] = {
    ".pdf": PdfDocumentAdapter(),
    ".docx": DocxDocumentAdapter(),
}


def extract_document(path: Path) -> ExtractedDocument:
    """Extract one supported document into a deterministic normalized stream."""
    adapter = DOCUMENT_ADAPTERS.get(path.suffix.casefold())
    if adapter is None:
        raise DocumentSourceError(
            "unsupported_document_type",
            "read_document supports only .pdf and .docx files.",
        )

    units = adapter.extract_units(path)
    if not units or not any(unit.text.strip() for unit in units):
        raise DocumentSourceError(
            "document_text_unavailable",
            f"Document contains no extractable text: {path.name}",
        )

    pieces: list[str] = []
    spans: list[DocumentUnitSpan] = []
    cursor = 0
    last_index = len(units) - 1
    for unit_index, unit in enumerate(units):
        piece = unit.text
        if unit_index != last_index:
            piece += adapter.separator
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        spans.append(
            DocumentUnitSpan(
                kind=unit.kind,
                index=unit.index,
                start=start,
                end=cursor,
            )
        )
    return ExtractedDocument(
        document_type=adapter.document_type,
        text="".join(pieces),
        spans=tuple(spans),
    )


def _normalize_unit(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _table_text(table: Table) -> str:
    return "\n".join(
        "\t".join(_normalize_unit(cell.text) for cell in row.cells)
        for row in table.rows
    ).strip()
