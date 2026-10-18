"""
Document ingestion pipeline for hybridRAG.

Supports .txt and .pdf files.  Every loaded chunk is a LangChain Document
whose ``page_content`` is clean, normalised plain-text and whose ``metadata``
dict carries:

    source      – absolute path of the originating file
    page        – 1-based page number (PDF) or logical block number (TXT)
    section     – nearest preceding section heading, or "" if none found
"""

import re
import unicodedata
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

def load_pdf(path: str) -> List[Document]:
    """Extract text from every page of a PDF.

    Each page becomes one :class:`~langchain_core.documents.Document`.
    Section headings are detected heuristically from the raw text before
    normalisation so that casing information is still available.
    """
    source = str(Path(path).resolve())
    reader = PdfReader(path)
    docs: List[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        section = _extract_section_heading(raw_text)
        clean_text = normalize(raw_text)

        if not clean_text:
            continue  # skip blank pages

        docs.append(
            Document(
                page_content=clean_text,
                metadata={
                    "source": source,
                    "page": page_num,
                    "section": section,
                },
            )
        )

    return docs

# ---------------------------------------------------------------------------
# Plain-text loader
# ---------------------------------------------------------------------------

def load_text(path: str) -> List[Document]:
    """Load a plain-text file and split it into logical page blocks.

    The file is split on sequences of two or more blank lines (common in
    books, articles, and README-style documents).  Each block becomes one
    Document.  A synthetic "page" counter is assigned to each block so the
    metadata schema stays consistent with the PDF loader.
    """
    source = str(Path(path).resolve())
    raw_content = Path(path).read_text(encoding="utf-8", errors="replace")

    # Split into logical blocks on 2+ consecutive blank lines
    blocks = re.split(r"\n{2,}", raw_content)

    docs: List[Document] = []
    page_num = 0

    for block in blocks:
        clean_block = normalize(block)
        if not clean_block:
            continue  # skip whitespace-only blocks

        page_num += 1
        section = _extract_section_heading(block)

        docs.append(
            Document(
                page_content=clean_block,
                metadata={
                    "source": source,
                    "page": page_num,
                    "section": section,
                },
            )
        )

    return docs


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Return clean, normalised plain-text suitable for embedding / indexing.

    Steps applied (in order):

    1. Unicode normalisation (NFC) so accented characters are consistent.
    2. Replace non-breaking spaces and soft hyphens with ASCII equivalents.
    3. Collapse all horizontal whitespace runs (spaces, tabs, form-feeds)
       into a single space *while preserving* newlines / paragraph structure.
    4. Strip leading / trailing whitespace from each line.
    5. Collapse runs of 3+ blank lines into a single blank line.
    6. Strip the overall string.

    Casing is intentionally *preserved* so downstream models can use it.
    """
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Replace typographic variants
    text = text.replace("\u00a0", " ")   # non-breaking space -> space
    text = text.replace("\u00ad", "-")   # soft hyphen -> hyphen

    # 3. Collapse horizontal whitespace within each line (keep newlines)
    text = re.sub(r"[ \t\f\r]+", " ", text)

    # 4. Strip each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # 5. Collapse 3+ consecutive blank lines into one blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Final strip
    return text.strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Heading patterns evaluated top-to-bottom; first match wins.
_HEADING_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Markdown-style  # H1 / ## H2 …
    ("markdown",  re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)),
    # Numbered section  "1.2  Introduction" / "3. Results"
    ("numbered",  re.compile(
        r"^(\d+(?:\.\d+)*\.?\s{1,4}[A-Z][^\n]{2,60})$", re.MULTILINE
    )),
    # ALL-CAPS short line  (3-80 chars, no sentence-ending punctuation mid-line)
    ("allcaps",   re.compile(r"^([A-Z][A-Z\s\-:]{2,79})$", re.MULTILINE)),
    # Title-Case line that is short and ends without sentence punctuation
    ("titlecase", re.compile(
        r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,10})$", re.MULTILINE
    )),
]


def _extract_section_heading(raw_text: str) -> str:
    """Return the *first* plausible section heading found in *raw_text*.

    Uses the un-normalised text so casing is intact for all pattern matches.
    Returns an empty string when no heading is detected.
    """
    for _name, pattern in _HEADING_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            heading = match.group(1) if match.lastindex else match.group(0)
            return heading.strip()
    return ""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(path: str) -> List[Document]:
    """Load *path* and return a list of normalised LangChain Documents.

    Dispatches to :func:`load_pdf` or :func:`load_text` based on the file
    extension.  Raises ``ValueError`` for unsupported extensions.
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_text(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            "Only .pdf and .txt files are accepted."
        )


if __name__ == "__main__":
    # Quick smoke-test – remove when pipeline.py is wired up
    pdf_path = Path(__file__).parent.parent / "ai_projects.pdf"
    docs = load_document(str(pdf_path))
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:120])
        print("---")