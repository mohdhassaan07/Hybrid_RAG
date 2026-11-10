from src.document_loader import load_document, _extract_section_heading
from pathlib import Path
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter)
from dotenv import load_dotenv
load_dotenv()


def _enrich_metadata(chunks, strategy_name: str):
    for idx, chunk in enumerate(chunks):
        # Prefer an explicitly assigned heading key (set by MarkdownHeaderTextSplitter).
        # Fall back to the page-level "section" propagated from the loader.
        # Last resort: scan the chunk's own text for a heading pattern — this
        # rescues fixed-size chunks whose parent page had no detectable heading
        # but whose content starts with one (common in PDFs split mid-page).
        heading = (
            chunk.metadata.get("section_heading")
            or chunk.metadata.get("section")
            or chunk.metadata.get("subsection")
            or _extract_section_heading(chunk.page_content)  # scan chunk text
            or ""
        )
        source = chunk.metadata.get(
            "source") or chunk.metadata.get("source_document") or ""

        chunk.metadata.update({
            "source": source,
            "chunk_index": idx,
            "section": heading,
            "chunking_strategy": strategy_name,
            "character_count": len(chunk.page_content),
        })
    return chunks


def chunk_docs(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview")

    # 1. Fixed-size chunking
    # RecursiveCharacterTextSplitter propagates parent metadata (incl. "section")
    # to every child chunk automatically.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    fixed_size_chunks = splitter.split_documents(docs)
    fixed_size_chunks = _enrich_metadata(fixed_size_chunks, "fixed_size")

    # 2. Markdown header chunking
    # MarkdownHeaderTextSplitter only sets "section"/"subsection" when the
    # content actually contains "# ..." / "## ..." lines.  For non-Markdown
    # sources (PDF, plain-text) the headers won't be present, so we fall back
    # to the "section" detected by document_loader (already in doc.metadata).
    headers_to_split_on = [
        ("#", "section"),
        ("##", "subsection"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)

    markdown_chunks = []
    for doc in docs:
        splits = md_splitter.split_text(doc.page_content)
        for split in splits:
            # doc.metadata carries the loader-detected "section"; split.metadata
            # may override it with a Markdown-header-derived value when present.
            split.metadata = {**doc.metadata, **split.metadata}
            markdown_chunks.append(split)

    markdown_chunks = _enrich_metadata(markdown_chunks, "markdown_header")

    # 3. Semantic chunking (optional)
    # semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_amount=0.8)
    # semantic_chunks = semantic_splitter.split_documents(docs)
    # semantic_chunks = _enrich_metadata(semantic_chunks, "semantic")

    return fixed_size_chunks, markdown_chunks
