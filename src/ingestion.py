import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.chunker import chunk_docs
from src.document_loader import load_document, normalize, _extract_section_heading

load_dotenv()

collection_name = "hybrid_rag2"
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


def build_qdrant_index(chunks, embeddings, collection_name=collection_name, url="http://localhost:6333"):
    """Build or update the dense Qdrant vector store with explicit chunk IDs."""
    ids = [chunk.id for chunk in chunks]
    return QdrantVectorStore.from_documents(
        documents=chunks,
        ids=ids,
        collection_name=collection_name,
        embedding=embeddings,
        url=url,
    )


def build_bm25_index(chunks, save_path: Path | None = None):
    """Build the sparse BM25 keyword index over the exact same chunks and persist it."""
    bm25_retriever = BM25Retriever.from_documents(documents=chunks)
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
    return bm25_retriever


SIMILARITY_THRESHOLD = 0.95


def drop_near_duplicates(chunks, vector_store, embeddings):
    accepted = []

    for chunk in chunks:
        vector = embeddings.embed_query(chunk.page_content)

        # k=1 is enough: only the closest existing chunk matters.
        matches = vector_store.similarity_search_with_score_by_vector(
            embedding=vector,
            k=1,
        )

        if matches and matches[0][1] > SIMILARITY_THRESHOLD:
            closest_chunk, score = matches[0]
            print(f"Skipping near-duplicate; cosine similarity={score:.3f}")
            continue

        accepted.append(chunk)

    return accepted


def _ingest_docs(docs: List[Document]) -> dict:
    """Shared indexing logic for both file and raw-text ingestion paths.

    Takes already-loaded Documents, chunks them, deduplicates against the
    existing Qdrant collection, assigns deterministic IDs, and upserts into
    both Qdrant (dense) and BM25 (sparse).

    Returns:
        A dict with ``indexed`` (number of new chunks added) and ``message``.
    """
    bm25_save_path = Path(__file__).parent.parent / "bm25_index.pkl"

    # 1. Chunk
    fixed_size_chunks, markdown_chunks = chunk_docs(docs)
    all_chunks = fixed_size_chunks + markdown_chunks

    # 2. Deduplicate against existing index
    client = QdrantClient(url="http://localhost:6333")
    existing_collections = [
        c.name for c in client.get_collections().collections]
    if collection_name in existing_collections:
        existing_store = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            embedding=embeddings,
            url="http://localhost:6333",
        )
        all_chunks = drop_near_duplicates(
            all_chunks, existing_store, embeddings)

    if not all_chunks:
        print("All chunks are duplicates of existing documents. Nothing new to index.")
        return {"indexed": 0, "message": "All chunks are duplicates. Nothing new to index."}

    # 3. Deterministic IDs
    for idx, chunk in enumerate(all_chunks):
        chunk_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{chunk.metadata.get('source')}_{chunk.metadata.get('chunking_strategy')}_{chunk.metadata.get('chunk_index', idx)}",
        ))
        chunk.id = chunk_id
        chunk.metadata["id"] = chunk_id

    # 4. Index in parallel
    print(
        f"Indexing {len(all_chunks)} chunks into Qdrant and BM25 concurrently...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        qdrant_future = executor.submit(
            build_qdrant_index, all_chunks, embeddings)
        bm25_future = executor.submit(
            build_bm25_index, all_chunks, bm25_save_path)
        qdrant_future.result()
        bm25_future.result()

    print("Vector store and BM25 index built in parallel and synchronized successfully!")
    print(f"BM25 index saved to: {bm25_save_path}")
    return {"indexed": len(all_chunks), "message": "Indexed successfully."}


def ingest(pdf_path: str):
    """Ingest a PDF or .txt file by path."""
    docs = load_document(str(pdf_path))
    _ingest_docs(docs)


def ingest_text(text: str, source_label: str = "inline_text") -> dict:
    """Ingest a raw text string directly (no file needed).

    Args:
        text:         The raw text content to index.
        source_label: An identifier stored in chunk metadata as ``source``.
                      Defaults to ``"inline_text"``; pass a document title or
                      any descriptive name so retrieved chunks are traceable.

    Returns:
        A dict with ``indexed`` (chunk count) and ``message``.
    """
    clean = normalize(text)
    # use raw text for heading detection
    section = _extract_section_heading(text)
    docs = [
        Document(
            page_content=clean,
            metadata={"source": source_label, "page": 1, "section": section},
        )
    ]
    return _ingest_docs(docs)
