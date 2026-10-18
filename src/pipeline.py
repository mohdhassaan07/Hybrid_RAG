import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from chunker import chunk_docs
from document_loader import load_document

load_dotenv()


def build_qdrant_index(chunks, embeddings, collection_name="hybrid_rag", url="http://localhost:6333"):
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


if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "six_projects-1-8.pdf"
    bm25_save_path = Path(__file__).parent.parent / "bm25_index.pkl"

    # 1. Load document
    docs = load_document(str(pdf_path))

    # 2. Chunk documents
    fixed_size_chunks, markdown_chunks = chunk_docs(docs)
    all_chunks = fixed_size_chunks + markdown_chunks

    # 3. Assign deterministic unique IDs so dense and sparse indexes stay in sync
    for idx, chunk in enumerate(all_chunks):
        chunk_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{chunk.metadata.get('source')}_{chunk.metadata.get('chunking_strategy')}_{chunk.metadata.get('chunk_index', idx)}"
        ))
        chunk.id = chunk_id
        chunk.metadata["id"] = chunk_id

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    print(f"Indexing {len(all_chunks)} chunks into Qdrant and BM25 concurrently...")

    # 4. Build both indexes in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        qdrant_future = executor.submit(build_qdrant_index, all_chunks, embeddings)
        bm25_future = executor.submit(build_bm25_index, all_chunks, bm25_save_path)

        vector_store = qdrant_future.result()
        bm25_retriever = bm25_future.result()

    print("Vector store and BM25 index built in parallel and synchronized successfully!")
    print(f"BM25 index saved to: {bm25_save_path}")
