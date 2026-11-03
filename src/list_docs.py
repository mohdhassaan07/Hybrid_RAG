"""Read indexed chunk metadata from the BM25 index.
Group chunks by source document.
Return source, page, section, and chunk counts."""


import pickle
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()


def list_indexed_documents():
    index_path = Path(__file__).parent.parent / "bm25_index.pkl"
    if not index_path.exists():
        return {"error": "bm25_index.pkl not found"}
    with open(index_path, "rb") as f:
        saved = pickle.load(f)
    docs = saved.docs if hasattr(saved, "docs") else (saved.get("docs", []) if isinstance(saved, dict) else [])
    chunks_by_source = defaultdict(list)
    for doc in docs:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        section = doc.metadata.get("section") or "Unknown section"
        chunks_by_source[(src, page, section)].append(doc)
    result = []
    for (src, page, section), chunks in chunks_by_source.items():
        result.append({
            "source": src,
            "page": page,
            "section": section,
            "chunk_count": len(chunks)
        })
    return {"documents": result}


def build_vector_index():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="hybrid_rag",
        embedding=embeddings,
        url="http://localhost:6333",
    )
    return vector_store


if __name__ == "__main__":
    indexed = list_indexed_documents()
    vector_store = build_vector_index()
    print("Vector store: ", vector_store)
    # print("[OK] Indexed documents")
    # for d in indexed["documents"]:
    #     print("- {source} (page {page}, {chunk_count} chunks)".format(**d))