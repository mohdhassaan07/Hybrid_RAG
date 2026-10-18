from document_loader import load_document
from chunker import chunk_docs
from pathlib import Path

pdf_path = Path(__file__).parent.parent / "six_projects-1-8.pdf"

docs = load_document(str(pdf_path))

fixed_size_chunks,markdown_chunks = chunk_docs(docs)

# print("Fixed Size")
# print(fixed_size_chunks[0])
# print("\n")
# print("Markdown")
# print(markdown_chunks[0])

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# store chunk in qdrant vector store
vector_store = QdrantVectorStore.from_documents(
    documents=[fixed_size_chunks,markdown_chunks],
    collection_name="Hybrid RAG",
    embedding=embeddings,
    url="http://localhost:6333",
)