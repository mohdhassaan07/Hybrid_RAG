from pathlib import Path
import pickle
import re
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from rank_bm25 import BM25Okapi
from groq import Groq
load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "use", "what", "when", "where", "which", "who", "with",
}


def tokenize(text: str) -> list[str]:
    """Normalise text before keyword matching."""
    return [
        token
        for token in re.findall(r"\b[\w'-]+\b", text.lower())
        if token not in STOP_WORDS
    ]


def qdrant_retriever(query: str):
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name="hybrid_rag",
        embedding=embeddings,
        url="http://localhost:6333",
    )
    return vector_store.similarity_search(query, k=5)


def bm25_retriever(
    query: str,
    k: int = 5,
    min_score: float = 0.0,
    required_terms: set[str] | None = None,
):
    """Return only positively-scored keyword matches from bm25_index.pkl."""
    index_path = Path(__file__).parent.parent / "bm25_index.pkl"

    with open(index_path, "rb") as file:
        saved_index = pickle.load(file)

    chunks = saved_index.docs
    corpus = [tokenize(chunk.page_content) for chunk in chunks]
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = BM25Okapi(corpus).get_scores(query_tokens)
    required_terms = {term.lower() for term in (required_terms or set())}
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

    results = []
    for index, score in ranked:
        if score <= min_score:
            break
        if required_terms and not required_terms.issubset(set(corpus[index])):
            continue

        results.append((chunks[index], float(score)))
        if len(results) == k:
            break
    return results

def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc.metadata.get("id") or doc.id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retriever(
    query: str,
    k: int = 6,
    rerank_top_n: int = 20,
):
    """
    Hybrid retriever combining BM25 and vector search using Reciprocal Rank Fusion.
    """
    # Step 1: Run BM25 retrieval
    bm25_results = bm25_retriever(query, k=rerank_top_n)
    bm25_docs = [doc for doc, score in bm25_results]
    bm25_rankings = [bm25_docs] if bm25_docs else []
    
    # Step 2: Run vector retrieval
    vector_docs = qdrant_retriever(query)
    vector_rankings = [vector_docs] if vector_docs else []
    
    # Step 3: Combine rankings using RRF
    all_rankings = bm25_rankings + vector_rankings
    combined_scores = reciprocal_rank_fusion(all_rankings, k=60)
    
    # Step 4: Get top N unique documents
    final_results = []
    seen_ids = set()
    
    for doc_id, score in combined_scores:
        if doc_id in seen_ids:
            continue
        
        # Find the original document object
        doc = None
        if bm25_docs and doc_id in [doc.metadata['id'] for doc in bm25_docs]:
            for d in bm25_docs:
                if d.metadata['id'] == doc_id:
                    doc = d
                    break
        if not doc and vector_docs and doc_id in [doc.metadata['id'] for doc in vector_docs]:
            for d in vector_docs:
                if d.metadata['id'] == doc_id:
                    doc = d
                    break
        
        if doc:
            final_results.append(doc)
            seen_ids.add(doc_id)
        
        if len(final_results) == k:
            break
    
    return final_results

def cross_encoder_reranker(query, chunks):
    """a cross-encoder reranker (LLM-as-judge) that scores each
    chunk’s relevance to the actual question. Keep the top 3. """

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    scores = {}
    for i, chunk in enumerate(chunks):
        system_prompt = f"""
    rate the relevence of the following chunk to the given query on a scale of 1 to 5.
    your response should only be a number between 1 and 5.
    1. query: {query}
    2. chunk: {chunk.page_content}
    """
        res = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        relevence = res.choices[0].message.content.strip()
        scores[chunk.id] = relevence
        
    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    reranked_docs = []
    for doc_id, score in reranked:
        doc = next((c for c in chunks if c.metadata['id'] == doc_id), None)
        if doc:
            doc.metadata['relevence_score'] = score
            reranked_docs.append(doc)
    return reranked_docs
        
        
if __name__ == "__main__":
    query = input("Enter query: ")
    hybrid_results = hybrid_retriever(query)
    reranked_results = cross_encoder_reranker(query, hybrid_results)
    for i, doc in enumerate(reranked_results, 1):
        print(f"Hybrid Top {i}:")
        print(doc.metadata)
        print(doc.page_content)
        print("-" * 80)
