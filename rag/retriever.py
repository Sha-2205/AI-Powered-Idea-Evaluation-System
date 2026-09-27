import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss

try:
    from rag.embeddings import get_embeddings
except ImportError:

    from embeddings import get_embeddings


def load_and_chunk_documents(doc_dir: str = "rag/documents") -> List[Dict[str, str]]:
    """
    Loads all .txt files from the documents directory and splits them
    into meaningful topic-based chunks.

    Args:
        doc_dir (str): Directory containing knowledge base .txt files.

    Returns:
        List[Dict[str, str]]: List of chunks with 'text' and 'source' metadata.
    """
    doc_path = Path(doc_dir)
    chunks = []

    if not doc_path.exists():
        raise FileNotFoundError(f"Documents directory '{doc_dir}' not found.")

    for file_path in sorted(doc_path.glob("*.txt")):
        filename = file_path.name
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        parts = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not parts:
            continue

        doc_title = parts[0]
        sections = parts[1:] if len(parts) > 1 else [doc_title]

        for section in sections:
           
            chunk_text = f"[{doc_title}] {section}"
            chunks.append({
                "text": chunk_text,
                "source": filename
            })

    return chunks


class SimpleRAGRetriever:
    """
    A beginner-friendly FAISS retriever that builds an in-memory
    similarity search index for knowledge base documents.
    """

    def __init__(self, doc_dir: str = "rag/documents"):
        self.doc_dir = doc_dir
        self.chunks: List[Dict[str, str]] = []
        self.index: faiss.Index = None
        self.build_index()

    def build_index(self) -> None:
        """
        Loads document chunks, computes embeddings, and populates the FAISS index.
        """
        self.chunks = load_and_chunk_documents(self.doc_dir)
        if not self.chunks:
            raise ValueError("No document chunks found to index.")

        chunk_texts = [chunk["text"] for chunk in self.chunks]
        embeddings = get_embeddings(chunk_texts)

       
        dim = embeddings.shape[1]

        
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Accepts a user query, converts it to an embedding, and retrieves
        the top_k most relevant chunks from FAISS.

        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Retrieved items with text, source, and score.
        """
        if not query or not query.strip():
            return []

        
        query_embedding = get_embeddings([query])

      
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                item = self.chunks[idx]
                results.append({
                    "text": item["text"],
                    "source": item["source"],
                    "score": float(score)
                })

        return results


if __name__ == "__main__":
    print("=" * 70)
    print("Building FAISS Index from rag/documents/ ...")
    retriever = SimpleRAGRetriever(doc_dir="rag/documents")
    print(f"Index built successfully with {len(retriever.chunks)} total chunks.")
    print("=" * 70)

   
    query_1 = "How can I evaluate whether customers really need my idea?"
    print(f"\n[Test 1] Query: '{query_1}'")
    print("-" * 70)
    results_1 = retriever.retrieve(query_1, top_k=3)
    for rank, res in enumerate(results_1, start=1):
        print(f"Rank {rank} | Source: {res['source']} | Similarity Score: {res['score']:.4f}")
        print(f"Content:\n{res['text']}\n")

   
    query_2 = "How technically difficult is the idea to build?"
    print("=" * 70)
    print(f"\n[Test 2] Query: '{query_2}'")
    print("-" * 70)
    results_2 = retriever.retrieve(query_2, top_k=3)
    for rank, res in enumerate(results_2, start=1):
        print(f"Rank {rank} | Source: {res['source']} | Similarity Score: {res['score']:.4f}")
        print(f"Content:\n{res['text']}\n")
    print("=" * 70)
